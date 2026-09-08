import json
import random
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import pytest
import yaml


tensorboard = types.ModuleType('torch.utils.tensorboard')
tensorboard.SummaryWriter = object
sys.modules.setdefault('torch.utils.tensorboard', tensorboard)
pycocotools = types.ModuleType('pycocotools')
pycocotools.__path__ = []
for name, class_name in [('mask', None), ('coco', 'COCO'), ('cocoeval', 'COCOeval')]:
    module = types.ModuleType(f'pycocotools.{name}')
    if class_name:
        setattr(module, class_name, type(class_name, (), {}))
    setattr(pycocotools, name, module)
    sys.modules.setdefault(f'pycocotools.{name}', module)
sys.modules.setdefault('pycocotools', pycocotools)
calflops = types.ModuleType('calflops')
calflops.calculate_flops = lambda *args, **kwargs: None
sys.modules.setdefault('calflops', calflops)

from engine.edgecrafter.decoder import ECTransformer
from engine.edgecrafter.denoising import get_contrastive_denoising_training_group
from engine.edgecrafter.utils import inverse_sigmoid
from engine.core.yaml_config import YAMLConfig
from engine.solver.ec_solver import ECSolver
from engine.solver.ec_engine import _apply_backbone_train_modes
from engine.solver.pareto_checkpoint import ParetoCheckpointManager
from engine.misc.dist_utils import atomic_save_on_master
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from tools.bee_e.eval_checkpoint_queue import (
    completed_with_matching_sha,
    epoch_paths,
)
from tools.bee_e.validate_formal_configs import validate_all


def _tiny_decoder(**kwargs):
    defaults = dict(
        num_classes=1,
        hidden_dim=16,
        num_queries=6,
        feat_channels=[16],
        feat_strides=[4],
        num_levels=1,
        num_points=2,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        num_denoising=0,
        num_keypoints=2,
        endpoint_decoder_type='decomposed',
        use_density_query=True,
        use_density_peaks=True,
        density_peak_ratio=0.5,
        use_pattern_query=True,
        num_query_patterns=4,
    )
    defaults.update(kwargs)
    return ECTransformer(**defaults)


def test_density_local_maxima_create_real_query_references():
    decoder = _tiny_decoder().train()
    density = torch.zeros(1, 1, 4, 4)
    density[0, 0, 0, 0] = 10
    density[0, 0, 0, 3] = 9
    density[0, 0, 3, 0] = 8
    p2 = torch.randn(1, 16, 4, 4)
    memory = p2.flatten(2).transpose(1, 2)
    content, references, _, _, diagnostics = decoder._get_decoder_input(
        memory,
        [[4, 4]],
        density_prior_flat=density.flatten(2).transpose(1, 2),
        density_prior=density,
        p2_feature=p2,
        domain_context=torch.tensor([[1.0, 0.0]]),
    )
    centers = references.sigmoid()[0, :3, :2]
    expected = torch.tensor([[0.125, 0.125], [0.875, 0.125], [0.125, 0.875]])
    assert torch.allclose(centers, expected, atol=1e-5)
    assert content.shape == (1, 6, 16)
    assert diagnostics['pred_pattern_weights'].shape == (1, 6, 4)
    assert diagnostics['query_spatial_coverage'].item() > 0


def test_pattern_router_uses_domain_context():
    decoder = _tiny_decoder(use_density_peaks=False).eval()
    memory = torch.randn(1, 16, 16)
    first = decoder._get_decoder_input(
        memory, [[4, 4]], domain_context=torch.tensor([[1.0, 0.0]])
    )[-1]['pred_pattern_weights']
    second = decoder._get_decoder_input(
        memory, [[4, 4]], domain_context=torch.tensor([[0.0, 1.0]])
    )[-1]['pred_pattern_weights']
    assert not torch.allclose(first, second)


def test_decomposed_endpoint_queries_use_shared_instance_attention_and_box_anchors():
    decoder = _tiny_decoder(endpoint_sampling_points=8).eval()
    endpoint = decoder.endpoint_decoder
    anchors = endpoint.base_sampling_offsets
    assert anchors.shape == (8, 2)
    for expected in (
        torch.tensor([0.0, 0.0]), torch.tensor([-0.5, 0.0]),
        torch.tensor([0.5, 0.0]), torch.tensor([0.0, -0.5]),
        torch.tensor([0.0, 0.5]),
    ):
        assert (anchors == expected).all(dim=1).any()
    instance_queries = torch.randn(1, 1, 6, 16)
    boxes = torch.tensor([[[[0.5, 0.5, 0.4, 0.2]] * 6]])
    features = [torch.randn(1, 16, 8, 8)]
    keypoints, visibility = endpoint(instance_queries, boxes, features)
    assert keypoints.shape == (1, 1, 6, 2, 2)
    assert visibility.shape == (1, 1, 6, 2)
    assert torch.isfinite(keypoints).all()
    assert endpoint.instance_sampling_attention_heads[0].out_features == 8


def test_keypoint_dn_is_scaled_swappable_and_bidirectionally_isolated():
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        'keypoints': torch.tensor([[[0.4, 0.5, 2.0], [0.6, 0.5, 2.0]]]),
    }]
    class_embed = torch.nn.Embedding(2, 8)
    _, _, attention, metadata = get_contrastive_denoising_training_group(
        targets,
        num_classes=1,
        num_queries=6,
        class_embed=class_embed,
        num_denoising=2,
        num_keypoints=2,
        keypoint_noise_scale=0.1,
        head_tail_swap_ratio=1.0,
    )
    dn_count = metadata['dn_num_split'][0]
    assert metadata['dn_keypoint_refs'].shape == (1, dn_count, 2, 2)
    assert metadata['dn_head_tail_swap_count'].item() == dn_count
    assert attention[dn_count:, :dn_count].all()
    assert attention[:dn_count, dn_count:].all()


def test_pareto_manifest_never_collapses_to_bbox_only(tmp_path):
    manager = ParetoCheckpointManager(
        tmp_path, {'bbox_ap': 'max', 'pose_nme': 'min'}
    )
    stats = [
        ([0.30], 0.20),
        ([0.40], 0.25),
        ([0.20], 0.30),
    ]
    for epoch, (bbox, nme) in enumerate(stats):
        checkpoint = tmp_path / f'checkpoint{epoch:04}.pth'
        checkpoint.write_bytes(f'epoch={epoch}'.encode())
        manager.record(epoch, checkpoint, {
            'coco_eval_bbox': bbox,
            'pose_all_nme_bbox_diagonal': nme,
        })
    manifest = yaml.safe_load((tmp_path / 'pareto_frontier.json').read_text())
    assert {item['epoch'] for item in manifest['frontier']} == {0, 1}
    assert manifest['selected']['epoch'] in {0, 1}
    epoch_record = yaml.safe_load(
        (tmp_path / 'checkpoint_metrics' / 'epoch0000.json').read_text()
    )
    assert epoch_record['eval_stats']['coco_eval_bbox'] == [0.3]
    assert epoch_record['eval_stats']['pose_all_nme_bbox_diagonal'] == 0.2
    assert not (tmp_path / 'best.pth').exists()

    resumed = ParetoCheckpointManager(
        tmp_path, {'bbox_ap': 'max', 'pose_nme': 'min'}
    )
    assert [record['epoch'] for record in resumed.records] == [0, 1, 2]
    checkpoint = tmp_path / 'checkpoint0003.pth'
    checkpoint.write_bytes(b'epoch=3')
    selected = resumed.record(3, checkpoint, {
        'coco_eval_bbox': [0.35],
        'pose_all_nme_bbox_diagonal': 0.18,
    })
    assert selected['epoch'] in {0, 1, 3}


def test_atomic_checkpoint_publishes_ready_after_complete_file(tmp_path):
    checkpoint = tmp_path / 'checkpoint0000.pth'
    atomic_save_on_master({'model': {'weight': torch.tensor([1.0])}}, checkpoint, True)
    assert checkpoint.is_file()
    assert checkpoint.with_suffix('.ready').read_text(encoding='utf-8') == 'ready\n'
    assert not (tmp_path / 'checkpoint0000.pth.tmp').exists()
    assert torch.load(checkpoint, weights_only=True)['model']['weight'].item() == 1.0


def test_async_done_marker_is_guarded_by_checkpoint_sha(tmp_path):
    paths = epoch_paths(tmp_path, 0)
    paths['checkpoint'].write_bytes(b'checkpoint-a')
    manager = ParetoCheckpointManager(tmp_path, {'bbox_ap': 'max'})
    manager.record(0, paths['checkpoint'], {'coco_eval_bbox': [0.5]})
    record = yaml.safe_load(paths['metrics'].read_text(encoding='utf-8'))
    paths['done'].write_text(
        json.dumps({'sha256': record['sha256']}), encoding='utf-8'
    )
    assert completed_with_matching_sha(paths)
    paths['checkpoint'].write_bytes(b'checkpoint-b')
    assert not completed_with_matching_sha(paths)


def test_detection_profile_does_not_construct_pose_or_tracking_metrics(monkeypatch):
    from engine.edgecrafter import pose_metrics as metric_module
    from engine.solver import ec_engine

    class SimpleMetric:
        def __init__(self):
            self.gt_instances = 0

        def update(self, *args):
            self.gt_instances += 1

        def state_dict(self):
            return {'gt_instances': self.gt_instances}

        def empty_copy(self):
            return type(self)()

        def merge_state(self, state):
            self.gt_instances += state['gt_instances']

        def summarize(self):
            return {'map': 0.5, 'query_utilization': 0.75}

    class ForbiddenMetric:
        def __init__(self, *args, **kwargs):
            raise AssertionError('detection profile constructed a forbidden metric')

    class FakeCocoEvaluator:
        iou_types = []
        labels = None

        def cleanup(self):
            pass

        def update(self, result):
            self.result = result

        def synchronize_between_processes(self):
            pass

        def accumulate(self):
            pass

        def summarize(self):
            pass

    class TinyModel(torch.nn.Module):
        def forward(self, samples):
            return {
                'pred_logits': torch.zeros(1, 2, 1),
                'pred_query_valid': torch.ones(1, 2, dtype=torch.bool),
            }

    class TinyPostprocessor(torch.nn.Module):
        def forward(self, outputs, target_sizes):
            return [{
                'boxes': torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                'scores': torch.tensor([0.9]),
                'labels': torch.tensor([0]),
            }]

    monkeypatch.setattr(metric_module, 'BeeDetectionMetrics', SimpleMetric)
    monkeypatch.setattr(metric_module, 'BeeQueryMetrics', SimpleMetric)
    monkeypatch.setattr(metric_module, 'BeePoseMetrics', ForbiddenMetric)
    tracking_module = sys.modules.get('engine.tracking')
    if tracking_module is not None:
        monkeypatch.setattr(
            tracking_module, 'TrackingMetricsAccumulator', ForbiddenMetric
        )

    samples = torch.zeros(1, 3, 4, 4)
    target = {
        'image_id': torch.tensor(1),
        'orig_size': torch.tensor([4, 4]),
        'boxes': torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        'labels': torch.tensor([0]),
        'domain_id': torch.tensor(0),
    }
    stats, _ = ec_engine.evaluate(
        TinyModel(),
        torch.nn.Identity(),
        TinyPostprocessor(),
        [(samples, [target])],
        FakeCocoEvaluator(),
        torch.device('cpu'),
        evaluation_profile='detection',
    )
    assert stats['detection_all_map'] == 0.5
    assert stats['query_utilization'] == 0.75
    assert not any(key.startswith(('pose_', 'tracking_', 'density_')) for key in stats)
    assert 'latency_mean_ms' not in stats


def test_inline_evaluation_cannot_advance_training_rng():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    with ECSolver._preserve_training_rng():
        random.random()
        np.random.rand()
        torch.rand(10)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_sync_and_async_training_state_equivalence():
    def run(inline_eval):
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        model = torch.nn.Linear(4, 2)
        ema = deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = FlatCosineLRScheduler(
            optimizer,
            lr_gamma=0.1,
            iter_per_epoch=1,
            total_epochs=2,
            warmup_iter=1,
            flat_epochs=1,
            no_aug_epochs=0,
        )
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        targets = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
        for epoch in range(2):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + 1, optimizer)
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema.parameters(), model.parameters()
                ):
                    ema_parameter.mul_(0.9).add_(parameter, alpha=0.1)
            if inline_eval:
                with ECSolver._preserve_training_rng():
                    model.eval()
                    _ = model(inputs)
                    random.random()
                    np.random.rand()
                    torch.rand(4)
                model.train()
        return {
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'last_epoch': 1,
        }

    synchronous = run(True)
    asynchronous = run(False)
    def assert_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right):
                assert_equal(left_item, right_item)
        else:
            assert left == right

    assert_equal(synchronous, asynchronous)


def test_backbone_schedule_freezes_then_unfreezes_deepest_groups():
    model = torch.nn.Module()
    model.backbone = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    )
    frozen, trainable, total = ECSolver._set_backbone_schedule(model, 0, 2, 1)
    assert frozen and trainable == 0 and total > 0
    frozen, trainable, total = ECSolver._set_backbone_schedule(model, 2, 2, 1)
    assert not frozen and 0 < trainable < total


def test_zero_freeze_trains_the_entire_pretrained_trunk():
    model = torch.nn.Module()
    model.backbone = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    )
    frozen, trainable, total = ECSolver._set_backbone_schedule(model, 0, 0, 2)
    assert not frozen and trainable == total
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_backbone_schedule_never_freezes_native_p2_adapter():
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.blocks = torch.nn.ModuleList([
                torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            ])
            self.native_p2_adapter = torch.nn.Linear(2, 2)
            self.history_p3_adapter = torch.nn.Linear(2, 2)

    model = torch.nn.Module()
    model.backbone = Adapter()
    frozen, trainable, total = ECSolver._set_backbone_schedule(model, 0, 10, 1)
    assert frozen and trainable == 0 and total > 0
    assert all(parameter.requires_grad for parameter in model.backbone.native_p2_adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.history_p3_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.backbone.parameters())


def test_gradual_unfreeze_keeps_frozen_blocks_in_eval_mode():
    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)
            )
            self.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)),
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)),
            ])

    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.backbone = Trunk()
    model.train()
    ECSolver._set_backbone_schedule(model, 1, 1, 1)
    _apply_backbone_train_modes(model)
    assert not model.backbone.backbone.patch_embed.training
    assert not model.backbone.backbone.blocks[0].training
    assert model.backbone.backbone.blocks[1].training


def test_formal_optimizer_assigns_exact_learning_rates_without_overlap():
    class FormalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.backbone = torch.nn.Module()
            self.backbone.backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.backbone.native_p2_adapter = torch.nn.Linear(2, 2)
            self.encoder = torch.nn.Linear(2, 2)

    root = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e'
    formal = yaml.safe_load((root / 'e_formal_contract_1280.yml').read_text(encoding='utf-8'))
    optimizer_cfg = dict(formal['optimizer'])
    optimizer_cfg['type'] = 'AdamW'
    model = FormalModel()
    groups = YAMLConfig.get_optim_params(optimizer_cfg, model)
    defaults = {
        'lr': optimizer_cfg['lr'],
        'weight_decay': optimizer_cfg['weight_decay'],
    }
    by_parameter = {}
    for group in groups:
        settings = {**defaults, **{key: value for key, value in group.items() if key != 'params'}}
        for parameter in group['params']:
            assert id(parameter) not in by_parameter
            by_parameter[id(parameter)] = settings

    named = dict(model.named_parameters())
    assert len(by_parameter) == len(named)
    assert by_parameter[id(named['backbone.backbone.blocks.0.weight'])]['lr'] == 5e-6
    assert by_parameter[id(named['backbone.backbone.blocks.0.bias'])]['lr'] == 5e-6
    assert by_parameter[id(named['backbone.backbone.blocks.0.bias'])]['weight_decay'] == 0
    assert by_parameter[id(named['backbone.native_p2_adapter.weight'])]['lr'] == 2e-4
    assert by_parameter[id(named['backbone.native_p2_adapter.bias'])]['lr'] == 2e-4
    assert by_parameter[id(named['backbone.native_p2_adapter.bias'])]['weight_decay'] == 0
    assert by_parameter[id(named['encoder.weight'])]['lr'] == 1e-4
    assert by_parameter[id(named['encoder.bias'])]['lr'] == 1e-4
    assert by_parameter[id(named['encoder.bias'])]['weight_decay'] == 0


def test_formal_warmup_uses_data_iterations_despite_gradient_accumulation():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([
        {'params': [parameter], 'lr': 1e-4},
    ])
    scheduler = FlatCosineLRScheduler(
        optimizer,
        lr_gamma=0.5,
        iter_per_epoch=90,
        total_epochs=74,
        warmup_iter=2000,
        flat_epochs=24,
        no_aug_epochs=2,
    )

    # With grad_accum_steps=16 the first update consumes 16 data iterations.
    scheduler.step(16, optimizer)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4 * (16 / 2000) ** 2)

    # Warm-up reaches the configured LR after 2000 data iterations, not after
    # 2000 optimizer updates (which this small formal run never has).
    scheduler.step(2000, optimizer)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    assert scheduler.state_dict()['current_iter'] == 2000


def test_formal_m2_contract_is_explicit():
    root = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e'
    base = yaml.safe_load((root / 'e_formal_contract_1280.yml').read_text(encoding='utf-8'))
    joint = yaml.safe_load((root / 'e2_formal_joint_1280.yml').read_text(encoding='utf-8'))
    query = yaml.safe_load((root / 'e3_formal_query_1280.yml').read_text(encoding='utf-8'))
    assert base['ECTransformer']['use_density_peaks'] is False
    assert base['ECTransformer']['keypoint_noise_scale'] > 0
    assert base['ECTransformer']['head_tail_swap_ratio'] > 0
    assert joint['ECTransformer']['endpoint_decoder_type'] == 'decomposed'
    assert query['ECTransformer']['use_density_peaks'] is True
    assert base['warmup_iter'] == 2000
    assert base['grad_accum_steps'] >= 16
    assert base['query_clip_max_norm'] == 0.1
    assert base['clip_max_norm'] == 1.0
    assert base['scaler']['init_scale'] == 8.0  # 当前4090版本的有限梯度热启动值。
    assert base['stop_aug_resume_checkpoint'] is None
    assert base['pareto_metrics']['rgb_ap'] == 'max'


def test_all_formal_stage_configs_pass_the_executable_contract():
    root = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e'
    records = validate_all(root)
    assert len(records) == 6
    assert all(record['input'] == 1280 for record in records)
    assert all(record['p2_channels'] == 256 for record in records)
    assert all(record['queries'] == 768 for record in records)
