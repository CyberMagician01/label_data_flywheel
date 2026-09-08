import sys
import types
import itertools
import json
import hashlib

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path


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
scipy = types.ModuleType('scipy')
scipy.__path__ = []
scipy_optimize = types.ModuleType('scipy.optimize')
def _exact_linear_sum_assignment(cost):
    cost = np.asarray(cost)
    row_count, column_count = cost.shape
    if row_count == 0 or column_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if column_count > row_count:
        columns, rows = _exact_linear_sum_assignment(cost.T)
        return rows, columns
    best = None
    for rows in itertools.permutations(range(row_count), column_count):
        score = sum(cost[row, column] for column, row in enumerate(rows))
        if best is None or score < best[0]:
            best = score, rows
    return np.asarray(best[1], dtype=np.int64), np.arange(column_count, dtype=np.int64)
scipy_optimize.linear_sum_assignment = _exact_linear_sum_assignment
scipy.optimize = scipy_optimize
sys.modules.setdefault('scipy', scipy)
sys.modules.setdefault('scipy.optimize', scipy_optimize)

from engine.edgecrafter.bee_e_layers import (
    DomainConditionedLayerNorm,
    DomainLowRankAdapter2d,
    DomainSpecificBatchNorm2d,
    StabilizedTemporalResidual,
    replace_layer_norms,
    set_explicit_domain,
)
from engine.edgecrafter.ecvit import ViTAdapter
from engine.edgecrafter.modeling import ECDet
from engine.edgecrafter.decoder import DecomposedEndpointDecoder, ECTransformer
from engine.edgecrafter.denoising import (
    _monotonic_denoising_count,
    get_contrastive_denoising_training_group,
)
from engine.edgecrafter.criterion import ECCriterion
from engine.edgecrafter.matcher import HungarianMatcher
from engine.edgecrafter.hybrid_encoder import HybridEncoder, P2LocalWindowInteraction
from engine.data._misc import convert_to_tv_tensor
from engine.data.transforms._transforms import (
    BeeDirectionRotation,
    BeeTargetCenteredCrop,
    IRPercentileNormalize,
    PrepareTemporalFrames,
    RandomHorizontalFlipWithKeypoints,
    SanitizeBoundingBoxes,
    StaticCameraStabilizer,
)
from engine.data.transforms.bee_e_augment import (
    BeeDensityConstrainedCrop,
    BeeDirectionGapRotation,
    BeeTrajectoryTailAugment,
    BeeTwoImageStitch,
)
from engine.data.transforms.container import Compose
from engine.solver.semi_supervised import OnlinePseudoLabeler
from engine.solver.stage_controller import BeeEStageController
from engine.solver.ec_engine import train_one_epoch
from engine.solver.ec_solver import ECSolver
from engine.core.yaml_utils import load_config
from engine.edgecrafter.inference import DensityResidualRechecker
from engine.edgecrafter.postprocessor import PostProcessor, structural_set_deduplicate
from engine.tracking import TrackerConfig, correct_head_tail_sequence
from tools.bee_e.calibrate_inference_contract import build_contract
from tools.bee_e.validate_formal_configs import validate_continuous_route
from tools.bee_e.validate_onnx import load_stratified_calibration_manifest
from tools.bee_e.split_protocol import build_fixed_511_manifest
from tools.bee_e.finalize_continuous_route import (
    domain_statistics_fingerprint,
    preprocessing_contract,
    validate_completed_checkpoint,
    validate_split_contract,
)
from tools.bee_e.design_evidence import validate_route_design_evidence
from tools.bee_e.prepare_public_handoff import render_template


def test_explicit_domain_layers_reject_implicit_routing():
    value = torch.randn(2, 4, 8, 8)
    with pytest.raises(RuntimeError, match='domain_id'):
        DomainSpecificBatchNorm2d(4)(value)
    with pytest.raises(RuntimeError, match='domain_id'):
        DomainLowRankAdapter2d(4, 2)(value)


def test_shared_foreground_prototype_uses_matched_p3_roi_and_equal_domains():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
        domain_foreground_quality_threshold=0.7,
    )
    feature = torch.zeros(2, 2, 8, 8, requires_grad=True)
    with torch.no_grad():
        feature[0, 0] = 1.0
        feature[1, 1] = 1.0
    outputs = {
        'pred_logits': torch.zeros(2, 1, 1),
        'pred_domain_p3_features': feature,
        # Deliberately use a different dtype from the activation feature to
        # exercise the AMP-safe FP32 prototype similarity path.
        'shared_bee_prototype': torch.tensor(
            [1.0, 1.0], dtype=torch.float64, requires_grad=True,
        ),
    }
    targets = []
    for domain in (0, 1):
        targets.append({
            'domain_id': torch.tensor([domain]),
            'boxes': torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            'is_pseudo': torch.tensor([False]),
            'inter_group_quality': torch.tensor([0.9]),
            'intra_group_quality': torch.tensor([0.8]),
            'hierarchy_quality': torch.tensor([0.95]),
        })
    matched = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    loss = criterion.loss_domain(outputs, targets, matched, 2)[
        'loss_shared_bee_prototype'
    ]
    assert torch.allclose(loss, torch.tensor(1.0 - 2 ** -0.5), atol=1e-5)
    loss.backward()
    assert feature.grad is not None
    assert outputs['shared_bee_prototype'].grad is not None


def test_domain_specific_batch_norm_routes_samples_and_freezes_statistics():
    module = DomainSpecificBatchNorm2d(2, momentum=1.0)
    module.train()
    module.set_domain_id(torch.tensor([0, 1]))
    value = torch.stack([
        torch.full((2, 4, 4), 2.0),
        torch.full((2, 4, 4), 9.0),
    ])
    module(value)
    assert torch.allclose(module.norms[0].running_mean, torch.full((2,), 2.0))
    assert torch.allclose(module.norms[1].running_mean, torch.full((2,), 9.0))

    frozen = DomainSpecificBatchNorm2d.from_batch_norm(
        nn.BatchNorm2d(2), freeze_running_stats=True,
    ).train()
    before = [norm.running_mean.clone() for norm in frozen.norms]
    frozen.set_domain_id(torch.tensor([0, 1]))
    frozen(value)
    assert all(torch.equal(norm.running_mean, old) for norm, old in zip(frozen.norms, before))


def test_conditioned_layer_norm_and_low_rank_adapter_are_pretrained_safe():
    layer_norm = nn.LayerNorm(4)
    conditioned = DomainConditionedLayerNorm.from_layer_norm(layer_norm)
    conditioned.set_domain_id(torch.tensor([0, 1]))
    value = torch.randn(2, 3, 4)
    assert torch.allclose(conditioned(value), layer_norm(value))

    adapter = DomainLowRankAdapter2d(4, 2)
    adapter.set_domain_id(torch.tensor([0, 1]))
    image = torch.randn(2, 4, 6, 6)
    assert torch.equal(adapter(image), image)


def test_recursive_domain_context_reaches_replaced_layer_norms():
    module = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    original = module[1](torch.randn(2, 3, 4))
    replace_layer_norms(module)
    replace_layer_norms(module)
    assert isinstance(module[1], DomainConditionedLayerNorm)
    assert isinstance(module[1].norm, nn.LayerNorm)
    set_explicit_domain(module, torch.tensor([0, 1]))
    value = torch.randn(2, 3, 4)
    assert torch.allclose(module[1](value), module[1].norm(value))
    assert original.shape == module[1](value).shape


def test_local_window_domain_routing_survives_checkpoint_recompute():
    module = P2LocalWindowInteraction(
        channels=8, window_size=4, num_heads=2, mlp_ratio=2.0,
    )
    replace_layer_norms(module)
    set_explicit_domain(module, torch.tensor([0, 1]))
    module.train()
    value = torch.randn(2, 8, 8, 8, requires_grad=True)

    output = torch.utils.checkpoint.checkpoint(
        module, value, use_reentrant=False,
    )
    output.square().mean().backward()

    assert value.grad is not None
    assert module.norm1._domain_id.tolist() == [0, 1]
    assert module.norm2._domain_id.tolist() == [0, 1]


def test_external_evaluation_requires_matching_checkpoint_sha(tmp_path):
    checkpoint = tmp_path / 'checkpoint0003.pth'
    torch.save({'model': {'weight': torch.ones(1)}}, checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics_dir = tmp_path / 'checkpoint_metrics'
    metrics_dir.mkdir()
    (metrics_dir / 'epoch0003.json').write_text(json.dumps({
        'sha256': checkpoint_sha,
        'eval_stats': {'coco_eval_bbox': [0.25], 'detection_rgb_recall': 0.5},
    }), encoding='utf-8')
    (metrics_dir / 'epoch0003.done').write_text(json.dumps({
        'sha256': checkpoint_sha,
        'status': 'done',
    }), encoding='utf-8')

    solver = ECSolver.__new__(ECSolver)
    solver.cfg = types.SimpleNamespace(yaml_cfg={
        'external_eval': {
            'enabled': True, 'timeout_seconds': 1, 'poll_seconds': 0.1,
        },
    })
    solver.output_dir = tmp_path
    result = solver._wait_for_external_evaluation(3, checkpoint)

    assert result['coco_eval_bbox'] == [0.25]


def test_stabilized_three_frame_residual_obeys_stage_multiplier_and_valid_mask():
    block = StabilizedTemporalResidual(4, 3)
    current = torch.randn(2, 4, 8, 8)
    support = torch.randn_like(current)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(2, 1, 1)
    valid = torch.ones(2, dtype=torch.bool)

    # The route starts as an exact identity before the temporal stage is enabled.
    assert torch.equal(
        block(current, support, support, identity, identity, valid, valid), current,
    )

    # Invalid support frames must contribute exactly zero, even after enabling the stage.
    block.set_stage_multiplier(1.0)
    nn.init.normal_(block.compress[-1].weight)
    invalid = torch.zeros(2, dtype=torch.bool)
    output = block(current, support, support, identity, identity, invalid, invalid)
    zero_input = torch.zeros(2, 16, 8, 8)
    expected = current + torch.sigmoid(block.gate(zero_input)) * block.compress(zero_input)
    assert torch.allclose(output, expected)


def test_stabilized_three_frame_residual_requires_two_explicit_transforms():
    block = StabilizedTemporalResidual(2, 2)
    current = torch.randn(1, 2, 5, 5)
    support = torch.randn_like(current)
    identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    valid = torch.ones(1, dtype=torch.bool)
    output = block(current, support, support, identity, identity, valid, valid)
    assert output.shape == current.shape


def test_p2_window_attention_preserves_shape_and_never_crosses_window_boundary():
    module = P2LocalWindowInteraction(
        channels=16, window_size=4, num_heads=4,
    )
    feature = torch.randn(1, 16, 7, 9, requires_grad=True)
    output = module(feature)
    assert output.shape == feature.shape
    output[0, :, 0, 0].sum().backward()
    assert torch.count_nonzero(feature.grad[:, :, 4:, :]) == 0
    assert torch.count_nonzero(feature.grad[:, :, :, 4:]) == 0


def test_hybrid_encoder_applies_p2_local_p3_fusion_and_domain_conditioned_norms():
    encoder = HybridEncoder(
        in_channels=[16, 16, 16, 16], feat_strides=[4, 8, 16, 32],
        hidden_dim=16, nhead=4, dim_feedforward=32,
        use_encoder_idx=[3], num_encoder_layers=1,
        expansion=0.5, depth_mult=0.34, csp_type='csp2',
        enable_p2_local_window=True, p2_window_size=4,
        p2_window_heads=4, enable_p2_p3_high_resolution_fusion=True,
        enable_dual_domain_norm=True, enable_domain_layernorm=True,
    ).train()
    set_explicit_domain(encoder, torch.tensor([0, 1]))
    outputs = encoder([
        torch.randn(2, 16, size, size) for size in (16, 8, 4, 2)
    ])
    assert [tuple(output.shape[-2:]) for output in outputs] == [
        (16, 16), (8, 8), (4, 4), (2, 2),
    ]
    assert any(
        isinstance(module, DomainConditionedLayerNorm)
        for module in encoder.modules()
    )
    assert any(
        isinstance(module, DomainSpecificBatchNorm2d)
        for module in encoder.p2_p3_high_resolution_fusion.modules()
    )


def test_vit_adapter_applies_explicit_domain_contract_to_stem_p2_p3_and_ln():
    model = ViTAdapter(
        name='ecvitt', skip_load_backbone=True,
        embed_dim=32, num_heads=4, interaction_indexes=[0, 1],
        depth=2, proj_dim=[8, 8, 8, 8], num_levels=4, use_native_p2=True,
        enable_dual_domain_norm=True, freeze_domain_bn_stats=True,
        domain_adapter_rank=4, enable_domain_layernorm=True,
    ).eval()
    explicit_norms = [
        module for module in model.modules()
        if isinstance(module, DomainSpecificBatchNorm2d)
    ]
    assert len(explicit_norms) >= 8
    assert isinstance(model.backbone.blocks[0].norm1, DomainConditionedLayerNorm)
    with pytest.raises(RuntimeError, match='domain_id'):
        model(torch.randn(2, 3, 64, 64))
    features = model(
        torch.randn(2, 3, 64, 64), domain_id=torch.tensor([0, 1]),
    )
    assert [feature.shape[-2:] for feature in features] == [
        (16, 16), (8, 8), (4, 4), (2, 2),
    ]


class _FullRouteBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_domain = None
        self.history_domain = None

    @staticmethod
    def _features(value):
        batch, _, height, width = value.shape
        return [
            torch.ones(batch, 4, height // stride, width // stride, device=value.device)
            for stride in (4, 8, 16, 32)
        ]

    def forward(self, value, return_shallow=False, domain_id=None):
        self.current_domain = domain_id
        features = self._features(value)
        return (features, features[:2]) if return_shallow else features

    def forward_shallow_history(self, value, domain_id=None):
        self.history_domain = domain_id
        return self._features(value)[:2]


class _FullRouteDecoder(nn.Module):
    num_queries = 8

    def forward(self, features, targets=None, **kwargs):
        self.domain_context = kwargs['domain_context']
        batch = features[0].shape[0]
        return {
            'pred_logits': torch.zeros(batch, self.num_queries, 1),
            'pred_boxes': torch.zeros(batch, self.num_queries, 4),
        }


def _full_route_model():
    return ECDet(
        backbone=_FullRouteBackbone(), encoder=nn.Identity(), decoder=_FullRouteDecoder(),
        enable_temporal=True, temporal_frames=3, enable_stabilized_temporal=True,
        temporal_feature_channels=[4, 4], temporal_bottleneck_channels=2,
        temporal_feature_levels=2, enable_explicit_domain=True,
    ).eval()


def test_ecdet_full_route_requires_stabilization_for_valid_support_frames():
    model = _full_route_model()
    with pytest.raises(RuntimeError, match='stabilization_theta'):
        model(
            torch.randn(2, 3, 3, 32, 32),
            temporal_valid_mask=torch.ones(2, 3, dtype=torch.bool),
            domain_id=torch.tensor([0, 1]),
        )


def test_ecdet_full_route_propagates_domain_and_three_frame_geometry():
    model = _full_route_model()
    identity = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    ).expand(2, 2, -1, -1).clone()
    output = model(
        torch.randn(2, 3, 3, 32, 32),
        temporal_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        domain_id=torch.tensor([0, 1]),
        stabilization_theta=identity,
    )
    assert torch.equal(model.backbone.current_domain, torch.tensor([0, 1]))
    assert torch.equal(model.backbone.history_domain, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(
        model.decoder.domain_context,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.equal(output['temporal_valid_ratio'], torch.ones(2))


def test_static_camera_stabilizer_translation_fallback_returns_grid_theta():
    support = np.zeros((64, 64), dtype=np.float32)
    support[16:48:4, 16:48:4] = 255.0
    current = np.roll(support, shift=4, axis=1)
    stabilizer = StaticCameraStabilizer(
        stable_translation=0.1, min_orb_matches=10_000, max_translation=16,
    )
    theta, diagnostic = stabilizer.estimate(support, current)
    assert diagnostic['branch'] == StaticCameraStabilizer.BRANCH_PHASE_TRANSLATION
    assert abs(diagnostic['forward_affine'][0, 2].item() - 4.0) < 0.5
    assert abs(theta[0, 2].item() - (-8.0 / 64.0)) < 0.02


def test_static_camera_stabilizer_accepts_route_contract_aliases():
    stabilizer = StaticCameraStabilizer(
        max_scale_change=0.15,
        max_rotation_deg=15,
    )
    assert stabilizer.min_scale == pytest.approx(0.85)
    assert stabilizer.max_scale == pytest.approx(1.15)
    assert stabilizer.max_rotation_degrees == pytest.approx(15.0)


def test_prepare_three_frames_records_stabilization_contract(tmp_path):
    path = tmp_path / 'frame.png'
    canvas = np.zeros((48, 64, 3), dtype=np.uint8)
    canvas[8:40:4, 8:56:4] = 255
    Image.fromarray(canvas).save(path)
    target = {
        'domain_id': torch.tensor([0]),
        'temporal_paths': [str(path), str(path), str(path)],
        'temporal_valid_mask': torch.tensor([False, True, True]),
    }
    current = torch.zeros(3, 64, 64)
    clip, output_target = PrepareTemporalFrames(
        size=(64, 64), stabilize=True,
    )((current, target))
    assert clip.shape == (3, 3, 64, 64)
    assert output_target['stabilization_theta'].shape == (2, 2, 3)
    assert output_target['stabilization_forward_affine'].shape == (2, 2, 3)
    assert output_target['stabilization_branch'][0] == StaticCameraStabilizer.BRANCH_INVALID
    assert output_target['stabilization_branch'][1] == StaticCameraStabilizer.BRANCH_IDENTITY


def test_ir_clip_uses_one_background_foreground_quantile_contract(tmp_path):
    frames = []
    for index, background in enumerate((20, 40, 60)):
        pixels = np.full((8, 8, 3), background, dtype=np.uint8)
        pixels[2:4, 2:4] = 220
        path = tmp_path / f'ir_{index}.png'
        Image.fromarray(pixels).save(path)
        frames.append(path)
    current = Image.open(frames[-1]).convert('RGB')
    normalized, target = IRPercentileNormalize(
        lower=0.0, upper=1.0, foreground_residual_quantile=0.75,
    )((current, {
        'domain_id': torch.tensor([1]),
        'temporal_paths': [str(path) for path in frames],
    }))
    bounds = target['ir_quantile_bounds']
    assert bounds.shape == (2,)
    assert bounds[0] <= 40.0
    assert bounds[1] == 220.0

    preparer = PrepareTemporalFrames(size=(8, 8), lower=0.0, upper=1.0)
    first = preparer._prepare_frame(
        Image.open(frames[0]).convert('RGB'), True,
        ir_quantile_bounds=bounds,
    )
    last = preparer._prepare_frame(
        Image.open(frames[-1]).convert('RGB'), True,
        ir_quantile_bounds=bounds,
    )
    # Identical foreground intensity must remain identical across support frames;
    # per-frame percentile normalization would violate this contract.
    assert torch.equal(first[:, 2, 2], last[:, 2, 2])
    assert normalized.size == (8, 8)


def test_full_endpoint_route_uses_body_axis_logit_refinement_and_uncertainty():
    decoder = DecomposedEndpointDecoder(
        hidden_dim=32, num_keypoints=2, num_layers=2, num_levels=4,
        num_sampling_points=3, use_full_route=True,
        body_axis_radius=0.2, endpoint_delta_limit=1.5,
    ).eval()
    instance = torch.randn(2, 1, 3, 32)
    boxes = torch.tensor([0.5, 0.5, 0.4, 0.2]).view(1, 1, 1, 4).repeat(2, 1, 3, 1)
    features = [
        torch.randn(1, 32, size, size) for size in (16, 8, 4, 2)
    ]
    keypoints, visibility, uncertainty, endpoint_features = decoder(
        instance, boxes, features,
    )
    assert keypoints.shape == (2, 1, 3, 2, 2)
    assert visibility.shape == uncertainty.shape == (2, 1, 3, 2)
    assert endpoint_features.shape == (2, 1, 3, 2, 32)
    assert torch.all(uncertainty > 0)
    assert torch.allclose(keypoints[0].mean(dim=2), boxes[0, ..., :2], atol=1e-5)
    expected_radius = 0.2 * boxes[0, ..., 2:].amax(dim=-1)
    actual_radius = (keypoints[0, ..., 0, :] - boxes[0, ..., :2]).norm(dim=-1)
    assert torch.allclose(actual_radius, expected_radius, atol=1e-5)


def test_full_query_route_starts_all_candidates_from_one_learnable_content():
    transformer = ECTransformer(
        num_classes=1, hidden_dim=32, num_queries=6,
        feat_channels=[32, 32, 32, 32], feat_strides=[4, 8, 16, 32],
        num_levels=4, nhead=4, num_layers=2, dim_feedforward=64,
        num_denoising=0, num_keypoints=2,
        endpoint_decoder_type='decomposed', use_full_endpoint_route=True,
        use_quality_head=True, use_shared_query_content=True,
    ).eval()
    memory = torch.randn(2, 7, 32)
    content, _, _, _, _ = transformer._get_decoder_input(
        memory, [[2, 2], [1, 1], [1, 1], [1, 1]],
    )
    assert content.shape == (2, 6, 32)
    assert torch.equal(content[:, 1:], content[:, :1].expand(-1, 5, -1))


def test_full_decoder_training_forward_carries_pose_ddf_aux_and_dn_contracts():
    transformer = ECTransformer(
        num_classes=1, hidden_dim=32, num_queries=6,
        feat_channels=[32, 32, 32, 32], feat_strides=[4, 8, 16, 32],
        num_levels=4, num_points=2, nhead=4, num_layers=2,
        dim_feedforward=64, num_denoising=2, num_keypoints=2,
        endpoint_decoder_type='decomposed', use_full_endpoint_route=True,
        use_quality_head=True, use_shared_query_content=True,
        keypoint_noise_scale=0.1, head_tail_swap_ratio=0.25,
        enable_domain_layernorm=True,
    ).train()
    set_explicit_domain(transformer, torch.tensor([0]))
    features = [
        torch.randn(1, 32, size, size) for size in (8, 4, 2, 1)
    ]
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.2]]),
        'keypoints': torch.tensor([[[0.43, 0.5, 2.0], [0.57, 0.5, 2.0]]]),
    }]
    output = transformer(features, targets=targets)
    required = {
        'pred_logits', 'pred_boxes', 'pred_corners', 'ref_points',
        'pred_keypoints', 'pred_visibility', 'pred_endpoint_uncertainty',
        'pred_quality', 'up', 'reg_scale',
    }
    assert required <= output.keys()
    assert len(output['aux_outputs']) == 1
    assert len(output['dn_outputs']) == 2
    for branch in [*output['aux_outputs'], *output['dn_outputs']]:
        assert {
            'pred_corners', 'ref_points', 'pred_keypoints',
            'pred_visibility', 'pred_endpoint_uncertainty',
            'pred_quality', 'up', 'reg_scale',
        } <= branch.keys()
        assert torch.isfinite(branch['pred_endpoint_uncertainty']).all()


def test_full_decoder_training_without_denoising_keeps_pre_outputs_defined():
    transformer = ECTransformer(
        num_classes=1, hidden_dim=32, num_queries=6,
        feat_channels=[32, 32, 32, 32], feat_strides=[4, 8, 16, 32],
        num_levels=4, num_points=2, nhead=4, num_layers=2,
        dim_feedforward=64, num_denoising=0, num_keypoints=2,
        endpoint_decoder_type='decomposed', use_full_endpoint_route=True,
        use_quality_head=True, use_shared_query_content=True,
    ).train()
    features = [
        torch.randn(1, 32, size, size) for size in (8, 4, 2, 1)
    ]
    output = transformer(features, targets=[{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.2]]),
        'keypoints': torch.tensor([[[0.43, 0.5, 2.0], [0.57, 0.5, 2.0]]]),
    }])

    assert 'pre_outputs' in output
    assert 'pred_masks' in output['pre_outputs']
    assert 'dn_outputs' not in output


def test_density_uncertainty_capacity_map_is_monotonic_and_stage_bounded():
    transformer = ECTransformer(
        num_classes=1, hidden_dim=32, num_queries=12,
        feat_channels=[32, 32, 32, 32], feat_strides=[4, 8, 16, 32],
        num_levels=4, nhead=4, num_layers=1, dim_feedforward=64,
        num_denoising=0, enable_monotonic_query_capacity=True,
        min_active_queries=3, stage_query_limit=10,
        density_capacity_ratio=1.5, density_capacity_padding=1,
        uncertainty_capacity_scale=4,
    )
    count = torch.tensor([0.0, 2.0, 20.0])
    uncertainty = torch.tensor([0.0, 0.5, 1.0])
    budget = transformer._monotonic_query_budget(count, uncertainty)
    assert budget.tolist() == [3, 6, 10]
    assert torch.all(
        transformer._monotonic_query_budget(count + 1, uncertainty) >= budget
    )
    assert torch.all(
        transformer._monotonic_query_budget(count, uncertainty + 0.1) >= budget
    )


def test_adaptive_density_gaussians_keep_unit_mass_at_image_boundaries():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
        use_adaptive_density_target=True,
        density_scale_bandwidth=0.2, density_neighbor_bandwidth=0.25,
    )
    prediction = torch.zeros(1, 1, 32, 32)
    targets = [{
        'boxes': torch.tensor([
            [0.01, 0.01, 0.05, 0.05],
            [0.50, 0.50, 0.30, 0.20],
        ]),
    }]
    density, counts = criterion._build_adaptive_density_target(prediction, targets)
    assert torch.allclose(density.sum(), torch.tensor(2.0), atol=1e-5)
    assert torch.equal(counts, torch.tensor([2.0]))
    losses = criterion.loss_density(
        {'pred_density': prediction + 1e-4}, targets, [], 2,
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_endpoint_uncertainty_is_supervised_only_on_visible_matched_points():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
    )
    outputs = {
        'pred_keypoints': torch.tensor([[[[0.4, 0.5], [0.6, 0.5]]]]),
        'pred_visibility': torch.zeros(1, 1, 2),
        'pred_endpoint_uncertainty': torch.full((1, 1, 2), 0.1),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 0.4, 0.2]]]),
    }
    targets = [{
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        'keypoints': torch.tensor([[[0.45, 0.5, 2.0], [0.55, 0.5, 2.0]]]),
        'pose_state': torch.tensor([2]),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    losses = criterion.loss_pose(outputs, targets, indices, 1)
    assert torch.isfinite(losses['loss_endpoint_uncertainty'])
    assert losses['loss_endpoint_uncertainty'] != 0


def test_pose_loss_keeps_ddp_keys_and_zero_gradients_for_detection_only_rank():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
    )
    outputs = {
        'pred_keypoints': torch.zeros(1, 2, 2, 2, requires_grad=True),
        'pred_visibility': torch.zeros(1, 2, 2, requires_grad=True),
        'pred_endpoint_uncertainty': torch.ones(1, 2, 2, requires_grad=True),
        'pred_boxes': torch.zeros(1, 2, 4, requires_grad=True),
    }
    targets = [{
        'labels': torch.zeros(0, dtype=torch.long),
        'boxes': torch.zeros(0, 4),
    }]
    empty = torch.zeros(0, dtype=torch.long)
    losses = criterion.loss_pose(outputs, targets, [(empty, empty)], num_boxes=1)
    assert set(losses) == {
        'loss_keypoint', 'loss_pose_oks', 'loss_pose_box',
        'loss_pose_length', 'loss_direction', 'loss_visibility',
        'loss_endpoint_uncertainty',
    }
    assert all(torch.isfinite(value) and value.item() == 0.0 for value in losses.values())
    sum(losses.values()).backward()
    for name, value in outputs.items():
        assert value.grad is not None, name
        assert torch.count_nonzero(value.grad) == 0, name


def test_hungarian_includes_pairwise_ddf_distribution_cost():
    matcher = HungarianMatcher(
        weight_dict={
            'cost_class': 1.0, 'cost_bbox': 0.0, 'cost_giou': 0.0,
            'cost_ddf': 1.0,
        },
        use_focal_loss=True,
    )
    outputs = {
        'pred_logits': torch.tensor([[[4.0], [1.0]]]),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 0.4, 0.2], [0.2, 0.2, 0.1, 0.1]]]),
        'pred_corners': torch.zeros(1, 2, 20),
        'ref_points': torch.tensor([[[0.5, 0.5, 0.4, 0.2], [0.2, 0.2, 0.1, 0.1]]]),
        'up': torch.tensor([0.5]),
        'reg_scale': torch.tensor([4.0]),
    }
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
    }]
    indices = matcher(outputs, targets)['indices'][0]
    assert indices[0].tolist() == [0]
    assert indices[1].tolist() == [0]


def test_auxiliary_positive_assignment_is_radius_limited_and_query_unique():
    matcher = HungarianMatcher(
        weight_dict={'cost_class': 1.0, 'cost_bbox': 2.0, 'cost_giou': 1.0},
        use_focal_loss=True, aux_positive_radius=1.0,
    )
    outputs = {
        'pred_logits': torch.full((1, 4, 1), 2.0),
        'pred_boxes': torch.tensor([[[
            0.50, 0.50, 0.20, 0.20,
        ], [
            0.55, 0.50, 0.20, 0.20,
        ], [
            0.90, 0.90, 0.20, 0.20,
        ], [
            0.10, 0.10, 0.20, 0.20,
        ]]]),
        'pred_query_valid': torch.tensor([[True, True, True, True]]),
    }
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
    }]
    standard = matcher(outputs, targets)['indices'][0]
    combined = matcher(outputs, targets, return_topk=3)
    assert torch.equal(combined['indices'][0][0], standard[0])
    assert torch.equal(combined['indices'][0][1], standard[1])
    sources, target_indices = combined['indices_o2m'][0]
    assert len(sources) == 2
    assert len(sources.unique()) == len(sources)
    assert target_indices.tolist() == [0, 0]


def test_auxiliary_matcher_reuses_the_combined_one_to_one_and_one_to_many_call():
    class CountingMatcher:
        def __init__(self):
            self.calls = []
            self.mask_point_sample_ratio = None

        def __call__(self, outputs, targets, return_topk=False):
            self.calls.append(return_topk)
            indices = [(torch.tensor([0]), torch.tensor([0]))]
            result = {'indices': indices}
            if return_topk:
                result['indices_o2m'] = indices
            return result

    matcher = CountingMatcher()
    criterion = ECCriterion(matcher, {}, [], one2many_topk=3)
    base = {
        'pred_logits': torch.zeros(1, 1, 1),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
    }
    outputs = {
        **base,
        'aux_outputs': [dict(base), dict(base)],
        'enc_aux_outputs': [],
        'enc_meta': {'class_agnostic': False},
    }
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }]

    assert criterion(outputs, targets) == {}
    assert matcher.calls == [False, 3, 3]


def test_ignore_boxes_follow_every_custom_geometric_transform():
    image = Image.new('RGB', (100, 100))
    target = {
        'boxes': convert_to_tv_tensor(
            torch.tensor([[40.0, 40.0, 60.0, 60.0]]), key='boxes',
            box_format='XYXY', spatial_size=(100, 100),
        ),
        'labels': torch.tensor([0]),
        'ignore_boxes': convert_to_tv_tensor(
            torch.tensor([[45.0, 45.0, 55.0, 55.0]]), key='boxes',
            box_format='XYXY', spatial_size=(100, 100),
        ),
    }
    cropped_image, cropped = BeeTargetCenteredCrop(
        p=1.0, scale=(0.5, 0.5), min_instances=1,
    )((image, target))
    assert cropped_image.size == (50, 50)
    assert torch.equal(
        cropped['ignore_boxes'].as_subclass(torch.Tensor),
        torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
    )

    _, flipped = RandomHorizontalFlipWithKeypoints(p=1.0)((cropped_image, cropped))
    assert torch.equal(
        flipped['ignore_boxes'].as_subclass(torch.Tensor),
        torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
    )

    _, rotated = BeeDirectionRotation(p=1.0, degrees=0.0)((cropped_image, flipped))
    assert torch.equal(
        rotated['ignore_boxes'].as_subclass(torch.Tensor),
        flipped['ignore_boxes'].as_subclass(torch.Tensor),
    )


def test_sanitize_keeps_independent_ignore_boxes_without_positive_instances():
    image = Image.new('RGB', (32, 32))
    target = {
        'boxes': convert_to_tv_tensor(
            torch.zeros(0, 4), key='boxes', box_format='XYXY', spatial_size=(32, 32),
        ),
        'ignore_boxes': convert_to_tv_tensor(
            torch.tensor([[-4.0, -3.0, 8.0, 9.0], [7.0, 7.0, 7.2, 7.3]]),
            key='boxes', box_format='XYXY', spatial_size=(32, 32),
        ),
    }
    _, cleaned = SanitizeBoundingBoxes()((image, target))
    assert torch.equal(
        cleaned['ignore_boxes'].as_subclass(torch.Tensor),
        torch.tensor([[0.0, 0.0, 8.0, 9.0]]),
    )


def test_ignore_region_queries_are_not_trained_as_background():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
        ignore_query_iou_threshold=0.3,
    )
    outputs = {
        'pred_logits': torch.zeros(1, 2, 1),
        'pred_boxes': torch.tensor([[[
            0.5, 0.5, 0.2, 0.2,
        ], [
            0.1, 0.1, 0.1, 0.1,
        ]]]),
    }
    empty = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    target = {'labels': torch.zeros(0, dtype=torch.long), 'boxes': torch.zeros(0, 4)}
    no_ignore = criterion.loss_labels_focal(outputs, [target], [empty], 1)['loss_focal']
    target['ignore_boxes'] = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    ignored = criterion.loss_labels_focal(outputs, [target], [empty], 1)['loss_focal']
    assert torch.allclose(no_ignore, ignored * 2.0)


def test_pseudo_label_requires_two_evidences_and_routes_partial_support_to_ignore():
    labeler = OnlinePseudoLabeler(ignore_score_threshold=0.9)
    candidates = {
        'boxes': torch.tensor([
            [0.3, 0.3, 0.2, 0.2],
            [0.7, 0.7, 0.2, 0.2],
        ]),
        'labels': torch.tensor([0, 0]),
        'scores': torch.tensor([0.8, 0.8]),
        'class_probabilities': torch.tensor([[0.8], [0.8]]),
        'keypoints': None,
        'visibility': None,
        'masks': None,
    }
    target = labeler._build_target(
        {}, candidates,
        set_score=torch.tensor([0.9, 0.9]),
        trajectory_score=torch.tensor([0.8, 0.0]),
        mask_size=(16, 16),
    )
    assert len(target['boxes']) == 1
    assert torch.equal(target['boxes'][0], candidates['boxes'][0])
    assert torch.equal(target['ignore_boxes'][0], candidates['boxes'][1])
    assert 'masks' not in target


def test_missing_instance_pseudo_merge_preserves_human_annotation_and_alignment():
    labeler = OnlinePseudoLabeler()
    human = {
        'boxes': torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
        'labels': torch.tensor([0]),
        'keypoints': torch.zeros(1, 2, 3),
        'masks': torch.zeros(1, 8, 8, dtype=torch.bool),
        'ignore_boxes': torch.tensor([[0.9, 0.9, 0.05, 0.05]]),
    }
    pseudo = labeler._empty_instances(human, torch.device('cpu'), mask_size=(8, 8))
    pseudo.update({
        'boxes': torch.tensor([[0.7, 0.7, 0.1, 0.1]]),
        'labels': torch.tensor([0]),
        'keypoints': torch.zeros(1, 2, 3),
        'masks': torch.ones(1, 8, 8, dtype=torch.bool),
        'is_pseudo': torch.ones(1, dtype=torch.bool),
        'pseudo_score': torch.tensor([0.8]),
        'teacher_class_probability': torch.tensor([[0.75]]),
        'pose_state': torch.zeros(1, dtype=torch.long),
        'pose_mask': torch.zeros(1),
        'pose_quality': torch.zeros(1),
        'track_geometry': torch.zeros(1, 6),
        'track_geometry_mask': torch.zeros(1, dtype=torch.bool),
        'track_id': torch.full((1,), -1, dtype=torch.long),
        'track_mask': torch.zeros(1, dtype=torch.bool),
        'track_quality': torch.zeros(1),
        'trajectory_stability': torch.tensor([0.7]),
        'supervision_mask': torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        'ignore_boxes': torch.tensor([[0.9, 0.9, 0.05, 0.05]]),
    })
    merged = labeler._merge_with_human_target(human, pseudo)
    assert len(merged['boxes']) == len(merged['labels']) == len(merged['masks']) == 2
    assert merged['is_pseudo'].tolist() == [False, True]
    assert torch.allclose(merged['pseudo_score'], torch.tensor([1.0, 0.8]))


def test_ssl_consistency_uses_teacher_soft_targets_only_for_pseudo_matches():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
    )
    outputs = {
        'pred_logits': torch.tensor([[[0.0], [0.0]]]),
        'pred_boxes': torch.tensor([[[
            0.2, 0.2, 0.1, 0.1,
        ], [
            0.7, 0.7, 0.1, 0.1,
        ]]]),
        'pred_keypoints': torch.tensor([[[
            [0.18, 0.2], [0.22, 0.2],
        ], [
            [0.68, 0.7], [0.72, 0.7],
        ]]]),
    }
    targets = [{
        'labels': torch.tensor([0, 0]),
        'boxes': outputs['pred_boxes'][0].clone(),
        'keypoints': torch.tensor([
            [[0.18, 0.2, 2.0], [0.22, 0.2, 2.0]],
            [[0.68, 0.7, 2.0], [0.72, 0.7, 2.0]],
        ]),
        'is_pseudo': torch.tensor([False, True]),
        'pseudo_score': torch.tensor([1.0, 0.8]),
        'teacher_class_probability': torch.tensor([[1.0], [0.8]]),
    }]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    losses = criterion.loss_consistency(outputs, targets, indices, 2)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(0.0), torch.tensor(0.8),
    )
    assert torch.allclose(losses['loss_consistency_class'], expected)
    assert losses['loss_consistency_box'] == 0
    assert losses['loss_consistency_pose'] == 0


def _stitch_target(image_id, domain, track_id, box):
    return {
        'boxes': convert_to_tv_tensor(
            torch.tensor([box], dtype=torch.float32), key='boxes',
            box_format='XYXY', spatial_size=(40, 60),
        ),
        'labels': torch.tensor([0]),
        'area': torch.tensor([(box[2] - box[0]) * (box[3] - box[1])], dtype=torch.float32),
        'iscrowd': torch.tensor([0]),
        'keypoints': torch.tensor([[[box[0] + 1, box[1] + 1, 2],
                                    [box[2] - 1, box[3] - 1, 2]]], dtype=torch.float32),
        'track_id': torch.tensor([track_id]),
        'track_mask': torch.tensor([True]),
        'image_id': torch.tensor([image_id]),
        'domain_id': torch.tensor([domain]),
    }


def test_two_image_stitch_is_domain_local_capacity_safe_and_field_aligned():
    transform = BeeTwoImageStitch(
        p=1.0, orientations=('horizontal',), effective_query_capacity=4,
        capacity_safety_ratio=0.75,
    )
    first = Image.new('RGB', (60, 40), color=(10, 20, 30))
    second = Image.new('RGB', (60, 40), color=(40, 50, 60))
    ir = Image.new('RGB', (60, 40), color=(70, 70, 70))
    _, first_target = transform(
        (first, _stitch_target(1, 0, 3, [10, 10, 30, 30]))
    )
    assert 'is_two_image_stitch' not in first_target
    _, ir_target = transform((ir, _stitch_target(2, 1, 3, [10, 10, 30, 30])))
    assert 'is_two_image_stitch' not in ir_target
    image, target = transform(
        (second, _stitch_target(4, 0, 3, [12, 8, 32, 28]))
    )
    assert image.size == (60, 40)
    assert target['is_two_image_stitch']
    assert target['stitched_image_ids'].tolist() == [4, 1]
    assert len(target['boxes']) == len(target['labels']) == len(target['keypoints']) == 2
    assert target['track_id'].tolist() == [3, 7]
    assert int(target['stitch_gt_count']) == 2
    assert int(target['stitch_capacity']) == 4
    assert bool(target['force_current_frame_history'])


def test_two_image_stitch_rejects_query_capacity_overflow():
    transform = BeeTwoImageStitch(
        p=1.0, effective_query_capacity=2, capacity_safety_ratio=0.75,
    )
    image = Image.new('RGB', (60, 40))
    transform((image, _stitch_target(1, 0, 1, [1, 1, 10, 10])))
    _, target = transform((image, _stitch_target(2, 0, 2, [1, 1, 10, 10])))
    assert 'is_two_image_stitch' not in target


def test_direction_gap_rotation_targets_an_undercovered_bin_and_keeps_endpoints():
    transform = BeeDirectionGapRotation(
        p=1.0, bins=4, max_degrees=180, min_endpoint_retention=1.0,
    )
    transform.update_coverage([10, 10, 0, 10])
    image = Image.new('RGB', (64, 64))
    target = _stitch_target(1, 0, 1, [20, 20, 44, 44])
    target['keypoints'] = torch.tensor([[[26.0, 32.0, 2.0], [38.0, 32.0, 2.0]]])
    _, output = transform((image, target))
    assert int(output['direction_gap_bin']) == 2
    assert (output['keypoints'][..., 2] > 0).all()
    assert abs(float(output['direction_rotation_angle'])) <= 180


def test_density_crop_respects_query_occupancy_and_endpoint_completeness():
    transform = BeeDensityConstrainedCrop(
        p=1.0, domains=(1,), scale=(0.75, 0.75), occupancy=(0.2, 0.8),
        effective_query_capacity=10, min_visible_ratio=1.0,
        require_complete_endpoints=True, attempts=100,
    )
    image = Image.new('RGB', (100, 100))
    boxes = torch.tensor([
        [20.0, 20.0, 30.0, 30.0], [35.0, 30.0, 45.0, 40.0],
        [50.0, 45.0, 60.0, 55.0], [65.0, 60.0, 75.0, 70.0],
    ])
    target = {
        'boxes': convert_to_tv_tensor(
            boxes, key='boxes', box_format='XYXY', spatial_size=(100, 100),
        ),
        'labels': torch.zeros(4, dtype=torch.long),
        'keypoints': torch.stack([
            torch.tensor([[box[0] + 2, box[1] + 2, 2], [box[2] - 2, box[3] - 2, 2]])
            for box in boxes
        ]),
        'area': torch.full((4,), 100.0),
        'domain_id': torch.tensor([1]),
        'image_id': torch.tensor([5]),
    }
    cropped, output = transform((image, target))
    assert cropped.size[0] == cropped.size[1] == 75
    assert 2 <= int(output['density_crop_gt_count']) <= 8
    assert 0.2 <= float(output['density_crop_occupancy']) <= 0.8
    assert (output['keypoints'][..., 2] > 0).all()


def _stage_spy(name, calls):
    def forward(self, sample):
        calls.append(name)
        return sample
    return type(name, (torch.nn.Module,), {'forward': forward})()


def test_stage_policy_explicitly_prohibits_stitch_after_s1_and_strong_aug_in_s5():
    calls = []
    transforms = [
        _stage_spy('BeeTwoImageStitch', calls),
        _stage_spy('BeeDirectionGapRotation', calls),
        _stage_spy('BeeTrajectoryTailAugment', calls),
        _stage_spy('PrepareTemporalFrames', calls),
    ]
    policy = Compose(transforms, policy='bee_e_staged', bee_e_stage='E-S2')
    sample = (Image.new('RGB', (8, 8)), {'domain_id': torch.tensor([0])})
    _, annotated = policy(sample)
    assert calls == ['BeeDirectionGapRotation', 'PrepareTemporalFrames']
    assert int(annotated['bee_e_stage_index']) == 2
    calls.clear()
    policy.set_stage('E-S3', progress=0.4)
    _, annotated = policy(sample)
    assert calls == ['BeeTrajectoryTailAugment', 'PrepareTemporalFrames']
    assert torch.allclose(annotated['temporal_stage_multiplier'], torch.tensor(0.4))
    calls.clear()
    policy.set_stage('E-S5', progress=1.0)
    policy(sample)
    assert calls == ['PrepareTemporalFrames']


def test_stage_resolution_and_pre_temporal_current_frame_contract(tmp_path):
    path = tmp_path / 'current.png'
    Image.new('RGB', (40, 20), color=(20, 40, 60)).save(path)
    policy = Compose(
        [{'type': 'PrepareTemporalFrames', 'size': (32, 32), 'stabilize': True}],
        policy='bee_e_staged', bee_e_stage='E-S0',
        bee_e_resolutions={'low': 32, 'mid': 48, 'full': 64},
    )
    prepare = policy.transforms[0]
    target = {
        'domain_id': torch.tensor([0]),
        'temporal_paths': [str(path), str(path), str(path)],
        'temporal_valid_mask': torch.tensor([True, True, True]),
    }
    current = prepare._prepare_frame(Image.open(path), False)
    clip, output = policy((current, target))
    assert clip.shape == (3, 3, 32, 32)
    assert torch.equal(clip[0], clip[1]) and torch.equal(clip[1], clip[2])
    assert output['temporal_valid_mask'].tolist() == [False, False, True]
    assert output['bee_e_input_size'].tolist() == [32, 32]
    policy.set_stage('E-S1', progress=0.5)
    assert policy._select_bee_e_resolution() == (48, 48)
    policy.set_stage('E-S1', progress=0.9)
    assert policy._select_bee_e_resolution() == (64, 64)


def test_btca_uses_empirical_tail_tube_and_appends_aligned_instance():
    height = width = 64
    clip = torch.zeros(3, 3, height, width)
    source_boxes = torch.tensor([
        [0.15, 0.30, 0.30, 0.45],
        [0.20, 0.30, 0.35, 0.45],
        [0.25, 0.30, 0.40, 0.45],
    ])
    for frame, box in enumerate(source_boxes):
        x1, y1, x2, y2 = (
            (box * torch.tensor([width, height, width, height])).long().tolist()
        )
        clip[frame, :, y1:y2, x1:x2] = 1.0
    boxes = convert_to_tv_tensor(
        torch.tensor([[0.325, 0.375, 0.15, 0.15]]), key='boxes',
        box_format='CXCYWH', spatial_size=(height, width),
    )
    target = {
        'boxes': boxes,
        'labels': torch.tensor([0]),
        'keypoints': torch.tensor([[[0.28, 0.375, 2.0], [0.37, 0.375, 2.0]]]),
        'track_id': torch.tensor([11]),
        'track_mask': torch.tensor([True]),
        'track_geometry': torch.zeros(1, 6),
        'track_geometry_mask': torch.tensor([True]),
        'area': torch.tensor([0.15 * 0.15 * height * width]),
        'domain_id': torch.tensor([0]),
        'btca_tubes': [{
            'track_id': 11,
            'boxes': source_boxes,
            'keypoints': torch.tensor([
                [[0.18, 0.375, 2.0], [0.27, 0.375, 2.0]],
                [[0.23, 0.375, 2.0], [0.32, 0.375, 2.0]],
                [[0.28, 0.375, 2.0], [0.37, 0.375, 2.0]],
            ]),
            'valid_mask': torch.tensor([True, True, True]),
            'quality': 0.95,
            'observation_length': 3,
            'short_track_threshold': 4,
            'tail_distribution': {
                'centers': torch.tensor([[0.70, 0.65]]),
                'velocities': torch.tensor([[0.04, 0.00]]),
                'sizes': torch.tensor([[0.16, 0.16]]),
                'axes': torch.tensor([[1.0, 0.0]]),
            },
        }],
    }
    augmented_clip, output = BeeTrajectoryTailAugment(
        p=1.0, residual_threshold=0.01, minimum_mask_mass=1.0,
    )((clip, target))
    assert bool(output['btca_applied'])
    assert len(output['boxes']) == len(output['labels']) == len(output['keypoints']) == 2
    assert output['track_id'].tolist() == [11, 12]
    assert output['btca_augmented'].tolist() == [False, True]
    assert float(augmented_clip[:, :, 30:55, 35:58].abs().sum()) > 0


def _stage_specs():
    return [
        {'name': f'E-S{index}', 'min_cycles': 3, 'max_cycles': 5,
         'initial_query_fraction': 0.25}
        for index in range(6)
    ]


def test_stage_controller_merges_global_and_stage_specific_constraints():
    controller = BeeEStageController(
        _stage_specs(),
        metric_directions={'ap': 'max', 'recall': 'max', 'swap': 'min'},
        official_primary_metric='ap',
        constraints={
            'all': {'recall': {'min': 0.7}},
            'E-S0': {'swap': {'max': 0.1}},
        },
        minimum_plateau_window=3,
    )
    assert controller._constraint_pass({'ap': 0.5, 'recall': 0.8, 'swap': 0.05})
    assert not controller._constraint_pass({'ap': 0.5, 'recall': 0.6, 'swap': 0.05})
    assert not controller._constraint_pass({'ap': 0.5, 'recall': 0.8, 'swap': 0.2})


def test_stage_controller_leaves_integer_parameter_caches_non_trainable():
    controller = BeeEStageController(
        _stage_specs(), metric_directions={'ap': 'max'},
        official_primary_metric='ap', constraints={}, minimum_plateau_window=3,
    )
    model = nn.Module()
    model.register_parameter(
        'integer_cache',
        nn.Parameter(torch.tensor([1], dtype=torch.long), requires_grad=False),
    )
    model.register_parameter('weight', nn.Parameter(torch.ones(1)))
    loader = types.SimpleNamespace(
        dataset=types.SimpleNamespace(), sampler=types.SimpleNamespace(),
    )
    controller.apply(model, types.SimpleNamespace(), loader)
    assert not model.integer_cache.requires_grad
    assert model.weight.requires_grad


def test_stage_controller_transitions_only_after_plateau_and_feasible_best():
    controller = BeeEStageController(
        _stage_specs(),
        metric_directions={'ap': 'max', 'recall': 'max'},
        official_primary_metric='ap',
        constraints={'all': {'recall': {'min': 0.7}}},
        minimum_plateau_window=3, bootstrap_samples=64,
    )
    assert not controller.record_cycle(
        {'ap': 0.5, 'recall': 0.8}, 'checkpoint0.pth',
    )['transition']
    assert not controller.record_cycle(
        {'ap': 0.5, 'recall': 0.8}, 'checkpoint1.pth',
    )['transition']
    decision = controller.record_cycle(
        {'ap': 0.5, 'recall': 0.8}, 'checkpoint2.pth',
    )
    assert decision['transition']
    assert decision['from_stage'] == 'E-S0'
    assert decision['to_stage'] == 'E-S1'
    assert decision['selected']['checkpoint'] == 'checkpoint0.pth'
    assert sum(spec['max_cycles'] for spec in controller.stage_specs) == 30
    restored = BeeEStageController(
        _stage_specs(),
        metric_directions={'ap': 'max', 'recall': 'max'},
        official_primary_metric='ap',
        constraints={'all': {'recall': {'min': 0.7}}},
        minimum_plateau_window=3,
    )
    restored.load_state_dict(controller.state_dict())
    assert restored.stage == 'E-S1'
    assert restored.global_cycle == 3


def test_decoder_denoising_curriculum_tracks_stage_and_matching_debt():
    transformer = ECTransformer(
        num_classes=1, hidden_dim=32, num_queries=200,
        feat_channels=[32, 32, 32, 32], feat_strides=[4, 8, 16, 32],
        num_levels=4, nhead=4, num_layers=1, dim_feedforward=64,
        num_denoising=100, label_noise_ratio=0.4, box_noise_scale=1.0,
        num_keypoints=2, keypoint_noise_scale=0.2,
        head_tail_swap_ratio=0.1, initial_denoising_fraction=0.25,
        denoising_debt_gain=0.5, max_denoising_multiplier=1.5,
    )
    s0 = transformer.set_stage_denoising('E-S0', 1.0)
    assert s0 == {
        'num_denoising': 25,
        'label_noise_ratio': 0.1,
        'box_noise_scale': 0.25,
        'keypoint_noise_scale': 0.0,
        'head_tail_swap_ratio': 0.0,
        'no_object_noise_ratio': 0.0,
    }
    s2 = transformer.set_stage_denoising('E-S2', 0.5)
    assert s2['num_denoising'] == 100
    assert s2['keypoint_noise_scale'] == pytest.approx(0.1)
    assert s2['head_tail_swap_ratio'] == pytest.approx(0.05)
    s4 = transformer.set_stage_denoising('E-S4', 1.0, matching_debt=1.0)
    assert s4['num_denoising'] == 150
    assert s4['box_noise_scale'] == pytest.approx(1.5)
    s5 = transformer.set_stage_denoising('E-S5', 1.0, matching_debt=10.0)
    assert s5['num_denoising'] == 100
    assert s5['box_noise_scale'] == pytest.approx(1.0)


def test_denoising_count_is_monotone_and_no_object_corruption_keeps_gt_targets():
    def count(num_gt, capacity, entropy):
        return _monotonic_denoising_count(
            num_gt, 100, capacity, entropy, 0.25, 0.75,
        )

    assert [count(n, 80, 0.4) for n in (1, 8, 20, 40)] == sorted(
        count(n, 80, 0.4) for n in (1, 8, 20, 40)
    )
    assert [count(8, cap, 0.4) for cap in (16, 32, 64, 80)] == sorted(
        count(8, cap, 0.4) for cap in (16, 32, 64, 80)
    )
    assert [count(8, 80, entropy) for entropy in (0.0, 0.25, 0.5, 1.0)] == sorted(
        count(8, 80, entropy) for entropy in (0.0, 0.25, 0.5, 1.0)
    )

    targets = [{
        'labels': torch.tensor([0, 0, 0]),
        'boxes': torch.tensor([
            [0.2, 0.2, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
            [0.8, 0.8, 0.1, 0.1],
        ]),
    }]
    class_embed = nn.Embedding(2, 4, padding_idx=1)
    with torch.no_grad():
        class_embed.weight[0].fill_(1.0)
        class_embed.weight[1].zero_()
    logits, _, attention, meta = get_contrastive_denoising_training_group(
        targets,
        num_classes=1,
        num_queries=20,
        class_embed=class_embed,
        num_denoising=20,
        label_noise_ratio=0.0,
        no_object_noise_ratio=1.0,
        effective_query_capacity=10,
        matching_entropy=1.0,
    )
    assert meta['dn_num_split'] == [10, 20]
    assert meta['dn_no_object_noise_count'].item() == 10
    assert torch.count_nonzero(logits) == 0
    assert meta['dn_positive_idx'][0].numel() == 5
    assert meta['dn_positive_target_idx'][0].numel() == 5
    matched = ECCriterion.get_cdn_matched_indices(meta, targets)
    assert torch.equal(matched[0][1], meta['dn_positive_target_idx'][0])
    assert attention[10:, :10].all() and attention[:10, 10:].all()


def test_denoising_uses_global_gt_capacity_when_local_rank_is_empty(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)

    def all_reduce_max(value, op=None):
        calls.append((int(value.item()), op))
        value.fill_(3)

    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce_max)
    class_embed = nn.Embedding(2, 4, padding_idx=1)
    with torch.no_grad():
        class_embed.weight[0].fill_(1.0)
        class_embed.weight[1].zero_()
    targets = [{
        'labels': torch.zeros(0, dtype=torch.long),
        'boxes': torch.zeros(0, 4),
    }]

    logits, boxes, attention, meta = get_contrastive_denoising_training_group(
        targets,
        num_classes=1,
        num_queries=20,
        class_embed=class_embed,
        num_denoising=20,
        label_noise_ratio=0.0,
        effective_query_capacity=10,
        matching_entropy=1.0,
    )

    assert calls == [(0, torch.distributed.ReduceOp.MAX)]
    assert meta['dn_num_split'] == [10, 20]
    assert meta['dn_positive_count'] == 0
    assert meta['dn_positive_idx'][0].numel() == 0
    assert logits.shape == (1, 10, 4)
    assert boxes.shape == (1, 10, 4)
    assert attention.shape == (30, 30)
    assert torch.count_nonzero(logits) == 0


def test_criterion_stage_curriculum_opens_every_loss_family_at_exact_stage():
    matcher = HungarianMatcher({
        'cost_class': 1, 'cost_bbox': 1, 'cost_giou': 1,
        'cost_keypoint': 1, 'cost_oks': 1, 'cost_direction': 1,
    })
    criterion = ECCriterion(matcher, {}, [], one2many_topk=3)
    criterion.set_stage_context('E-S0', 0.5, 0.5)
    assert criterion.stage_loss_multipliers == {
        'pose': 0.5, 'structure_domain': 0.0,
        'temporal_density_track': 0.0, 'consistency': 0.0,
    }
    criterion.set_stage_context('E-S2', 0.25, 1.0)
    assert criterion.stage_loss_multipliers['structure_domain'] == 0.25
    assert criterion.stage_loss_multipliers['temporal_density_track'] == 0.0
    criterion.set_stage_context('E-S3', 0.4, 1.0)
    assert criterion.stage_loss_multipliers['temporal_density_track'] == 0.4
    assert criterion.stage_loss_multipliers['consistency'] == 1.0
    criterion.set_stage_context('E-S5', 1.0, 1.0)
    assert criterion.stage_loss_multipliers['consistency'] == 0.0
    assert criterion.one2many_topk == 3


def test_quality_target_combines_class_correctness_iou_and_endpoint_oks():
    matcher = HungarianMatcher({'cost_class': 1, 'cost_bbox': 1, 'cost_giou': 1})
    criterion = ECCriterion(matcher, {}, [])
    target = {
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        'keypoints': torch.tensor([[[0.4, 0.5, 2.0], [0.6, 0.5, 2.0]]]),
        'pose_state': torch.tensor([1]),
        'domain_id': torch.tensor([0]),
    }
    base = {
        'pred_logits': torch.tensor([[[10.0]]]),
        'pred_boxes': target['boxes'][None].clone(),
        'pred_quality': torch.tensor([[10.0]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    perfect = dict(base, pred_keypoints=target['keypoints'][None, ..., :2].clone())
    displaced = dict(
        base,
        pred_keypoints=torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]]),
    )
    perfect_loss = criterion.loss_quality(perfect, [target], indices, 1)['loss_quality']
    displaced_loss = criterion.loss_quality(displaced, [target], indices, 1)['loss_quality']
    assert perfect_loss < displaced_loss


def test_quality_balancing_weight_scales_loss_without_leaving_probability_range(monkeypatch):
    matcher = HungarianMatcher({'cost_class': 1, 'cost_bbox': 1, 'cost_giou': 1})
    criterion = ECCriterion(matcher, {}, [])
    target = {
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        'domain_id': torch.tensor([0]),
    }
    outputs = {
        'pred_logits': torch.tensor([[[10.0]]]),
        'pred_boxes': target['boxes'][None].clone(),
        'pred_quality': torch.tensor([[0.0]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    def set_weight(value):
        monkeypatch.setattr(
            ECCriterion,
            '_matched_instance_weight',
            staticmethod(lambda targets, indices, device, mode='base': torch.tensor(
                [value], dtype=torch.float32, device=device,
            )),
        )

    set_weight(1.0)
    base_loss = criterion.loss_quality(outputs, [target], indices, 1)['loss_quality']
    set_weight(4.0)
    weighted_loss = criterion.loss_quality(outputs, [target], indices, 1)['loss_quality']

    assert torch.isfinite(weighted_loss)
    assert weighted_loss >= 0
    torch.testing.assert_close(weighted_loss, base_loss * 4.0)


def test_ddf_kl_is_finite_for_extreme_but_finite_fp16_corner_logits():
    matcher = HungarianMatcher({'cost_class': 1, 'cost_bbox': 1, 'cost_giou': 1})
    criterion = ECCriterion(matcher, {}, [], reg_max=32)
    pred_corners = torch.tensor(
        [[([65504.0] + [-65504.0] * 32) * 4]],
        dtype=torch.float16,
        requires_grad=True,
    )
    outputs = {
        'pred_corners': pred_corners,
        'teacher_corners': torch.tensor(
            [[([65504.0] + [0.0] * 32) * 4]], dtype=torch.float16,
        ),
        'teacher_logits': torch.zeros(1, 1, 1, dtype=torch.float16),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        'ref_points': torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        'reg_scale': torch.tensor([4.0]),
        'up': torch.tensor([0.5]),
    }
    target = {
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = criterion.loss_local(outputs, [target], indices, 1)['loss_ddf']
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss >= 0
    loss.backward()
    assert torch.isfinite(pred_corners.grad).all()


def test_ddf_zero_teacher_branch_does_not_overflow_fp16_reduction():
    matcher = HungarianMatcher({'cost_class': 1, 'cost_bbox': 1, 'cost_giou': 1})
    criterion = ECCriterion(matcher, {}, [], reg_max=32)
    pred_corners = torch.full(
        (1, 1, 4 * 33), 65504.0, dtype=torch.float16, requires_grad=True,
    )
    outputs = {
        'pred_corners': pred_corners,
        'teacher_corners': pred_corners.detach().clone(),
        'teacher_logits': torch.zeros(1, 1, 1, dtype=torch.float16),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        'ref_points': torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        'reg_scale': torch.tensor([4.0]),
        'up': torch.tensor([0.5]),
    }
    target = {
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = criterion.loss_local(outputs, [target], indices, 1)['loss_ddf']
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss == 0
    loss.backward()
    assert torch.equal(pred_corners.grad, torch.zeros_like(pred_corners.grad))


class _NoSyncTrainSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.no_sync_calls = 0

    def no_sync(self):
        model = self

        class Context:
            def __enter__(self):
                model.no_sync_calls += 1

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        return Context()

    def forward(self, samples, targets=None):
        batch = samples.shape[0]
        return {'pred_boxes': self.weight * torch.ones(batch, 1, 4)}


class _ScalarCriterion(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return {'loss_test': outputs['pred_boxes'].sum()}


class _GradientConflictTrainSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)

    def forward(self, samples, targets=None):
        batch = samples.shape[0]
        value = self.backbone.weight.reshape(1, 1, 1)
        return {'pred_boxes': value.expand(batch, 1, 4)}


def _domain_microbatch(domain_id):
    return (
        torch.ones(1, 3, 2, 2),
        [{'domain_id': torch.tensor([domain_id]), 'image_id': torch.tensor([domain_id])}],
    )


def test_optimizer_update_enforces_equal_domains_and_uses_ddp_no_sync_once():
    model = _NoSyncTrainSpy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stats = train_one_epoch(
        False, None, model, _ScalarCriterion(),
        [_domain_microbatch(0), _domain_microbatch(1)],
        optimizer, torch.device('cpu'), 0,
        grad_accum_steps=2, enable_ddp_no_sync=True,
        require_equal_domain_updates=True, print_freq=100,
    )
    assert model.no_sync_calls == 1
    assert stats['rgb_views_per_update'] == 1
    assert stats['ir_views_per_update'] == 1


def test_optimizer_update_rejects_unequal_domain_views():
    model = _NoSyncTrainSpy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match='equal non-zero RGB/IR'):
        train_one_epoch(
            False, None, model, _ScalarCriterion(),
            [_domain_microbatch(0), _domain_microbatch(0)],
            optimizer, torch.device('cpu'), 0,
            grad_accum_steps=2, enable_ddp_no_sync=True,
            require_equal_domain_updates=True, print_freq=100,
        )


def test_ddp_disables_pre_backward_gradient_conflict_diagnostics(monkeypatch):
    model = _GradientConflictTrainSpy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(
        'engine.solver.ec_engine.dist_utils.is_dist_available_and_initialized',
        lambda: True,
    )
    monkeypatch.setattr(
        'engine.solver.ec_engine.dist_utils.get_world_size', lambda: 2,
    )
    monkeypatch.setattr(
        'engine.solver.ec_engine.dist_utils.is_main_process', lambda: True,
    )
    monkeypatch.setattr(
        'engine.solver.ec_engine.dist_utils.reduce_dict', lambda values: values,
    )

    def reject_diagnostic_grad(*args, **kwargs):
        raise AssertionError('DDP training must not run pre-backward autograd.grad')

    monkeypatch.setattr(torch.autograd, 'grad', reject_diagnostic_grad)
    stats = train_one_epoch(
        False, None, model, _ScalarCriterion(),
        [_domain_microbatch(0)], optimizer, torch.device('cpu'), 0,
        grad_accum_steps=1, log_gradient_conflicts_every=1, print_freq=100,
    )
    assert torch.isfinite(torch.tensor(stats['loss']))


def test_shared_prototype_forward_output_is_graph_connected_for_ddp():
    class Backbone(nn.Module):
        def forward(self, value, **kwargs):
            return [value, value]

    class Decoder(nn.Module):
        def forward(self, features, targets=None, **kwargs):
            return {'pred_boxes': features[0].mean((2, 3), keepdim=True)}

    model = ECDet(
        Backbone(), nn.Identity(), Decoder(),
        enable_shared_foreground_prototype=True,
        domain_channels=3,
        domain_levels=1,
    )
    output = model(torch.ones(1, 3, 2, 2))['shared_bee_prototype']

    assert not output.is_leaf
    assert output.grad_fn is not None
    output.sum().backward()
    assert torch.equal(
        model.shared_bee_prototype.grad,
        torch.ones_like(model.shared_bee_prototype),
    )


def test_structural_set_dedup_requires_all_four_conditions_and_runs_once():
    result = {
        'scores': torch.tensor([0.9, 0.8, 0.7]),
        'labels': torch.zeros(3, dtype=torch.long),
        'boxes': torch.tensor([
            [10.0, 10.0, 30.0, 30.0],
            [10.5, 10.0, 30.5, 30.0],
            [10.0, 10.0, 30.0, 30.0],
        ]),
        'keypoints': torch.tensor([
            [[12.0, 20.0], [28.0, 20.0]],
            [[12.5, 20.0], [28.5, 20.0]],
            [[28.0, 20.0], [12.0, 20.0]],
        ]),
        'query_indices': torch.tensor([0, 1, 2]),
    }
    filtered = structural_set_deduplicate(result, {
        'box_iou_min': 0.7, 'endpoint_oks_min': 0.7,
        'relative_length_max': 0.2, 'oriented_axis_deg_max': 20.0,
    })
    assert filtered['query_indices'].tolist() == [0, 2]
    assert int(filtered['set_dedup_removed']) == 1
    # 已去重结果再次调用不应继续删除真实的反向遮挡个体。
    repeated = structural_set_deduplicate(filtered, {
        'box_iou_min': 0.7, 'endpoint_oks_min': 0.7,
        'relative_length_max': 0.2, 'oriented_axis_deg_max': 20.0,
    })
    assert repeated['query_indices'].tolist() == [0, 2]


def test_density_residual_recheck_uses_one_square_region_and_same_callback():
    image = torch.zeros(3, 64, 64)
    density = torch.zeros(8, 8)
    density[5:7, 5:7] = 0.75
    initial = {
        'scores': torch.tensor([0.9]),
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[2.0, 2.0, 12.0, 12.0]]),
        'keypoints': torch.tensor([[[4.0, 7.0], [10.0, 7.0]]]),
        'query_indices': torch.tensor([0]),
        'density': density,
    }
    calls = []

    def infer(crop, domain_id):
        calls.append((tuple(crop.shape), domain_id))
        return {
            'scores': torch.tensor([0.8]),
            'labels': torch.tensor([0]),
            'boxes': torch.tensor([[8.0, 8.0, 24.0, 24.0]]),
            'keypoints': torch.tensor([[[10.0, 16.0], [22.0, 16.0]]]),
            'query_indices': torch.tensor([4]),
        }

    rechecker = DensityResidualRechecker(
        calibration_by_domain={
            'rgb': {
                'trigger_count': 1.0, 'residual_threshold': 0.5,
                'expansion_rate': 0.1, 'gaussian_sigma_ratio': 0.2,
            },
            'ir': {
                'trigger_count': 1.0, 'residual_threshold': 0.5,
                'expansion_rate': 0.1, 'gaussian_sigma_ratio': 0.2,
            },
        },
        input_size=32, max_passes=1,
    )
    output = rechecker(image, initial, infer, domain_id=0)
    assert calls == [((3, 32, 32), 0)]
    assert output['density_recheck_passes'] == 1
    assert len(output['scores']) == 2
    assert output['recheck_pass'].tolist() == [0, 1]


def test_offline_head_tail_viterbi_backtracks_full_track():
    records = []
    for frame, reversed_pose in enumerate((False, True, False)):
        points = torch.tensor([[0.0 + frame, 0.0], [10.0 + frame, 0.0]])
        if reversed_pose:
            points = points.flip(0)
        records.append({
            'track_id': 1, 'frame_index': frame,
            'box': torch.tensor([0.0 + frame, -2.0, 10.0 + frame, 2.0]),
            'keypoints': points,
        })
    corrected = correct_head_tail_sequence(records)
    assert all(float(record['keypoints'][0, 0]) < float(record['keypoints'][1, 0])
               for record in corrected)


def test_continuous_route_config_retains_every_fixed_contract():
    config_path = Path(__file__).parents[2] / 'configs' / 'bee_e' / 'e_route_continuous_1280.yml'
    config = load_config(str(config_path))
    assert sum(stage['max_cycles'] for stage in config['bee_e_stage_specs']) == 180
    assert [stage['name'] for stage in config['bee_e_stage_specs']] == [
        'E-S0', 'E-S1', 'E-S2', 'E-S3', 'E-S4', 'E-S5'
    ]
    assert config['ECDet']['temporal_frames'] == 3
    assert config['ECTransformer']['num_queries'] == 768
    assert config['ECTransformer']['num_layers'] == 4
    assert config['eval_spatial_size'] == [1280, 1280]
    assert config['pareto_metrics'] is None
    assert config['require_equal_domain_updates']
    assert config['enable_ddp_no_sync']
    assert config['bee_e_initialization']['source'] == 'official_ecdet_coco'
    assert config['bee_e_initialization']['selected_scale'] == 'M'
    assert set(config['bee_e_input_distribution_evidence']) == {'path', 'sha256'}
    assert set(config['bee_e_initialization']['reinitialize_output_heads']) == {
        'category', 'quality', 'density', 'endpoints',
    }
    assert not config['PostProcessor']['enable_set_dedup']
    assert config['bee_e_density_recheck']['deduplicate_after_merge']
    assert config['train_dataloader']['dataset']['temporal_offsets'] == [-4, -1, 0]
    assert config['val_dataloader']['dataset']['temporal_offsets'] == [-4, -1, 0]
    assert config['train_dataloader']['dataset']['expected_ann_sha256'] == config['bee_e_split_contract']['train_annotation_sha256']
    assert validate_continuous_route(config_path)['status'] == 'ok'


def test_finalizer_freezes_exact_training_preprocessing_contract():
    config_path = Path(__file__).parents[2] / 'configs' / 'bee_e' / 'e_route_continuous_1280.yml'
    contract = preprocessing_contract(load_config(str(config_path)))
    assert contract['temporal_frames'] == 3
    assert contract['temporal_offsets'] == [-4, -1, 0]
    assert contract['input_size'] == [1280, 1280]
    assert contract['ir_normalization'] == {
        'lower': 0.01, 'upper': 0.99,
        'foreground_residual_quantile': 0.75,
    }
    assert contract['stabilizer'] == {
        'stable_translation': 0.35,
        'min_phase_response': 0.05,
        'orb_features': 1000,
        'min_orb_matches': 12,
        'ransac_threshold': 2.0,
        'max_translation': 96.0,
        'max_scale_change': 0.15,
        'max_rotation_deg': 15.0,
    }


def test_continuous_route_uses_exact_p3_shared_foreground_prototype():
    config_path = Path(__file__).parents[2] / 'configs' / 'bee_e' / 'e_route_continuous_1280.yml'
    config = load_config(str(config_path))
    adapter = config['ViTAdapter']
    detector = config['ECDet']
    criterion = config['ECCriterion']

    assert adapter['domain_adapter_rank'] == 16
    assert adapter['enable_dual_domain_norm'] is True
    assert adapter['enable_domain_layernorm'] is True
    assert detector['enable_domain_adapter'] is False
    assert detector['enable_shared_foreground_prototype'] is True
    assert criterion['domain_foreground_quality_threshold'] == 0.7
    assert {
        key for key, value in criterion['weight_dict'].items()
        if (key == 'loss_domain' or key.startswith('loss_domain_') or
            key == 'loss_shared_bee_prototype') and float(value) > 0.0
    } == {'loss_shared_bee_prototype'}


def test_route_design_choices_require_nondominated_calibration_evidence():
    config_path = Path(__file__).parents[2] / 'configs' / 'bee_e' / 'e_route_continuous_1280.yml'
    config = load_config(str(config_path))
    operations = config['train_dataloader']['dataset']['transforms']['ops']
    stabilizer = next(
        operation['stabilizer'] for operation in operations
        if operation['type'] == 'PrepareTemporalFrames'
    )
    ir = next(
        operation for operation in operations
        if operation['type'] == 'IRPercentileNormalize'
    )
    ir_selected = {
        key: float(ir[key])
        for key in ('lower', 'upper', 'foreground_residual_quantile')
    }
    selected = {
        'p2_channels': 256,
        'endpoint_sampling_points': 8,
        'domain_adapter_rank': 16,
        'foreground_prototype_quality_threshold': 0.7,
        'stabilizer': stabilizer,
        'ir_normalization': ir_selected,
    }
    strong = {
        'weakest_video_recall': 0.9, 'pose_nme': 0.1,
        'peak_memory_mb': 1000, 'end_to_end_latency_ms': 10,
        'head_tail_swap_rate': 0.01, 'rgb_score': 0.8, 'ir_score': 0.8,
        'residual_spectrum_explained': 0.95, 'mask_reproduction': 0.95,
        'false_motion_rate': 0.01, 'alignment_residual': 0.1,
        'foreground_recall': 0.95, 'background_false_response': 0.01,
        'temporal_scale_residual': 0.01,
        'rgb_foreground_alignment': 0.90,
        'ir_foreground_alignment': 0.88,
        'weak_annotation_contamination': 0.01,
    }
    weak = {
        key: value - 0.1 if key in {
            'weakest_video_recall', 'rgb_score', 'ir_score',
            'residual_spectrum_explained', 'mask_reproduction', 'foreground_recall',
            'rgb_foreground_alignment', 'ir_foreground_alignment',
        } else value + 0.1
        for key, value in strong.items()
    }
    frontiers = {}
    axis_metrics = {
        'p2_channels': ('weakest_video_recall', 'pose_nme', 'peak_memory_mb',
                        'end_to_end_latency_ms'),
        'endpoint_sampling_points': (
            'weakest_video_recall', 'pose_nme', 'head_tail_swap_rate',
            'peak_memory_mb', 'end_to_end_latency_ms',
        ),
        'domain_adapter_rank': (
            'rgb_score', 'ir_score', 'residual_spectrum_explained',
            'peak_memory_mb', 'end_to_end_latency_ms',
        ),
        'stabilizer': (
            'mask_reproduction', 'false_motion_rate', 'alignment_residual',
            'end_to_end_latency_ms',
        ),
        'ir_normalization': (
            'foreground_recall', 'background_false_response',
            'temporal_scale_residual',
        ),
        'foreground_prototype_quality_threshold': (
            'rgb_foreground_alignment', 'ir_foreground_alignment',
            'weak_annotation_contamination',
        ),
    }
    alternative_values = {
        'p2_channels': 128,
        'endpoint_sampling_points': 4,
        'domain_adapter_rank': 8,
        'stabilizer': {**stabilizer, 'max_translation': 64.0},
        'ir_normalization': {**ir_selected, 'upper': 0.98},
        'foreground_prototype_quality_threshold': 0.5,
    }
    for axis, metrics in axis_metrics.items():
        frontiers[axis] = [
            {'value': selected[axis], **{name: strong[name] for name in metrics}},
            {'value': alternative_values[axis], **{name: weak[name] for name in metrics}},
        ]
    optimizer = config['optimizer']
    evidence = {
        'evidence_type': 'measured_design_evidence',  # 合成单元测试记录，不是实验结果。
        'split': 'calibration',
        'calibration_annotation_sha256': config['val_dataloader']['dataset'][
            'expected_ann_sha256'
        ],
        'selected': selected,
        'frontiers': frontiers,
        'optimization': {
            'selected': {
                'optimizer': 'AdamW', 'base_lr': float(optimizer['lr']),
                'weight_decay': float(optimizer['weight_decay']),
                'total_batch_size': 2, 'grad_accum_steps': 16,
                'effective_batch_size': 32, 'use_amp': True,
                'clip_max_norm': 1.0, 'query_clip_max_norm': 0.1,
                'ema_decay': 0.9999,
                'loss_weights': config['ECCriterion']['weight_dict'],
            },
            'gradient_trace': [{
                'parameter_group': 'shared', 'gradient_norm': 1.0,
                'train_calibration_gap': 0.01,
            }],
            'mixed_precision_test': {
                'optimizer_step_equivalent': True,
                'max_loss_error': 1e-4, 'max_gradient_error': 1e-3,
            },
        },
        'stage_budget': [{
            'stage': stage['name'], 'min_cycles': stage['min_cycles'],
            'max_cycles': stage['max_cycles'], 'coverage_speed': 1.0,
            'metric_autocorrelation': 0.0, 'bootstrap_noise_bound': 0.01,
        } for stage in config['bee_e_stage_specs']],
    }
    assert validate_route_design_evidence(evidence, config)['stages'] == [
        'E-S0', 'E-S1', 'E-S2', 'E-S3', 'E-S4', 'E-S5',
    ]
    evidence['frontiers']['p2_channels'][0]['weakest_video_recall'] = 0.1
    evidence['frontiers']['p2_channels'][0]['pose_nme'] = 1.0
    evidence['frontiers']['p2_channels'][0]['peak_memory_mb'] = 2000
    evidence['frontiers']['p2_channels'][0]['end_to_end_latency_ms'] = 20
    with pytest.raises(ValueError, match='dominated'):
        validate_route_design_evidence(evidence, config)


def test_fresh_continuous_lineage_requires_verified_coco_checkpoint(tmp_path):
    checkpoint = tmp_path / 'ecdet_m_coco.pth'
    torch.save({'model': {'weight': torch.ones(1)}}, checkpoint)
    import hashlib
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evidence = tmp_path / 'scale_frontier.json'
    evidence.write_text(json.dumps({
        'dataset':'COCO2017', 'extra_supervision':'--',
        'candidates': [
            {'scale':scale, 'official_coco_ap':50.+index, 'parameters_million':10+index,
             'official_t4_trt_fp16_latency_ms':5+index, 'official_checkpoint_url':f'https://example.invalid/{scale}.pth',
             'official_score': 0.1 + index,
             'weakest_video_recall': 0.7 + index * 0.01,
             'pose_nme': 0.2 - index * 0.01,
             'parameters': 10 + index,
             'end_to_end_latency_ms': 5 + index}
            for index, scale in enumerate(('S', 'M', 'L'))
        ],
        'pareto_frontier': ['S', 'M', 'L'],
        'selected_scale': 'M',
    }), encoding='utf-8')
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    solver = ECSolver.__new__(ECSolver)
    solver.cfg = types.SimpleNamespace(
        enable_continuous_stage_training=True,
        resume=None,
        tuning=None,
        bee_e_initialization={
            'source': 'official_ecdet_coco',
            'selected_scale': 'M',
            'selection_evidence': str(evidence),
            'selection_evidence_sha256': evidence_digest,
            'checkpoint': str(checkpoint),
            'sha256': digest,
            'reinitialize_output_heads': [
                'category', 'quality', 'density', 'endpoints',
            ],
        },
    )
    verified = solver._validate_continuous_initialization()
    assert verified['selected_scale'] == 'M'
    assert verified['sha256'] == digest
    assert solver.cfg.tuning == str(checkpoint.resolve())
    forged = json.loads(evidence.read_text(encoding='utf-8'))
    forged['pareto_frontier'] = ['M']
    evidence.write_text(json.dumps(forged), encoding='utf-8')
    solver.cfg.bee_e_initialization['selection_evidence_sha256'] = hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match='official three-objective frontier'):
        solver._validate_continuous_initialization()
    evidence.write_text(json.dumps({
        **forged, 'pareto_frontier': ['S', 'M', 'L'],
    }), encoding='utf-8')
    solver.cfg.bee_e_initialization['selection_evidence_sha256'] = hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()
    solver.cfg.bee_e_initialization['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        solver._validate_continuous_initialization()


def test_objects365_task_heads_are_explicitly_reinitialized():
    expected = {
        'decoder.enc_score_head.weight': 'category',
        'decoder.dec_score_head.0.bias': 'category',
        'decoder.denoising_class_embed.weight': 'category',
        'decoder.dec_quality_head.0.layers.1.weight': 'quality',
        'density_head.3.bias': 'density',
        'decoder.endpoint_decoder.endpoint_offsets.0.weight': 'endpoints',
        'decoder.dec_visibility_head.0.weight': 'endpoints',
    }
    assert {
        name: ECSolver._bee_e_output_head_group(name) for name in expected
    } == expected
    assert ECSolver._bee_e_output_head_group('backbone.backbone.blocks.0.weight') is None


def test_public_handoff_replaces_overlapping_placeholders_without_corrupting_sha():
    rendered = render_template(
        'evidence: PUBLIC_PREADAPTATION_EVIDENCE\n'
        'sha256: PUBLIC_PREADAPTATION_EVIDENCE_SHA256\n',
        {
            'PUBLIC_PREADAPTATION_EVIDENCE': '/tmp/evidence.json',
            'PUBLIC_PREADAPTATION_EVIDENCE_SHA256': 'a' * 64,
        },
    )
    assert 'evidence: /tmp/evidence.json' in rendered
    assert f"sha256: {'a' * 64}" in rendered
    assert '/tmp/evidence.json_SHA256' not in rendered


def test_public_preadaptation_initialization_requires_verified_full_lineage(tmp_path):
    def write_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    complete_state = {
        'model': {'weight': torch.ones(1)},
        'optimizer': {}, 'lr_scheduler': {}, 'scaler': {},
        'ema': {'module': {'weight': torch.ones(1)}},
    }
    p0a_path = tmp_path / 'checkpoint0000.pth'
    torch.save({**complete_state, 'last_epoch': 0}, p0a_path)
    p0a_sha = hashlib.sha256(p0a_path.read_bytes()).hexdigest()
    selected_path = tmp_path / 'checkpoint0008.pth'
    torch.save({**complete_state, 'last_epoch': 8}, selected_path)
    selected_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()

    scale_evidence, scale_sha = write_json('scale_frontier.json', {
        'dataset': 'COCO2017', 'extra_supervision': '--',
        'selected_scale': 'M', 'pareto_frontier': ['S', 'M', 'L'],
        'candidates': [
            {'scale': 'S', 'official_coco_ap': 40.0, 'parameters_million': 20.0,
             'official_t4_trt_fp16_latency_ms': 10.0,
             'official_checkpoint_url': 'https://example.invalid/s.pth'},
            {'scale': 'M', 'official_coco_ap': 45.0, 'parameters_million': 30.0,
             'official_t4_trt_fp16_latency_ms': 8.0,
             'official_checkpoint_url': 'https://example.invalid/m.pth'},
            {'scale': 'L', 'official_coco_ap': 50.0, 'parameters_million': 40.0,
             'official_t4_trt_fp16_latency_ms': 12.0,
             'official_checkpoint_url': 'https://example.invalid/l.pth'},
        ],
    })
    public_manifest, manifest_sha = write_json('public_manifest.json', {
        'public_datasets': ['BEE24', 'BeePose', 'MendeleyBeePose'],
        'leakage_checks': {'train_val_overlap': 0},
    })
    public_evidence, public_evidence_sha = write_json('public_evidence.json', {
        'schema_version': 1,
        'base_official_checkpoint_sha256': (
            'c4cdf8bcd3b27c7903e422acd03caf733b5e1bfd664550bce43e50e2a3bbdc6e'
        ),
        'public_datasets': ['BEE24', 'BeePose', 'MendeleyBeePose'],
        'public_data_manifest': str(public_manifest),
        'public_data_manifest_sha256': manifest_sha,
        'p0a': {
            'checkpoint': str(p0a_path), 'checkpoint_sha256': p0a_sha,
            'last_epoch': 0,
        },
        'p0b': {
            'selected_epoch': 8, 'selected_checkpoint': str(selected_path),
            'last_epoch': 8, 'metrics': {},
        },
        'selected_checkpoint_sha256': selected_sha,
        'handoff_policy': 'model_and_ema_tuning_with_fresh_E-S0_optimizer_scheduler',
    })
    solver = ECSolver.__new__(ECSolver)
    solver.cfg = types.SimpleNamespace(
        enable_continuous_stage_training=True, resume=None, tuning=None,
        bee_e_initialization={
            'source': 'public_bee_preadaptation', 'selected_scale': 'M',
            'selection_evidence': str(scale_evidence),
            'selection_evidence_sha256': scale_sha,
            'checkpoint': str(selected_path), 'sha256': selected_sha,
            'reinitialize_output_heads': [],
            'public_preadaptation_evidence': str(public_evidence),
            'public_preadaptation_evidence_sha256': public_evidence_sha,
        },
    )
    verified = solver._validate_continuous_initialization()
    assert verified['source'] == 'public_bee_preadaptation'
    assert verified['sha256'] == selected_sha
    assert solver.cfg.tuning == str(selected_path.resolve())

    forged = json.loads(public_evidence.read_text(encoding='utf-8'))
    forged['selected_checkpoint_sha256'] = '0' * 64
    public_evidence.write_text(json.dumps(forged), encoding='utf-8')
    solver.cfg.bee_e_initialization['public_preadaptation_evidence_sha256'] = (
        hashlib.sha256(public_evidence.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match='does not bind'):
        solver._validate_continuous_initialization()


def test_continuous_split_contract_binds_5_1_1_roles_to_annotation_files(tmp_path):
    config=_schema_v2_test_contract(tmp_path)
    solver=ECSolver.__new__(ECSolver)
    solver.cfg=types.SimpleNamespace(enable_continuous_stage_training=True,
        bee_e_split_contract=config['bee_e_split_contract'],yaml_cfg=config)
    verified=solver._validate_continuous_split_contract()
    assert verified['role_counts']=={'train':5,'calibration':1,'dev_holdout':1}
    config['val_dataloader']['dataset']['expected_ann_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA-bound'):
        solver._validate_continuous_split_contract()


def test_fresh_continuous_lineage_requires_passed_sha_bound_input_report(tmp_path):
    import hashlib

    shas = {role: str(index) * 64 for index, role in enumerate(
        ('train', 'calibration', 'dev_holdout'), start=1,
    )}
    report = {
        'query_capacity': 768,
        'splits': {role: {} for role in shas},
        'gates': {
            'dual_domain_coverage': True,
            'video_coverage': True,
            'group_isolation': True,
            'effective_sample_size': True,
            'query_capacity': True,
            'supervision_masks': True,
        },
        'ready_for_training': True,
        'source_artifacts': {
            role: {'path': f'{role}.json', 'sha256': digest}
            for role, digest in shas.items()
        },
    }
    path = tmp_path / 'input_distribution.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    solver = ECSolver.__new__(ECSolver)
    solver.cfg = types.SimpleNamespace(
        enable_continuous_stage_training=True,
        resume=None,
        bee_e_input_distribution_evidence={
            'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        bee_e_split_contract={
            'dev_holdout_annotation_sha256': shas['dev_holdout'],
        },
        yaml_cfg={
            'ECTransformer': {'num_queries': 768},
            'train_dataloader': {'dataset': {
                'expected_ann_sha256': shas['train'],
            }},
            'val_dataloader': {'dataset': {
                'expected_ann_sha256': shas['calibration'],
            }},
        },
    )
    verified = solver._validate_continuous_input_distribution_evidence()
    assert verified['query_capacity'] == 768
    report['gates']['group_isolation'] = False
    path.write_text(json.dumps(report), encoding='utf-8')
    solver.cfg.bee_e_input_distribution_evidence['sha256'] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match='failed gates'):
        solver._validate_continuous_input_distribution_evidence()


def test_inference_calibration_contract_is_fit_only_on_calibration(tmp_path):
    candidates = []
    pairs = []
    density_trials = []
    for domain_id in (0, 1):
        candidates.extend([
            {'domain_id': domain_id, 'class_score': 0.9, 'quality': 0.9,
             'pose_visibility': 0.8, 'is_true_positive': True},
            {'domain_id': domain_id, 'class_score': 0.4, 'quality': 0.3,
             'pose_visibility': 0.4, 'is_true_positive': False},
        ])
        pairs.extend([
            {'domain_id': domain_id, 'box_iou': 0.9, 'endpoint_oks': 0.9,
             'relative_length_difference': 0.05,
             'oriented_axis_difference_deg': 5.0, 'is_duplicate': True},
            {'domain_id': domain_id, 'box_iou': 0.9, 'endpoint_oks': 0.5,
             'relative_length_difference': 0.3,
             'oriented_axis_difference_deg': 80.0, 'is_duplicate': False},
        ])
        density_trials.append({
            'domain_id': domain_id, 'trigger_count': 1.0,
            'residual_threshold': 0.1, 'expansion_rate': 0.2,
            'gaussian_sigma_ratio': 0.2, 'max_passes': 1,
            'new_true_positives': 3, 'new_false_positives': 0,
            'latency_ms': 10,
        })
    calibration_annotation = tmp_path / 'fixed_calibration_annotation.json'
    calibration_annotation.write_text(
        json.dumps({'images': [], 'annotations': [], 'categories': []}),
        encoding='utf-8',
    )
    calibration_annotation_sha = hashlib.sha256(
        calibration_annotation.read_bytes()
    ).hexdigest()
    observations = {
        'split': 'calibration', 'latency_budget_ms': 20,
        'calibration_annotation': {
            'path': str(calibration_annotation),
            'sha256': calibration_annotation_sha,
        },
        'candidates': candidates, 'duplicate_pairs': pairs,
        'density_recheck_trials': density_trials,
        'query_capacity_trials': [{
            'density_capacity_ratio': 1.25,
            'density_capacity_padding': 32,
            'uncertainty_capacity_scale': 64,
            'min_active_queries': 128,
            'stage_query_limit': 768,
            'rgb_recall': 0.91,
            'ir_recall': 0.90,
            'capacity_truncation_rate_450plus': 0.01,
            'query_utilization': 0.72,
            'latency_ms': 12,
        }],
        'tracker_trials': [{
            'parameters': TrackerConfig().__dict__,
            'mota': 0.7, 'idf1': 0.8, 'hota': 0.75, 'cast': 0.7,
            'fragment_rate': 0.1,
        }],
    }
    source = tmp_path / 'calibration.json'
    source.write_text(json.dumps(observations), encoding='utf-8')
    contract = build_contract(observations, source)
    assert contract['effective_stage'] == 'E-S5'
    assert set(contract['domains']) == {'rgb', 'ir'}
    assert contract['domains']['rgb']['set_deduplication']['distinct_preservation'] == 1.0
    assert contract['tracker']['source_data'] == 'calibration'
    assert contract['query_capacity']['parameters']['stage_query_limit'] == 768
    assert contract['source']['calibration_annotation_sha256'] == calibration_annotation_sha
    postprocessor = PostProcessor(num_classes=1).apply_calibration_contract(contract)
    assert postprocessor.enable_calibrated_filter
    assert postprocessor.score_exponents_by_domain['rgb'] == (
        contract['domains']['rgb']['ranking']['score_exponents']
    )
    rechecker = DensityResidualRechecker.from_calibration_contract(
        contract, input_size=1280,
    )
    assert rechecker.calibration_by_domain['ir']['max_passes'] == 1
    assert rechecker.dedup_thresholds_by_domain['rgb']['box_iou_min'] == (
        contract['domains']['rgb']['set_deduplication']['box_iou_min']
    )
    transformer = ECTransformer(
        num_classes=1, hidden_dim=32, num_queries=768,
        feat_channels=[32, 32, 32, 32], feat_strides=[4, 8, 16, 32],
        num_levels=4, nhead=4, num_layers=1, dim_feedforward=64,
    ).apply_query_capacity_contract(contract)
    assert transformer.min_active_queries == 128
    assert TrackerConfig.from_calibration_contract(contract) == TrackerConfig()
    observations['split'] = 'dev_holdout'
    with pytest.raises(ValueError, match='calibration split'):
        build_contract(observations, source)


def test_finalizer_requires_completed_es5_and_fingerprints_both_domain_stats(tmp_path):
    model_state = {
        'backbone.stem.bn.norms.0.running_mean': torch.tensor([1.0, 2.0]),
        'backbone.stem.bn.norms.0.running_var': torch.tensor([3.0, 4.0]),
        'backbone.stem.bn.norms.1.running_mean': torch.tensor([5.0, 6.0]),
        'backbone.stem.bn.norms.1.running_var': torch.tensor([7.0, 8.0]),
        'decoder.weight': torch.ones(1),
    }
    checkpoint = tmp_path / 'completed_es5.pth'
    state = {
        'model': {'weight': torch.ones(1)},
        'optimizer': {'state': {}, 'param_groups': []},
        'lr_scheduler': {'last_epoch': 20},
        'scaler': {'scale': torch.tensor(65536.0)},
        'ema': {'module': model_state, 'updates': 20},
        'last_epoch': 20,
        'stage_controller': {
            'completed': True,
            'current_stage_index': 5,
            'best': {'E-S5': {'checkpoint': str(checkpoint)}},
            'provenance': {
                'initialization': {
                    'sha256': '1' * 64,
                    'selection_evidence_sha256': '2' * 64,
                },
                'design_evidence': {'sha256': '3' * 64},
                'split_contract': {
                    'source_manifest_sha256': '4' * 64,
                    'derived_manifest_sha256': '5' * 64,
                    'train_annotation_sha256': '8' * 64,
                    'calibration_annotation_sha256': '9' * 64,
                    'dev_holdout_annotation_sha256': '6' * 64,
                },
                'input_distribution': {'sha256': '7' * 64},
            },
        },
    }
    torch.save(state, checkpoint)
    loaded, checkpoint_sha = validate_completed_checkpoint(checkpoint)
    assert loaded['last_epoch'] == 20
    assert checkpoint_sha == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fingerprint = domain_statistics_fingerprint(model_state)
    assert fingerprint['domains'] == ['rgb', 'ir']
    assert fingerprint['tensor_count'] == 4
    assert len(fingerprint['sha256']) == 64

    state['stage_controller']['completed'] = False
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='not a completed'):
        validate_completed_checkpoint(checkpoint)


def test_finalizer_rechecks_all_fixed_split_role_ids_and_hashes(tmp_path):
    config=_schema_v2_test_contract(tmp_path)
    verified=validate_split_contract(config)
    assert verified['calibration'][1]==config['bee_e_split_contract']['calibration_annotation_sha256']
    path=Path(config['bee_e_split_contract']['calibration_annotation'])
    data=json.loads(path.read_text());data['images'][0]['source_json']='forged.json'
    path.write_text(json.dumps(data))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    config['bee_e_split_contract']['calibration_annotation_sha256']=digest
    config['val_dataloader']['dataset']['expected_ann_sha256']=digest
    with pytest.raises(ValueError,match='identities disagree'):
        validate_split_contract(config)


def test_onnx_numerical_acceptance_requires_stratified_rgb_ir_calibration(tmp_path):
    rgb = tmp_path / 'rgb.png'
    ir = tmp_path / 'ir.png'
    Image.new('RGB', (8, 8), color=(30, 60, 90)).save(rgb)
    Image.new('RGB', (8, 8), color=(120, 120, 120)).save(ir)
    annotation = tmp_path / 'calibration_annotation.json'
    annotation.write_text(
        json.dumps({'images': [], 'annotations': [], 'categories': []}),
        encoding='utf-8',
    )
    manifest = {
        'split': 'calibration',
        'calibration_annotation': {
            'path': str(annotation),
            'sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
        },
        'frames': [
            {
                'domain': 'rgb', 'image': str(rgb),
                'strata': {
                    'video_id': 'rgb-1', 'density_bin': 'low',
                    'scale_bin': 'small', 'occlusion_bin': 'clear',
                },
            },
            {
                'domain': 'ir', 'image': str(ir), 'clip': [str(ir)] * 3,
                'strata': {
                    'video_id': 'ir-1', 'density_bin': 'high',
                    'scale_bin': 'tiny', 'occlusion_bin': 'occluded',
                },
            },
        ],
    }
    path = tmp_path / 'onnx_calibration.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    records = load_stratified_calibration_manifest(path)
    assert [record['domain'] for record in records] == ['rgb', 'ir']
    assert records[0]['image_sha256']
    manifest['split'] = 'dev_holdout'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='split=calibration'):
        load_stratified_calibration_manifest(path)


def _schema_v2_test_contract(tmp_path):
    # 七个合成区段：验证5:1:1隔离、文件哈希和执行输入的一致性。
    roles=['train']*5+['calibration','dev_holdout']
    rows=[{'source_json':f'synthetic/{i}.json','video_id':'video-rgb',
           'section_id':f'v/s{i}','near_duplicate_cluster':f'n{i}','track_scope':f't{i}',
           'split':role,'id':i+1,'canonical_image_sha256':hashlib.sha256(str(i).encode()).hexdigest()}
          for i,role in enumerate(roles)]
    contract={'protocol':'video_section_near_duplicate_track_scope_fixed_5_1_1'}
    for name in ('source_manifest','derived_manifest'):
        path=tmp_path/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        contract[name]=str(path);contract[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    config={'bee_e_split_contract':contract}
    for role in ('train','calibration','dev_holdout'):
        path=tmp_path/(role+'.json')
        path.write_text(json.dumps({'images':[r for r in rows if r['split']==role],'annotations':[]}),encoding='utf-8')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        contract[role+'_annotation']=str(path);contract[role+'_annotation_sha256']=digest
        if role!='dev_holdout':config['train_dataloader' if role=='train' else 'val_dataloader']={'dataset':{'ann_file':str(path),'expected_ann_sha256':digest}}
    return config
