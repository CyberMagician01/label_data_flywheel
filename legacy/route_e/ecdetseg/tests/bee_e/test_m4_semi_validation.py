import itertools
import sys
import types

import numpy as np
import torch


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

from engine.data.dataloader import DataLoader, DomainBalancedSampler
from engine.misc import dist_utils
from engine.solver.semi_supervised import OnlinePseudoLabeler
from tools.bee_e.experiment_matrix import (
    REQUIRED_METRICS,
    staged_jobs,
    validate_result_contract,
)
from tools.bee_e.split_protocol import (
    build_fixed_511_manifest,
    build_paired_four_fold_manifest,
    summarize_seed_fold_results,
    validate_fixed_511_manifest,
    validate_split_manifest,
)
from tools.bee_e.input_distribution_report import analyze_input_distribution


class _Teacher(torch.nn.Module):
    def forward(self, samples, temporal_valid_mask=None):
        batch = samples.shape[0]
        device = samples.device
        return {
            'pred_logits': torch.full((batch, 2, 1), 8.0, device=device),
            'pred_quality': torch.full((batch, 2), 8.0, device=device),
            'pred_query_valid': torch.tensor([[True, False]], device=device).expand(batch, -1),
            'pred_boxes': torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]], device=device
            ).expand(batch, -1, -1),
            'pred_keypoints': torch.tensor(
                [[[[0.45, 0.5], [0.55, 0.5]], [[0.78, 0.8], [0.82, 0.8]]]], device=device
            ).expand(batch, -1, -1, -1),
            'pred_visibility': torch.full((batch, 2, 2), 8.0, device=device),
        }


def _empty_unlabelled_target():
    return {
        'is_unlabeled': torch.tensor([True]),
        'image_id': torch.tensor([1]),
        'domain_id': torch.tensor([1]),
        'trex2_boxes': torch.tensor([[0.1, 0.1, 0.05, 0.05]]),
        'trex2_scores': torch.tensor([0.99]),
    }


def test_online_pseudo_uses_set_and_trajectory_without_query_ids():
    samples = torch.zeros(1, 5, 3, 32, 32)
    targets, stats = OnlinePseudoLabeler(endpoint_threshold=1.0).generate(
        _Teacher(), samples, [_empty_unlabelled_target()]
    )
    assert stats['pseudo_instances'] == 1
    assert len(targets[0]['boxes']) == 1
    assert 'query_id' not in targets[0]
    assert targets[0]['trajectory_stability'][0] == 1
    assert 0 < targets[0]['pseudo_score'][0] <= 1
    assert targets[0]['track_id'][0] == -1
    assert not targets[0]['track_mask'][0]
    assert not targets[0]['track_geometry_mask'][0]


def test_online_pseudo_rejects_padding_only_history():
    samples = torch.zeros(1, 5, 3, 32, 32)
    target = _empty_unlabelled_target()
    target['temporal_valid_mask'] = torch.tensor([False, False, False, False, True])
    targets, stats = OnlinePseudoLabeler(endpoint_threshold=1.0).generate(
        _Teacher(), samples, [target]
    )
    assert stats['pseudo_instances'] == 0
    assert len(targets[0]['boxes']) == 0


class _SamplerDataset:
    def __len__(self):
        return 8

    domain_indices = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    labelled_indices = {0: [0, 1], 1: [4, 5]}
    unlabelled_indices = {0: [2, 3], 1: [6, 7]}
    hard_negative_indices = {0: [], 1: []}


def test_sampler_balances_domains_and_can_separate_unlabelled_pool():
    indices = list(DomainBalancedSampler(
        _SamplerDataset(), unlabelled_ratio=1.0, hard_negative_ratio=0.0, seed=1
    ))
    assert set(indices[:8]) == set(range(8))
    assert indices[-2] in (2, 3)
    assert indices[-1] in (6, 7)
    assert all(index in (0, 1, 2, 3) for index in indices[0::2])
    assert all(index in (4, 5, 6, 7) for index in indices[1::2])


def test_sampler_pairs_rgb_ir_across_two_ddp_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    rank = {'value': 0}
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank['value'])
    rank0 = list(DomainBalancedSampler(
        _SamplerDataset(), unlabelled_ratio=0.0, hard_negative_ratio=0.0, seed=2,
    ))
    rank['value'] = 1
    rank1 = list(DomainBalancedSampler(
        _SamplerDataset(), unlabelled_ratio=0.0, hard_negative_ratio=0.0, seed=2,
    ))
    assert all(index in _SamplerDataset.domain_indices[0] for index in rank0)
    assert all(index in _SamplerDataset.domain_indices[1] for index in rank1)


def test_ddp_loader_wrap_preserves_worker_performance_options(monkeypatch):
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )
    monkeypatch.setattr(dist_utils, 'is_dist_available_and_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    wrapped = dist_utils.warp_loader(loader, shuffle=True)

    assert wrapped.num_workers == 1
    assert wrapped.pin_memory
    assert wrapped.persistent_workers
    assert wrapped.prefetch_factor == 3


def _coco_protocol_fixture():
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1
    for environment in ('inside', 'outside'):
        for sequence_index in range(4):
            sequence = f'{environment}_{sequence_index}'
            for frame in range(2):
                images.append({
                    'id': image_id,
                    'sequence_id': sequence,
                    'environment': environment,
                    'frame_id': frame,
                    'derived_group_id': f'{sequence}_{frame}',
                })
                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'track_id': sequence_index,
                })
                image_id += 1
                annotation_id += 1
    return {'images': images, 'annotations': annotations}


def test_four_fold_protocol_is_sequence_disjoint():
    coco = _coco_protocol_fixture()
    manifest = build_paired_four_fold_manifest(coco)
    assert len(manifest['folds']) == 4
    assert validate_split_manifest(coco, manifest)
    for fold in manifest['folds']:
        roles = fold['roles']
        assert all(len(roles[role]['inside_sequences']) == 1 for role in roles)
        assert all(len(roles[role]['outside_sequences']) == 1 for role in roles)


def test_fixed_511_protocol_keeps_linked_groups_and_every_video_in_each_role():
    images = []
    image_id = 1
    for video in ('video_a', 'video_b'):
        for section in range(7):
            images.append({
                'id': image_id,
                'sequence_id': video,
                'section_id': section,
                'near_duplicate_cluster': f'{video}_dup_{section}',
                'track_scope': f'{video}_track_{section}',
                'temporal_group_id': f'{video}_tube_{section}',
                'derived_group_id': f'{video}_derived_{section}',
            })
            image_id += 1
    coco = {'images': images, 'annotations': []}
    manifest = build_fixed_511_manifest(coco, seed=2026)
    assert validate_fixed_511_manifest(coco, manifest)
    for video in ('video_a', 'video_b'):
        assert len(manifest['roles']['train']['video_sections'][video]) == 5
        assert len(manifest['roles']['calibration']['video_sections'][video]) == 1
        assert len(manifest['roles']['dev_holdout']['video_sections'][video]) == 1


def test_three_seed_four_fold_statistics_and_worst_fold():
    records = [
        {'seed': seed, 'fold': fold, 'metrics': {'ir_ap': 0.5 + fold * 0.01}}
        for seed in (2026, 3407, 827) for fold in range(4)
    ]
    summary = summarize_seed_fold_results(records, {'ir_ap': 'max'})
    assert summary['runs'] == 12
    assert summary['metrics']['ir_ap']['worst_fold'] == 0


def test_experiment_matrix_uses_distinct_selection_calibration_test():
    jobs = staged_jobs('formal.yml')
    assert jobs
    first = jobs[0]
    assert len({
        first['selection_manifest'], first['calibration_manifest'], first['test_manifest']
    }) == 3
    metrics = {name: 0.0 for name in REQUIRED_METRICS}
    assert validate_result_contract({'metrics': metrics})
    assert {
        'query_capacity', 'p2_channels', 'endpoint_sampling_points',
        'domain_adapter_rank', 'module_removal',
    } <= {job['axis'] for job in jobs}


def _distribution_fixture():
    result = {}
    annotation_id = 1
    for split_index, split in enumerate(('train', 'calibration', 'dev_holdout')):
        images, annotations = [], []
        for video_index, (video, domain) in enumerate(
            (('rgb-video', 'RGB'), ('ir-video', 'IR'))
        ):
            image_id = split_index * 10 + video_index + 1
            images.append({
                'id': image_id, 'width': 100, 'height': 100,
                'domain': domain, 'sequence_id': video,
                'section_id': split_index,
                'near_duplicate_cluster': f'{split}-{video}',
                'track_scope': f'{split}-{video}', 'source': 'self',
                'sampling_weight': 1.0,
            })
            annotations.append({
                'id': annotation_id, 'image_id': image_id,
                'bbox': [10, 10, 20, 20],
                'track_id': split_index * 10 + video_index,
                'supervision_mask': [True, True, True, True],
            })
            annotation_id += 1
        result[split] = {'images': images, 'annotations': annotations}
    return result


def test_input_distribution_report_covers_contract_axes_and_rejects_leakage():
    datasets = _distribution_fixture()
    report = analyze_input_distribution(datasets, query_capacity=4)
    assert report['ready_for_training']
    assert set(report['splits']['train']['domains']) == {'RGB', 'IR'}
    assert report['splits']['train']['ess_ratio'] == 1.0
    assert report['group_isolation']['leakage_count'] == 0
    datasets['train']['images'][0]['support_group_id'] = 'leak'
    datasets['calibration']['images'][0]['support_group_id'] = 'leak'
    leaked = analyze_input_distribution(datasets, query_capacity=4)
    assert not leaked['ready_for_training']
    assert leaked['group_isolation']['leakages']['temporal_support_groups'] == ['leak']
