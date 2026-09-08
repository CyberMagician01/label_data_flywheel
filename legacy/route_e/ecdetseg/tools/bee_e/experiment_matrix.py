"""Executable experiment-matrix contract for BeePoseTrack-E."""

import itertools
import json
from pathlib import Path


EXPERIMENT_AXES = {
    'detector_scale': ['s', 'm', 'l'],
    'input_size': [640, 960, 1280],
    'query_capacity': [300, 512, 768],
    'p2_channels': [128, 192, 256],
    'endpoint_sampling_points': [4, 8, 12],
    'domain_adapter_rank': [0, 4, 8, 16, 32],
    'foreground_prototype_quality_threshold': [0.5, 0.7, 0.85],
    'task_adaptation': [
        'native_capacity', 'ncap', 'p2_ncap', 'p2_ncap_joint_endpoints',
    ],
    'pose_decoder': ['direct', 'global_endpoints', 'decomposed_local_endpoints'],
    'query_initialization': ['learned', 'encoder_topk', 'density_peak_encoder_topk'],
    'temporal_injection': ['encoder', 'decoder_query', 'encoder_and_decoder'],
    'temporal_mechanism': [
        'single_frame', 'unstabilized_three_frame_mean',
        'unstabilized_three_frame_residual', 'stabilized_three_frame_residual',
    ],
    'domain_model': ['shared_norm', 'dual_domain_norm', 'dual_domain_norm_low_rank'],
    'structure_knowledge': ['none', 'local_pose', 'full_morphology_temporal'],
    'semi_supervised': ['ema', 'trex2', 'set_trajectory_consistency'],
    'missing_annotation_robustness': [
        'original_no_object', 'ignore_queries', 'ignore_queries_soft_pseudo_set',
    ],
    'trajectory_tail': [
        'original_distribution', 'geometric_spatiotemporal_tube',
        'residual_soft_mask_dual_domain_btca',
    ],
    'module_removal': [
        'full', 'without_p2', 'without_density_query',
        'without_stabilized_temporal', 'without_domain_adapter',
        'without_structural_losses', 'without_quality_aux_assignment',
    ],
}


REQUIRED_METRICS = (
    'official_score', 'rgb_ap', 'ir_ap', 'rgb_recall', 'ir_recall',
    'weakest_video_recall', 'pose_nme', 'head_tail_swap_rate',
    'query_utilization', 'unmatched_gt_rate', 'duplicate_queries_per_frame',
    'capacity_truncation_rate_450plus', 'mota_fixed_tracker',
    'idf1_fixed_tracker', 'hota_fixed_tracker', 'cast_fixed_tracker',
    'fragment_rate_fixed_tracker', 'onnx_parameters', 'onnx_size_mb',
    'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
    'peak_memory_mb', 'endpoint_decoder_flops',
)


def staged_jobs(base_config, seeds=(2026, 3407, 827), folds=range(4)):
    """Expand one axis at a time so the matrix stays auditable and affordable."""
    jobs = []
    for axis, values in EXPERIMENT_AXES.items():
        for value, seed, fold in itertools.product(values, seeds, folds):
            jobs.append({
                'base_config': str(base_config),
                'axis': axis,
                'value': value,
                'seed': int(seed),
                'fold': int(fold),
                'selection_manifest': f'fold{fold}_selection.json',
                'calibration_manifest': f'fold{fold}_calibration.json',
                'test_manifest': f'fold{fold}_test.json',
                'status': 'pending',
            })
    return jobs


def validate_result_contract(result):
    missing = [metric for metric in REQUIRED_METRICS if metric not in result.get('metrics', {})]
    if missing:
        raise ValueError(f'missing experiment metrics: {missing}')
    if result.get('tracker_parameters_tuned_on_test', False):
        raise ValueError('tracker parameters must be fixed from the calibration split')
    if result.get('checkpoint_selected_on_test', False):
        raise ValueError('checkpoint must be selected on the selection split')
    return True


def write_matrix(output, base_config):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema_version': 1,
        'axes': EXPERIMENT_AXES,
        'required_metrics': REQUIRED_METRICS,
        'jobs': staged_jobs(base_config),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
