"""Validate evidence-derived BeePoseTrack-E architecture and training choices."""

import json


FRONTIER_METRICS = {
    'p2_channels': {
        'weakest_video_recall': 'max', 'pose_nme': 'min',
        'peak_memory_mb': 'min', 'end_to_end_latency_ms': 'min',
    },
    'endpoint_sampling_points': {
        'weakest_video_recall': 'max', 'pose_nme': 'min',
        'head_tail_swap_rate': 'min', 'peak_memory_mb': 'min',
        'end_to_end_latency_ms': 'min',
    },
    'domain_adapter_rank': {
        'rgb_score': 'max', 'ir_score': 'max',
        'residual_spectrum_explained': 'max',
        'peak_memory_mb': 'min', 'end_to_end_latency_ms': 'min',
    },
    'stabilizer': {
        'mask_reproduction': 'max', 'false_motion_rate': 'min',
        'alignment_residual': 'min', 'end_to_end_latency_ms': 'min',
    },
    'ir_normalization': {
        'foreground_recall': 'max', 'background_false_response': 'min',
        'temporal_scale_residual': 'min',
    },
    'foreground_prototype_quality_threshold': {
        'rgb_foreground_alignment': 'max',
        'ir_foreground_alignment': 'max',
        'weak_annotation_contamination': 'min',
    },
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _dominates(left, right, directions):
    no_worse = True
    strictly_better = False
    for metric, direction in directions.items():
        left_value = float(left[metric])
        right_value = float(right[metric])
        if direction == 'max':
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
    return bool(no_worse and strictly_better)


def _selected_frontier_trial(axis, trials, selected):
    if len(trials) < 2:
        raise ValueError(f'{axis} evidence requires at least two measured trials')
    directions = FRONTIER_METRICS[axis]
    selected_trials = [
        trial for trial in trials
        if _canonical(trial.get('value')) == _canonical(selected)
    ]
    if len(selected_trials) != 1:
        raise ValueError(f'{axis} evidence must contain the selected value exactly once')
    for trial in trials:
        missing = set(directions) - set(trial)
        if missing:
            raise ValueError(f'{axis} trial misses {sorted(missing)}')
    chosen = selected_trials[0]
    if any(_dominates(other, chosen, directions) for other in trials if other is not chosen):
        raise ValueError(f'selected {axis} is dominated by another measured trial')
    return chosen


def _transform_operation(config, operation_type):
    operations = config.get('train_dataloader', {}).get(
        'dataset', {},
    ).get('transforms', {}).get('ops', [])
    matches = [operation for operation in operations if operation.get('type') == operation_type]
    if len(matches) != 1:
        raise ValueError(f'configuration requires exactly one {operation_type}')
    return matches[0]


def validate_route_design_evidence(evidence, config):
    """Reject a route whose fixed design values are not evidence-backed."""
    if evidence.get('split') != 'calibration':
        raise ValueError('route design evidence must be measured on calibration')
    expected_annotation_sha = config.get('val_dataloader', {}).get(
        'dataset', {},
    ).get('expected_ann_sha256')
    if evidence.get('calibration_annotation_sha256') != expected_annotation_sha:
        raise ValueError('route design evidence is not bound to the calibration annotation')

    adapter = config['ViTAdapter']
    projection = adapter['proj_dim']
    p2_channels = int(projection[0] if isinstance(projection, list) else projection)
    decoder = config['ECTransformer']
    selected = {
        'p2_channels': p2_channels,
        'endpoint_sampling_points': int(decoder['endpoint_sampling_points']),
        'domain_adapter_rank': int(adapter['domain_adapter_rank']),
        'foreground_prototype_quality_threshold': float(
            config['ECCriterion']['domain_foreground_quality_threshold']
        ),
    }
    prepare = _transform_operation(config, 'PrepareTemporalFrames')
    selected['stabilizer'] = dict(prepare.get('stabilizer', {}))
    ir_normalize = _transform_operation(config, 'IRPercentileNormalize')
    selected['ir_normalization'] = {
        name: float(ir_normalize[name])
        for name in ('lower', 'upper', 'foreground_residual_quantile')
    }
    if evidence.get('selected') != selected:
        raise ValueError('route design selected values disagree with the executable config')
    evidence_type = evidence.get('evidence_type')
    frontiers = evidence.get('frontiers', {})
    if evidence_type == 'measured_design_evidence':
        for axis, value in selected.items():
            _selected_frontier_trial(axis, frontiers.get(axis, []), value)
    elif evidence_type == 'preregistered_design_experiment':
        if evidence.get('status') != 'registered_before_training':
            raise ValueError('preregistered design evidence has an invalid status')
        experiments = evidence.get('experiments', {})
        if set(experiments) != set(selected):
            raise ValueError('preregistered design experiments must cover every fixed axis')
        for axis, value in selected.items():
            experiment = experiments[axis]
            candidates = experiment.get('candidate_values', [])
            if len(candidates) < 2 or not any(
                _canonical(candidate) == _canonical(value) for candidate in candidates
            ):
                raise ValueError(f'{axis} preregistration misses its selected value or alternative')
            if experiment.get('metrics') != list(FRONTIER_METRICS[axis]):
                raise ValueError(f'{axis} preregistration metrics disagree with the route contract')
            if experiment.get('measurement_status') != 'pending_training_measurement':
                raise ValueError(f'{axis} must remain explicitly pending before training')
    else:
        raise ValueError('design evidence must be measured or honestly preregistered')

    optimizer = config['optimizer']
    train_loader = config['train_dataloader']
    criterion = config['ECCriterion']
    optimization = evidence.get('optimization', {})
    expected_optimization = {
        'optimizer': 'AdamW',
        'base_lr': float(optimizer['lr']),
        'weight_decay': float(optimizer['weight_decay']),
        'total_batch_size': int(train_loader['total_batch_size']),
        'grad_accum_steps': int(config['grad_accum_steps']),
        'effective_batch_size': int(
            train_loader['total_batch_size'] * config['grad_accum_steps']
        ),
        'use_amp': bool(config['use_amp']),
        'clip_max_norm': float(config['clip_max_norm']),
        'query_clip_max_norm': float(config['query_clip_max_norm']),
        'ema_decay': float(config.get('ema_decay', 0.9999)),
        'loss_weights': criterion['weight_dict'],
    }
    if optimization.get('selected') != expected_optimization:
        raise ValueError('evidence-derived optimization values disagree with config')
    if evidence_type == 'measured_design_evidence':
        gradient_trace = optimization.get('gradient_trace', [])
        if not gradient_trace or any(
            {'parameter_group', 'gradient_norm', 'train_calibration_gap'} - set(record)
            for record in gradient_trace
        ):
            raise ValueError('optimization evidence requires S0 gradient and gap traces')
        mixed_precision = optimization.get('mixed_precision_test', {})
        if not mixed_precision.get('optimizer_step_equivalent') or not {
            'max_loss_error', 'max_gradient_error',
        } <= set(mixed_precision):
            raise ValueError('mixed-precision numerical equivalence evidence is incomplete')
    else:
        protocol = optimization.get('measurement_protocol', {})
        if protocol.get('status') != 'pending_training_measurement':
            raise ValueError('optimization measurement protocol must be pending before training')
        if set(protocol.get('gradient_trace_fields', [])) != {
            'parameter_group', 'gradient_norm', 'train_calibration_gap',
        }:
            raise ValueError('optimization preregistration misses gradient trace fields')
        if set(protocol.get('mixed_precision_fields', [])) != {
            'optimizer_step_equivalent', 'max_loss_error', 'max_gradient_error',
        }:
            raise ValueError('optimization preregistration misses mixed-precision fields')

    stage_budget = evidence.get('stage_budget', [])
    expected_stages = {
        stage['name']: (int(stage['min_cycles']), int(stage['max_cycles']))
        for stage in config['bee_e_stage_specs']
    }
    actual_stages = {
        record.get('stage'): (int(record.get('min_cycles', -1)),
                              int(record.get('max_cycles', -1)))
        for record in stage_budget
    }
    if actual_stages != expected_stages:
        raise ValueError('stage coverage evidence disagrees with E-S0..E-S5 budgets')
    required_stage_metrics = {
        'coverage_speed', 'metric_autocorrelation', 'bootstrap_noise_bound',
    }
    if evidence_type == 'measured_design_evidence':
        if any(required_stage_metrics - set(record) for record in stage_budget):
            raise ValueError('stage budget evidence misses convergence measurements')
    elif any(
        record.get('measurement_status') != 'pending_training_measurement'
        or set(record.get('metrics', [])) != required_stage_metrics
        for record in stage_budget
    ):
        raise ValueError('stage budget preregistration misses convergence measurement fields')
    return {
        'selected': selected,
        'evidence_type': evidence_type,
        'frontiers': sorted(frontiers) if frontiers else [],
        'stages': list(expected_stages),
    }


__all__ = ['validate_route_design_evidence']
