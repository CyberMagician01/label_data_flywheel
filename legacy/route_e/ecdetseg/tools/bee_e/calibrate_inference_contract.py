"""Derive and freeze every BeePoseTrack-E inference parameter on calibration only."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _average_precision(labels, scores):
    ordered = sorted(zip(scores, labels), reverse=True)
    positives = sum(bool(label) for label in labels)
    if positives == 0:
        return 0.0
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _score(record, exponents):
    alpha, beta, gamma = exponents
    return (
        max(float(record['class_score']), 1e-8) ** alpha
        * max(float(record['quality']), 1e-8) ** beta
        * max(float(record['pose_visibility']), 1e-8) ** gamma
    )


def _best_threshold(labels, scores):
    best = None
    for threshold in sorted(set(scores)):
        predicted = [score >= threshold for score in scores]
        tp = sum(prediction and label for prediction, label in zip(predicted, labels))
        fp = sum(prediction and not label for prediction, label in zip(predicted, labels))
        fn = sum(not prediction and label for prediction, label in zip(predicted, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        candidate = (f1, recall, precision, -threshold)
        if best is None or candidate > best[0]:
            best = candidate, threshold
    return float(best[1]), {
        'f1': best[0][0], 'recall': best[0][1], 'precision': best[0][2],
    }


def calibrate_ranking(candidates, exponent_grid=(0.5, 1.0, 1.5, 2.0)):
    labels = [bool(record['is_true_positive']) for record in candidates]
    if not candidates or len(set(labels)) < 2:
        raise ValueError('ranking calibration requires both TP and FP candidates')
    best = None
    for exponents in itertools.product(exponent_grid, repeat=3):
        scores = [_score(record, exponents) for record in candidates]
        ap = _average_precision(labels, scores)
        threshold, threshold_metrics = _best_threshold(labels, scores)
        candidate = (ap, threshold_metrics['recall'], -sum(exponents))
        if best is None or candidate > best[0]:
            best = candidate, exponents, threshold, threshold_metrics
    return {
        'score_exponents': list(best[1]),
        'score_threshold': best[2],
        'calibration_ap': best[0][0],
        **best[3],
    }


def calibrate_deduplication(pairs):
    if not pairs or len({bool(pair['is_duplicate']) for pair in pairs}) < 2:
        raise ValueError('dedup calibration requires duplicate and overlapping-distinct pairs')
    candidates = {
        'box_iou_min': sorted({float(pair['box_iou']) for pair in pairs}),
        'endpoint_oks_min': sorted({float(pair['endpoint_oks']) for pair in pairs}),
        'relative_length_max': sorted({float(pair['relative_length_difference']) for pair in pairs}),
        'oriented_axis_deg_max': sorted({float(pair['oriented_axis_difference_deg']) for pair in pairs}),
    }
    best = None
    for values in itertools.product(*candidates.values()):
        thresholds = dict(zip(candidates, values))
        predicted = [
            float(pair['box_iou']) >= thresholds['box_iou_min']
            and float(pair['endpoint_oks']) >= thresholds['endpoint_oks_min']
            and float(pair['relative_length_difference']) <= thresholds['relative_length_max']
            and float(pair['oriented_axis_difference_deg']) <= thresholds['oriented_axis_deg_max']
            for pair in pairs
        ]
        labels = [bool(pair['is_duplicate']) for pair in pairs]
        tp = sum(prediction and label for prediction, label in zip(predicted, labels))
        tn = sum(not prediction and not label for prediction, label in zip(predicted, labels))
        fp = sum(prediction and not label for prediction, label in zip(predicted, labels))
        fn = sum(not prediction and label for prediction, label in zip(predicted, labels))
        duplicate_recall = tp / max(tp + fn, 1)
        distinct_preservation = tn / max(tn + fp, 1)
        balanced_accuracy = 0.5 * (duplicate_recall + distinct_preservation)
        candidate = (balanced_accuracy, distinct_preservation, duplicate_recall)
        if best is None or candidate > best[0]:
            best = candidate, thresholds
    return {
        **best[1], 'balanced_accuracy': best[0][0],
        'distinct_preservation': best[0][1], 'duplicate_recall': best[0][2],
    }


def select_density_recheck(trials, latency_budget_ms):
    feasible = [
        trial for trial in trials
        if float(trial['latency_ms']) <= float(latency_budget_ms)
    ]
    if not feasible:
        raise ValueError('no density recheck trial satisfies the latency budget')
    selected = max(feasible, key=lambda trial: (
        float(trial['new_true_positives']) - float(trial['new_false_positives']),
        float(trial['new_true_positives']), -float(trial['latency_ms']),
    ))
    return {
        name: selected[name] for name in (
            'trigger_count', 'residual_threshold', 'expansion_rate',
            'gaussian_sigma_ratio', 'max_passes',
        )
    } | {
        'new_true_positives': selected['new_true_positives'],
        'new_false_positives': selected['new_false_positives'],
        'latency_ms': selected['latency_ms'],
    }


def select_query_capacity(trials, latency_budget_ms):
    """Freeze the monotone density/uncertainty query map on calibration."""
    if not trials:
        raise ValueError('query-capacity calibration trials are required')
    required = {
        'density_capacity_ratio', 'density_capacity_padding',
        'uncertainty_capacity_scale', 'min_active_queries',
        'stage_query_limit', 'rgb_recall', 'ir_recall',
        'capacity_truncation_rate_450plus', 'query_utilization', 'latency_ms',
    }
    for trial in trials:
        missing = required - set(trial)
        if missing:
            raise ValueError(f'query-capacity trial misses {sorted(missing)}')
        if not 1 <= int(trial['min_active_queries']) <= int(trial['stage_query_limit']):
            raise ValueError('query-capacity trial violates min <= stage limit')
    feasible = [
        trial for trial in trials
        if float(trial['latency_ms']) <= float(latency_budget_ms)
    ]
    if not feasible:
        raise ValueError('no query-capacity trial satisfies the latency budget')
    selected = max(feasible, key=lambda trial: (
        min(float(trial['rgb_recall']), float(trial['ir_recall'])),
        -float(trial['capacity_truncation_rate_450plus']),
        float(trial['query_utilization']),
        -float(trial['latency_ms']),
    ))
    return {
        'parameters': {
            name: selected[name] for name in (
                'density_capacity_ratio', 'density_capacity_padding',
                'uncertainty_capacity_scale', 'min_active_queries',
                'stage_query_limit',
            )
        },
        'metrics': {
            name: selected[name] for name in (
                'rgb_recall', 'ir_recall',
                'capacity_truncation_rate_450plus',
                'query_utilization', 'latency_ms',
            )
        },
    }


def select_tracker(trials):
    if not trials:
        raise ValueError('tracker calibration trials are required')
    required_parameters = {
        'high_score_threshold', 'low_score_threshold', 'new_track_threshold',
        'max_missed', 'min_hits', 'mahalanobis_gate', 'max_assignment_cost',
        'process_noise', 'measurement_noise', 'observation_velocity_blend',
        'swap_transition_penalty', 'weight_mahalanobis', 'weight_iou',
        'weight_endpoint_oks', 'weight_axis', 'weight_length_ratio',
        'weight_region',
    }
    for trial in trials:
        missing = required_parameters - set(trial.get('parameters', {}))
        if missing:
            raise ValueError(f'tracker calibration trial misses {sorted(missing)}')
        metric_missing = {
            'mota', 'idf1', 'hota', 'cast', 'fragment_rate',
        } - set(trial)
        if metric_missing:
            raise ValueError(f'tracker calibration trial misses {sorted(metric_missing)}')
    def score(trial):
        return (
            float(trial['idf1']) + float(trial['hota']) + float(trial['cast'])
            + float(trial['mota']) - float(trial['fragment_rate'])
        )
    selected = max(trials, key=lambda trial: (score(trial), float(trial['idf1'])))
    return {'parameters': selected['parameters'], 'metrics': {
        name: selected[name] for name in (
            'mota', 'idf1', 'hota', 'cast', 'fragment_rate'
        )
    }}


def _verified_calibration_annotation(observations, expected_sha256=None):
    annotation = observations.get('calibration_annotation', {})
    path_value = annotation.get('path')
    expected = str(annotation.get('sha256') or '').lower()
    if not path_value or len(expected) != 64:
        raise ValueError(
            'calibration observations must bind calibration_annotation.path and SHA256'
        )
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'calibration annotation not found: {path}')
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f'calibration annotation SHA256 mismatch: expected={expected}, actual={actual}'
        )
    if expected_sha256 is not None and actual != str(expected_sha256).lower():
        raise ValueError(
            'calibration observations are not bound to the fixed split calibration file'
        )
    return path, actual


def build_contract(observations, source_path,
                   expected_calibration_annotation_sha256=None):
    if observations.get('split') != 'calibration':
        raise ValueError('inference parameters may only be fitted on the calibration split')
    calibration_annotation, calibration_annotation_sha = (
        _verified_calibration_annotation(
            observations, expected_calibration_annotation_sha256,
        )
    )
    result = {
        'schema_version': 1,
        'source': {
            'path': str(Path(source_path).resolve()),
            'sha256': _sha256(source_path),
            'split': 'calibration',
            'calibration_annotation': str(calibration_annotation),
            'calibration_annotation_sha256': calibration_annotation_sha,
        },
        'effective_stage': 'E-S5',
        'domains': {},
        'query_capacity': select_query_capacity(
            observations.get('query_capacity_trials', []),
            observations['latency_budget_ms'],
        ) | {
            'source_data': 'calibration',
            'evaluation_target': (
                'worst-domain Recall, 450+ capacity truncation, query utilization, and latency'
            ),
            'effective_stage': 'E-S5',
        },
    }
    for domain_name, domain_id in (('rgb', 0), ('ir', 1)):
        candidates = [
            record for record in observations['candidates']
            if int(record['domain_id']) == domain_id
        ]
        pairs = [
            record for record in observations['duplicate_pairs']
            if int(record['domain_id']) == domain_id
        ]
        trials = [
            record for record in observations['density_recheck_trials']
            if int(record['domain_id']) == domain_id
        ]
        result['domains'][domain_name] = {
            'ranking': calibrate_ranking(candidates),
            'set_deduplication': calibrate_deduplication(pairs),
            'density_recheck': select_density_recheck(
                trials, observations['latency_budget_ms']
            ),
            'source_data': 'calibration',
            'evaluation_target': (
                'official score, Recall, distinct-overlap preservation, and added TP saturation'
            ),
            'effective_stage': 'E-S5',
        }
    result['tracker'] = select_tracker(observations['tracker_trials'])
    result['tracker'].update({
        'source_data': 'calibration',
        'evaluation_target': 'MOTA/IDF1/HOTA/CAST with minimum fragment rate',
        'effective_stage': 'E-S5',
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observations = json.loads(args.observations.read_text(encoding='utf-8'))
    contract = build_contract(observations, args.observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
