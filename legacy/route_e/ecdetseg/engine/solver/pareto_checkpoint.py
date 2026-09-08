import hashlib
import json
from pathlib import Path

import torch


DEFAULT_DIRECTIONS = {
    'bbox_ap': 'max',
    'rgb_ap': 'max',
    'ir_ap': 'max',
    'pose_pck10': 'max',
    'pose_nme': 'min',
    'orientation_accuracy': 'max',
    'head_tail_swap_rate': 'min',
    'query_utilization': 'max',
    'latency_p99_ms': 'min',
    'density_relative_mae': 'min',
}


def flatten_eval_stats(stats):
    bbox = stats.get('coco_eval_bbox', [])
    metrics = {
        'bbox_ap': bbox[0] if len(bbox) > 0 else None,
        'bbox_ap50': bbox[1] if len(bbox) > 1 else None,
        'rgb_ap': stats.get('detection_rgb_map'),
        'ir_ap': stats.get('detection_ir_map'),
        'pose_pck10': stats.get('pose_all_pck@0.10'),
        'pose_nme': stats.get('pose_all_nme_bbox_diagonal'),
        'orientation_accuracy': stats.get('pose_all_orientation_accuracy'),
        'head_tail_swap_rate': (
            1.0 - stats['pose_all_orientation_accuracy']
            if 'pose_all_orientation_accuracy' in stats else None
        ),
        'angle_over45_rate': stats.get('pose_all_angle_over_45deg'),
        'query_utilization': stats.get('query_utilization'),
        'query_spatial_coverage': stats.get('query_spatial_coverage'),
        'query_collapse': stats.get('query_collapse'),
        'pattern_entropy': stats.get('pattern_entropy'),
        'latency_mean_ms': stats.get('latency_mean_ms'),
        'latency_p99_ms': stats.get('latency_p99_ms'),
        'density_relative_mae': stats.get('density_all_relative_mae'),
        'density_mae': stats.get('density_all_mae'),
        'idf1_fixed_tracker': stats.get('tracking_all_idf1'),
    }
    return {key: float(value) for key, value in metrics.items() if value is not None}


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def atomic_write_json(path, payload):
    """Publish JSON only after the complete payload is on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    temporary.replace(path)


class ParetoCheckpointManager:
    """Persist per-checkpoint metrics and select only from the Pareto frontier."""

    def __init__(self, output_dir, metric_directions=None):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / 'checkpoint_metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metric_directions = metric_directions or DEFAULT_DIRECTIONS
        self.records = self._load_records()

    def _load_records(self):
        """Restore completed epoch records so resume keeps the same frontier."""
        records = []
        for path in sorted(self.metrics_dir.glob('epoch*.json')):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                checkpoint = Path(record['checkpoint'])
                if not checkpoint.is_file() or not record.get('sha256'):
                    continue
                if self._sha256(checkpoint) != record['sha256']:
                    continue
                if not isinstance(record.get('metrics'), dict):
                    continue
                records.append(record)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return records

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _complete(self, record):
        return all(name in record['metrics'] for name in self.metric_directions)

    def _dominates(self, left, right):
        no_worse = True
        strictly_better = False
        for name, direction in self.metric_directions.items():
            left_value = left['metrics'][name]
            right_value = right['metrics'][name]
            if direction == 'max':
                no_worse &= left_value >= right_value
                strictly_better |= left_value > right_value
            else:
                no_worse &= left_value <= right_value
                strictly_better |= left_value < right_value
        return no_worse and strictly_better

    def _frontier(self):
        complete = [record for record in self.records if self._complete(record)]
        return [
            record for record in complete
            if not any(self._dominates(other, record) for other in complete if other is not record)
        ]

    def _balanced_score(self, record, complete):
        scores = []
        for name, direction in self.metric_directions.items():
            values = [item['metrics'][name] for item in complete]
            low, high = min(values), max(values)
            if high == low:
                score = 1.0
            else:
                score = (record['metrics'][name] - low) / (high - low)
                if direction == 'min':
                    score = 1.0 - score
            scores.append(score)
        return sum(scores) / len(scores)

    def record(self, epoch, checkpoint_path, eval_stats):
        checkpoint_path = Path(checkpoint_path)
        record = {
            'epoch': int(epoch),
            'checkpoint': str(checkpoint_path.resolve()),
            'sha256': self._sha256(checkpoint_path),
            'metrics': flatten_eval_stats(eval_stats),
            'eval_stats': _jsonable(eval_stats),
        }
        self.records = [item for item in self.records if item.get('epoch') != int(epoch)]
        self.records.append(record)
        self.records.sort(key=lambda item: item['epoch'])
        atomic_write_json(self.metrics_dir / f'epoch{epoch:04}.json', record)

        complete = [item for item in self.records if self._complete(item)]
        frontier = self._frontier()
        for item in complete:
            item['balanced_score'] = self._balanced_score(item, complete)
        selected = max(frontier, key=lambda item: item['balanced_score']) if frontier else None
        manifest = {
            'selector': 'pareto_balanced_v2_resume_safe',
            'metric_directions': self.metric_directions,
            'frontier': frontier,
            'selected': selected,
            'note': 'Stage chaining must read selected.checkpoint; best.pth is not authoritative.',
        }
        atomic_write_json(self.output_dir / 'pareto_frontier.json', manifest)
        atomic_write_json(self.output_dir / 'selected_checkpoint.json', selected or {})
        return selected
