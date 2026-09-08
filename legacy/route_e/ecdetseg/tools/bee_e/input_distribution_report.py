"""Pre-training input-distribution audit for the BeePoseTrack-E route."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch


SPLITS = ('train', 'calibration', 'dev_holdout')
SCALE_BINS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, float('inf'))
DENSITY_BINS = (0, 8, 32, 64, 128, 256, float('inf'))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _domain(image):
    value = str(image.get('domain', image.get('modality', ''))).upper()
    if value in ('IR', 'INFRARED', 'THERMAL'):
        return 'IR'
    if value in ('RGB', 'VISIBLE', 'VIS'):
        return 'RGB'
    raise ValueError(f"image {image.get('id')} has unsupported domain {value!r}")


def _video(image):
    # Coverage is defined on the original video.  ``sequence_id`` is more
    # restrictive because schema-v2 scopes it to one annotated section, and
    # therefore belongs to temporal/track isolation rather than video coverage.
    value = image.get('video_id', image.get('video', image.get('sequence_id')))
    if value is None:
        raise ValueError(f"image {image.get('id')} has no video identifier")
    return str(value)


def _sequence(image):
    value = image.get('sequence_id', image.get('video_id'))
    if value is None:
        raise ValueError(f"image {image.get('id')} has no sequence identifier")
    return str(value)


def _source(image):
    return str(image.get('source', image.get('dataset_source', 'self')))


def _bin(value, edges):
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


def _quantiles(values):
    if not values:
        return {'min': 0.0, 'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        'min': float(tensor.min()), 'p50': float(torch.quantile(tensor, 0.50)),
        'p90': float(torch.quantile(tensor, 0.90)),
        'p95': float(torch.quantile(tensor, 0.95)),
        'p99': float(torch.quantile(tensor, 0.99)), 'max': float(tensor.max()),
    }


def _box_iou(left, right):
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry),
    )
    union = lw * lh + rw * rh - intersection
    return intersection / max(union, 1e-12)


def _total_variation(counter, reference):
    keys = set(counter) | set(reference)
    left_total = max(sum(counter.values()), 1)
    right_total = max(sum(reference.values()), 1)
    return 0.5 * sum(
        abs(counter.get(key, 0) / left_total - reference.get(key, 0) / right_total)
        for key in keys
    )


def _split_records(coco_by_split):
    records = {}
    for split, coco in coco_by_split.items():
        annotations = defaultdict(list)
        for annotation in coco.get('annotations', []):
            annotations[annotation['image_id']].append(annotation)
        records[split] = [
            (image, annotations.get(image['id'], [])) for image in coco.get('images', [])
        ]
    return records


def analyze_input_distribution(coco_by_split, query_capacity,
                               capacity_safety_ratio=0.80,
                               minimum_ess_ratio=0.50,
                               duplicate_iou_threshold=0.95):
    """Return the complete distribution report and hard pre-training gates."""
    missing_splits = set(SPLITS) - set(coco_by_split)
    if missing_splits:
        raise ValueError(f'missing fixed splits: {sorted(missing_splits)}')
    records = _split_records(coco_by_split)
    report = {
        'schema_version': 1,
        'query_capacity': int(query_capacity),
        'capacity_safety_ratio': float(capacity_safety_ratio),
        'splits': {},
    }
    group_owners = defaultdict(set)
    support_owners = defaultdict(set)
    derived_owners = defaultdict(set)
    track_owners = defaultdict(set)
    split_videos = {}
    global_joint = Counter()
    video_joint = defaultdict(Counter)
    all_duplicate_pairs = 0
    all_box_pairs = 0
    for split in SPLITS:
        images = records[split]
        domains = Counter()
        sources = Counter()
        videos = Counter()
        counts = []
        pressures = []
        scales = []
        weights = []
        joint = Counter()
        supervision = Counter()
        cluster_counts = Counter()
        duplicate_pairs = box_pairs = 0
        for image, annotations in images:
            domain = _domain(image)
            video = _video(image)
            source = _source(image)
            domains[domain] += 1
            videos[video] += 1
            sources[source] += 1
            count = len([
                annotation for annotation in annotations
                if not bool(annotation.get('ignore_region', annotation.get('ignore', False)))
            ])
            counts.append(count)
            pressures.append(count / max(int(query_capacity), 1))
            weight = float(image.get('sampling_weight', 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"image {image['id']} has invalid sampling_weight={weight}")
            weights.append(weight)
            image_scales = []
            for annotation in annotations:
                if bool(annotation.get('ignore_region', annotation.get('ignore', False))):
                    continue
                box = annotation['bbox']
                relative_scale = math.sqrt(
                    max(float(box[2]) * float(box[3]), 0.0)
                    / max(float(image['width']) * float(image['height']), 1.0)
                )
                scales.append(relative_scale)
                image_scales.append(relative_scale)
                mask = annotation.get('supervision_mask')
                if mask is None:
                    mask = [True, bool(annotation.get('pose_mask', False)),
                            bool(annotation.get('track_mask', False)), True]
                for name, enabled in zip(('box', 'pose', 'track', 'quality'), mask):
                    supervision[name] += int(bool(enabled))
            scale_value = sum(image_scales) / max(len(image_scales), 1)
            joint_key = (_bin(scale_value, SCALE_BINS), _bin(count, DENSITY_BINS))
            joint[joint_key] += 1
            global_joint[joint_key] += 1
            video_joint[video][joint_key] += 1
            boxes = [annotation['bbox'] for annotation in annotations]
            for left_index, left_box in enumerate(boxes):
                for right_box in boxes[left_index + 1:]:
                    box_pairs += 1
                    duplicate_pairs += int(
                        _box_iou(left_box, right_box) >= duplicate_iou_threshold
                    )
            cluster = image.get('near_duplicate_cluster')
            if cluster is not None:
                cluster_counts[str(cluster)] += 1
            group = '|'.join([
                video, str(image.get('section_id')),
                str(image.get('near_duplicate_cluster')),
                str(image.get('track_scope')),
            ])
            group_owners[group].add(split)
            for field, owners in (
                ('temporal_group_id', support_owners),
                ('support_group_id', support_owners),
                ('derived_group_id', derived_owners),
                ('source_image_id', derived_owners),
            ):
                if image.get(field) is not None:
                    owners[str(image[field])].add(split)
            for annotation in annotations:
                track_id = int(annotation.get('track_id', -1))
                if track_id >= 0:
                    track_owners[f'{_sequence(image)}:{track_id}'].add(split)
        weight_sum = sum(weights)
        ess = weight_sum ** 2 / max(sum(weight ** 2 for weight in weights), 1e-12)
        near_duplicate_images = sum(
            count for count in cluster_counts.values() if count > 1
        )
        split_videos[split] = set(videos)
        report['splits'][split] = {
            'images': len(images), 'annotations': sum(counts),
            'domains': dict(sorted(domains.items())),
            'sources': dict(sorted(sources.items())),
            'videos': dict(sorted(videos.items())),
            'instances_per_image': _quantiles(counts),
            'relative_scale': _quantiles(scales),
            'query_pressure': _quantiles(pressures),
            'scale_density_joint': {
                f'{scale_bin}:{density_bin}': value
                for (scale_bin, density_bin), value in sorted(joint.items())
            },
            'raw_samples': len(images),
            'effective_weight_sum': weight_sum,
            'effective_sample_size': ess,
            'ess_ratio': ess / max(len(images), 1),
            'supervision': dict(supervision),
            'near_duplicate_image_rate': near_duplicate_images / max(len(images), 1),
            'within_image_duplicate_pair_rate': duplicate_pairs / max(box_pairs, 1),
        }
        all_duplicate_pairs += duplicate_pairs
        all_box_pairs += box_pairs

    leakages = {
        'indivisible_groups': sorted(key for key, owners in group_owners.items() if len(owners) > 1),
        'temporal_support_groups': sorted(key for key, owners in support_owners.items() if len(owners) > 1),
        'derived_views': sorted(key for key, owners in derived_owners.items() if len(owners) > 1),
        'track_scopes': sorted(key for key, owners in track_owners.items() if len(owners) > 1),
    }
    report['group_isolation'] = {
        'leakages': leakages,
        'leakage_count': sum(len(values) for values in leakages.values()),
    }
    report['video_drift_total_variation'] = {
        video: _total_variation(counter, global_joint)
        for video, counter in sorted(video_joint.items())
    }
    report['duplicate_pair_rate'] = all_duplicate_pairs / max(all_box_pairs, 1)
    all_videos = set.union(*split_videos.values()) if split_videos else set()
    gates = {
        'dual_domain_coverage': all(
            set(report['splits'][split]['domains']) == {'RGB', 'IR'} for split in SPLITS
        ),
        'video_coverage': all(split_videos[split] == all_videos for split in SPLITS),
        'group_isolation': report['group_isolation']['leakage_count'] == 0,
        'effective_sample_size': all(
            report['splits'][split]['ess_ratio'] >= minimum_ess_ratio for split in SPLITS
        ),
        'query_capacity': all(
            report['splits'][split]['query_pressure']['max'] <= capacity_safety_ratio
            for split in SPLITS
        ),
        'supervision_masks': all(
            report['splits'][split]['supervision'].get('box', 0)
            == report['splits'][split]['annotations']
            for split in SPLITS
        ),
    }
    report['gates'] = gates
    report['ready_for_training'] = all(gates.values())
    return report


def _load_inputs(dataset_args, coco_path=None, manifest_path=None):
    if dataset_args:
        result = {}
        for item in dataset_args:
            split, path = item.split('=', 1)
            result[split] = json.loads(Path(path).read_text(encoding='utf-8'))
        return result
    if not coco_path or not manifest_path:
        raise ValueError('provide --dataset split=path or both --coco and --manifest')
    coco = json.loads(Path(coco_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    images = {image['id']: image for image in coco['images']}
    annotations = defaultdict(list)
    for annotation in coco.get('annotations', []):
        annotations[annotation['image_id']].append(annotation)
    result = {}
    for split in SPLITS:
        ids = set(manifest['roles'][split]['image_ids'])
        result[split] = dict(coco)
        result[split]['images'] = [images[image_id] for image_id in sorted(ids)]
        result[split]['annotations'] = [
            annotation for image_id in sorted(ids) for annotation in annotations[image_id]
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append', default=[])
    parser.add_argument('--coco')
    parser.add_argument('--manifest')
    parser.add_argument('--query-capacity', type=int, required=True)
    parser.add_argument('--capacity-safety-ratio', type=float, default=0.80)
    parser.add_argument('--minimum-ess-ratio', type=float, default=0.50)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    datasets = _load_inputs(args.dataset, args.coco, args.manifest)
    report = analyze_input_distribution(
        datasets,
        query_capacity=args.query_capacity,
        capacity_safety_ratio=args.capacity_safety_ratio,
        minimum_ess_ratio=args.minimum_ess_ratio,
    )
    source_artifacts = {}
    for item in args.dataset:
        split, value = item.split('=', 1)
        path = Path(value).expanduser().resolve()
        source_artifacts[split] = {
            'path': str(path), 'sha256': _sha256(path),
        }
    if args.coco:
        source = Path(args.coco).expanduser().resolve()
        source_artifacts['source_annotation'] = {
            'path': str(source), 'sha256': _sha256(source),
        }
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        source_artifacts['split_manifest'] = {
            'path': str(manifest), 'sha256': _sha256(manifest),
        }
    report['source_artifacts'] = source_artifacts
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    if not report['ready_for_training']:
        raise SystemExit('input distribution gates failed')


if __name__ == '__main__':
    main()
