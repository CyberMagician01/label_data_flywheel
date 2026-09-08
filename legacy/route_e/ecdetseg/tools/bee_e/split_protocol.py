"""Strict four-fold paired-sequence protocol and leakage checks."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROLES = ('train', 'selection', 'calibration', 'test')
FIXED_511_ROLES = ('train', 'calibration', 'dev_holdout')


def _sequence_key(image):
    sequence = image.get('sequence_id', image.get('video_id'))
    if sequence is None:
        raise ValueError(f"image {image.get('id')} has no sequence_id/video_id")
    return str(sequence)


def _environment(image):
    value = str(image.get('environment', image.get('scene', ''))).strip().lower()
    aliases = {
        'inside': 'inside', 'indoor': 'inside', 'nest_inside': 'inside', '巢内': 'inside',
        'outside': 'outside', 'outdoor': 'outside', 'nest_outside': 'outside', '巢外': 'outside',
    }
    if value not in aliases:
        raise ValueError(f"image {image.get('id')} has unsupported environment={value!r}")
    return aliases[value]


def _fixed_group_key(image):
    required = ('section_id', 'near_duplicate_cluster', 'track_scope')
    missing = [key for key in required if image.get(key) is None]
    if missing:
        raise ValueError(
            f"image {image.get('id')} is missing fixed-group fields: {', '.join(missing)}"
        )
    return '|'.join([
        _sequence_key(image),
        str(image['section_id']),
        str(image['near_duplicate_cluster']),
        str(image['track_scope']),
    ])


def build_fixed_511_manifest(coco, seed=2026):
    """Assign every video's seven indivisible sections with a 5:1:1 quota."""
    import random

    images_by_video_section = defaultdict(lambda: defaultdict(list))
    for image in coco['images']:
        video = _sequence_key(image)
        section = str(image.get('section_id'))
        _fixed_group_key(image)
        images_by_video_section[video][section].append(image)

    roles = {role: {'groups': [], 'image_ids': [], 'video_sections': {}}
             for role in FIXED_511_ROLES}
    for video in sorted(images_by_video_section):
        sections = sorted(images_by_video_section[video])
        if len(sections) != 7:
            raise ValueError(
                f'video {video!r} must contain exactly seven annotated sections, got {len(sections)}'
            )
        parent = {section: section for section in sections}

        def find(section):
            while parent[section] != section:
                parent[section] = parent[parent[section]]
                section = parent[section]
            return section

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        link_fields = (
            'near_duplicate_cluster', 'track_scope',
            'temporal_group_id', 'support_group_id',
            'derived_group_id', 'source_image_id',
        )
        for field in link_fields:
            field_sections = defaultdict(set)
            for section, section_images in images_by_video_section[video].items():
                for image in section_images:
                    value = image.get(field)
                    if value is not None:
                        field_sections[str(value)].add(section)
            for linked_sections in field_sections.values():
                linked_sections = sorted(linked_sections)
                for other in linked_sections[1:]:
                    union(linked_sections[0], other)

        components = defaultdict(list)
        for section in sections:
            components[find(section)].append(section)
        singleton_sections = sorted(
            values[0] for values in components.values() if len(values) == 1
        )
        if len(singleton_sections) < 2:
            raise ValueError(
                f'video {video!r} cannot satisfy 5:1:1 without splitting linked sections'
            )
        generator = random.Random(int(seed) + sum(ord(char) for char in video))
        generator.shuffle(singleton_sections)
        calibration_section, holdout_section = singleton_sections[:2]
        assignment = {
            'train': [
                section for section in sections
                if section not in {calibration_section, holdout_section}
            ],
            'calibration': [calibration_section],
            'dev_holdout': [holdout_section],
        }
        for role, assigned_sections in assignment.items():
            roles[role]['video_sections'][video] = assigned_sections
            for section in assigned_sections:
                section_images = images_by_video_section[video][section]
                roles[role]['image_ids'].extend(image['id'] for image in section_images)
                roles[role]['groups'].extend(_fixed_group_key(image) for image in section_images)
    for role in roles.values():
        role['groups'] = sorted(set(role['groups']))
        role['image_ids'] = sorted(set(role['image_ids']))
    manifest = {
        'schema_version': 2,
        'protocol': 'video_section_near_duplicate_track_scope_fixed_5_1_1',
        'seed': int(seed),
        'roles': roles,
    }
    validate_fixed_511_manifest(coco, manifest)
    return manifest


def validate_fixed_511_manifest(coco, manifest):
    """Reject frame, section, track, support-group and derived-view leakage."""
    if manifest.get('protocol') != 'video_section_near_duplicate_track_scope_fixed_5_1_1':
        raise ValueError('unexpected fixed 5:1:1 protocol')
    image_by_id = {image['id']: image for image in coco['images']}
    owner = {}
    errors = []
    for role in FIXED_511_ROLES:
        data = manifest.get('roles', {}).get(role, {})
        for image_id in data.get('image_ids', []):
            if image_id not in image_by_id:
                errors.append(f'{role}: unknown image {image_id}')
                continue
            if image_id in owner:
                errors.append(f'image {image_id} appears in {owner[image_id]} and {role}')
            owner[image_id] = role

    if set(owner) != set(image_by_id):
        errors.append('every image must belong to exactly one fixed split')

    leakage_fields = (
        ('indivisible_group', lambda image: _fixed_group_key(image)),
        ('section', lambda image: (_sequence_key(image), str(image['section_id']))),
        ('track_scope', lambda image: (_sequence_key(image), str(image['track_scope']))),
        ('temporal_support', lambda image: image.get(
            'temporal_group_id', image.get('support_group_id')
        )),
        ('derived_view', lambda image: image.get(
            'derived_group_id', image.get('source_image_id')
        )),
    )
    for field_name, extractor in leakage_fields:
        field_owners = defaultdict(set)
        for image_id, image in image_by_id.items():
            value = extractor(image)
            if value is not None and image_id in owner:
                field_owners[str(value)].add(owner[image_id])
        for value, owners in field_owners.items():
            if len(owners) > 1:
                errors.append(f'{field_name} leak {value}: {sorted(owners)}')

    videos = {_sequence_key(image) for image in coco['images']}
    for role in FIXED_511_ROLES:
        role_videos = {
            _sequence_key(image_by_id[image_id])
            for image_id in manifest['roles'][role]['image_ids']
        }
        if role_videos != videos:
            errors.append(f'{role} must contain every video')
    for video in videos:
        quotas = {
            role: len(manifest['roles'][role]['video_sections'].get(video, []))
            for role in FIXED_511_ROLES
        }
        if quotas != {'train': 5, 'calibration': 1, 'dev_holdout': 1}:
            errors.append(f'video {video}: invalid section quota {quotas}')
    if errors:
        raise ValueError('\n'.join(errors))
    return True


def build_paired_four_fold_manifest(coco, pair_map=None, seeds=(2026, 3407, 827)):
    """Assign whole inside/outside sequences to four non-overlapping roles.

    Each fold uses one paired sequence for each role. This is deliberately
    conservative: no frame or track from a sequence can leak between roles.
    """
    groups = defaultdict(set)
    for image in coco['images']:
        groups[_environment(image)].add(_sequence_key(image))
    inside = sorted(groups['inside'])
    outside = sorted(groups['outside'])
    if len(inside) != 4 or len(outside) != 4:
        raise ValueError('strict paired LOSO requires exactly four inside and four outside sequences')

    if pair_map is None:
        pairs = list(zip(inside, outside))
    else:
        pairs = []
        for inside_sequence in inside:
            outside_sequence = str(pair_map[inside_sequence])
            if outside_sequence not in outside:
                raise ValueError(f'unknown outside sequence in pair map: {outside_sequence}')
            pairs.append((inside_sequence, outside_sequence))
        if len({pair[1] for pair in pairs}) != 4:
            raise ValueError('pair map must use each outside sequence exactly once')

    folds = []
    for fold_index in range(4):
        role_pairs = {
            'test': pairs[fold_index],
            'calibration': pairs[(fold_index + 1) % 4],
            'selection': pairs[(fold_index + 2) % 4],
            'train': pairs[(fold_index + 3) % 4],
        }
        folds.append({
            'fold': fold_index,
            'roles': {
                role: {
                    'inside_sequences': [pair[0]],
                    'outside_sequences': [pair[1]],
                }
                for role, pair in role_pairs.items()
            },
        })
    manifest = {
        'schema_version': 1,
        'protocol': 'paired_inside_outside_leave_one_sequence_out',
        'seeds': [int(seed) for seed in seeds],
        'folds': folds,
    }
    validate_split_manifest(coco, manifest)
    return manifest


def validate_split_manifest(coco, manifest):
    images = coco['images']
    image_by_id = {image['id']: image for image in images}
    annotations_by_image = defaultdict(list)
    for annotation in coco.get('annotations', []):
        annotations_by_image[annotation['image_id']].append(annotation)

    known_sequences = {_sequence_key(image) for image in images}
    errors = []
    for fold in manifest.get('folds', []):
        role_sequences = {}
        for role in ROLES:
            role_data = fold.get('roles', {}).get(role, {})
            sequences = {
                str(value)
                for key in ('inside_sequences', 'outside_sequences')
                for value in role_data.get(key, [])
            }
            unknown = sequences - known_sequences
            if unknown:
                errors.append(f"fold {fold.get('fold')} {role}: unknown sequences {sorted(unknown)}")
            role_sequences[role] = sequences

        for left_index, left_role in enumerate(ROLES):
            for right_role in ROLES[left_index + 1:]:
                overlap = role_sequences[left_role] & role_sequences[right_role]
                if overlap:
                    errors.append(
                        f"fold {fold.get('fold')}: sequence leak {left_role}/{right_role}: {sorted(overlap)}"
                    )

        ownership = {
            sequence: role
            for role, sequences in role_sequences.items()
            for sequence in sequences
        }
        track_owners = defaultdict(set)
        derived_owners = defaultdict(set)
        for image_id, image in image_by_id.items():
            role = ownership.get(_sequence_key(image))
            if role is None:
                continue
            derived = image.get('derived_group_id', image.get('source_image_id'))
            if derived is not None:
                derived_owners[str(derived)].add(role)
            for annotation in annotations_by_image[image_id]:
                track_id = annotation.get('track_id')
                if track_id is not None and int(track_id) >= 0:
                    track_owners[(_sequence_key(image), int(track_id))].add(role)
        for key, owners in track_owners.items():
            if len(owners) > 1:
                errors.append(f'fold {fold.get("fold")}: track leak {key}: {sorted(owners)}')
        for key, owners in derived_owners.items():
            if len(owners) > 1:
                errors.append(f'fold {fold.get("fold")}: derived-view leak {key}: {sorted(owners)}')

    if errors:
        raise ValueError('\n'.join(errors))
    return True


def summarize_seed_fold_results(records, metric_directions):
    """Aggregate 3-seed x 4-fold records with CI and worst-fold values."""
    expected = {(int(seed), int(fold)) for seed in {r['seed'] for r in records}
                for fold in {r['fold'] for r in records}}
    actual = {(int(record['seed']), int(record['fold'])) for record in records}
    if expected != actual or len({r['seed'] for r in records}) != 3 or len({r['fold'] for r in records}) != 4:
        raise ValueError('results must contain a complete 3-seed x 4-fold matrix')
    summary = {'runs': len(records), 'metrics': {}}
    for metric, direction in metric_directions.items():
        values = [float(record['metrics'][metric]) for record in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        std = math.sqrt(variance)
        fold_means = {
            fold: sum(float(r['metrics'][metric]) for r in records if int(r['fold']) == fold) / 3
            for fold in range(4)
        }
        summary['metrics'][metric] = {
            'mean': mean,
            'std': std,
            'ci95_half_width': 1.96 * std / math.sqrt(len(values)),
            'worst_fold': min(fold_means, key=fold_means.get) if direction == 'max'
                          else max(fold_means, key=fold_means.get),
            'worst_fold_value': min(fold_means.values()) if direction == 'max'
                                else max(fold_means.values()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pair-map', type=Path)
    parser.add_argument(
        '--protocol', choices=('fixed-5-1-1', 'paired-four-fold'),
        default='fixed-5-1-1',
    )
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    coco = json.loads(args.annotations.read_text(encoding='utf-8'))
    pair_map = json.loads(args.pair_map.read_text(encoding='utf-8')) if args.pair_map else None
    manifest = (
        build_fixed_511_manifest(coco, seed=args.seed)
        if args.protocol == 'fixed-5-1-1'
        else build_paired_four_fold_manifest(coco, pair_map=pair_map)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
