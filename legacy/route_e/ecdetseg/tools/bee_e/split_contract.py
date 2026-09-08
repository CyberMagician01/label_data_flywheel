"""Validate the frozen schema-v2 train/calibration/dev-holdout contract."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROLES = ('train', 'calibration', 'dev_holdout')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(path_value, sha_value, label):
    path = Path(path_value or '').expanduser().resolve()
    expected = str(sha_value or '').lower()
    if not path.is_file() or len(expected) != 64:
        raise ValueError(f'{label} requires an existing file and complete SHA256.')
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f'{label} SHA256 mismatch: expected={expected}, actual={actual}')
    return path, actual


def _read_jsonl(path):
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSONL at {path}:{line_number}') from exc
    return rows


def validate_schema_v2_split_contract(contract, train_dataset, calibration_dataset):
    """Bind executable inputs to one immutable, leakage-free 5:1:1 view."""
    protocol = 'video_section_near_duplicate_track_scope_fixed_5_1_1'
    required = {
        'protocol', 'source_manifest', 'source_manifest_sha256',
        'derived_manifest', 'derived_manifest_sha256',
        'train_annotation', 'train_annotation_sha256',
        'calibration_annotation', 'calibration_annotation_sha256',
        'dev_holdout_annotation', 'dev_holdout_annotation_sha256',
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError('Continuous BeePoseTrack-E split contract misses: ' + ', '.join(missing))
    if contract['protocol'] != protocol:
        raise ValueError(f'Continuous BeePoseTrack-E requires split protocol={protocol}.')

    source_manifest, source_manifest_sha = _verified_path(
        contract['source_manifest'], contract['source_manifest_sha256'], 'source manifest',
    )
    derived_manifest, derived_manifest_sha = _verified_path(
        contract['derived_manifest'], contract['derived_manifest_sha256'], 'derived manifest',
    )
    role_paths = {}
    role_shas = {}
    role_documents = {}
    for role in ROLES:
        path, digest = _verified_path(
            contract[f'{role}_annotation'], contract[f'{role}_annotation_sha256'],
            f'{role} annotation',
        )
        role_paths[role] = path
        role_shas[role] = digest
        role_documents[role] = json.loads(path.read_text(encoding='utf-8'))

    for role, dataset in {
        'train': train_dataset, 'calibration': calibration_dataset,
    }.items():
        configured_path = Path(dataset.get('ann_file', '')).expanduser().resolve()
        configured_sha = str(dataset.get('expected_ann_sha256') or '').lower()
        if configured_path != role_paths[role] or configured_sha != role_shas[role]:
            raise ValueError(
                f'executable {role} annotation does not match the SHA-bound split contract.'
            )

    source_rows = _read_jsonl(source_manifest)
    derived_rows = _read_jsonl(derived_manifest)
    source_ids = [str(row.get('source_json', '')) for row in source_rows]
    derived_ids = [str(row.get('source_json', '')) for row in derived_rows]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError('source manifest source_json identities must be non-empty and unique.')
    if len(derived_ids) != len(set(derived_ids)) or set(derived_ids) != set(source_ids):
        raise ValueError('derived manifest must preserve every source identity exactly once.')

    rows_by_role = {role: [] for role in ROLES}
    for row in derived_rows:
        role = row.get('split')
        if role not in rows_by_role:
            raise ValueError(f'derived manifest contains unsupported split={role!r}.')
        rows_by_role[role].append(row)

    annotation_identities = {}
    for role, document in role_documents.items():
        images = document.get('images', [])
        if any(image.get('split') != role for image in images):
            raise ValueError(f'{role} annotation contains an image from another split.')
        identities = [str(image.get('source_json', '')) for image in images]
        if not identities or len(identities) != len(set(identities)):
            raise ValueError(f'{role} source_json identities must be non-empty and unique.')
        expected = {str(row['source_json']) for row in rows_by_role[role]}
        if set(identities) != expected:
            raise ValueError(f'{role} annotation identities disagree with the derived manifest.')
        annotation_identities[role] = set(identities)

    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1:]:
            if annotation_identities[left] & annotation_identities[right]:
                raise ValueError(f'source_json leakage detected between {left} and {right}.')

    for field in ('section_id', 'near_duplicate_cluster', 'track_scope'):
        values = {
            role: {str(row.get(field, '')) for row in rows_by_role[role]}
            for role in ROLES
        }
        if any('' in role_values for role_values in values.values()):
            raise ValueError(f'derived manifest has an empty {field}.')
        for index, left in enumerate(ROLES):
            for right in ROLES[index + 1:]:
                if values[left] & values[right]:
                    raise ValueError(f'{field} leakage detected between {left} and {right}.')

    hashes = {
        role: {
            str(image.get('canonical_image_sha256', ''))
            for image in role_documents[role].get('images', [])
        }
        for role in ROLES
    }
    if any('' in role_hashes for role_hashes in hashes.values()):
        raise ValueError('canonical image SHA256 is required for exact-duplicate isolation.')
    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1:]:
            if hashes[left] & hashes[right]:
                raise ValueError(f'exact-image leakage detected between {left} and {right}.')

    video_sections = defaultdict(lambda: {role: set() for role in ROLES})
    for row in derived_rows:
        video_sections[str(row['video_id'])][row['split']].add(str(row['section_id']))
    expected_quota = {'train': 5, 'calibration': 1, 'dev_holdout': 1}
    invalid_videos = {
        video: {role: len(sections) for role, sections in role_map.items()}
        for video, role_map in video_sections.items()
        if {role: len(sections) for role, sections in role_map.items()} != expected_quota
    }
    if invalid_videos:
        raise ValueError(f'per-video 5:1:1 section quota failed: {invalid_videos}')

    return {
        'protocol': protocol,
        'source_manifest': str(source_manifest),
        'source_manifest_sha256': source_manifest_sha,
        'derived_manifest': str(derived_manifest),
        'derived_manifest_sha256': derived_manifest_sha,
        **{f'{role}_annotation': str(role_paths[role]) for role in ROLES},
        **{f'{role}_annotation_sha256': role_shas[role] for role in ROLES},
        'role_counts': {role: len(annotation_identities[role]) for role in ROLES},
        'video_count': len(video_sections),
    }


__all__ = ['validate_schema_v2_split_contract']
