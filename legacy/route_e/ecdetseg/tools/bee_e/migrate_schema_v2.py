"""Create a strict BeePoseTrack-E schema-v2 view without changing source labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_state(annotation: dict) -> int:
    keypoints = annotation.get("keypoints") or []
    pose_mask = bool(annotation.get("pose_mask", annotation.get("num_keypoints", 0) > 0))
    if not pose_mask:
        if keypoints:
            raise ValueError(
                f"annotation {annotation.get('id')} disables pose supervision but contains keypoints"
            )
        return 0
    if len(keypoints) != 6:
        raise ValueError(
            f"annotation {annotation.get('id')} must contain two COCO endpoints"
        )
    return 2 if any(float(value) > 0 for value in keypoints[2::3]) else 1


def _sequence_id(image: dict, domain: str) -> str:
    """Scope temporal identities to one manually annotated video segment."""
    source = str(image.get("source", "unknown"))
    scene = str(image.get("scene", "unknown"))
    video = str(image.get("video", "unknown"))
    source_json = str(image.get("source_json", "")).replace("\\", "/")
    segment = Path(source_json).parent.name or "unknown_segment"
    return f"{source}:scene_{scene}:video_{video}:{segment}:{domain.lower()}"


def _valid_source_group(value) -> bool:
    return str(value).strip().lower() not in {"", "none", "null", "-1"}


def _group_sort_key(value: str):
    text = str(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def _endpoint_pair(annotation: dict):
    keypoints = annotation.get("keypoints") or []
    if len(keypoints) != 6 or not all(float(value) > 0 for value in keypoints[2::3]):
        return None
    return (
        (float(keypoints[0]), float(keypoints[1])),
        (float(keypoints[3]), float(keypoints[4])),
    )


def _track_geometry(previous: dict, current: dict):
    """Return motion/scale/current-axis geometry for two adjacent observations."""
    previous_points = _endpoint_pair(previous)
    current_points = _endpoint_pair(current)
    if previous_points is None or current_points is None:
        return None

    px, py, pw, ph = (float(value) for value in previous["bbox"])
    cx, cy, cw, ch = (float(value) for value in current["bbox"])
    if min(pw, ph, cw, ch) <= 0:
        return None

    previous_center = (px + pw / 2.0, py + ph / 2.0)
    current_center = (cx + cw / 2.0, cy + ch / 2.0)
    displacement_scale = max(math.hypot(pw, ph), 1.0)
    dx = (current_center[0] - previous_center[0]) / displacement_scale
    dy = (current_center[1] - previous_center[1]) / displacement_scale
    dw = math.log(cw / pw)
    dh = math.log(ch / ph)

    head, tail = current_points
    axis_x, axis_y = tail[0] - head[0], tail[1] - head[1]
    axis_norm = math.hypot(axis_x, axis_y)
    if axis_norm <= 1e-6:
        return None
    return [
        dx, dy, dw, dh,
        axis_x / axis_norm, axis_y / axis_norm,
    ]


def migrate(source: Path, image_root: Optional[Path] = None):
    dataset = json.loads(source.read_text(encoding="utf-8"))
    if int(dataset.get("schema_version", dataset.get("info", {}).get("schema_version", 0))) == 2:
        raise ValueError(f"{source} is already schema v2")

    missing_images = []
    image_by_id = {}
    for image in dataset.get("images", []):
        domain = str(image.get("domain", "")).upper()
        if domain not in {"RGB", "IR"}:
            raise ValueError(f"image {image.get('id')} has invalid domain {domain!r}")
        source_name = str(image.get("source", "unknown"))
        scene = str(image.get("scene", "unknown"))
        video = str(image.get("video", "unknown"))
        frame = int(image.get("frame", -1))
        if frame < 0:
            raise ValueError(f"image {image.get('id')} has no valid frame number")

        image["domain"] = domain
        image["sensor_id"] = domain.lower()
        image["camera_id"] = f"scene_{scene}_{domain.lower()}"
        image["sequence_id"] = _sequence_id(image, domain)
        image["frame_id"] = frame
        image["annotator_id"] = source_name
        image["track_supervised"] = False
        image_by_id[image["id"]] = image
        if image_root is not None and not (image_root / image["file_name"]).is_file():
            missing_images.append(image["file_name"])

    pose_counts = {0: 0, 1: 0, 2: 0}
    pairing_counts: dict[str, int] = {}
    source_groups = defaultdict(set)
    annotation_groups = {}
    for annotation in dataset.get("annotations", []):
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            raise ValueError(
                f"annotation {annotation.get('id')} references an unknown image"
            )
        state = _pose_state(annotation)
        pose_counts[state] += 1
        pairing = str(annotation.get("pairing_method", "unknown"))
        pairing_counts[pairing] = pairing_counts.get(pairing, 0) + 1
        original_group = annotation.get("track_id")
        group = str(original_group)
        if _valid_source_group(original_group):
            source_groups[image["sequence_id"]].add(group)
            annotation_groups[annotation["id"]] = group

        annotation["pose_state"] = state
        annotation["pose_mask"] = state > 0
        annotation["source_group_id"] = group
        annotation["track_id"] = -1
        annotation["track_mask"] = False
        annotation["track_geometry_mask"] = False
        annotation.pop("track_geometry", None)
        annotation["supervision_mask"] = [True, state > 0, False, True]
        annotation["annotator_id"] = str(annotation.get("source", "unknown"))

        # No numeric quality scores exist in the source.  A value of one is the
        # neutral weight for manual, geometry-checked labels; provenance remains
        # explicit so later reviewed scores can replace it without ambiguity.
        annotation["inter_group_quality"] = 1.0
        annotation["intra_group_quality"] = 1.0
        annotation["hierarchy_quality"] = 1.0
        annotation["pose_quality"] = 1.0 if state > 0 else 0.0
        annotation["track_quality"] = 0.0
        annotation["trajectory_stability"] = 0.0
        annotation["quality_provenance"] = "manual_unrated_neutral_weight"

    track_maps = {
        sequence: {
            group: track_id
            for track_id, group in enumerate(sorted(groups, key=_group_sort_key))
        }
        for sequence, groups in source_groups.items()
    }
    observations = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        group = annotation_groups.get(annotation["id"])
        if group is None:
            continue
        image = image_by_id[annotation["image_id"]]
        sequence = image["sequence_id"]
        annotation["track_id"] = track_maps[sequence][group]
        annotation["track_mask"] = True
        annotation["track_quality"] = 1.0
        annotation["trajectory_stability"] = 1.0
        annotation["supervision_mask"][2] = True
        image["track_supervised"] = True
        observations[(sequence, annotation["track_id"])].append(
            (int(image["frame_id"]), annotation)
        )

    sequence_steps = {}
    frames_by_sequence = defaultdict(list)
    for image in image_by_id.values():
        frames_by_sequence[image["sequence_id"]].append(int(image["frame_id"]))
    for sequence, frames in frames_by_sequence.items():
        ordered = sorted(set(frames))
        differences = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
        sequence_steps[sequence] = Counter(differences).most_common(1)[0][0] if differences else None

    geometry_count = 0
    for (sequence, track_id), items in observations.items():
        items.sort(key=lambda item: item[0])
        frames = [frame for frame, _ in items]
        if len(frames) != len(set(frames)):
            raise ValueError(
                f"sequence {sequence} has duplicate track_id={track_id} in one frame"
            )
        step = sequence_steps[sequence]
        for (previous_frame, previous), (current_frame, current) in zip(items, items[1:]):
            if step is None or current_frame - previous_frame != step:
                continue
            geometry = _track_geometry(previous, current)
            if geometry is None:
                continue
            current["track_geometry"] = geometry
            current["track_geometry_mask"] = True
            geometry_count += 1

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing below {image_root}: {preview}"
        )

    info = dict(dataset.get("info", {}))
    info.update(
        {
            "schema_version": 2,
            "schema_name": "BeePoseTrack-E",
            "source_annotation": str(source.resolve()),
            "source_sha256": _sha256(source),
            "migration": "tools/bee_e/migrate_schema_v2.py",
            "track_supervision": "enabled: manually persistent group_id scoped by annotated sequence segment",
            "track_geometry": "[center_dx, center_dy, log_width_ratio, log_height_ratio, current_axis_x, current_axis_y]",
            "quality_provenance": "manual labels without numeric review scores use neutral weight 1.0",
        }
    )
    dataset["schema_version"] = 2
    dataset["info"] = info
    summary = {
        "images": len(dataset.get("images", [])),
        "annotations": len(dataset.get("annotations", [])),
        "pose_state": pose_counts,
        "pairing_method": pairing_counts,
        "track_supervised_annotations": sum(
            bool(item.get("track_mask")) for item in dataset.get("annotations", [])
        ),
        "track_geometry_annotations": geometry_count,
        "track_sequences": len(source_groups),
        "tracklets": len(observations),
        "source_sha256": info["source_sha256"],
    }
    return dataset, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset, summary = migrate(args.input, args.image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
