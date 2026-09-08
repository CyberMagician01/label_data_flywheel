#!/usr/bin/env python3
"""Build strict Schema-v2 public preadaptation views for BeePoseTrack-E."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CATEGORY = {
    "id": 1,
    "name": "bee",
    "supercategory": "insect",
    "keypoints": ["head", "tail"],
    "skeleton": [[1, 2]],
}
QUALITY_FIELDS = (
    "inter_group_quality",
    "intra_group_quality",
    "hierarchy_quality",
    "pose_quality",
    "track_quality",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def base_dataset(description: str) -> dict:
    return {
        "schema_version": 2,
        "info": {"schema_version": 2, "description": description},
        "images": [],
        "annotations": [],
        "categories": [CATEGORY],
    }


def quality_payload(pose: bool, track: bool) -> dict:
    return {
        "inter_group_quality": 1.0,
        "intra_group_quality": 1.0,
        "hierarchy_quality": 1.0,
        "pose_quality": 1.0 if pose else 0.0,
        "track_quality": 1.0 if track else 0.0,
    }


def convert_public_pose(unified_root: Path, split: str) -> tuple[dict, dict]:
    source_name = "public_bee_pose_train.json" if split == "train" else "public_bee_pose_val.json"
    source_path = unified_root / "views/E/public_pose_coco/annotations" / source_name
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("categories") != [CATEGORY]:
        raise ValueError("Public pose category/keypoint order is not exact [head, tail].")

    output = base_dataset(f"BeePose + Mendeley public pose {split}, strict Schema-v2")
    image_ids = set()
    missing = []
    source_counts = defaultdict(int)
    for image in source["images"]:
        image_id = int(image["id"])
        if image_id in image_ids:
            raise ValueError(f"Duplicate public pose image id: {image_id}")
        image_ids.add(image_id)
        relative = Path("views/E/public_pose_coco/images") / image["file_name"]
        if not (unified_root / relative).is_file():
            missing.append(str(relative))
        source_dataset = str(image.get("source_dataset", "unknown"))
        source_counts[source_dataset] += 1
        output["images"].append({
            "id": image_id,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "file_name": relative.as_posix(),
            "domain": "RGB",
            "sensor_id": source_dataset,
            "sequence_id": f"{source_dataset}_{split}",
            "frame_id": image_id,
            "track_supervised": False,
            "source_dataset": source_dataset,
        })

    dropped_partial_pose = 0
    full_pose = 0
    annotation_ids = set()
    for annotation in source["annotations"]:
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate public pose annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(f"Unknown public pose image id: {annotation['image_id']}")
        keypoints = list(annotation.get("keypoints", []))
        complete_pose = (
            bool(annotation.get("pose_mask", False))
            and int(annotation.get("num_keypoints", 0)) == 2
            and len(keypoints) == 6
            and any(float(value) > 0 for value in keypoints[2::3])
        )
        if complete_pose:
            pose_state = 2
            full_pose += 1
        else:
            keypoints = []
            pose_state = 0
            dropped_partial_pose += 1
        converted = {
            "id": annotation_id,
            "image_id": int(annotation["image_id"]),
            "category_id": 1,
            "iscrowd": int(annotation.get("iscrowd", 0)),
            "bbox": [float(value) for value in annotation["bbox"]],
            "area": float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])),
            "pose_state": pose_state,
            "pose_mask": complete_pose,
            "track_id": -1,
            "track_mask": False,
            "track_geometry_mask": False,
            "supervision_mask": [True, complete_pose, False, True],
            "annotator_id": "public_pose",
            "source": "public_pose",
            **quality_payload(complete_pose, False),
        }
        if complete_pose:
            converted["keypoints"] = [float(value) for value in keypoints]
            converted["num_keypoints"] = 2
        output["annotations"].append(converted)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} public pose images; first={missing[0]}")
    return output, {
        "source_json": str(source_path),
        "source_json_sha256": sha256(source_path),
        "images": len(output["images"]),
        "annotations": len(output["annotations"]),
        "full_pose_annotations": full_pose,
        "box_only_partial_pose_annotations": dropped_partial_pose,
        "source_counts": dict(sorted(source_counts.items())),
        "missing_images": 0,
    }


def parse_mot_gt(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, width, height = (float(value) for value in row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            object_class = int(float(row[7])) if len(row) > 7 else 1
            if confidence <= 0 or object_class != 1 or width <= 0 or height <= 0:
                continue
            by_frame[frame].append((track_id, x, y, width, height))
    return by_frame


def convert_bee24(unified_root: Path, split: str) -> tuple[dict, dict]:
    bee_root = unified_root / "sources/BEE24" / split
    sequences = sorted(path for path in bee_root.glob("BEE24-*") if path.is_dir())
    output = base_dataset(f"BEE24 official {split}, detection+tracking strict Schema-v2")
    image_id = 0
    annotation_id = 0
    empty_images = 0
    clipped_boxes = 0
    ambiguous_track_annotations = 0
    track_observations = defaultdict(list)
    for sequence in sequences:
        gt_path = sequence / "gt/gt.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing BEE24 GT: {gt_path}")
        gt = parse_mot_gt(gt_path)
        for image_path in sorted((sequence / "img1").glob("*")):
            if not image_path.is_file():
                continue
            try:
                frame_id = int(image_path.stem)
            except ValueError as error:
                raise ValueError(f"Non-numeric BEE24 frame: {image_path}") from error
            with Image.open(image_path) as image:
                width, height = image.size
            image_id += 1
            relative = image_path.relative_to(unified_root)
            rows = gt.get(frame_id, [])
            track_counts = Counter(row[0] for row in rows)
            frame_has_track_supervision = any(
                count == 1 for count in track_counts.values()
            )
            if not rows:
                empty_images += 1
            output["images"].append({
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": relative.as_posix(),
                "domain": "RGB",
                "sensor_id": "BEE24_visible",
                "sequence_id": sequence.name,
                "frame_id": frame_id,
                "track_supervised": frame_has_track_supervision,
                "source_dataset": "BEE24",
                "hard_negative": not rows,
            })
            for track_id, x, y, box_width, box_height in rows:
                track_supervised = track_counts[track_id] == 1
                if not track_supervised:
                    ambiguous_track_annotations += 1
                x1 = min(max(x, 0.0), float(width))
                y1 = min(max(y, 0.0), float(height))
                x2 = min(max(x + box_width, 0.0), float(width))
                y2 = min(max(y + box_height, 0.0), float(height))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x1, y1, x2 - x1, y2 - y1) != (x, y, box_width, box_height):
                    clipped_boxes += 1
                annotation_id += 1
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "pose_state": 0,
                    "pose_mask": False,
                    "track_id": track_id if track_supervised else -1,
                    "track_mask": track_supervised,
                    "track_geometry_mask": False,
                    "track_axis_mask": False,
                    "supervision_mask": [True, False, track_supervised, True],
                    "annotator_id": "BEE24_GT",
                    "source": "BEE24_GT",
                    **quality_payload(False, track_supervised),
                }
                output["annotations"].append(annotation)
                if track_supervised:
                    track_observations[(sequence.name, track_id)].append(
                        (frame_id, annotation)
                    )

    # BEE24 has persistent identities and boxes, but it does not label head/tail
    # keypoints.  Adjacent unique observations can therefore supervise motion
    # and scale; the body-axis component stays explicitly masked.
    track_geometry_annotations = 0
    for observations in track_observations.values():
        observations.sort(key=lambda item: item[0])
        for (previous_frame, previous), (current_frame, current) in zip(
            observations, observations[1:]
        ):
            if current_frame - previous_frame != 1:
                continue
            px, py, pw, ph = (float(value) for value in previous["bbox"])
            cx, cy, cw, ch = (float(value) for value in current["bbox"])
            if min(pw, ph, cw, ch) <= 0:
                continue
            previous_center_x = px + pw / 2.0
            previous_center_y = py + ph / 2.0
            current_center_x = cx + cw / 2.0
            current_center_y = cy + ch / 2.0
            displacement_scale = max(math.hypot(pw, ph), 1.0)
            current["track_geometry"] = [
                (current_center_x - previous_center_x) / displacement_scale,
                (current_center_y - previous_center_y) / displacement_scale,
                math.log(cw / pw),
                math.log(ch / ph),
                0.0,
                0.0,
            ]
            current["track_geometry_mask"] = True
            current["track_axis_mask"] = False
            track_geometry_annotations += 1
    return output, {
        "sequences": [sequence.name for sequence in sequences],
        "images": len(output["images"]),
        "annotations": len(output["annotations"]),
        "empty_images": empty_images,
        "clipped_boxes": clipped_boxes,
        "ambiguous_track_annotations_masked": ambiguous_track_annotations,
        "track_geometry_annotations": track_geometry_annotations,
        "missing_images": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args()
    unified_root = args.unified_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {}
    reports = {}
    for split in ("train", "val"):
        dataset, report = convert_public_pose(unified_root, split)
        path = output_dir / f"public_pose_{split}_schema_v2.json"
        write_json(path, dataset)
        generated[f"public_pose_{split}"] = path
        reports[f"public_pose_{split}"] = report
    for split in ("train", "test"):
        dataset, report = convert_bee24(unified_root, split)
        path = output_dir / f"bee24_{split}_schema_v2.json"
        write_json(path, dataset)
        generated[f"bee24_{split}"] = path
        reports[f"bee24_{split}"] = report

    pose_train = json.loads(generated["public_pose_train"].read_text(encoding="utf-8"))
    pose_val = json.loads(generated["public_pose_val"].read_text(encoding="utf-8"))
    pose_train_files = {item["file_name"] for item in pose_train["images"]}
    pose_val_files = {item["file_name"] for item in pose_val["images"]}
    if pose_train_files & pose_val_files:
        raise ValueError("Public pose train/val image leakage detected.")
    bee_train_sequences = set(reports["bee24_train"]["sequences"])
    bee_test_sequences = set(reports["bee24_test"]["sequences"])
    if bee_train_sequences & bee_test_sequences:
        raise ValueError("BEE24 train/test sequence leakage detected.")

    manifest = {
        "schema_version": 1,
        "unified_root": str(unified_root),
        "keypoint_order": ["head", "tail"],
        "public_datasets": ["BEE24", "BeePose", "MendeleyBeePose"],
        "supervision_policy": {
            "BEE24": [
                "detection",
                "density",
                "track_id",
                "adjacent_box_center_scale",
                "axis_masked_without_keypoints",
            ],
            "BeePose_Mendeley": ["detection", "density", "pose_when_complete"],
            "partial_single_endpoint": "box_only_no_pose_loss",
        },
        "leakage_checks": {
            "public_pose_train_val_file_overlap": 0,
            "bee24_train_test_sequence_overlap": 0,
        },
        "reports": reports,
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sorted(generated.items())
        },
    }
    manifest_path = output_dir / "public_preadaptation_manifest.json"
    write_json(manifest_path, manifest)
    if args.config_dir is not None:
        replacements = {
            "BEE24_TRAIN_SHA256": manifest["files"]["bee24_train"]["sha256"],
            "BEE24_TEST_SHA256": manifest["files"]["bee24_test"]["sha256"],
            "PUBLIC_POSE_TRAIN_SHA256": manifest["files"]["public_pose_train"]["sha256"],
            "PUBLIC_POSE_VAL_SHA256": manifest["files"]["public_pose_val"]["sha256"],
        }
        for name in ("e_p0a_bee24_dettrack_1280.yml", "e_p0b_public_pose_1280.yml"):
            config_path = args.config_dir / name
            rendered = config_path.read_text(encoding="utf-8")
            for placeholder, value in replacements.items():
                rendered = rendered.replace(placeholder, value)
            if "_SHA256" in rendered and any(
                placeholder in rendered for placeholder in replacements
            ):
                raise ValueError(f"Unresolved annotation SHA placeholder in {config_path}")
            atomic = config_path.with_suffix(config_path.suffix + ".tmp")
            atomic.write_text(rendered, encoding="utf-8")
            atomic.replace(config_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "reports": reports,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
