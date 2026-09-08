#!/usr/bin/env python3
"""Prepare VnBeeTracking and Mendeley detection curricula as strict Schema-v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tarfile
import zipfile
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def annotation_payload(
    annotation_id: int,
    image_id: int,
    box: list[float],
    source: str,
    track_id: int = -1,
    track_mask: bool = False,
) -> dict:
    x, y, width, height = box
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": box,
        "area": width * height,
        "pose_state": 0,
        "pose_mask": False,
        "track_id": track_id if track_mask else -1,
        "track_mask": track_mask,
        "track_geometry_mask": False,
        "track_axis_mask": False,
        "supervision_mask": [True, False, track_mask, True],
        "annotator_id": source,
        "source": source,
        "inter_group_quality": 1.0,
        "intra_group_quality": 1.0,
        "hierarchy_quality": 1.0,
        "pose_quality": 0.0,
        "track_quality": 1.0 if track_mask else 0.0,
    }


def extract_vn(archive: Path, prepared_root: Path) -> Path:
    target = prepared_root / "vnbeetracking"
    allowed = ("VnBeeTracking/Image_Label/", "VnBeeTracking/GroundTruth/")
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile() or not member.name.startswith(allowed):
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES | {".txt"}:
                continue
            relative = Path(member.name)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = source.extractfile(member)
            if handle is None:
                raise OSError(f"Cannot read tar member: {member.name}")
            with handle, destination.open("wb") as output:
                shutil.copyfileobj(handle, output)
    return target


def extract_mendeley(archive: Path, prepared_root: Path) -> Path:
    target = prepared_root / "mendeley_detection"
    marker = "/detection/"
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir() or marker not in member.filename:
                continue
            relative_text = member.filename.split(marker, 1)[1]
            relative = Path("detection") / relative_text
            if relative.suffix.lower() not in IMAGE_SUFFIXES | {".txt"}:
                continue
            if len(relative.parts) < 4 or relative.parts[2] not in {"images", "labels"}:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as handle, destination.open("wb") as output:
                shutil.copyfileobj(handle, output)
    return target


def parse_mot(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    rows: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            frame, track_id = int(float(row[0])), int(float(row[1]))
            x, y, width, height = (float(value) for value in row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            category = int(float(row[7])) if len(row) > 7 else 1
            if confidence > 0 and category == 1 and width > 0 and height > 0:
                rows[frame].append((track_id, x, y, width, height))
    return rows


def clip_box(x: float, y: float, width: float, height: float, image_width: int, image_height: int):
    x1, y1 = max(0.0, min(x, image_width)), max(0.0, min(y, image_height))
    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))
    return None if x2 <= x1 or y2 <= y1 else [x1, y1, x2 - x1, y2 - y1]


def build_vn(root: Path, sequences: list[str], split: str) -> tuple[dict, dict]:
    dataset = base_dataset(f"VnBeeTracking {split}, outdoor detection+tracking Schema-v2")
    annotation_id = 0
    clipped = 0
    dropped = 0
    ground_truth = {
        sequence: parse_mot(root / "VnBeeTracking/GroundTruth" / f"{sequence}.txt")
        for sequence in sequences
    }
    def frame_number(path: Path) -> int:
        digits = "".join(character for character in path.stem if character.isdigit())
        if not digits:
            raise ValueError(f"VnBeeTracking image has no numeric frame id: {path}")
        return int(digits)

    track_observations: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for image_id, (sequence, image_path) in enumerate(
        (
            (sequence, image_path)
            for sequence in sequences
            for image_path in sorted(
                (root / "VnBeeTracking/Image_Label" / sequence / "images").glob("*"),
                key=frame_number,
            )
            if image_path.suffix.lower() in IMAGE_SUFFIXES
        ),
        start=1,
    ):
        frame_id = frame_number(image_path)
        rows = ground_truth[sequence].get(frame_id, [])
        counts = Counter(row[0] for row in rows)
        with Image.open(image_path) as image:
            width, height = image.size
        dataset["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path.relative_to(root).as_posix(),
            "domain": "RGB",
            "scene": "A",
            "environment": "outdoor",
            "sensor_id": "VnBeeTracking_visible",
            "sequence_id": sequence,
            "frame_id": frame_id,
            "track_supervised": bool(rows),
            "source_dataset": "VnBeeTracking",
            "hard_negative": not rows,
        })
        for track_id, x, y, box_width, box_height in rows:
            box = clip_box(x, y, box_width, box_height, width, height)
            if box is None:
                dropped += 1
                continue
            clipped += box != [x, y, box_width, box_height]
            annotation_id += 1
            track_mask = counts[track_id] == 1
            annotation = annotation_payload(
                annotation_id, image_id, box, "VnBeeTracking_GT", track_id, track_mask
            )
            dataset["annotations"].append(annotation)
            if track_mask:
                track_observations[(sequence, track_id)].append((frame_id, annotation))

    geometry = 0
    for observations in track_observations.values():
        observations.sort(key=lambda item: item[0])
        for (previous_frame, previous), (current_frame, current) in zip(observations, observations[1:]):
            if current_frame - previous_frame != 1:
                continue
            px, py, pw, ph = previous["bbox"]
            cx, cy, cw, ch = current["bbox"]
            scale = max(math.hypot(pw, ph), 1.0)
            current["track_geometry"] = [
                (cx + cw / 2 - px - pw / 2) / scale,
                (cy + ch / 2 - py - ph / 2) / scale,
                math.log(cw / pw),
                math.log(ch / ph),
                0.0,
                0.0,
            ]
            current["track_geometry_mask"] = True
            geometry += 1
    return dataset, {
        "sequences": sequences,
        "images": len(dataset["images"]),
        "annotations": len(dataset["annotations"]),
        "clipped_boxes": clipped,
        "dropped_boxes": dropped,
        "track_geometry_annotations": geometry,
    }


def build_mendeley(root: Path, sequences: list[str], split: str) -> tuple[dict, dict]:
    dataset = base_dataset(f"Mendeley landing-board {split}, outdoor detection Schema-v2")
    annotation_id = 0
    missing_labels = 0
    bad_rows = 0
    def frame_number(path: Path, sequence: str) -> int:
        suffix = path.stem.removeprefix(sequence)
        digits = "".join(character for character in suffix if character.isdigit())
        return int(digits or 0)

    for image_id, (sequence, image_path) in enumerate(
        (
            (sequence, image_path)
            for sequence in sequences
            for image_path in sorted(
                (root / "detection" / f"_bee_{sequence}" / "images").glob("*"),
                key=lambda path: frame_number(path, sequence),
            )
            if image_path.suffix.lower() in IMAGE_SUFFIXES
        ),
        start=1,
    ):
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        rows = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) < 5 or int(float(values[0])) != 0:
                    bad_rows += 1
                    continue
                _, center_x, center_y, box_width, box_height = map(float, values[:5])
                box = clip_box(
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    box_width * width,
                    box_height * height,
                    width,
                    height,
                )
                if box is None:
                    bad_rows += 1
                    continue
                rows.append(box)
        else:
            missing_labels += 1
        dataset["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path.relative_to(root).as_posix(),
            "domain": "RGB",
            "scene": "A",
            "environment": "outdoor",
            "sensor_id": "Mendeley_landing_board",
            "sequence_id": sequence,
            "frame_id": frame_number(image_path, sequence),
            "track_supervised": False,
            "source_dataset": "MendeleyBeeDetection",
            "hard_negative": not rows,
        })
        for box in rows:
            annotation_id += 1
            dataset["annotations"].append(
                annotation_payload(annotation_id, image_id, box, "Mendeley_YOLO")
            )
    return dataset, {
        "sequences": sequences,
        "images": len(dataset["images"]),
        "annotations": len(dataset["annotations"]),
        "missing_label_files_treated_as_empty": missing_labels,
        "bad_label_rows": bad_rows,
    }


def monitor_subset(dataset: dict, size: int, seed: int) -> dict:
    if len(dataset["images"]) <= size:
        return dataset
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for image in dataset["images"]:
        by_sequence[str(image["sequence_id"])].append(image)
    selected = []
    remaining = size
    sequences = sorted(by_sequence)
    for index, sequence in enumerate(sequences):
        images = by_sequence[sequence]
        quota = remaining if index == len(sequences) - 1 else max(1, round(size * len(images) / len(dataset["images"])))
        ranked = sorted(
            images,
            key=lambda item: hashlib.sha256(f"{seed}:{sequence}:{item['id']}".encode()).hexdigest(),
        )
        chosen = ranked[: min(quota, len(ranked), remaining)]
        selected.extend(chosen)
        remaining -= len(chosen)
    selected_ids = {item["id"] for item in selected}
    subset = dict(dataset)
    subset["info"] = dict(dataset["info"], monitor_size=len(selected_ids), monitor_seed=seed)
    subset["images"] = sorted(selected, key=lambda item: item["id"])
    subset["annotations"] = [item for item in dataset["annotations"] if item["image_id"] in selected_ids]
    return subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vn-archive", type=Path, required=True)
    parser.add_argument("--mendeley-archive", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    vn_root = extract_vn(args.vn_archive.resolve(), args.prepared_root.resolve())
    mendeley_root = extract_mendeley(args.mendeley_archive.resolve(), args.prepared_root.resolve())
    vn_train = ["2022-04-04-10-00", "2022-04-05-08-00", "2022-04-05-10-30", "2022-04-06-10-30"]
    vn_val = ["2022-04-08-12-30"]
    men_train = ["20230609a", "20230609b", "20230609c", "20230609d", "20230711a", "20230711b"]
    men_val = ["20230609e", "20230711c"]
    generated = {}
    reports = {}
    for name, builder, root, sequences in (
        ("vn_train", build_vn, vn_root, vn_train),
        ("vn_val", build_vn, vn_root, vn_val),
        ("mendeley_train", build_mendeley, mendeley_root, men_train),
        ("mendeley_val", build_mendeley, mendeley_root, men_val),
    ):
        split = "train" if name.endswith("train") else "val"
        dataset, report = builder(root, sequences, split)
        path = args.annotation_dir / f"{name}_schema_v2.json"
        write_json(path, dataset)
        generated[name] = path
        reports[name] = report
        if split == "val":
            monitor = monitor_subset(dataset, 256, args.seed)
            monitor_path = args.annotation_dir / f"{name}_monitor256_seed{args.seed}_schema_v2.json"
            write_json(monitor_path, monitor)
            generated[f"{name}_monitor"] = monitor_path

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "scene_contract": {"all_public_curricula": "A/outdoor", "B_indoor_used": False},
        "split_contract": {
            "vn_train_sequences": vn_train,
            "vn_val_sequences": vn_val,
            "mendeley_train_sequences": men_train,
            "mendeley_val_sequences": men_val,
            "sequence_overlap": 0,
        },
        "archive_sha256": {
            "vn": sha256(args.vn_archive),
            "mendeley": sha256(args.mendeley_archive),
        },
        "prepared_roots": {"vn": str(vn_root), "mendeley": str(mendeley_root)},
        "reports": reports,
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in generated.items()},
    }
    manifest_path = args.annotation_dir / "curriculum_public_detection_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
