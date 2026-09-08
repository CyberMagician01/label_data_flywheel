#!/usr/bin/env python3
"""Derive a YOLO-Pose Y view from unified canonical LabelMe sections."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SECTION_RE = re.compile(r"(?P<video>[AB]-5-\d+).*?(?P<section>\d+)$", re.IGNORECASE)
FRAME_RE = re.compile(r"_frame_0*(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, default=Path("/data/bee26/datasets/bee_e_y_unified_20260901"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy-rgb", action="store_true")
    parser.add_argument(
        "--supervision-mode",
        choices=["detposemask", "pose_only"],
        default="detposemask",
        help=(
            "detposemask keeps det-only bee boxes as bbox/class positives with invisible keypoints. "
            "pose_only drops instances without complete head+tail supervision from labels and manifest."
        ),
    )
    parser.add_argument(
        "--split-policy",
        choices=[
            "annotator01_val_03_test",
            "annotator03_train_test",
            "video_holdout_train_test",
            "section_5_1_1",
        ],
        default="annotator01_val_03_test",
        help=(
            "Formal split policy. annotator01_val_03_test maps annotator_01 to val "
            "(protocol calibration), annotator_03 to test (protocol dev_holdout), and all other annotators to train."
        ),
    )
    return parser.parse_args()


def annotator_id(path: Path) -> str | None:
    for part in path.resolve().parts:
        normalized = part.replace("标注员", "").replace("_", "")
        if normalized in {"01", "02", "03", "004", "05"}:
            return normalized
    return None


def split_for_record(video_id: str, section_num: int, split_policy: str, json_path: Path) -> str:
    video_id = video_id.upper()
    if split_policy == "annotator01_val_03_test":
        annotator = annotator_id(json_path)
        if annotator == "01":
            return "val"
        if annotator == "03":
            return "test"
        if annotator in {"02", "004", "05"}:
            return "train"
        raise ValueError(f"unexpected or missing annotator id for formal split: {json_path}")
    if split_policy == "annotator03_train_test":
        return "test" if annotator_id(json_path) == "03" else "train"
    if split_policy == "video_holdout_train_test":
        if video_id in {"A-5-1", "A-5-2", "A-5-3", "B-5-1", "B-5-2", "B-5-3"}:
            return "train"
        if video_id in {"A-5-4", "B-5-4"}:
            return "test"
        raise ValueError(f"unexpected video id for train/test split: {video_id}")
    if split_policy == "section_5_1_1":
        if 1 <= section_num <= 5:
            return "train"
        if section_num == 6:
            return "calibration"
        if section_num == 7:
            return "dev_holdout"
        raise ValueError(f"unexpected section number: {section_num}")
    raise ValueError(f"unexpected split policy: {split_policy}")


def safe_section_name(section_id: str, video_id: str, section_num: int) -> str:
    return f"{video_id.upper()}_section{section_num:02d}"


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_ir_normalized(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = gray.astype(np.float32)
    lo, hi = np.percentile(gray, [1.0, 99.0])
    if hi <= lo:
        norm = np.zeros_like(gray, dtype=np.uint8)
    else:
        norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
        norm = (norm * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(dst), bgr):
        raise RuntimeError(f"failed to write {dst}")


def shape_xyxy(shape: dict[str, Any]) -> list[float]:
    pts = np.asarray(shape.get("points", []), dtype=float)
    if pts.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def point_xy(shape: dict[str, Any]) -> list[float] | None:
    pts = shape.get("points", [])
    if not pts:
        return None
    return [float(pts[0][0]), float(pts[0][1])]


def point_inside_box(point: list[float] | None, box: list[float], slack: float = 0.0) -> bool:
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = box
    return x1 - slack <= x <= x2 + slack and y1 - slack <= y <= y2 + slack


def point_box_distance(point: list[float], box: list[float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
    return ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / scale


def assign_ungrouped_points(
    grouped: dict[str, dict[str, Any]],
    bee_rectangles: list[tuple[str, dict[str, Any]]],
    points: list[dict[str, Any]],
    point_label: str,
) -> dict[str, int]:
    """Assign ungrouped points one-to-one to bee boxes containing them."""
    used_points: set[int] = set()
    assigned_boxes: set[str] = {
        gid for gid, pack in grouped.items() if point_label in pack
    }
    candidates = []
    for point_index, point_shape in enumerate(points):
        point = point_xy(point_shape)
        if point is None:
            continue
        for box_order, (gid, rect) in enumerate(bee_rectangles):
            if gid in assigned_boxes:
                continue
            box = shape_xyxy(rect)
            if not point_inside_box(point, box):
                continue
            x1, y1, x2, y2 = box
            area = max((x2 - x1) * (y2 - y1), 1.0)
            candidates.append((point_box_distance(point, box), area, box_order, point_index, gid, point_shape))

    by_box: dict[str, int] = {}
    for _, _, _, point_index, gid, point_shape in sorted(candidates):
        if point_index in used_points or gid in assigned_boxes:
            continue
        grouped[gid][point_label] = point_shape
        used_points.add(point_index)
        assigned_boxes.add(gid)
        by_box[gid] = point_index
    return by_box


def yolo_row(inst: dict[str, Any], width: float, height: float) -> str | None:
    if not inst.get("det_mask"):
        return None
    x1, y1, x2, y2 = inst["bbox_xyxy"]
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    if inst.get("pose_mask"):
        head, tail = inst["keypoints"]
        head_vis, tail_vis = 2, 2
    else:
        head, tail = [0.0, 0.0], [0.0, 0.0]
        head_vis, tail_vis = 0, 0
    vals = [
        0,
        (x1 + w / 2) / width,
        (y1 + h / 2) / height,
        w / width,
        h / height,
        head[0] / width,
        head[1] / height,
        head_vis,
        tail[0] / width,
        tail[1] / height,
        tail_vis,
    ]
    vals[1:] = [v if i in {7, 10} else min(max(float(v), 0.0), 1.0) for i, v in enumerate(vals[1:], 1)]
    return " ".join(str(v) if isinstance(v, int) else f"{v:.8f}" for v in vals)


def image_for_json(path: Path, data: dict[str, Any]) -> Path | None:
    raw = data.get("imagePath")
    candidates = []
    if raw:
        candidates.append(path.parent / str(raw))
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidates.append(path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_frame(
    section_dir: Path,
    json_path: Path,
    split: str,
    video_id: str,
    section_id: str,
    section_num: int,
    out_dir: Path,
    copy_rgb: bool,
    supervision_mode: str,
) -> dict[str, Any] | None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image = image_for_json(json_path, data)
    if image is None:
        return None
    width = float(data.get("imageWidth") or 0)
    height = float(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        height, width = img.shape[:2]
        width, height = float(width), float(height)
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    rectangle_records: list[tuple[str, dict[str, Any]]] = []
    ungrouped_points: dict[str, list[dict[str, Any]]] = {"head": [], "tail": []}
    dropped_grouped_points: dict[str, int] = {"head": 0, "tail": 0}
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).lower()
        gid = shape.get("group_id")
        if gid is None:
            if shape.get("shape_type") == "point" and label in {"head", "tail"}:
                ungrouped_points[label].append(shape)
            continue
        key = str(gid)
        if shape.get("shape_type") == "rectangle" and label in {"bee", "bee_shadow"}:
            grouped[key].setdefault("rects", []).append(shape)
            rectangle_records.append((key, shape))
        elif shape.get("shape_type") == "point" and label in {"head", "tail"}:
            if label in grouped[key]:
                dropped_grouped_points[label] += 1
            else:
                grouped[key][label] = shape

    bee_rectangles = [
        (gid, rect)
        for gid, rect in rectangle_records
        if str(rect.get("label", "bee")).lower() == "bee"
    ]
    spatial_assignments = {
        "head": assign_ungrouped_points(grouped, bee_rectangles, ungrouped_points["head"], "head"),
        "tail": assign_ungrouped_points(grouped, bee_rectangles, ungrouped_points["tail"], "tail"),
    }
    frame_pairing_audit = {
        "ungrouped_head_points": len(ungrouped_points["head"]),
        "ungrouped_tail_points": len(ungrouped_points["tail"]),
        "spatial_head_assignments": len(spatial_assignments["head"]),
        "spatial_tail_assignments": len(spatial_assignments["tail"]),
        "dropped_duplicate_grouped_head_points": dropped_grouped_points["head"],
        "dropped_duplicate_grouped_tail_points": dropped_grouped_points["tail"],
    }

    instances = []
    incomplete_pose_bee_boxes = 0
    for gid, pack in grouped.items():
        rects = pack.get("rects") or []
        if not rects:
            continue
        rect = max(rects, key=lambda s: (shape_xyxy(s)[2] - shape_xyxy(s)[0]) * (shape_xyxy(s)[3] - shape_xyxy(s)[1]))
        label = str(rect.get("label", "bee")).lower()
        box = shape_xyxy(rect)
        pairing_method = "group_id"
        head = point_xy(pack["head"]) if "head" in pack else None
        tail = point_xy(pack["tail"]) if "tail" in pack else None
        pose_mask = int(label == "bee" and head is not None and tail is not None)
        if pose_mask and (
            pack["head"].get("group_id") is None
            or pack["tail"].get("group_id") is None
        ):
            pairing_method = "spatial_ungrouped_point"
        if label == "bee" and not pose_mask:
            incomplete_pose_bee_boxes += 1
        if supervision_mode == "pose_only" and not pose_mask:
            continue
        instances.append(
            {
                "bbox_xyxy": box,
                "class_id": 0,
                "keypoints": [head, tail] if pose_mask else [],
                "visibility": [2, 2] if pose_mask else [],
                "track_id": f"{safe_section_name(section_id, video_id, section_num)}:{gid}",
                "source_group_id": gid,
                "pairing_method": pairing_method if pose_mask else "box_only",
                "det_mask": 1,
                "pose_mask": pose_mask,
                "track_mask": 1,
                "quality": 1.0,
                "source_dataset": "labelme_5_sections",
                "source_label": label,
            }
        )
    if not instances:
        return None

    domain = "RGB" if video_id.upper().startswith("A") else "IR"
    frame_match = FRAME_RE.search(json_path.stem)
    frame_id = int(frame_match.group(1)) if frame_match else -1
    stem = f"{safe_section_name(section_id, video_id, section_num)}__frame{frame_id:06d}"
    dst_img = out_dir / "images" / split / f"{stem}{image.suffix.lower()}"
    dst_lab = out_dir / "labels" / split / f"{stem}.txt"
    if domain == "IR":
        write_ir_normalized(image, dst_img)
    else:
        link_or_copy(image, dst_img, copy_rgb)
    labels = [row for inst in instances if (row := yolo_row(inst, width, height)) is not None]
    dst_lab.parent.mkdir(parents=True, exist_ok=True)
    dst_lab.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return {
        "source_json": str(json_path.resolve()),
        "source_image_path": str(image.resolve()),
        "image_path": str(dst_img.absolute()),
        "video_id": video_id.upper(),
        "section_id": section_id,
        "frame_id": frame_id,
        "domain": domain,
        "split": split,
        "width": int(width),
        "height": int(height),
        "instances": instances,
        "pairing_audit": {
            **frame_pairing_audit,
            "incomplete_pose_bee_boxes": incomplete_pose_bee_boxes,
            "pose_instances": sum(int(item["pose_mask"]) for item in instances),
            "det_instances": len(instances),
        },
    }


def write_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Unified LabelMe Y Dataset Summary",
        "",
        f"unified_root: `{summary['unified_root']}`",
        f"out_dir: `{summary['out_dir']}`",
        f"split_policy: `{summary['split_policy']}`",
        "",
        "| split | frames | RGB frames | IR frames | det instances | pose instances | track instances | empty pose label frames | incomplete bee boxes | spatial head | spatial tail | max/frame |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in summary["splits"].items():
        audit = row["pairing_audit"]
        lines.append(
            f"| {split} | {row['frames']} | {row['domains'].get('RGB', 0)} | {row['domains'].get('IR', 0)} | "
            f"{row['det_instances']} | {row['pose_instances']} | {row['track_instances']} | {row['empty_pose_label_frames']} | "
            f"{audit['incomplete_pose_bee_boxes']} | {audit['spatial_head_assignments']} | {audit['spatial_tail_assignments']} | {row['max_instances_per_frame']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.unified_root / "views" / "Y" / "labelme_5_sections"
    if not root.exists():
        raise FileNotFoundError(root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for section_dir in sorted(root.iterdir(), key=lambda p: p.name):
        match = SECTION_RE.match(section_dir.name)
        if not match:
            continue
        video_id = match.group("video").upper()
        section_num = int(match.group("section"))
        for dirpath, _, filenames in os.walk(section_dir, followlinks=True):
            for filename in sorted(filenames):
                if not filename.endswith(".json") or ".bak" in filename:
                    continue
                json_path = Path(dirpath) / filename
                split = split_for_record(video_id, section_num, args.split_policy, json_path)
                row = parse_frame(
                    section_dir,
                    json_path,
                    split,
                    video_id,
                    section_dir.name,
                    section_num,
                    args.out_dir,
                    args.copy_rgb,
                    args.supervision_mode,
                )
                if row:
                    rows.append(row)

    if args.split_policy == "annotator01_val_03_test":
        ordered_splits = ["train", "val", "test"]
    elif args.split_policy in {"annotator03_train_test", "video_holdout_train_test"}:
        ordered_splits = ["train", "test"]
    else:
        ordered_splits = ["train", "calibration", "dev_holdout"]
    split_lists: dict[str, list[str]] = {split: [] for split in ordered_splits}
    for row in rows:
        split_lists[row["split"]].append(row["image_path"])
    for split, images in split_lists.items():
        (args.out_dir / f"{split}_images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")

    with (args.out_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.split_policy == "annotator01_val_03_test":
        val_list = "val_images.txt"
        test_list = "test_images.txt"
    elif args.split_policy in {"annotator03_train_test", "video_holdout_train_test"}:
        val_list = "test_images.txt"
        test_list = "test_images.txt"
    else:
        val_list = "calibration_images.txt"
        test_list = "dev_holdout_images.txt"
    yaml_text = f"""path: {args.out_dir.resolve()}
train: train_images.txt
val: {val_list}
test: {test_list}

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
    (args.out_dir / "bee_yolo_pose_strict.yaml").write_text(yaml_text, encoding="utf-8")

    if args.split_policy == "annotator01_val_03_test":
        split_policy_text = (
            "annotator frozen train/val/test: source_json resolved under annotator_01=val "
            "(alignment protocol calibration), annotator_03=test (alignment protocol dev_holdout), "
            "annotator_004/02/05=train; tracking IDs are scoped by section_id and source LabelMe group_id"
        )
    elif args.split_policy == "annotator03_train_test":
        split_policy_text = (
            "annotator holdout train/test: source_json resolved under annotator_03=test; "
            "all other annotators=train; tracking IDs are scoped by section_id and source LabelMe group_id"
        )
    elif args.split_policy == "video_holdout_train_test":
        split_policy_text = (
            "video holdout train/test: A-5-1,A-5-2,A-5-3,B-5-1,B-5-2,B-5-3=train; "
            "A-5-4,B-5-4=test; tracking IDs are scoped by section_id and never merged across sections"
        )
    else:
        split_policy_text = "section ordinal 5:1:1 per video: 01-05=train, 06=calibration, 07=dev_holdout"

    summary: dict[str, Any] = {
        "unified_root": str(args.unified_root),
        "out_dir": str(args.out_dir),
        "split_policy": split_policy_text,
        "supervision_mode": args.supervision_mode,
        "splits": {},
        "manifest": str(args.out_dir / "dataset_manifest.jsonl"),
        "yaml": str(args.out_dir / "bee_yolo_pose_strict.yaml"),
        "label_note": (
            "detposemask: YOLO labels include every det_mask=1 instance. pose_mask=1 instances carry visible head/tail keypoints; "
            "det-only instances keep bbox/class and set both keypoint visibility flags to 0 so they contribute detection supervision but not pose loss. "
            "pose_only: instances without complete head+tail are dropped from labels and manifest, matching the old schema-v2 pose-only semantics. "
            "Pairing prefers matching group_id; ungrouped head/tail points are assigned one-to-one only to bee boxes that contain the point. "
            "Manifest track_id is scoped as section:LabelMe group_id."
        ),
    }
    for split in ordered_splits:
        split_rows = [r for r in rows if r["split"] == split]
        summary["splits"][split] = {
            "frames": len(split_rows),
            "domains": {
                "RGB": sum(1 for r in split_rows if r["domain"] == "RGB"),
                "IR": sum(1 for r in split_rows if r["domain"] == "IR"),
            },
            "det_instances": sum(len(r["instances"]) for r in split_rows),
            "pose_instances": sum(sum(int(i["pose_mask"]) for i in r["instances"]) for r in split_rows),
            "track_instances": sum(sum(int(i["track_mask"]) for i in r["instances"]) for r in split_rows),
            "empty_pose_label_frames": sum(1 for r in split_rows if not any(i["pose_mask"] for i in r["instances"])),
            "max_instances_per_frame": max((len(r["instances"]) for r in split_rows), default=0),
            "pairing_audit": {
                "ungrouped_head_points": sum(r["pairing_audit"]["ungrouped_head_points"] for r in split_rows),
                "ungrouped_tail_points": sum(r["pairing_audit"]["ungrouped_tail_points"] for r in split_rows),
                "spatial_head_assignments": sum(r["pairing_audit"]["spatial_head_assignments"] for r in split_rows),
                "spatial_tail_assignments": sum(r["pairing_audit"]["spatial_tail_assignments"] for r in split_rows),
                "dropped_duplicate_grouped_head_points": sum(r["pairing_audit"]["dropped_duplicate_grouped_head_points"] for r in split_rows),
                "dropped_duplicate_grouped_tail_points": sum(r["pairing_audit"]["dropped_duplicate_grouped_tail_points"] for r in split_rows),
                "incomplete_pose_bee_boxes": sum(r["pairing_audit"]["incomplete_pose_bee_boxes"] for r in split_rows),
            },
            "videos": sorted({r["video_id"] for r in split_rows}),
            "sections": sorted({r["section_id"] for r in split_rows}),
        }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary, args.out_dir / "dataset_summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
