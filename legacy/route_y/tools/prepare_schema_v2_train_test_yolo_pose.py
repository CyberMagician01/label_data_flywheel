#!/usr/bin/env python3
"""Prepare a single train/test YOLO pose dataset from E-route schema-v2 COCO files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_PATTERN = re.compile(r"([AB]-5-\d+)", re.IGNORECASE)
ANNOTATOR_PATTERN = re.compile(r"annotator[_-](\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy-rgb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy_rgb(src: Path, dst: Path, copy_rgb: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_rgb:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.absolute(), dst)


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


def infer_video_id(image: dict[str, Any]) -> str:
    if image.get("video_id"):
        return str(image["video_id"]).upper()
    match = VIDEO_PATTERN.search(str(image.get("file_name", "")))
    if not match:
        raise ValueError(f"cannot infer video_id from {image}")
    return match.group(1).upper()


def infer_frame_id(image: dict[str, Any]) -> int:
    for key in ("frame_id", "frame"):
        if image.get(key) is not None:
            return int(image[key])
    match = re.search(r"_frame_?0*(\d+)", str(image.get("file_name", "")), re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot infer frame_id from {image}")
    return int(match.group(1))


def infer_annotator_id(image: dict[str, Any]) -> str:
    raw = str(image.get("annotator_id") or image.get("source") or "")
    match = ANNOTATOR_PATTERN.search(raw) or ANNOTATOR_PATTERN.search(str(image.get("file_name", "")))
    return match.group(1) if match else raw or "na"


def xyxy_from_ann(annotation: dict[str, Any]) -> list[float]:
    if "bbox_xyxy" in annotation:
        return [float(x) for x in annotation["bbox_xyxy"]]
    x, y, w, h = [float(x) for x in annotation["bbox"]]
    return [x, y, x + w, y + h]


def keypoints_from_ann(annotation: dict[str, Any]) -> tuple[list[list[float]], list[int]]:
    flat = annotation.get("keypoints") or []
    if len(flat) >= 6:
        return [[float(flat[0]), float(flat[1])], [float(flat[3]), float(flat[4])]], [int(flat[2]), int(flat[5])]
    return [], []


def manifest_instance(annotation: dict[str, Any]) -> dict[str, Any]:
    kpts, vis = keypoints_from_ann(annotation)
    pose_mask = bool(annotation.get("pose_mask", int(len(kpts) >= 2 and annotation.get("num_keypoints", 0) > 0)))
    track_id = annotation.get("track_id")
    return {
        "bbox_xyxy": xyxy_from_ann(annotation),
        "class_id": 0,
        "keypoints": kpts,
        "visibility": vis,
        "track_id": None if track_id is None else str(track_id),
        "det_mask": int(annotation.get("det_mask", 1)),
        "pose_mask": int(pose_mask),
        "track_mask": int(bool(annotation.get("track_mask", track_id not in [None, -1, "-1", ""]))),
        "quality": float(annotation.get("pose_quality", annotation.get("quality", 1.0))),
        "source_group_id": annotation.get("source_group_id"),
        "pose_state": annotation.get("pose_state"),
    }


def yolo_row(inst: dict[str, Any], width: float, height: float) -> str | None:
    if int(inst.get("pose_mask", 0)) <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in inst["bbox_xyxy"]]
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    kpts = inst.get("keypoints") or []
    vis = inst.get("visibility") or []
    if len(kpts) < 2 or len(vis) < 2:
        return None
    hv, tv = int(vis[0]), int(vis[1])
    if hv <= 0 or tv <= 0:
        return None
    hx, hy = float(kpts[0][0]), float(kpts[0][1])
    tx, ty = float(kpts[1][0]), float(kpts[1][1])
    vals = [
        0,
        (x1 + w / 2) / width,
        (y1 + h / 2) / height,
        w / width,
        h / height,
        hx / width,
        hy / height,
        min(hv, 2),
        tx / width,
        ty / height,
        min(tv, 2),
    ]
    vals[1:] = [v if i in {7, 10} else min(max(float(v), 0.0), 1.0) for i, v in enumerate(vals[1:], 1)]
    return " ".join(str(v) if isinstance(v, int) else f"{v:.8f}" for v in vals)


def out_name(row: dict[str, Any]) -> str:
    src = Path(row["source_image_path"])
    suffix = src.suffix or ".jpg"
    annotator = row.get("annotator_id", "na")
    video = row.get("video_id", "unknown")
    frame = int(row["frame_id"])
    return f"ann{annotator}__{video}__frame{frame:06d}{suffix}"


def write_split(
    data: dict[str, Any],
    split: str,
    out_dir: Path,
    image_root: Path,
    copy_rgb: bool,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    image_paths: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    missing_images = 0
    skipped_pose_instances = 0
    det_instances = 0
    pose_instances = 0
    track_instances = 0

    for image in data.get("images", []):
        rel_path = str(image["file_name"]).replace("\\", "/")
        src = image_root / rel_path
        if not src.exists():
            missing_images += 1
            continue
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        if width <= 0 or height <= 0:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is None:
                missing_images += 1
                continue
            height, width = img.shape[:2]
            width, height = float(width), float(height)

        row = {
            "source_image_path": rel_path,
            "image_path": "",
            "video_id": infer_video_id(image),
            "frame_id": infer_frame_id(image),
            "domain": image.get("domain", "unknown"),
            "annotator_id": infer_annotator_id(image),
            "split": split,
            "width": int(width),
            "height": int(height),
            "schema_version": data.get("info", {}).get("schema_version"),
            "track_supervised": bool(image.get("track_supervised", False)),
            "instances": [],
        }
        instances = [manifest_instance(ann) for ann in annotations_by_image.get(int(image["id"]), [])]
        instances = [inst for inst in instances if int(inst.get("det_mask", 1)) > 0]
        labels = []
        for inst in instances:
            det_instances += 1
            pose_instances += int(inst.get("pose_mask", 0) > 0)
            track_instances += int(inst.get("track_mask", 0) > 0)
            label = yolo_row(inst, width, height)
            if label is None:
                skipped_pose_instances += 1
            else:
                labels.append(label)
        if not instances:
            continue

        name = out_name(row)
        dst_img = out_dir / "images" / split / name
        dst_lab = out_dir / "labels" / split / Path(name).with_suffix(".txt").name
        if row["domain"] == "IR":
            write_ir_normalized(src, dst_img)
        else:
            link_or_copy_rgb(src, dst_img, copy_rgb)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        row["image_path"] = str(dst_img.absolute())
        row["instances"] = instances
        manifest_rows.append(row)
        image_paths.append(str(dst_img.absolute()))

    stats = {
        "images": len(image_paths),
        "det_instances": det_instances,
        "pose_instances": pose_instances,
        "track_instances": track_instances,
        "missing_images": missing_images,
        "skipped_pose_instances": skipped_pose_instances,
        "empty_pose_label_images": sum(
            1 for row in manifest_rows if not (out_dir / "labels" / split / Path(row["image_path"]).with_suffix(".txt").name).read_text(encoding="utf-8").strip()
        ),
        "domains": {
            "RGB": sum(1 for row in manifest_rows if row.get("domain") == "RGB"),
            "IR": sum(1 for row in manifest_rows if row.get("domain") == "IR"),
        },
        "videos": sorted({str(row.get("video_id")) for row in manifest_rows}),
    }
    return image_paths, manifest_rows, stats


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_coco(args.train_json)
    test_data = load_coco(args.test_json)
    train_images, train_manifest, train_stats = write_split(train_data, "train", args.out_dir, args.image_root, args.copy_rgb)
    test_images, test_manifest, test_stats = write_split(test_data, "test", args.out_dir, args.image_root, args.copy_rgb)

    (args.out_dir / "train_images.txt").write_text("\n".join(train_images) + "\n", encoding="utf-8")
    (args.out_dir / "test_images.txt").write_text("\n".join(test_images) + "\n", encoding="utf-8")
    # Compatibility for Ultralytics and older helper scripts that use "val" terminology.
    (args.out_dir / "val_images.txt").write_text("\n".join(test_images) + "\n", encoding="utf-8")

    with (args.out_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in train_manifest + test_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    yaml_text = f"""path: {args.out_dir.resolve()}
train: train_images.txt
val: test_images.txt

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
    (args.out_dir / "bee_yolo_pose_strict.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "schema_v2_train_json": str(args.train_json),
        "schema_v2_train_sha256": sha256(args.train_json),
        "schema_v2_test_json": str(args.test_json),
        "schema_v2_test_sha256": sha256(args.test_json),
        "image_root": str(args.image_root),
        "train": train_stats,
        "test": test_stats,
        "yaml": str(args.out_dir / "bee_yolo_pose_strict.yaml"),
        "manifest": str(args.out_dir / "dataset_manifest.jsonl"),
        "label_note": "YOLO-Pose labels include only pose_mask=1 instances with visible head/tail; manifest keeps all det_mask=1 instances for detection and tracking evaluation.",
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
