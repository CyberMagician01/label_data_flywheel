#!/usr/bin/env python3
"""Prepare YOLO pose A1 folds directly from the shared Y/E manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--copy-rgb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
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


def yolo_row(inst: dict[str, Any], width: float, height: float) -> str | None:
    x1, y1, x2, y2 = [float(v) for v in inst["bbox_xyxy"]]
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    if int(inst.get("pose_mask", 1)) <= 0:
        return None
    kpts = inst.get("keypoints", [])
    vis = inst.get("visibility", [2, 2])
    if len(kpts) < 2:
        return None
    hx, hy = float(kpts[0][0]), float(kpts[0][1])
    tx, ty = float(kpts[1][0]), float(kpts[1][1])
    hv, tv = int(vis[0]), int(vis[1])
    if hv <= 0 or tv <= 0:
        return None
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


def image_size(src: Path, row: dict[str, Any]) -> tuple[float, float]:
    width, height = row.get("width"), row.get("height")
    if width and height:
        return float(width), float(height)
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(src)
    h, w = img.shape[:2]
    return float(w), float(h)


def out_name(row: dict[str, Any]) -> str:
    src = Path(row["image_path"])
    stem = src.stem
    suffix = src.suffix or ".jpg"
    annotator = row.get("annotator_id", "na")
    video = row.get("video_id", "unknown")
    frame = row.get("frame_id")
    frame_tag = f"{int(frame):06d}" if isinstance(frame, int) else stem
    return f"ann{annotator}__{video}__frame{frame_tag}{suffix}"


def write_split(
    fold_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
    image_root: Path,
    copy_rgb: bool,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    image_paths: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    skipped_instances = 0
    missing_images = 0
    for row in rows:
        src = image_root / row["image_path"]
        if not src.exists():
            missing_images += 1
            continue
        width, height = image_size(src, row)
        labels = []
        kept_instances = []
        for inst in row.get("instances", []):
            label = yolo_row(inst, width, height)
            if label is None:
                skipped_instances += 1
                continue
            labels.append(label)
            kept_instances.append(inst)
        if not labels:
            continue
        name = out_name(row)
        dst_img = fold_dir / "images" / split / name
        dst_lab = fold_dir / "labels" / split / Path(name).with_suffix(".txt").name
        if row.get("domain") == "IR":
            write_ir_normalized(src, dst_img)
        else:
            link_or_copy_rgb(src, dst_img, copy_rgb)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.write_text("\n".join(labels) + "\n", encoding="utf-8")
        frozen = dict(row)
        frozen["shared_manifest_image_path"] = row["image_path"]
        frozen["image_path"] = str(dst_img.absolute())
        frozen["instances"] = kept_instances
        frozen["split"] = split
        manifest_rows.append(frozen)
        image_paths.append(str(dst_img.absolute()))
    stats = {
        "images": len(image_paths),
        "instances": sum(len(r["instances"]) for r in manifest_rows),
        "missing_images": missing_images,
        "skipped_instances": skipped_instances,
        "domains": {
            "RGB": sum(1 for r in manifest_rows if r.get("domain") == "RGB"),
            "IR": sum(1 for r in manifest_rows if r.get("domain") == "IR"),
        },
    }
    return image_paths, manifest_rows, stats


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summary = {
        "shared_manifest": str(args.manifest),
        "shared_manifest_sha256": sha256(args.manifest),
        "image_root": str(args.image_root),
        "folds": {},
    }
    for fold in args.folds:
        fold_dir = args.out_root / f"fold{fold}"
        train_rows = [r for r in records if int(r["split_fold"]) != fold]
        val_rows = [r for r in records if int(r["split_fold"]) == fold]
        train_images, train_manifest, train_stats = write_split(fold_dir, "train", train_rows, args.image_root, args.copy_rgb)
        val_images, val_manifest, val_stats = write_split(fold_dir, "val", val_rows, args.image_root, args.copy_rgb)
        (fold_dir / "train_images.txt").write_text("\n".join(train_images) + "\n", encoding="utf-8")
        (fold_dir / "val_images.txt").write_text("\n".join(val_images) + "\n", encoding="utf-8")
        with (fold_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as f:
            for row in train_manifest + val_manifest:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        yaml_text = f"""path: {fold_dir.resolve()}
train: train_images.txt
val: val_images.txt

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
        (fold_dir / "bee_yolo_pose_strict.yaml").write_text(yaml_text, encoding="utf-8")
        all_summary["folds"][f"fold{fold}"] = {
            "train": train_stats,
            "val": val_stats,
            "yaml": str(fold_dir / "bee_yolo_pose_strict.yaml"),
            "manifest": str(fold_dir / "dataset_manifest.jsonl"),
        }
    (args.out_root / "fold_dataset_summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
