#!/usr/bin/env python3
"""Build a YOLO pose dataset from BeePoseTrack COCO head/tail annotations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-json", action="append", required=True)
    parser.add_argument("--val-json", action="append", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--tag", default="public_bee_pose", help="Dataset tag written to the summary.")
    parser.add_argument("--keep-partial-keypoints", action="store_true")
    parser.add_argument("--include-empty-images", action="store_true")
    return parser.parse_args()


def safe_name(prefix: str, file_name: str, image_id: int) -> str:
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix or ".jpg"
    return f"{prefix}__{image_id:08d}__{stem}{suffix}"


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def yolo_row(ann: dict[str, Any], width: float, height: float, keep_partial: bool) -> str | None:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    kpts = ann.get("keypoints", [])
    if len(kpts) < 6:
        return None
    hx, hy, hv = float(kpts[0]), float(kpts[1]), int(kpts[2])
    tx, ty, tv = float(kpts[3]), float(kpts[4]), int(kpts[5])
    if not keep_partial and (hv <= 0 or tv <= 0):
        return None
    if keep_partial and hv <= 0 and tv <= 0:
        return None
    if hv <= 0:
        hx, hy = 0.0, 0.0
    if tv <= 0:
        tx, ty = 0.0, 0.0
    vals = [
        0,
        (x + w / 2) / width,
        (y + h / 2) / height,
        w / width,
        h / height,
        hx / width,
        hy / height,
        min(max(hv, 0), 2),
        tx / width,
        ty / height,
        min(max(tv, 0), 2),
    ]
    vals[1:] = [
        v if i in {7, 10} else min(max(float(v), 0.0), 1.0)
        for i, v in enumerate(vals[1:], 1)
    ]
    return " ".join(str(v) if isinstance(v, int) else f"{v:.8f}" for v in vals)


def convert_split(
    ann_root: Path,
    image_root: Path,
    json_paths: list[str],
    out_dir: Path,
    split: str,
    copy: bool,
    keep_partial: bool,
    include_empty: bool,
) -> dict[str, int]:
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    image_count = 0
    instance_count = 0
    skipped = 0
    missing_images = 0
    empty_images = 0
    for json_name in json_paths:
        ann_path = ann_root / json_name
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        images = {int(img["id"]): img for img in data["images"]}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        prefix = ann_path.stem
        for image_id, img in sorted(images.items()):
            src = image_root / img["file_name"]
            if not src.exists():
                missing_images += 1
                skipped += len(anns_by_image.get(image_id, []))
                continue
            rows = []
            width, height = float(img["width"]), float(img["height"])
            for ann in anns_by_image.get(image_id, []):
                row = yolo_row(ann, width, height, keep_partial)
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)
            if not rows and not include_empty:
                continue
            dst_name = safe_name(prefix, img["file_name"], image_id)
            link_or_copy(src.absolute(), image_dir / dst_name, copy)
            (label_dir / Path(dst_name).with_suffix(".txt").name).parent.mkdir(parents=True, exist_ok=True)
            label_text = "\n".join(rows) + ("\n" if rows else "")
            (label_dir / Path(dst_name).with_suffix(".txt").name).write_text(label_text, encoding="utf-8")
            image_count += 1
            instance_count += len(rows)
            if not rows:
                empty_images += 1
    return {
        "images": image_count,
        "instances": instance_count,
        "skipped": skipped,
        "missing_images": missing_images,
        "empty_images": empty_images,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = convert_split(
        args.ann_root,
        args.image_root,
        args.train_json,
        args.out_dir,
        "train",
        args.copy,
        args.keep_partial_keypoints,
        args.include_empty_images,
    )
    val = convert_split(
        args.ann_root,
        args.image_root,
        args.val_json,
        args.out_dir,
        "val",
        args.copy,
        args.keep_partial_keypoints,
        args.include_empty_images,
    )
    yaml_text = f"""path: {args.out_dir.resolve()}
train: images/train
val: images/val

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
    (args.out_dir / "bee_yolo_pose.yaml").write_text(yaml_text, encoding="utf-8")
    summary = {
        "tag": args.tag,
        "ann_root": str(args.ann_root),
        "image_root": str(args.image_root),
        "train_json": args.train_json,
        "val_json": args.val_json,
        "keep_partial_keypoints": args.keep_partial_keypoints,
        "include_empty_images": args.include_empty_images,
        "train": train,
        "val": val,
        "yaml": str(args.out_dir / "bee_yolo_pose.yaml"),
    }
    (args.out_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
