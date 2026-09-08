#!/usr/bin/env python3
"""Build the public-all YOLO pose dataset: BeePose/Mendeley pose + BEE24 detection.

The output is a YOLO pose view with the same 2-keypoint head/tail schema used by
the Y route. BEE24 contributes detection supervision only: its labels are written
with valid bbox/class fields and head/tail visibility set to 0.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-pose-dir", type=Path, required=True)
    parser.add_argument("--bee24-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bee24-val-ratio", type=float, default=0.2)
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def stable_score(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def yolo_pose_bbox_only_row(cx: float, cy: float, w: float, h: float) -> str | None:
    vals = [cx, cy, w, h]
    if any(v < 0 for v in vals) or w <= 0 or h <= 0:
        return None
    cx, cy, w, h = [min(max(v, 0.0), 1.0) for v in vals]
    return (
        f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f} "
        "0.00000000 0.00000000 0 0.00000000 0.00000000 0"
    )


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def copy_public_pose_split(public_pose_dir: Path, out_dir: Path, split: str, copy: bool) -> dict:
    src_image_dir = public_pose_dir / "images" / split
    src_label_dir = public_pose_dir / "labels" / split
    dst_image_dir = out_dir / "images" / split
    dst_label_dir = out_dir / "labels" / split
    images: list[str] = []
    image_count = 0
    instance_count = 0
    empty_images = 0

    for src_img in image_files(src_image_dir):
        dst_name = f"pose__{src_img.name}"
        dst_img = dst_image_dir / dst_name
        dst_label = dst_label_dir / Path(dst_name).with_suffix(".txt").name
        src_label = src_label_dir / src_img.with_suffix(".txt").name
        link_or_copy(src_img, dst_img, copy)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        text = src_label.read_text(encoding="utf-8") if src_label.exists() else ""
        dst_label.write_text(text, encoding="utf-8")
        rows = [ln for ln in text.splitlines() if ln.strip()]
        image_count += 1
        instance_count += len(rows)
        empty_images += int(not rows)
        images.append(str(dst_img.absolute()))

    return {
        "images": image_count,
        "instances": instance_count,
        "empty_images": empty_images,
        "image_list": images,
    }


def read_seq_size(seq: Path) -> tuple[float, float] | None:
    seqinfo = seq / "seqinfo.ini"
    if not seqinfo.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(seqinfo, encoding="utf-8")
    try:
        return float(parser["Sequence"]["imWidth"]), float(parser["Sequence"]["imHeight"])
    except KeyError:
        return None


def read_gt_rows(gt_path: Path, image_width: float, image_height: float) -> dict[int, list[str]]:
    rows_by_frame: dict[int, list[str]] = {}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        x, y, w, h = [float(v) for v in parts[2:6]]
        if w <= 0 or h <= 0:
            continue
        rows_by_frame.setdefault(frame_id, []).append(
            ((x + w / 2.0) / image_width, (y + h / 2.0) / image_height, w / image_width, h / image_height)
        )
    return rows_by_frame


def bee24_sequences(bee24_root: Path, split: str) -> list[Path]:
    base = bee24_root / split
    seqs = []
    for seq in sorted(p for p in base.iterdir() if p.is_dir()):
        if (
            (seq / "img1").is_dir()
            and (seq / "seqinfo.ini").exists()
            and ((seq / "labels_with_ids").is_dir() or (seq / "gt" / "gt.txt").exists())
        ):
            seqs.append(seq)
    return seqs


def split_bee24_train_sequences(seqs: list[Path], seed: int, val_ratio: float) -> tuple[list[Path], list[Path]]:
    ordered = sorted(seqs, key=lambda p: (stable_score(p.name, seed), p.name))
    val_n = max(1, round(len(ordered) * val_ratio)) if ordered else 0
    val = sorted(ordered[:val_n], key=lambda p: p.name)
    train = sorted(ordered[val_n:], key=lambda p: p.name)
    return train, val


def convert_bee24_sequences(
    seqs: Iterable[Path],
    out_dir: Path,
    split: str,
    copy: bool,
) -> dict:
    dst_image_dir = out_dir / "images" / split
    dst_label_dir = out_dir / "labels" / split
    images: list[str] = []
    image_count = 0
    instance_count = 0
    empty_images = 0
    sequence_summary = {}

    for seq in sorted(seqs, key=lambda p: p.name):
        seq_size = read_seq_size(seq)
        gt_by_frame = {}
        if seq_size:
            gt_by_frame = read_gt_rows(seq / "gt" / "gt.txt", seq_size[0], seq_size[1])
        seq_images = image_files(seq / "img1")
        seq_instances = 0
        seq_empty = 0
        for src_img in seq_images:
            try:
                frame_id = int(src_img.stem)
            except ValueError:
                frame_id = int(src_img.stem.split("_")[-1])
            rows = []
            label_with_ids = seq / "labels_with_ids" / src_img.with_suffix(".txt").name
            if label_with_ids.exists():
                for line in label_with_ids.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    # BEE24 labels_with_ids: class, track_id, cx, cy, w, h.
                    _, _, cx, cy, w, h = parts[:6]
                    row = yolo_pose_bbox_only_row(float(cx), float(cy), float(w), float(h))
                    if row:
                        rows.append(row)
            else:
                for cx, cy, w, h in gt_by_frame.get(frame_id, []):
                    row = yolo_pose_bbox_only_row(cx, cy, w, h)
                    if row:
                        rows.append(row)
            dst_name = f"bee24__{seq.name}__{src_img.name}"
            dst_img = dst_image_dir / dst_name
            dst_label = dst_label_dir / Path(dst_name).with_suffix(".txt").name
            link_or_copy(src_img, dst_img, copy)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            image_count += 1
            instance_count += len(rows)
            seq_instances += len(rows)
            empty_images += int(not rows)
            seq_empty += int(not rows)
            images.append(str(dst_img.absolute()))
        sequence_summary[seq.name] = {
            "images": len(seq_images),
            "instances": seq_instances,
            "empty_images": seq_empty,
        }

    return {
        "images": image_count,
        "instances": instance_count,
        "empty_images": empty_images,
        "image_list": images,
        "sequences": sequence_summary,
    }


def write_list(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.public_pose_dir.exists():
        raise FileNotFoundError(args.public_pose_dir)
    if not args.bee24_root.exists():
        raise FileNotFoundError(args.bee24_root)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    public_train = copy_public_pose_split(args.public_pose_dir, args.out_dir, "train", args.copy)
    public_val = copy_public_pose_split(args.public_pose_dir, args.out_dir, "val", args.copy)

    bee24_train_all = bee24_sequences(args.bee24_root, "train")
    bee24_train_seqs, bee24_val_seqs = split_bee24_train_sequences(
        bee24_train_all, args.seed, args.bee24_val_ratio
    )
    bee24_train = convert_bee24_sequences(bee24_train_seqs, args.out_dir, "train", args.copy)
    bee24_val = convert_bee24_sequences(bee24_val_seqs, args.out_dir, "val", args.copy)

    train_images = public_train["image_list"] + bee24_train["image_list"]
    val_images = public_val["image_list"] + bee24_val["image_list"]
    train_images = sorted(train_images, key=lambda s: (stable_score(s, args.seed), s))
    val_images = sorted(val_images, key=lambda s: (stable_score(s, args.seed), s))
    write_list(args.out_dir / "train_images.txt", train_images)
    write_list(args.out_dir / "val_images.txt", val_images)

    yaml_text = f"""path: {args.out_dir.resolve()}
train: train_images.txt
val: val_images.txt

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
    (args.out_dir / "bee_yolo_pose.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "tag": "BeePose_Mendeley_BEE24_public_EY_aligned",
        "public_pose_dir": str(args.public_pose_dir.resolve()),
        "bee24_root": str(args.bee24_root.resolve()),
        "seed": args.seed,
        "bee24_val_ratio": args.bee24_val_ratio,
        "supervision": {
            "BeePose_Mendeley": "det_mask=1, pose_mask=1 where keypoints are visible; source labels preserved",
            "BEE24": "det_mask=1, pose_mask=0, track_mask=0; head/tail visibility is 0",
        },
        "train": {
            "images": len(train_images),
            "instances": public_train["instances"] + bee24_train["instances"],
            "public_pose": {k: v for k, v in public_train.items() if k != "image_list"},
            "bee24_detection": {k: v for k, v in bee24_train.items() if k != "image_list"},
        },
        "val": {
            "images": len(val_images),
            "instances": public_val["instances"] + bee24_val["instances"],
            "public_pose": {k: v for k, v in public_val.items() if k != "image_list"},
            "bee24_detection": {k: v for k, v in bee24_val.items() if k != "image_list"},
        },
        "bee24_sequence_split": {
            "train": [p.name for p in bee24_train_seqs],
            "val": [p.name for p in bee24_val_seqs],
            "official_test_reserved": [p.name for p in bee24_sequences(args.bee24_root, "test") if (p / "seqinfo.ini").exists()],
        },
        "yaml": str((args.out_dir / "bee_yolo_pose.yaml").resolve()),
    }
    (args.out_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
