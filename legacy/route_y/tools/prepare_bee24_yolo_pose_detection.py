#!/usr/bin/env python3
"""Build a BEE24 detection-only YOLO pose view for continuation pretraining."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bee24-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--no-val", action="store_true", help="Use all official BEE24 train sequences for training.")
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


def stable_score(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


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


def read_gt_rows(seq: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    size = read_seq_size(seq)
    if not size:
        return {}
    width, height = size
    rows_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    gt_path = seq / "gt" / "gt.txt"
    if not gt_path.exists():
        return rows_by_frame
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
            ((x + w / 2.0) / width, (y + h / 2.0) / height, w / width, h / height)
        )
    return rows_by_frame


def yolo_pose_bbox_only_row(cx: float, cy: float, w: float, h: float) -> str | None:
    if w <= 0 or h <= 0:
        return None
    cx, cy, w, h = [min(max(v, 0.0), 1.0) for v in (cx, cy, w, h)]
    return (
        f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f} "
        "0.00000000 0.00000000 0 0.00000000 0.00000000 0"
    )


def usable_train_sequences(bee24_root: Path) -> list[Path]:
    seqs = []
    for seq in sorted((bee24_root / "train").iterdir()):
        if not seq.is_dir():
            continue
        if not (seq / "img1").is_dir() or not (seq / "seqinfo.ini").exists():
            continue
        if not ((seq / "labels_with_ids").is_dir() or (seq / "gt" / "gt.txt").exists()):
            continue
        seqs.append(seq)
    return seqs


def split_sequences(seqs: list[Path], seed: int, val_ratio: float) -> tuple[list[Path], list[Path]]:
    ordered = sorted(seqs, key=lambda p: (stable_score(p.name, seed), p.name))
    val_n = max(1, round(len(ordered) * val_ratio))
    return sorted(ordered[val_n:], key=lambda p: p.name), sorted(ordered[:val_n], key=lambda p: p.name)


def convert_sequences(seqs: list[Path], out_dir: Path, split: str, copy: bool) -> dict:
    image_list = []
    total_instances = 0
    empty_images = 0
    per_seq = {}
    for seq in seqs:
        gt_rows = read_gt_rows(seq)
        seq_instances = 0
        seq_empty = 0
        for src_img in image_files(seq / "img1"):
            dst_name = f"bee24__{seq.name}__{src_img.name}"
            dst_img = out_dir / "images" / split / dst_name
            dst_label = out_dir / "labels" / split / Path(dst_name).with_suffix(".txt").name
            rows = []
            label_with_ids = seq / "labels_with_ids" / src_img.with_suffix(".txt").name
            if label_with_ids.exists():
                for line in label_with_ids.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    _, _, cx, cy, w, h = parts[:6]
                    row = yolo_pose_bbox_only_row(float(cx), float(cy), float(w), float(h))
                    if row:
                        rows.append(row)
            else:
                try:
                    frame_id = int(src_img.stem)
                except ValueError:
                    frame_id = int(src_img.stem.split("_")[-1])
                for cx, cy, w, h in gt_rows.get(frame_id, []):
                    row = yolo_pose_bbox_only_row(cx, cy, w, h)
                    if row:
                        rows.append(row)
            link_or_copy(src_img, dst_img, copy)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            image_list.append(str(dst_img.absolute()))
            total_instances += len(rows)
            seq_instances += len(rows)
            empty_images += int(not rows)
            seq_empty += int(not rows)
        per_seq[seq.name] = {
            "images": len(image_files(seq / "img1")),
            "instances": seq_instances,
            "empty_images": seq_empty,
        }
    return {
        "images": len(image_list),
        "instances": total_instances,
        "empty_images": empty_images,
        "image_list": sorted(image_list, key=lambda s: (stable_score(s, 2026), s)),
        "sequences": per_seq,
    }


def official_test_sequences(bee24_root: Path) -> list[str]:
    names = []
    for seq in sorted((bee24_root / "test").iterdir()):
        if seq.is_dir() and (seq / "img1").is_dir() and (seq / "seqinfo.ini").exists() and (seq / "gt" / "gt.txt").exists():
            names.append(seq.name)
    return names


def write_list(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    all_train_seqs = usable_train_sequences(args.bee24_root)
    if args.no_val:
        train_seqs, val_seqs = all_train_seqs, []
    else:
        train_seqs, val_seqs = split_sequences(all_train_seqs, args.seed, args.val_ratio)
    train = convert_sequences(train_seqs, args.out_dir, "train", args.copy)
    val = convert_sequences(val_seqs, args.out_dir, "val", args.copy)
    write_list(args.out_dir / "train_images.txt", train["image_list"])
    write_list(args.out_dir / "val_images.txt", val["image_list"] if val["image_list"] else train["image_list"])
    (args.out_dir / "bee_yolo_pose.yaml").write_text(
        f"""path: {args.out_dir.resolve()}
train: train_images.txt
val: val_images.txt

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
""",
        encoding="utf-8",
    )
    summary = {
        "tag": "BEE24_detection_only_for_public_continuation",
        "bee24_root": str(args.bee24_root.resolve()),
        "seed": args.seed,
        "val_ratio": 0.0 if args.no_val else args.val_ratio,
        "no_val": args.no_val,
        "supervision": "det_mask=1, pose_mask=0, track_mask=0; labels are 11-column YOLO pose rows with head/tail visibility 0",
        "train": {k: v for k, v in train.items() if k != "image_list"},
        "val": {k: v for k, v in val.items() if k != "image_list"},
        "sequence_split": {
            "train": [p.name for p in train_seqs],
            "val": [p.name for p in val_seqs],
            "official_test_reserved": official_test_sequences(args.bee24_root),
        },
        "yaml": str((args.out_dir / "bee_yolo_pose.yaml").resolve()),
    }
    (args.out_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
