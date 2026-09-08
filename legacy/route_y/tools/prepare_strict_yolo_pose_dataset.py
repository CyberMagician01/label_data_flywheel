#!/usr/bin/env python3
"""Prepare the strict BeePoseTrack-Y A1 single-frame aligned dataset.

This writes:
- YOLO pose images/labels with fixed head,tail order.
- Explicit dataset_manifest.jsonl.
- Frozen fold train/val image lists.
- A dataset yaml that points to the frozen lists.

IR images are percentile-normalized once during preparation, then saved as
three-channel images so YOLO training sees the documented shared input.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-json", action="append", required=True)
    parser.add_argument("--val-json", action="append", required=True)
    parser.add_argument("--fold-id", default="fold4")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--balance-domains", action="store_true", help="Freeze equal RGB/IR frame counts per split.")
    parser.add_argument("--copy-rgb", action="store_true", help="Copy RGB images instead of symlinking.")
    return parser.parse_args()


def video_domain(file_name: str) -> tuple[str, str]:
    for marker in ("A-5-1", "A-5-2", "A-5-3", "A-5-4", "B-5-1", "B-5-2", "B-5-3", "B-5-4"):
        if marker in file_name:
            return marker, "RGB" if marker.startswith("A-") else "IR"
    return "unknown", "RGB"


def frame_id_from_name(file_name: str) -> int | None:
    stem = Path(file_name).stem
    if "_frame_" not in stem:
        return None
    tail = stem.rsplit("_frame_", 1)[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def safe_name(prefix: str, file_name: str, image_id: int) -> str:
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix or ".jpg"
    return f"{prefix}__{image_id:08d}__{stem}{suffix}"


def link_or_copy_rgb(src: Path, dst: Path, copy_rgb: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_rgb:
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
    ok = cv2.imwrite(str(dst), bgr)
    if not ok:
        raise RuntimeError(f"failed to write {dst}")


def yolo_row(ann: dict[str, Any], width: float, height: float) -> tuple[str | None, dict[str, Any] | None]:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None, None
    kpts = ann.get("keypoints", [])
    if len(kpts) < 6:
        return None, None
    hx, hy, hv = float(kpts[0]), float(kpts[1]), int(kpts[2])
    tx, ty, tv = float(kpts[3]), float(kpts[4]), int(kpts[5])
    if hv <= 0 or tv <= 0:
        pose_mask = 0
    else:
        pose_mask = 1
    if not pose_mask:
        return None, None
    vals = [
        0,
        (x + w / 2) / width,
        (y + h / 2) / height,
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
    row = " ".join(str(v) if isinstance(v, int) else f"{v:.8f}" for v in vals)
    inst = {
        "bbox_xyxy": [x, y, x + w, y + h],
        "class_id": 0,
        "keypoints": [[hx, hy], [tx, ty]],
        "visibility": [min(hv, 2), min(tv, 2)],
        "track_id": ann.get("track_id"),
        "det_mask": 1,
        "pose_mask": pose_mask,
        "track_mask": 1 if ann.get("track_id") is not None else 0,
        "quality": float(ann.get("quality", 1.0)),
    }
    return row, inst


def load_records(coco_root: Path, json_paths: list[str], split: str, fold_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_name in json_paths:
        ann_path = coco_root / json_name
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        images = {int(img["id"]): img for img in data["images"]}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        prefix = ann_path.stem
        for image_id, img in sorted(images.items()):
            src = coco_root / img["file_name"]
            if not src.exists():
                continue
            rows: list[str] = []
            instances: list[dict[str, Any]] = []
            width, height = float(img["width"]), float(img["height"])
            for ann in anns_by_image.get(image_id, []):
                row, inst = yolo_row(ann, width, height)
                if row is not None and inst is not None:
                    rows.append(row)
                    instances.append(inst)
            if not rows:
                continue
            video_id, domain = video_domain(img["file_name"])
            records.append(
                {
                    "src": src,
                    "dst_name": safe_name(prefix, img["file_name"], image_id),
                    "label_rows": rows,
                    "manifest": {
                        "image_path": None,
                        "video_id": video_id,
                        "frame_id": frame_id_from_name(img["file_name"]),
                        "domain": domain,
                        "annotator_id": "manual",
                        "split_fold": fold_id,
                        "split": split,
                        "source_json": json_name,
                        "instances": instances,
                    },
                }
            )
    return records


def balance_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {"RGB": [], "IR": []}
    for rec in records:
        by_domain.setdefault(rec["manifest"]["domain"], []).append(rec)
    if not by_domain.get("RGB") or not by_domain.get("IR"):
        return records
    n = min(len(by_domain["RGB"]), len(by_domain["IR"]))
    rng = random.Random(seed)
    balanced = []
    for domain in ("RGB", "IR"):
        items = list(by_domain[domain])
        rng.shuffle(items)
        balanced.extend(sorted(items[:n], key=lambda r: (r["manifest"]["video_id"], r["manifest"]["frame_id"] or -1)))
    return sorted(balanced, key=lambda r: (r["manifest"]["domain"], r["manifest"]["video_id"], r["manifest"]["frame_id"] or -1))


def write_split(out_dir: Path, split: str, records: list[dict[str, Any]], copy_rgb: bool) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_rows: list[dict[str, Any]] = []
    image_paths: list[str] = []
    for rec in records:
        image_path = out_dir / "images" / split / rec["dst_name"]
        label_path = out_dir / "labels" / split / Path(rec["dst_name"]).with_suffix(".txt").name
        domain = rec["manifest"]["domain"]
        if domain == "IR":
            write_ir_normalized(rec["src"], image_path)
        else:
            link_or_copy_rgb(rec["src"], image_path, copy_rgb)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(rec["label_rows"]) + "\n", encoding="utf-8")
        rec_manifest = dict(rec["manifest"])
        rec_manifest["image_path"] = str(image_path.absolute())
        manifest_rows.append(rec_manifest)
        image_paths.append(str(image_path.absolute()))
    return manifest_rows, image_paths


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_records(args.coco_root, args.train_json, "train", args.fold_id)
    val_records = load_records(args.coco_root, args.val_json, "val", args.fold_id)
    if args.balance_domains:
        train_records = balance_records(train_records, args.seed)
        val_records = balance_records(val_records, args.seed)

    train_manifest, train_images = write_split(args.out_dir, "train", train_records, args.copy_rgb)
    val_manifest, val_images = write_split(args.out_dir, "val", val_records, args.copy_rgb)

    (args.out_dir / "train_images.txt").write_text("\n".join(train_images) + "\n", encoding="utf-8")
    (args.out_dir / "val_images.txt").write_text("\n".join(val_images) + "\n", encoding="utf-8")
    with (args.out_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in train_manifest + val_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    yaml_text = f"""path: {args.out_dir.resolve()}
train: train_images.txt
val: val_images.txt

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
    (args.out_dir / "bee_yolo_pose_strict.yaml").write_text(yaml_text, encoding="utf-8")

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for domain in ("RGB", "IR"):
            subset = [r for r in rows if r["domain"] == domain]
            out[domain] = {
                "images": len(subset),
                "instances": sum(len(r["instances"]) for r in subset),
            }
        return out

    summary = {
        "fold_id": args.fold_id,
        "seed": args.seed,
        "balance_domains": args.balance_domains,
        "train": summarize(train_manifest),
        "val": summarize(val_manifest),
        "yaml": str(args.out_dir / "bee_yolo_pose_strict.yaml"),
        "manifest": str(args.out_dir / "dataset_manifest.jsonl"),
    }
    (args.out_dir / "strict_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
