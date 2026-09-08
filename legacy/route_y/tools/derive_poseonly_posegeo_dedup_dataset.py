#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/data/bee26/beeposetrack_y_20260827/"
    "datasets/y_unified_20260901_labelme5_a1_strict_v8_annotator01_val_03_test_poseonly"
)
DEFAULT_OUT = Path(
    "/data/bee26/beeposetrack_y_20260827/"
    "datasets/y_unified_20260901_labelme5_a1_strict_v8_annotator01_val_03_test_poseonly_posegeo_dedup"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pose-dist-ratio", type=float, default=0.15)
    parser.add_argument("--contain-ratio", type=float, default=0.75)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def diag(box: list[float]) -> float:
    return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def intersect_area(a: list[float], b: list[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def valid_point(point: Any) -> bool:
    return (
        isinstance(point, list)
        and len(point) >= 2
        and math.isfinite(float(point[0]))
        and math.isfinite(float(point[1]))
    )


def point_in_box(point: list[float], box: list[float]) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 <= x <= x2 and y1 <= y <= y2


def valid_pose_instance(inst: dict[str, Any]) -> bool:
    box = inst.get("bbox_xyxy")
    kpts = inst.get("keypoints") or []
    if not isinstance(box, list) or len(box) != 4 or area(box) <= 1.0:
        return False
    if len(kpts) < 2 or not valid_point(kpts[0]) or not valid_point(kpts[1]):
        return False
    return point_in_box(kpts[0], box) and point_in_box(kpts[1], box)


def point_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def min_kpt_score(inst: dict[str, Any]) -> float:
    keys = ("min_keypoint_score", "pose_score", "head_score", "tail_score", "quality")
    values = []
    for key in keys:
        if key in inst:
            try:
                values.append(float(inst[key]))
            except (TypeError, ValueError):
                pass
    vis = inst.get("visibility") or []
    if vis:
        try:
            values.append(min(float(v) for v in vis))
        except (TypeError, ValueError):
            pass
    return min(values) if values else 1.0


def det_score(inst: dict[str, Any]) -> float:
    for key in ("score", "det_score", "quality"):
        if key in inst:
            try:
                return float(inst[key])
            except (TypeError, ValueError):
                pass
    return 1.0


def choose_duplicate_drop(a: dict[str, Any], b: dict[str, Any], contain_ratio: float) -> int:
    box_a = a["bbox_xyxy"]
    box_b = b["bbox_xyxy"]
    area_a = area(box_a)
    area_b = area(box_b)
    small = max(min(area_a, area_b), 1e-9)
    contain = intersect_area(box_a, box_b) / small
    if contain >= contain_ratio and area_a != area_b:
        return 0 if area_a < area_b else 1

    k_a, k_b = min_kpt_score(a), min_kpt_score(b)
    if k_a != k_b:
        return 0 if k_a < k_b else 1
    d_a, d_b = det_score(a), det_score(b)
    if d_a != d_b:
        return 0 if d_a < d_b else 1
    if area_a != area_b:
        return 0 if area_a < area_b else 1
    return 1


def dedup_instances(instances: list[dict[str, Any]], dist_ratio: float, contain_ratio: float) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    valid = []
    for inst in instances:
        if valid_pose_instance(inst):
            valid.append(inst)
        else:
            stats["invalid_pose_removed"] += 1

    dropped: set[int] = set()
    candidates = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            min_diag = max(min(diag(a["bbox_xyxy"]), diag(b["bbox_xyxy"])), 1e-9)
            threshold = dist_ratio * min_diag
            head_dist = point_dist(a["keypoints"][0], b["keypoints"][0])
            tail_dist = point_dist(a["keypoints"][1], b["keypoints"][1])
            if head_dist < threshold and tail_dist < threshold:
                candidates.append((head_dist + tail_dist, i, j))

    for _, i, j in sorted(candidates):
        if i in dropped or j in dropped:
            continue
        rel_drop = choose_duplicate_drop(valid[i], valid[j], contain_ratio)
        dropped.add(i if rel_drop == 0 else j)
        stats["duplicate_pose_removed"] += 1

    kept = []
    for idx, inst in enumerate(valid):
        if idx in dropped:
            continue
        item = dict(inst)
        item["det_mask"] = 1
        item["pose_mask"] = 1
        item["track_mask"] = int(item.get("track_mask", 1))
        item["visibility"] = [2, 2]
        kept.append(item)
    return kept, stats


def yolo_row(inst: dict[str, Any], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = [float(v) for v in inst["bbox_xyxy"]]
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    head, tail = inst["keypoints"]
    values: list[float | int] = [
        0,
        (x1 + w / 2.0) / width,
        (y1 + h / 2.0) / height,
        w / width,
        h / height,
        float(head[0]) / width,
        float(head[1]) / height,
        2,
        float(tail[0]) / width,
        float(tail[1]) / height,
        2,
    ]
    clipped = [values[0]]
    for idx, value in enumerate(values[1:], 1):
        if idx in {7, 10}:
            clipped.append(value)
        else:
            clipped.append(min(max(float(value), 0.0), 1.0))
    return " ".join(str(v) if isinstance(v, int) else f"{v:.8f}" for v in clipped)


def relabel_path(source_dataset: Path, out_dataset: Path, image_path: str) -> Path:
    path = Path(image_path)
    try:
        rel = path.relative_to(source_dataset / "images")
    except ValueError:
        split = path.parent.name
        rel = Path(split) / path.name
    return out_dataset / "labels" / rel.with_suffix(".txt")


def out_image_path(source_dataset: Path, out_dataset: Path, image_path: str) -> Path:
    path = Path(image_path)
    try:
        rel = path.relative_to(source_dataset / "images")
    except ValueError:
        split = path.parent.name
        rel = Path(split) / path.name
    return out_dataset / "images" / rel


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def split_key(row: dict[str, Any]) -> str:
    split = row.get("split")
    if split == "calibration":
        return "val"
    if split == "dev_holdout":
        return "test"
    return str(split)


def derive(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.source / "dataset_manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows = []
    stats = Counter()
    by_split: dict[str, list[str]] = defaultdict(list)
    per_frame_removed = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = split_key(row)
            original_instances = row.get("instances") or []
            kept, row_stats = dedup_instances(original_instances, args.pose_dist_ratio, args.contain_ratio)
            stats.update(row_stats)
            if not kept:
                stats["empty_frames_removed"] += 1
                per_frame_removed.append({
                    "source_image": row.get("image_path"),
                    "split": split,
                    "domain": row.get("domain"),
                    "section_id": row.get("section_id"),
                    "frame_id": row.get("frame_id"),
                    "before": len(original_instances),
                    "after": 0,
                    **row_stats,
                })
                continue

            src_image = Path(row["image_path"])
            dst_image = out_image_path(args.source, args.out, row["image_path"])
            link_or_copy(src_image, dst_image, args.copy_images)
            labels = [yolo_row(inst, int(row["width"]), int(row["height"])) for inst in kept]
            labels = [label for label in labels if label is not None]
            label_path = relabel_path(args.source, args.out, row["image_path"])
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

            new_row = dict(row)
            new_row["image_path"] = str(dst_image)
            new_row["split"] = split
            new_row["instances"] = kept
            audit = dict(new_row.get("pairing_audit") or {})
            audit.update({
                "posegeo_invalid_pose_removed": int(row_stats["invalid_pose_removed"]),
                "posegeo_duplicate_pose_removed": int(row_stats["duplicate_pose_removed"]),
                "posegeo_instances_before": len(original_instances),
                "posegeo_instances_after": len(kept),
                "pose_instances": len(kept),
                "det_instances": len(kept),
            })
            new_row["pairing_audit"] = audit
            rows.append(new_row)
            by_split[split].append(str(dst_image))
            if row_stats:
                per_frame_removed.append({
                    "source_image": row.get("image_path"),
                    "new_image": str(dst_image),
                    "split": split,
                    "domain": row.get("domain"),
                    "section_id": row.get("section_id"),
                    "frame_id": row.get("frame_id"),
                    "before": len(original_instances),
                    "after": len(kept),
                    **row_stats,
                })

    for split in ("train", "val", "test"):
        (args.out / f"{split}_images.txt").write_text("\n".join(by_split[split]) + ("\n" if by_split[split] else ""), encoding="utf-8")
    with (args.out / "dataset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    yaml_text = f"""path: {args.out.resolve()}
train: train_images.txt
val: val_images.txt
test: test_images.txt

names:
  0: bee

kpt_shape: [2, 3]
flip_idx: [0, 1]
"""
    (args.out / "bee_yolo_pose_strict.yaml").write_text(yaml_text, encoding="utf-8")
    (args.out / "posegeo_filter_audit.json").write_text(
        json.dumps(per_frame_removed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = summarize(args, rows, stats)
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(summary, args.out / "dataset_summary.md")
    return summary


def summarize(args: argparse.Namespace, rows: list[dict[str, Any]], stats: Counter) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_dataset": str(args.source),
        "out_dir": str(args.out),
        "supervision_mode": "pose_only_posegeo_dedup",
        "filter_rule": {
            "remove_invalid_pose_box": "head and tail must be finite and both must lie inside the corresponding bbox",
            "pose_duplicate_distance": f"head distance and tail distance both < {args.pose_dist_ratio} * smaller bbox diagonal",
            "contain_ratio": args.contain_ratio,
            "keypoint_confidence_hard_threshold": None,
            "duplicate_policy": "if containment >= threshold, keep larger bbox; otherwise keep higher min keypoint confidence, then higher detection confidence, then larger bbox",
        },
        "removed": dict(stats),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        frame_counts = [len(row.get("instances") or []) for row in split_rows]
        summary["splits"][split] = {
            "frames": len(split_rows),
            "domains": {
                "RGB": sum(1 for row in split_rows if row.get("domain") == "RGB"),
                "IR": sum(1 for row in split_rows if row.get("domain") == "IR"),
            },
            "det_instances": sum(frame_counts),
            "pose_instances": sum(frame_counts),
            "track_instances": sum(sum(int(inst.get("track_mask", 1)) for inst in row.get("instances", [])) for row in split_rows),
            "min_instances_per_frame": min(frame_counts) if frame_counts else 0,
            "max_instances_per_frame": max(frame_counts) if frame_counts else 0,
            "empty_pose_label_frames": sum(1 for count in frame_counts if count == 0),
            "videos": sorted({row.get("video_id") for row in split_rows}),
            "sections": sorted({row.get("section_id") for row in split_rows}),
        }
    return summary


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# strict-v8 pose-only posegeo_dedup 数据摘要",
        "",
        f"source_dataset: `{summary['source_dataset']}`",
        f"out_dir: `{summary['out_dir']}`",
        "",
        "过滤规则：删除 head/tail 坐标无效或越出自身 bbox 的实例；head 和 tail 都小于较小框对角线 15% 时进入重复判断；包含率不低于 0.75 时保留大框，否则按最低关键点置信度、检测置信度和面积决策；不使用关键点置信度硬阈值。",
        "",
        f"removed: `{json.dumps(summary['removed'], ensure_ascii=False)}`",
        "",
        "| split | frames | RGB | IR | instances | min/frame | max/frame |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in summary["splits"].items():
        lines.append(
            f"| {split} | {row['frames']} | {row['domains'].get('RGB', 0)} | {row['domains'].get('IR', 0)} | "
            f"{row['pose_instances']} | {row['min_instances_per_frame']} | {row['max_instances_per_frame']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = derive(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
