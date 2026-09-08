#!/usr/bin/env python3
"""Evaluate BeePoseTrack JSONL pose predictions against COCO head/tail labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


FRAME_RE = re.compile(r"([AB]-5-\d+)_frame_(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--coco", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    return parser.parse_args()


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return np.asarray([x, y, x + w, y + h], dtype=float)


def iou(a, b) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def angle_error(pred_head, pred_tail, gt_head, gt_tail) -> float:
    pred = np.asarray(pred_tail) - np.asarray(pred_head)
    gt = np.asarray(gt_tail) - np.asarray(gt_head)
    denom = np.linalg.norm(pred) * np.linalg.norm(gt)
    if denom <= 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(pred, gt) / denom, -1.0, 1.0))))


def image_key(file_name: str) -> tuple[str, int] | None:
    match = FRAME_RE.search(Path(file_name).stem)
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2))


def load_gt(paths: list[Path]):
    gt = defaultdict(list)
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        images = {img["id"]: img for img in data["images"]}
        for ann in data["annotations"]:
            img = images[ann["image_id"]]
            key = image_key(img["file_name"])
            if key is None:
                continue
            k = ann["keypoints"]
            gt[key].append(
                {
                    "bbox": xywh_to_xyxy(ann["bbox"]),
                    "head": np.asarray(k[0:2], dtype=float),
                    "tail": np.asarray(k[3:5], dtype=float),
                    "diag": max(math.hypot(float(ann["bbox"][2]), float(ann["bbox"][3])), 1.0),
                    "domain": "RGB" if key[0].startswith("A") else "IR",
                }
            )
    return gt


def summarize(records):
    if not records:
        return {"matched": 0}
    nme = np.asarray([r["nme"] for r in records], dtype=float)
    angles = np.asarray([r["angle_error_deg"] for r in records], dtype=float)
    swapped = np.asarray([r["head_tail_swapped"] for r in records], dtype=bool)
    summary = {
        "matched": len(records),
        "nme_bbox_diagonal": round(float(nme.mean()), 6),
        "pck@0.05": round(float(np.mean(nme <= 0.05)), 6),
        "pck@0.10": round(float(np.mean(nme <= 0.10)), 6),
        "pck@0.20": round(float(np.mean(nme <= 0.20)), 6),
        "orientation_accuracy": round(float(np.mean(~swapped)), 6),
        "head_tail_swap_rate": round(float(np.mean(swapped)), 6),
        "mean_angle_error_deg": round(float(angles.mean()), 4),
        "median_angle_error_deg": round(float(np.median(angles)), 4),
        "angle_over_45deg": round(float(np.mean(angles > 45)), 6),
        "angle_over_90deg": round(float(np.mean(angles > 90)), 6),
    }
    return summary


def main() -> None:
    args = parse_args()
    gt = load_gt(args.coco)
    records = []
    total_pred = 0
    total_gt = sum(len(v) for v in gt.values())
    for line in args.pred_jsonl.read_text(encoding="utf-8").splitlines():
        pred_frame = json.loads(line)
        key = (str(pred_frame["video_id"]).upper(), int(pred_frame["frame_id"]))
        gt_items = gt.get(key, [])
        pred_items = [d for d in pred_frame.get("detections", []) if d.get("head") and d.get("tail")]
        total_pred += len(pred_items)
        if not gt_items or not pred_items:
            continue
        cost = np.ones((len(gt_items), len(pred_items)), dtype=float)
        for gi, g in enumerate(gt_items):
            for pi, p in enumerate(pred_items):
                cost[gi, pi] = 1.0 - iou(g["bbox"], np.asarray(p["bbox_xyxy"], dtype=float))
        rows, cols = linear_sum_assignment(cost)
        for gi, pi in zip(rows, cols):
            matched_iou = 1.0 - float(cost[gi, pi])
            if matched_iou < args.iou_thresh:
                continue
            g = gt_items[gi]
            p = pred_items[pi]
            ph = np.asarray(p["head"], dtype=float)
            pt = np.asarray(p["tail"], dtype=float)
            direct = np.linalg.norm(ph - g["head"]) + np.linalg.norm(pt - g["tail"])
            swapped = np.linalg.norm(ph - g["tail"]) + np.linalg.norm(pt - g["head"])
            records.append(
                {
                    "video_id": key[0],
                    "frame_id": key[1],
                    "domain": g["domain"],
                    "iou": matched_iou,
                    "nme": float((np.linalg.norm(ph - g["head"]) + np.linalg.norm(pt - g["tail"])) / (2 * g["diag"])),
                    "angle_error_deg": angle_error(ph, pt, g["head"], g["tail"]),
                    "head_tail_swapped": bool(swapped < direct),
                }
            )
    output = {
        "summary": summarize(records),
        "by_domain": {
            domain: summarize([r for r in records if r["domain"] == domain])
            for domain in ["RGB", "IR"]
        },
        "total_gt": total_gt,
        "total_pred_with_pose": total_pred,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
