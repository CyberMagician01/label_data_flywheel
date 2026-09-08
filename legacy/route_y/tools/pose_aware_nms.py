"""Pose-aware Soft-NMS for BeePoseTrack-Y prediction JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def det_angle(det: dict) -> float:
    if det.get("head") is None or det.get("tail") is None:
        return np.nan
    h = np.asarray(det["head"], dtype=float)
    t = np.asarray(det["tail"], dtype=float)
    v = t - h
    if np.linalg.norm(v) <= 1e-9:
        return np.nan
    return math.atan2(float(v[1]), float(v[0]))


def quality(det: dict) -> float:
    return float(det.get("score", 0.0)) * max(float(det.get("pose_score", 0.0)), 0.25)


def pairwise_iou(best_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(best_box[0], boxes[:, 0])
    y1 = np.maximum(best_box[1], boxes[:, 1])
    x2 = np.minimum(best_box[2], boxes[:, 2])
    y2 = np.minimum(best_box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_best = max(0.0, float(best_box[2] - best_box[0])) * max(0.0, float(best_box[3] - best_box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_best + area_boxes - inter, 1e-9)


def angle_gap(best_angle: float, angles: np.ndarray) -> np.ndarray:
    if np.isnan(best_angle):
        return np.zeros_like(angles)
    gaps = np.abs((best_angle - angles + math.pi) % (2 * math.pi) - math.pi) / math.pi
    gaps[np.isnan(gaps)] = 0.0
    return gaps


def soft_nms(dets: list[dict], iou_thr: float, angle_keep: float, max_det: int) -> list[dict]:
    if not dets:
        return []

    work = [dict(d) for d in dets]
    boxes = np.asarray([d["bbox_xyxy"] for d in work], dtype=float)
    scores = np.asarray([float(d.get("score", 0.0)) for d in work], dtype=float)
    pose_scores = np.asarray([max(float(d.get("pose_score", 0.0)), 0.25) for d in work], dtype=float)
    angles = np.asarray([det_angle(d) for d in work], dtype=float)
    active = np.ones(len(work), dtype=bool)
    keep: list[dict] = []

    while active.any() and len(keep) < max_det:
        active_idx = np.flatnonzero(active)
        qualities = scores[active_idx] * pose_scores[active_idx]
        best_idx = int(active_idx[int(np.argmax(qualities))])

        best = dict(work[best_idx])
        best["score"] = round(float(scores[best_idx]), 6)
        keep.append(best)
        active[best_idx] = False

        remain_idx = np.flatnonzero(active)
        if remain_idx.size == 0:
            break

        overlaps = pairwise_iou(boxes[best_idx], boxes[remain_idx])
        gaps = angle_gap(float(angles[best_idx]), angles[remain_idx])
        decay_mask = (overlaps > iou_thr) & (gaps < angle_keep)
        if decay_mask.any():
            target_idx = remain_idx[decay_mask]
            scores[target_idx] *= np.exp(-((overlaps[decay_mask] * overlaps[decay_mask]) / 0.5))
            for idx in target_idx:
                work[int(idx)]["score"] = round(float(scores[int(idx)]), 6)

        active[remain_idx[scores[remain_idx] <= 1e-4]] = False

    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--angle-keep", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=768)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            row["detections"] = soft_nms(row.get("detections", []), args.iou, args.angle_keep, args.max_det)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
