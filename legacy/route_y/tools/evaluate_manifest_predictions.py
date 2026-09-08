#!/usr/bin/env python3
"""Evaluate Y-route predictions against the E-aligned fold manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-iou", type=float, default=0.3)
    parser.add_argument("--pr-iou", type=float, default=0.5)
    parser.add_argument("--eval-split", default="val")
    return parser.parse_args()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_gt = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    area_pred = np.maximum(0.0, pred_boxes[:, 2] - pred_boxes[:, 0]) * np.maximum(0.0, pred_boxes[:, 3] - pred_boxes[:, 1])
    return inter / np.maximum(area_gt[:, None] + area_pred[None, :] - inter, 1e-9)


def canonical(path: str) -> str:
    return str(Path(path).resolve())


def load_manifest(path: Path, eval_split: str) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != eval_split:
            continue
        gt = []
        for inst in row.get("instances", []):
            if not inst.get("det_mask", 1):
                continue
            kpts = inst.get("keypoints") or []
            gt.append(
                {
                    "bbox": np.asarray(inst["bbox_xyxy"], dtype=float),
                    "head": np.asarray(kpts[0], dtype=float) if len(kpts) > 0 else None,
                    "tail": np.asarray(kpts[1], dtype=float) if len(kpts) > 1 else None,
                    "visibility": inst.get("visibility", [0, 0]),
                    "track_id": inst.get("track_id"),
                }
            )
        frames[canonical(row["image_path"])] = {
            "image_path": canonical(row["image_path"]),
            "video_id": row.get("video_id"),
            "section_id": row.get("section_id", row.get("video_id")),
            "frame_id": row.get("frame_id"),
            "domain": row.get("domain", "unknown"),
            "gt": gt,
        }
    return frames


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[canonical(row["image_path"])] = row.get("detections", [])
    return out


def ap_for_threshold(items: list[dict[str, Any]], threshold: float) -> float:
    total_gt = sum(len(x["gt"]) for x in items)
    if total_gt == 0:
        return 0.0
    preds = []
    for frame_index, item in enumerate(items):
        for pred in item["pred"]:
            preds.append((float(pred.get("score", 0.0)), frame_index, pred))
    preds.sort(key=lambda x: x[0], reverse=True)
    used: dict[int, set[int]] = defaultdict(set)
    tp = np.zeros(len(preds), dtype=float)
    fp = np.zeros(len(preds), dtype=float)
    for i, (_, frame_index, pred) in enumerate(preds):
        matrix = items[frame_index]["iou_matrix"]
        best_iou = 0.0
        best_gt = -1
        for gi in range(matrix.shape[0]):
            if gi in used[frame_index]:
                continue
            score = float(matrix[gi, pred["_pred_index"]])
            if score > best_iou:
                best_iou = score
                best_gt = gi
        if best_iou >= threshold and best_gt >= 0:
            tp[i] = 1.0
            used[frame_index].add(best_gt)
        else:
            fp[i] = 1.0
    if len(preds) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        mask = recall >= r
        ap += float(np.max(precision[mask])) if np.any(mask) else 0.0
    return ap / 101.0


def pr_at_threshold(items: list[dict[str, Any]], threshold: float) -> tuple[float, float]:
    total_gt = sum(len(x["gt"]) for x in items)
    total_pred = sum(len(x["pred"]) for x in items)
    used: dict[int, set[int]] = defaultdict(set)
    tp = 0
    for frame_index, item in enumerate(items):
        preds = sorted(item["pred"], key=lambda p: float(p.get("score", 0.0)), reverse=True)
        matrix = item["iou_matrix"]
        for pred in preds:
            best_iou = 0.0
            best_gt = -1
            for gi in range(matrix.shape[0]):
                if gi in used[frame_index]:
                    continue
                score = float(matrix[gi, pred["_pred_index"]])
                if score > best_iou:
                    best_iou = score
                    best_gt = gi
            if best_iou >= threshold and best_gt >= 0:
                tp += 1
                used[frame_index].add(best_gt)
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return precision, recall


def angle_error(pred_head: np.ndarray, pred_tail: np.ndarray, gt_head: np.ndarray, gt_tail: np.ndarray) -> float:
    pred_axis = pred_head - pred_tail
    gt_axis = gt_head - gt_tail
    denom = np.linalg.norm(pred_axis) * np.linalg.norm(gt_axis)
    if denom <= 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(pred_axis, gt_axis) / denom, -1.0, 1.0))))


def pose_matches(items: list[dict[str, Any]], pose_iou: float) -> list[dict[str, float]]:
    records = []
    for item in items:
        used: set[int] = set()
        preds = [p for p in item["pred"] if p.get("head") is not None and p.get("tail") is not None]
        preds.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
        matrix = item["iou_matrix"]
        for pred in preds:
            best_iou = 0.0
            best_gt = -1
            for gi, gt in enumerate(item["gt"]):
                if gi in used or gt["head"] is None or gt["tail"] is None:
                    continue
                score = float(matrix[gi, pred["_pred_index"]])
                if score > best_iou:
                    best_iou = score
                    best_gt = gi
            if best_iou < pose_iou or best_gt < 0:
                continue
            used.add(best_gt)
            gt = item["gt"][best_gt]
            ph = np.asarray(pred["head"], dtype=float)
            pt = np.asarray(pred["tail"], dtype=float)
            diag = max(float(np.linalg.norm(gt["bbox"][2:] - gt["bbox"][:2])), 1.0)
            direct = np.linalg.norm(ph - gt["head"]) + np.linalg.norm(pt - gt["tail"])
            swapped = np.linalg.norm(ph - gt["tail"]) + np.linalg.norm(pt - gt["head"])
            angle = angle_error(ph, pt, gt["head"], gt["tail"])
            records.append(
                {
                    "nme": float(direct / (2.0 * diag)),
                    "swapped": float(swapped < direct),
                    "angle_error_deg": float(angle),
                }
            )
    return records


def summarize(items: list[dict[str, Any]], pose_iou: float, pr_iou: float) -> dict[str, Any]:
    thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
    aps = {f"AP{int(t * 100)}": ap_for_threshold(items, t) for t in thresholds}
    precision, recall = pr_at_threshold(items, pr_iou)
    poses = pose_matches(items, pose_iou)
    nme = np.asarray([x["nme"] for x in poses], dtype=float)
    angles = np.asarray([x["angle_error_deg"] for x in poses], dtype=float)
    swapped = np.asarray([x["swapped"] for x in poses], dtype=float)
    out: dict[str, Any] = {
        "frames": len(items),
        "gt_instances": sum(len(x["gt"]) for x in items),
        "pred_instances": sum(len(x["pred"]) for x in items),
        "mAP50-95": round(float(np.mean(list(aps.values()))), 6),
        "AP50": round(float(aps["AP50"]), 6),
        "AP75": round(float(aps["AP75"]), 6),
        "precision_iou50": round(float(precision), 6),
        "recall_iou50": round(float(recall), 6),
        "pose_matched": len(poses),
    }
    if len(poses):
        out.update(
            {
                "NME_bbox_diag": round(float(nme.mean()), 6),
                "PCK@0.05": round(float(np.mean(nme <= 0.05)), 6),
                "PCK@0.10": round(float(np.mean(nme <= 0.10)), 6),
                "PCK@0.20": round(float(np.mean(nme <= 0.20)), 6),
                "direction_acc_angle_le_90": round(float(np.mean(angles <= 90.0)), 6),
                "mean_angle_error_deg": round(float(angles.mean()), 4),
                "median_angle_error_deg": round(float(np.median(angles)), 4),
                "angle_over_45deg": round(float(np.mean(angles > 45.0)), 6),
                "angle_over_90deg": round(float(np.mean(angles > 90.0)), 6),
                "head_tail_swap_rate": round(float(swapped.mean()), 6),
            }
        )
    return out


def prepare_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        for index, pred in enumerate(item["pred"]):
            pred["_pred_index"] = index
        gt_boxes = np.asarray([gt["bbox"] for gt in item["gt"]], dtype=float)
        pred_boxes = np.asarray([pred["bbox_xyxy"] for pred in item["pred"]], dtype=float)
        if pred_boxes.size == 0:
            pred_boxes = pred_boxes.reshape(0, 4)
        if gt_boxes.size == 0:
            gt_boxes = gt_boxes.reshape(0, 4)
        item["iou_matrix"] = iou_matrix(gt_boxes, pred_boxes)


def main() -> None:
    args = parse_args()
    frames = load_manifest(args.manifest, args.eval_split)
    preds = load_predictions(args.pred_jsonl)
    items = []
    missing_predictions = 0
    for key, frame in frames.items():
        pred = preds.get(key)
        if pred is None:
            missing_predictions += 1
            pred = []
        item = dict(frame)
        item["pred"] = pred
        items.append(item)
    prepare_items(items)
    output = {
        "manifest": str(args.manifest),
        "pred_jsonl": str(args.pred_jsonl),
        "pose_iou": args.pose_iou,
        "pr_iou": args.pr_iou,
        "missing_prediction_frames": missing_predictions,
        "all": summarize(items, args.pose_iou, args.pr_iou),
        "by_domain": {
            domain: summarize([x for x in items if x["domain"] == domain], args.pose_iou, args.pr_iou)
            for domain in ("RGB", "IR")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
