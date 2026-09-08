#!/usr/bin/env python3
"""Evaluate Track-by-Detection JSONL against the E-aligned fold manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--track-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--eval-split", default="val")
    return parser.parse_args()


def box_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
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


def load_gt(path: Path, eval_split: str) -> dict[tuple[str, int], dict[str, Any]]:
    frames = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != eval_split:
            continue
        gt = []
        for inst in row.get("instances", []):
            if not inst.get("track_mask", 1):
                continue
            if inst.get("track_id") is None:
                continue
            gt.append({"track_id": str(inst["track_id"]), "bbox": np.asarray(inst["bbox_xyxy"], dtype=float)})
        section_id = str(row.get("section_id", row["video_id"]))
        frames[(section_id, int(row["frame_id"]))] = {
            "video_id": str(row["video_id"]),
            "section_id": section_id,
            "frame_id": int(row["frame_id"]),
            "domain": row.get("domain", "unknown"),
            "gt": gt,
        }
    return frames


def load_tracks(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequence_id = str(row.get("section_id", row["video_id"]))
        out[(sequence_id, int(row["frame_id"]))] = row.get("tracks", [])
    return out


def evaluate_frames(frames: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total_gt = 0
    fp = 0
    fn = 0
    idsw = 0
    matches_total = 0
    idtp = 0
    idfp = 0
    idfn = 0
    last_pred_for_gt: dict[tuple[str, str], int] = {}
    seen_gt: set[tuple[str, str]] = set()
    prev_present: set[tuple[str, str]] = set()
    frag = 0
    ass_scores = []

    for frame in sorted(frames, key=lambda x: (x.get("section_id", x["video_id"]), x["frame_id"])):
        sequence_id = frame.get("section_id", frame["video_id"])
        gt = frame["gt"]
        pred = frame["pred"]
        seen_before_frame = set(seen_gt)
        total_gt += len(gt)
        gt_boxes = np.asarray([g["bbox"] for g in gt], dtype=float).reshape(len(gt), 4)
        pred_boxes = np.asarray([p["bbox_xyxy"] for p in pred], dtype=float).reshape(len(pred), 4)
        mat = box_iou_matrix(gt_boxes, pred_boxes)
        cost = 1.0 - mat
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        if len(gt) and len(pred):
            rows, cols = linear_sum_assignment(cost)
            for gi, pi in zip(rows, cols):
                if float(mat[gi, pi]) < threshold:
                    continue
                matched_gt.add(int(gi))
                matched_pred.add(int(pi))
                matches_total += 1
                gt_key = (sequence_id, str(gt[gi]["track_id"]))
                pred_id = int(pred[pi]["track_id"])
                if gt_key in last_pred_for_gt and last_pred_for_gt[gt_key] != pred_id:
                    idsw += 1
                last_pred_for_gt[gt_key] = pred_id
                idtp += 1
                ass_scores.append(float(mat[gi, pi]))
                seen_gt.add(gt_key)
        current_present = {(sequence_id, str(g["track_id"])) for i, g in enumerate(gt) if i in matched_gt}
        for gt_key in current_present:
            if gt_key in seen_before_frame and gt_key not in prev_present:
                # Count re-appearance after a missed frame as a fragmentation event.
                frag += 1
        prev_present = current_present
        frame_fp = len(pred) - len(matched_pred)
        frame_fn = len(gt) - len(matched_gt)
        fp += frame_fp
        fn += frame_fn
        idfp += frame_fp
        idfn += frame_fn

    mota = 1.0 - (fn + fp + idsw) / total_gt if total_gt else 0.0
    idf1 = (2 * idtp) / max(2 * idtp + idfp + idfn, 1)
    deta = matches_total / max(total_gt + fp, 1)
    assa = float(np.mean(ass_scores)) if ass_scores else 0.0
    hota_proxy = math.sqrt(max(deta, 0.0) * max(assa, 0.0))
    return {
        "frames": len(frames),
        "gt_instances": total_gt,
        "matches": matches_total,
        "FP": fp,
        "FN": fn,
        "MOTA": round(float(mota), 6),
        "IDF1": round(float(idf1), 6),
        "HOTA_proxy": round(float(hota_proxy), 6),
        "IDSW": idsw,
        "FRAG_proxy": frag,
    }


def main() -> None:
    args = parse_args()
    gt = load_gt(args.manifest, args.eval_split)
    tracks = load_tracks(args.track_jsonl)
    items = []
    for key, frame in gt.items():
        item = dict(frame)
        item["pred"] = tracks.get(key, [])
        items.append(item)
    output = {
        "manifest": str(args.manifest),
        "track_jsonl": str(args.track_jsonl),
        "iou": args.iou,
        "note": "Evaluation keys are (section_id, frame_id). HOTA_proxy and FRAG_proxy are internal approximations; use TrackEval for final official-style HOTA/FRAG.",
        "all": evaluate_frames(items, args.iou),
        "by_domain": {
            domain: evaluate_frames([x for x in items if x["domain"] == domain], args.iou)
            for domain in ("RGB", "IR")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
