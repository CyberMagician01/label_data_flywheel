#!/usr/bin/env python3
"""Evaluate flywheel boxes against the manually annotated A-5 LabelMe frames."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


VIDEOS = ("A-5-1", "A-5-2", "A-5-3", "A-5-4")


def xyxy_from_shape(shape):
    if shape.get("shape_type") != "rectangle" or shape.get("label") != "bee":
        return None
    points = shape.get("points", [])
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    box = [min(xs), min(ys), max(xs), max(ys)]
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def load_gt(root):
    gt = {}
    source_files = {}
    for video in VIDEOS:
        files = sorted(root.glob(f"{video}_区段_*/*_frame_*.json"))
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            image_name = Path(data.get("imagePath") or path.with_suffix(".jpg").name).name
            frame_digits = image_name.rsplit("_", 1)[-1].split(".")[0]
            key = (video, int(frame_digits))
            if key in gt:
                raise RuntimeError(f"duplicate GT frame: {key}")
            boxes = []
            for shape in data.get("shapes", []):
                box = xyxy_from_shape(shape)
                if box is not None:
                    boxes.append(box)
            gt[key] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            source_files[key] = str(path)
    return gt, source_files


def load_predictions(root, gt_keys, max_dets):
    pred = {}
    missing = []
    for video, frame_id in sorted(gt_keys):
        path = root / video / "frames" / f"frame_{frame_id:08d}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for det in data.get("detections", []):
            box = [float(v) for v in det["bbox_xyxy"]]
            score = float(det.get("det_confidence", 1.0))
            if all(math.isfinite(v) for v in box) and math.isfinite(score) and box[2] > box[0] and box[3] > box[1]:
                rows.append((score, box))
        rows.sort(key=lambda x: x[0], reverse=True)
        rows = rows[:max_dets]
        pred[(video, frame_id)] = rows
    if missing:
        raise RuntimeError(f"missing {len(missing)} prediction files; first={missing[0]}")
    return pred


def iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / np.maximum(a1[:, None] + a2[None, :] - inter, 1e-12)


def match_at_iou(gt, pred, keys, threshold, score_floor=-math.inf, area_range=None):
    total_gt = 0
    records = []
    for key in keys:
        g = gt[key]
        if area_range is not None:
            area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
            g = g[(area >= area_range[0]) & (area < area_range[1])]
        total_gt += len(g)
        rows = [(s, b) for s, b in pred[key] if s >= score_floor]
        pboxes = np.asarray([b for _, b in rows], dtype=np.float64).reshape(-1, 4)
        ious = iou_matrix(pboxes, g)
        used = np.zeros(len(g), dtype=bool)
        for pi, (score, _) in enumerate(rows):
            best = -1
            best_iou = threshold
            for gi in range(len(g)):
                if not used[gi] and ious[pi, gi] >= best_iou:
                    best_iou = ious[pi, gi]
                    best = gi
            if best >= 0:
                used[best] = True
                records.append((score, 1, 0))
            else:
                records.append((score, 0, 1))
    records.sort(key=lambda x: x[0], reverse=True)
    return total_gt, records


def ap101(total_gt, records):
    if total_gt == 0:
        return None, 0.0, 0.0, None
    if not records:
        return 0.0, 0.0, 0.0, None
    tp = np.cumsum([r[1] for r in records], dtype=np.float64)
    fp = np.cumsum([r[2] for r in records], dtype=np.float64)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    sampled = []
    for r in np.linspace(0, 1, 101):
        candidates = np.flatnonzero(recall >= r)
        sampled.append(float(envelope[candidates[0]]) if len(candidates) else 0.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    best_point = {
        "score": float(records[best][0]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }
    return float(np.mean(sampled)), float(recall[-1]), float(precision[-1]), best_point


def point_metrics(total_gt, records):
    tp = sum(r[1] for r in records)
    fp = sum(r[2] for r in records)
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(gt, pred, keys):
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    aps, recalls = [], []
    ap50 = ap75 = None
    best_f1 = None
    total_predictions = sum(len(pred[k]) for k in keys)
    total_gt = sum(len(gt[k]) for k in keys)
    for threshold in iou_thresholds:
        n_gt, records = match_at_iou(gt, pred, keys, float(threshold))
        ap, recall, _, best = ap101(n_gt, records)
        aps.append(ap)
        recalls.append(recall)
        if abs(threshold - 0.50) < 1e-6:
            ap50, best_f1 = ap, best
        if abs(threshold - 0.75) < 1e-6:
            ap75 = ap
    fixed = {}
    for score in (0.25, 0.50, 0.75):
        n_gt, records = match_at_iou(gt, pred, keys, 0.50, score_floor=score)
        fixed[f"score_{score:.2f}"] = point_metrics(n_gt, records)
    return {
        "images": len(keys),
        "gt_boxes": total_gt,
        "pred_boxes": total_predictions,
        "max_dets_per_image": max(len(pred[k]) for k in keys),
        "map_50_95": float(np.mean(aps)),
        "map_50": ap50,
        "map_75": ap75,
        "average_recall_50_95": float(np.mean(recalls)),
        "recall_50": float(recalls[0]),
        "best_f1_at_iou50": best_f1,
        "fixed_score_at_iou50": fixed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stats", type=Path)
    parser.add_argument("--max-dets", type=int, default=1000)
    args = parser.parse_args()
    if len(args.pred) != len(args.name):
        raise SystemExit("--pred and --name counts differ")
    gt, gt_files = load_gt(args.gt)
    expected = {(v, f) for v in VIDEOS for f in [k[1] for k in gt if k[0] == v]}
    if gt.keys() != expected or len(gt) != 420:
        raise RuntimeError(f"expected exactly 420 unique A-5 manual frames, got {len(gt)}")
    per_video_gt = defaultdict(int)
    for (video, _), boxes in gt.items():
        per_video_gt[video] += len(boxes)
    result = {
        "protocol": {
            "gt": str(args.gt),
            "manual_frames": len(gt),
            "videos": list(VIDEOS),
            "class": "bee_class_agnostic",
            "iou_thresholds": [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)],
            "ap_interpolation": "101-point precision envelope",
            "max_dets_per_image": args.max_dets,
            "gt_boxes_per_video": dict(per_video_gt),
        },
        "results": {},
    }
    annotator_keys = {}
    if args.frame_stats:
        with args.frame_stats.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            for annotator in ("01", "02", "03", "04"):
                annotator_keys[annotator] = []
            for row in rows:
                annotator = row["annotator"].zfill(2)
                video = row["video"]
                if annotator in annotator_keys and row["modality"] == "RGB" and video in VIDEOS:
                    key = (video, int(row["frame_no"]))
                    if key not in gt:
                        raise RuntimeError(f"frame-stat key missing from GT: {key}")
                    annotator_keys[annotator].append(key)
        for annotator, keys in annotator_keys.items():
            if len(keys) != len(set(keys)):
                raise RuntimeError(f"duplicate frame-stat rows for annotator {annotator}")
        result["protocol"]["manual_frames_per_annotator"] = {
            key: len(value) for key, value in annotator_keys.items()
        }
    for name, root in zip(args.name, args.pred):
        pred = load_predictions(root, gt.keys(), args.max_dets)
        entry = {"overall": evaluate(gt, pred, sorted(gt))}
        entry["per_video"] = {
            video: evaluate(gt, pred, sorted(k for k in gt if k[0] == video)) for video in VIDEOS
        }
        if annotator_keys:
            entry["per_annotator"] = {
                annotator: evaluate(gt, pred, sorted(keys))
                for annotator, keys in annotator_keys.items()
            }
        result["results"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
