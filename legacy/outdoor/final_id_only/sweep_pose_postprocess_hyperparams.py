#!/usr/bin/env python3
"""Sweep pose-box postprocessing parameters on annotators 01-04 independently."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from evaluate_detection_against_manual_gt import VIDEOS, evaluate, load_gt


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(min(area(a), area(b)), 1e-12)


def valid(det, score_threshold):
    box = det["bbox_xyxy"]
    for name in ("head", "abdomen_tip"):
        point = det["keypoints"][name]
        if not all(math.isfinite(float(x)) for x in point):
            return False
        if not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
            return False
        if score_threshold is not None and point[2] < score_threshold:
            return False
    return True


def same_pose(a, b, distance_ratio):
    ba, bb = a["bbox_xyxy"], b["bbox_xyxy"]
    scale = min(math.hypot(ba[2] - ba[0], ba[3] - ba[1]), math.hypot(bb[2] - bb[0], bb[3] - bb[1]))
    limit = distance_ratio * max(scale, 1e-12)
    return all(
        math.hypot(
            a["keypoints"][name][0] - b["keypoints"][name][0],
            a["keypoints"][name][1] - b["keypoints"][name][1],
        ) <= limit
        for name in ("head", "abdomen_tip")
    )


def deduplicate(detections, distance_ratio, containment_threshold):
    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    contained = set()
    for i in range(n):
        for j in range(i + 1, n):
            if same_pose(detections[i], detections[j], distance_ratio):
                union(i, j)
                if containment(detections[i]["bbox_xyxy"], detections[j]["bbox_xyxy"]) >= containment_threshold:
                    contained.add((i, j))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    selected = []
    for members in groups.values():
        if any(i in members and j in members for i, j in contained):
            winner = max(members, key=lambda i: area(detections[i]["bbox_xyxy"]))
        else:
            winner = max(members, key=lambda i: (
                min(detections[i]["keypoints"]["head"][2], detections[i]["keypoints"]["abdomen_tip"][2]),
                detections[i].get("det_confidence", 1.0),
            ))
        selected.append(winner)
    return [detections[i] for i in sorted(selected)]


def load_raw(root, keys):
    raw = {}
    for video, frame in keys:
        path = root / video / "frames" / f"frame_{frame:08d}.json"
        raw[(video, frame)] = json.loads(path.read_text(encoding="utf-8"))["detections"]
    return raw


def postprocess(raw, distance_ratio, containment_threshold, score_threshold):
    output = {}
    for key, detections in raw.items():
        kept = [x for x in detections if valid(x, score_threshold)]
        kept = deduplicate(kept, distance_ratio, containment_threshold)
        output[key] = sorted(
            [(float(x.get("det_confidence", 1.0)), [float(v) for v in x["bbox_xyxy"]]) for x in kept],
            reverse=True,
        )
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--frame-stats", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    gt, _ = load_gt(args.gt)
    groups = {x: [] for x in ("01", "02", "03", "04")}
    with args.frame_stats.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ann = row["annotator"].zfill(2)
            key = (row["video"], int(row["frame_no"]))
            if ann in groups and row["modality"] == "RGB" and row["video"] in VIDEOS:
                groups[ann].append(key)
    keys = sorted(set(itertools.chain.from_iterable(groups.values())))
    raw = load_raw(args.pred, keys)
    distance_values = (0.08, 0.10, 0.12, 0.15, 0.18, 0.20)
    containment_values = (0.60, 0.70, 0.75, 0.80, 0.90)
    keypoint_score_values = (None, 0.05, 0.10, 0.15, 0.20)
    rows = []
    for distance_ratio, containment_threshold, score_threshold in itertools.product(
        distance_values, containment_values, keypoint_score_values
    ):
        pred = postprocess(raw, distance_ratio, containment_threshold, score_threshold)
        metrics = {ann: evaluate(gt, pred, sorted(group_keys)) for ann, group_keys in groups.items()}
        row = {
            "keypoint_distance_ratio": distance_ratio,
            "containment_threshold": containment_threshold,
            "keypoint_score_threshold": score_threshold,
            "per_annotator": metrics,
            "macro": {
                name: float(np.mean([metrics[a][name] for a in groups]))
                for name in ("map_50_95", "map_50", "map_75", "average_recall_50_95", "recall_50")
            },
        }
        row["macro"]["best_f1_at_iou50"] = float(np.mean([metrics[a]["best_f1_at_iou50"]["f1"] for a in groups]))
        row["boxes"] = {a: metrics[a]["pred_boxes"] for a in groups}
        rows.append(row)
    baseline = next(r for r in rows if r["keypoint_distance_ratio"] == 0.15 and r["containment_threshold"] == 0.75 and r["keypoint_score_threshold"] is None)
    eligible = [r for r in rows if all(
        r["per_annotator"][a]["recall_50"] >= baseline["per_annotator"][a]["recall_50"] - 0.01
        for a in groups
    )]
    ranking = sorted(eligible, key=lambda r: (r["macro"]["map_50_95"], r["macro"]["best_f1_at_iou50"]), reverse=True)
    best_per_annotator = {
        a: max(eligible, key=lambda r: (r["per_annotator"][a]["map_50_95"], r["per_annotator"][a]["best_f1_at_iou50"]))
        for a in groups
    }
    result = {
        "datasets": {a: {"images": len(groups[a]), "gt_boxes": sum(len(gt[k]) for k in groups[a])} for a in groups},
        "search_space": {
            "keypoint_distance_ratio": distance_values,
            "containment_threshold": containment_values,
            "keypoint_score_threshold": keypoint_score_values,
            "combinations": len(rows),
        },
        "selection_constraint": "each annotator recall50 must be no more than 1 percentage point below current baseline",
        "baseline": baseline,
        "best_macro": ranking[0],
        "best_per_annotator": best_per_annotator,
        "top10_macro": ranking[:10],
        "all_results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({k: result[k] for k in ("datasets", "search_space", "selection_constraint", "baseline", "best_macro", "best_per_annotator")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
