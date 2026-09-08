#!/usr/bin/env python3
"""室内 YOLO + 动态复检 + 姿态审核/扩框/去重，并按标注来源评测。"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from indoor_adaptive_recheck_4090 import (
    PoseVerifier,
    centered_crop_with_padding,
    dynamic_low_density_centers,
    iou,
    read_yolo_labels,
)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(min(area(a), area(b)), 1e-12)


def same_pose(a, b, distance_ratio):
    # 低置信姿态不能作为去重证据；尤其不能据此删除原始 YOLO 框。
    if not a["pose_valid"] or not b["pose_valid"]:
        return False
    ba, bb = a["box"], b["box"]
    scale = min(math.hypot(ba[2] - ba[0], ba[3] - ba[1]),
                math.hypot(bb[2] - bb[0], bb[3] - bb[1]))
    limit = distance_ratio * max(scale, 1e-12)
    ka, kb = a["keypoints"], b["keypoints"]
    return all(math.hypot(ka[i][0] - kb[i][0], ka[i][1] - kb[i][1]) <= limit for i in (0, 1))


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
                if containment(detections[i]["box"], detections[j]["box"]) >= containment_threshold:
                    contained.add((i, j))
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    winners = []
    for members in groups.values():
        has_containment = any(i in members and j in members for i, j in contained)
        if has_containment:
            winner = max(members, key=lambda i: area(detections[i]["box"]))
        else:
            winner = max(members, key=lambda i: (
                min(detections[i]["keypoints"][:, 2]), detections[i]["score"]
            ))
        winners.append(winner)
    return [detections[i] for i in sorted(winners)]


def load_coco(paths):
    images, gt = [], {}
    seen = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        anns = defaultdict(list)
        for ann in data["annotations"]:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            anns[ann["image_id"]].append([x, y, x + w, y + h])
        for image in data["images"]:
            key = (image["source"], image["file_name"])
            if key in seen:
                raise RuntimeError(f"重复图像: {key}")
            seen.add(key)
            record = dict(image)
            record["eval_key"] = key
            images.append(record)
            gt[key] = np.asarray(anns[image["id"]], dtype=np.float64).reshape(-1, 4)
    return images, gt


def frame_paths(package_root, file_name):
    stem = Path(file_name).stem
    video = stem.split("_frame_")[0]
    frame = int(stem.rsplit("_", 1)[-1])
    return (package_root / "frames" / video / f"frame_{frame:08d}.jpg",
            package_root / "labels" / video / f"frame_{frame:08d}.txt")


def make_recheck_candidates(model, image, baseline, args):
    height, width = image.shape[:2]
    centers = dynamic_low_density_centers(
        baseline, width, height, args.cell_w, args.cell_h, args.neighbor_threshold,
        max_candidates=args.max_candidates,
    )
    crops, metadata = [], []
    for cx, cy, density in centers:
        crop, ox, oy = centered_crop_with_padding(image, cx, cy, args.recheck_w, args.recheck_h)
        crops.append(crop)
        metadata.append((cx, cy, density, ox, oy))
    candidates = []
    if crops:
        results = model.predict(
            crops, imgsz=640, conf=args.recheck_confidence, iou=0.45,
            device=args.device, half=True, batch=min(16, len(crops)), verbose=False,
        )
        for result, (cx, cy, _, ox, oy) in zip(results, metadata):
            for box, score in zip(result.boxes.xyxy.detach().cpu().numpy(),
                                  result.boxes.conf.detach().cpu().numpy()):
                x1, y1, x2, y2 = [float(v) for v in box]
                if x1 <= 3 or y1 <= 3 or x2 >= args.recheck_w - 3 or y2 >= args.recheck_h - 3:
                    continue
                full = [x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(score)]
                bx, by = (full[0] + full[2]) / 2, (full[1] + full[3]) / 2
                if not (cx - args.cell_w / 2 <= bx < cx + args.cell_w / 2 and
                        cy - args.cell_h / 2 <= by < cy + args.cell_h / 2):
                    continue
                full[:4] = [max(0.0, full[0]), max(0.0, full[1]),
                            min(float(width), full[2]), min(float(height), full[3])]
                candidates.append(full)
    candidates.sort(key=lambda x: x[4], reverse=True)
    kept = []
    for candidate in candidates:
        if any(iou(candidate[:4], old[:4]) >= 0.45 for old in kept):
            continue
        if any(iou(candidate[:4], old[:4]) >= 0.45 for old in baseline):
            continue
        kept.append(candidate)
    return kept, centers


def pose_filter_expand(image, boxes, origins, pose_model, args):
    height, width = image.shape[:2]
    poses = pose_model(image, boxes, batch_size=args.pose_batch)
    kept, counters = [], Counter()
    for box, origin, keypoints in zip(boxes, origins, poses):
        x1, y1, x2, y2 = box[:4]
        pose_min = float(keypoints[:, 2].min())
        pose_mean = float(keypoints[:, 2].mean())
        diagonal = max(math.hypot(x2 - x1, y2 - y1), 1e-6)
        separation = float(np.linalg.norm(keypoints[0, :2] - keypoints[1, :2]) / diagonal)
        pose_valid = (pose_min >= args.pose_min_confidence and
                      pose_mean >= args.pose_mean_confidence and
                      separation >= args.pose_min_separation_ratio)
        if not pose_valid and origin == "recheck":
            reason = "low_pose" if (pose_min < args.pose_min_confidence or
                                    pose_mean < args.pose_mean_confidence) else "short_pose"
            counters[f"removed_{reason}_{origin}"] += 1
            continue
        if not pose_valid:
            counters["kept_baseline_despite_weak_pose"] += 1

        # 只有可靠姿态才允许扩框；弱姿态的原始框原样保留。
        expanded = (not args.disable_pose_expansion) and pose_valid and not all(
            x1 <= p[0] <= x2 and y1 <= p[1] <= y2 for p in keypoints
        )
        if expanded:
            mx = args.expand_margin * (x2 - x1)
            my = args.expand_margin * (y2 - y1)
            x1 = max(0.0, min(x1, float(keypoints[:, 0].min()) - mx))
            y1 = max(0.0, min(y1, float(keypoints[:, 1].min()) - my))
            x2 = min(float(width), max(x2, float(keypoints[:, 0].max()) + mx))
            y2 = min(float(height), max(y2, float(keypoints[:, 1].max()) + my))
            counters[f"expanded_{origin}"] += 1
        kept.append({"box": [x1, y1, x2, y2], "score": float(box[4]),
                     "origin": origin, "keypoints": keypoints,
                     "pose_valid": pose_valid, "expanded": expanded})
    return kept, counters


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def records_at_iou(gt, pred, keys, threshold):
    records, total_gt = [], 0
    for key in keys:
        g = gt[key]
        total_gt += len(g)
        rows = sorted(pred[key], key=lambda x: x[0], reverse=True)
        boxes = np.asarray([x[1] for x in rows], dtype=np.float64).reshape(-1, 4)
        overlaps, used = iou_matrix(boxes, g), np.zeros(len(g), dtype=bool)
        for pi, (score, _) in enumerate(rows):
            best, best_iou = -1, threshold
            for gi in range(len(g)):
                if not used[gi] and overlaps[pi, gi] >= best_iou:
                    best, best_iou = gi, overlaps[pi, gi]
            if best >= 0:
                used[best] = True
                records.append((score, 1, 0))
            else:
                records.append((score, 0, 1))
    records.sort(key=lambda x: x[0], reverse=True)
    return total_gt, records


def ap101(total_gt, records):
    if not records:
        return 0.0, 0.0, {"score": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = np.cumsum([x[1] for x in records], dtype=np.float64)
    fp = np.cumsum([x[2] for x in records], dtype=np.float64)
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = np.mean([envelope[np.flatnonzero(recall >= r)[0]] if np.any(recall >= r) else 0.0
                  for r in np.linspace(0, 1, 101)])
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(ap), float(recall[-1]), {
        "score": float(records[best][0]), "precision": float(precision[best]),
        "recall": float(recall[best]), "f1": float(f1[best]),
    }


def evaluate(gt, pred, keys):
    aps, recalls, best50 = [], [], None
    for threshold in np.arange(0.50, 0.96, 0.05):
        total, records = records_at_iou(gt, pred, keys, float(threshold))
        ap, recall, best = ap101(total, records)
        aps.append(ap)
        recalls.append(recall)
        if abs(threshold - 0.50) < 1e-6:
            ap50, best50 = ap, best
        if abs(threshold - 0.75) < 1e-6:
            ap75 = ap
    return {"images": len(keys), "gt_boxes": sum(len(gt[k]) for k in keys),
            "pred_boxes": sum(len(pred[k]) for k in keys), "map_50_95": float(np.mean(aps)),
            "map_50": ap50, "map_75": ap75, "average_recall_50_95": float(np.mean(recalls)),
            "recall_50": recalls[0], "best_f1_at_iou50": best50}


def render(path, image, gt_boxes, baseline, final):
    left = image.copy()
    for box in gt_boxes:
        cv2.rectangle(left, tuple(np.rint(box[:2]).astype(int)), tuple(np.rint(box[2:]).astype(int)), (40, 210, 40), 2)
    right = image.copy()
    for det in final:
        color = (255, 255, 0) if det["origin"] == "recheck" else ((0, 215, 255) if det.get("expanded") else (220, 0, 220))
        box = np.rint(det["box"]).astype(int)
        cv2.rectangle(right, tuple(box[:2]), tuple(box[2:]), color, 2)
        for point, c in zip(det["keypoints"], ((0, 0, 255), (255, 180, 0))):
            cv2.circle(right, tuple(np.rint(point[:2]).astype(int)), 3, c, -1)
    cv2.putText(left, f"GT {len(gt_boxes)}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(right, f"YOLO {len(baseline)} -> fused {len(final)}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.hconcat([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 94])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, action="append", required=True)
    p.add_argument("--source", action="append", help="只评测指定标注来源，可重复传入")
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vitpose-root", type=Path, required=True)
    p.add_argument("--pose-config", type=Path, required=True)
    p.add_argument("--pose-weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--recheck-confidence", type=float, default=0.05)
    p.add_argument("--pose-min-confidence", type=float, default=0.15)
    p.add_argument("--pose-mean-confidence", type=float, default=0.25)
    p.add_argument("--pose-min-separation-ratio", type=float, default=0.15)
    p.add_argument("--expand-margin", type=float, default=0.10)
    p.add_argument("--disable-pose-expansion", action="store_true",
                   help="消融：关键点出框时也保持原检测框不变")
    p.add_argument("--pose-distance-ratio", type=float, default=0.15)
    p.add_argument("--containment-threshold", type=float, default=0.75)
    p.add_argument("--cell-w", type=int, default=160)
    p.add_argument("--cell-h", type=int, default=112)
    p.add_argument("--neighbor-threshold", type=int, default=2)
    p.add_argument("--recheck-w", type=int, default=320)
    p.add_argument("--recheck-h", type=int, default=224)
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--pose-batch", type=int, default=64)
    p.add_argument("--visuals-per-annotator", type=int, default=3)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "visuals").mkdir(exist_ok=True)
    images, gt = load_coco(args.annotations)
    if args.source:
        selected_sources = set(args.source)
        images = [image for image in images if image["source"] in selected_sources]
        selected_keys = {image["eval_key"] for image in images}
        gt = {key: boxes for key, boxes in gt.items() if key in selected_keys}
        missing_sources = selected_sources - {image["source"] for image in images}
        if missing_sources:
            raise RuntimeError(f"指定来源没有图像: {sorted(missing_sources)}")
    yolo = YOLO(str(args.weights))
    pose = PoseVerifier(args.vitpose_root, args.pose_config, args.pose_weights, args.device)
    baseline_pred, final_pred = {}, {}
    counters = defaultdict(Counter)
    details = []
    prediction_records = []
    visual_keys = set()
    rng = random.Random(2026)
    by_source = defaultdict(list)
    for image in images:
        by_source[image["source"]].append(image["eval_key"])
    for source, keys in by_source.items():
        visual_keys.update(rng.sample(keys, min(args.visuals_per_annotator, len(keys))))

    for index, info in enumerate(images, 1):
        key, source = info["eval_key"], info["source"]
        image_path, label_path = frame_paths(args.package_root, info["file_name"])
        image = cv2.imread(str(image_path))
        if image is None or not label_path.is_file():
            raise FileNotFoundError(f"缺少图像或检测标签: {image_path} / {label_path}")
        h, w = image.shape[:2]
        baseline = read_yolo_labels(label_path, w, h, args.confidence)
        additions, centers = make_recheck_candidates(yolo, image, baseline, args)
        boxes = baseline + additions
        origins = ["baseline"] * len(baseline) + ["recheck"] * len(additions)
        filtered, frame_counts = pose_filter_expand(image, boxes, origins, pose, args)
        before_dedup = len(filtered)
        final = deduplicate(filtered, args.pose_distance_ratio, args.containment_threshold)
        frame_counts["removed_duplicate"] += before_dedup - len(final)
        frame_counts["baseline"] += len(baseline)
        frame_counts["recheck_candidates"] += len(additions)
        frame_counts["final"] += len(final)
        counters[source].update(frame_counts)
        baseline_pred[key] = [(float(x[4]), [float(v) for v in x[:4]]) for x in baseline]
        final_pred[key] = [(x["score"], x["box"]) for x in final]
        prediction_records.append({
            "source": source,
            "file_name": info["file_name"],
            "width": w,
            "height": h,
            "detections": [{
                "category_id": 0,
                "bbox_xyxy": [float(v) for v in det["box"]],
                "score": float(det["score"]),
                "origin": det["origin"],
                "pose_valid": bool(det["pose_valid"]),
                "expanded": bool(det["expanded"]),
                "keypoints": [[float(v) for v in point] for point in det["keypoints"]],
            } for det in final],
        })
        detail = {"source": source, "file_name": info["file_name"], "baseline": len(baseline),
                  "density_regions": len(centers), "recheck_candidates": len(additions),
                  "final": len(final), "counts": dict(frame_counts)}
        details.append(detail)
        if key in visual_keys:
            render(args.output / "visuals" / f"{source}_{Path(info['file_name']).stem}.jpg",
                   image, gt[key], baseline, final)
        print(json.dumps({"progress": f"{index}/{len(images)}", **detail}, ensure_ascii=False), flush=True)

    sources = sorted(by_source)
    parameters = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, list):
            value = [str(item) if isinstance(item, Path) else item for item in value]
        parameters[key] = value
    result = {
        "parameters": parameters,
        "dataset": {s: {"images": len(by_source[s]), "gt_boxes": sum(len(gt[k]) for k in by_source[s])}
                    for s in sources},
        "metrics": {
            s: {"baseline": evaluate(gt, baseline_pred, by_source[s]),
                "fused": evaluate(gt, final_pred, by_source[s])} for s in sources
        },
        "overall": {"baseline": evaluate(gt, baseline_pred, list(gt)),
                    "fused": evaluate(gt, final_pred, list(gt))},
        "counters": {s: dict(counters[s]) for s in sources},
        "details": details,
        "caveat": "人工边缘漏标会把部分真实新增框计为FP；annotator_04的关键点为ordered_heuristic配对。",
    }
    temp = args.output / "full_eval.json.tmp"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(args.output / "full_eval.json")
    predictions_path = args.output / "fused_predictions.json"
    predictions_temp = args.output / "fused_predictions.json.tmp"
    predictions_temp.write_text(
        json.dumps({"parameters": parameters, "images": prediction_records},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions_temp.replace(predictions_path)
    print(json.dumps({"dataset": result["dataset"], "metrics": result["metrics"],
                      "overall": result["overall"], "counters": result["counters"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
