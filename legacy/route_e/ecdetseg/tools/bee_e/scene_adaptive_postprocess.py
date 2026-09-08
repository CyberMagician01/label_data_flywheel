#!/usr/bin/env python3
"""Evaluate and visualize reversible scene-adaptive duplicate suppression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _ordered_box(box: list[float]) -> list[float]:
    return [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]


def _iou(a: list[float], b: list[float]) -> float:
    a, b = _ordered_box(a), _ordered_box(b)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _duplicate_geometry(a: list[float], b: list[float], iou_threshold: float) -> bool:
    a, b = _ordered_box(a), _ordered_box(b)
    overlap = _iou(a, b)
    if overlap < iou_threshold:
        return False
    aw, ah = max(a[2] - a[0], 1e-6), max(a[3] - a[1], 1e-6)
    bw, bh = max(b[2] - b[0], 1e-6), max(b[3] - b[1], 1e-6)
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    center_distance = math.hypot(acx - bcx, acy - bcy)
    center_ratio = center_distance / max(min(math.hypot(aw, ah), math.hypot(bw, bh)), 1.0)
    scale_delta = max(abs(math.log(aw / bw)), abs(math.log(ah / bh)))
    return center_ratio <= 0.35 and scale_delta <= math.log(1.75)


def suppress_duplicates(
    predictions: list[dict], score_threshold: float, iou_threshold: float
) -> tuple[list[dict], list[dict]]:
    candidates = [item for item in predictions if float(item["score"]) >= score_threshold]
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict] = []
    removed: list[dict] = []
    for prediction in candidates:
        if any(_duplicate_geometry(prediction["box"], prior["box"], iou_threshold) for prior in kept):
            removed.append(prediction)
        else:
            kept.append(prediction)
    return kept, removed


def match_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    predictions = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    matched_gt: set[int] = set()
    duplicate_count = 0
    for prediction in predictions:
        overlaps = [_iou(prediction["box"], gt["box"]) for gt in ground_truth]
        order = sorted(range(len(overlaps)), key=overlaps.__getitem__, reverse=True)
        unmatched = next((index for index in order if overlaps[index] >= 0.5 and index not in matched_gt), None)
        if unmatched is not None:
            matched_gt.add(unmatched)
        elif any(overlaps[index] >= 0.5 and index in matched_gt for index in order):
            duplicate_count += 1
    tp = len(matched_gt)
    fp = len(predictions) - tp
    miss = len(ground_truth) - tp
    return {
        "predictions": len(predictions),
        "ground_truth": len(ground_truth),
        "tp": tp,
        "fp": fp,
        "duplicate_fp": duplicate_count,
        "miss": miss,
        "precision": tp / len(predictions) if predictions else 0.0,
        "recall": tp / len(ground_truth) if ground_truth else 0.0,
    }


def removed_independent_gt(removed: list[dict], kept: list[dict], ground_truth: list[dict]) -> int:
    protected = {
        index
        for index, gt in enumerate(ground_truth)
        if any(_iou(item["box"], gt["box"]) >= 0.5 for item in kept)
    }
    at_risk = {
        index
        for index, gt in enumerate(ground_truth)
        if index not in protected and any(_iou(item["box"], gt["box"]) >= 0.5 for item in removed)
    }
    return len(at_risk)


def _draw_panel(image: Image.Image, gt: list[dict], predictions: list[dict], title: str) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for item in gt:
        draw.rectangle(_ordered_box(item["box"]), outline=(0, 255, 0), width=2)
    for item in predictions:
        box = _ordered_box(item["box"])
        draw.rectangle(box, outline=(255, 50, 50), width=2)
        draw.text((box[0], max(0, box[1] - 11)), f'{float(item["score"]):.3f}', fill=(255, 255, 0), font=font)
    draw.rectangle((0, 0, panel.width, 22), fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 255, 255), font=font)
    return panel


def _resolve_image(sample: dict, image_root: Path | None) -> Path:
    remote_path = Path(sample["image_path"])
    if remote_path.is_file():
        return remote_path
    if image_root is not None:
        candidate = image_root / sample["file_name"]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Image not found for image_id={sample["image_id"]}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--indoor-score", type=float, default=0.02)
    parser.add_argument("--indoor-iou", type=float, default=0.85)
    parser.add_argument("--outdoor-score", type=float, default=0.02)
    parser.add_argument("--outdoor-iou", type=float, default=0.70)
    args = parser.parse_args()

    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": source.get("checkpoint"),
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "policy": {
            "indoor_B": {"scene_id": 0, "score": args.indoor_score, "iou": args.indoor_iou},
            "outdoor_A": {"scene_id": 1, "score": args.outdoor_score, "iou": args.outdoor_iou},
            "endpoint_oks_used": False,
            "geometry_gate": "box_iou AND normalized_center_distance<=0.35 AND scale_ratio<=1.75",
        },
        "samples": [],
    }
    totals = {name: {key: 0 for key in ("predictions", "ground_truth", "tp", "fp", "duplicate_fp", "miss")} for name in ("raw", "adaptive")}
    for sample in source["samples"]:
        scene_id = int(sample["route_scene_id"])
        score_threshold = args.indoor_score if scene_id == 0 else args.outdoor_score
        iou_threshold = args.indoor_iou if scene_id == 0 else args.outdoor_iou
        raw = [item for item in sample["predictions"] if float(item["score"]) >= score_threshold]
        kept, removed = suppress_duplicates(sample["predictions"], score_threshold, iou_threshold)
        raw_metrics = match_metrics(raw, sample["gt"])
        adaptive_metrics = match_metrics(kept, sample["gt"])
        for name, metrics in (("raw", raw_metrics), ("adaptive", adaptive_metrics)):
            for key in totals[name]:
                totals[name][key] += metrics[key]
        item_report = {
            "image_id": int(sample["image_id"]),
            "role": sample.get("role"),
            "scene_id": scene_id,
            "raw": raw_metrics,
            "adaptive": adaptive_metrics,
            "removed": len(removed),
            "removed_independent_gt": removed_independent_gt(removed, kept, sample["gt"]),
        }
        report["samples"].append(item_report)
        image = Image.open(_resolve_image(sample, args.image_root)).convert("RGB")
        left = _draw_panel(image, sample["gt"], raw, f'RAW image_id={sample["image_id"]}')
        right = _draw_panel(image, sample["gt"], kept, f'ADAPTIVE image_id={sample["image_id"]}')
        comparison = Image.new("RGB", (left.width * 2, left.height))
        comparison.paste(left, (0, 0))
        comparison.paste(right, (left.width, 0))
        comparison.save(args.output_dir / f'image_{int(sample["image_id"])}_raw_vs_adaptive.jpg', quality=92)

    for name in ("raw", "adaptive"):
        total = totals[name]
        total["precision"] = total["tp"] / total["predictions"] if total["predictions"] else 0.0
        total["recall"] = total["tp"] / total["ground_truth"] if total["ground_truth"] else 0.0
    report["totals"] = totals
    report_path = args.output_dir / "scene_adaptive_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
