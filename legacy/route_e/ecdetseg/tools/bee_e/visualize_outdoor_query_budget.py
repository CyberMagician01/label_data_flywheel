#!/usr/bin/env python3
"""Render fixed outdoor samples for query-budget and NMS comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont


COLORS = {"gt": "#f4a300", "tp": "#00e5ff", "fp": "#ff304f", "dup": "#ff00d4"}


def _classify(predictions, gt_boxes):
    gt = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)
    claimed = set()
    labelled = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        if not len(gt):
            labelled.append((prediction, "fp"))
            continue
        box = torch.tensor(prediction["box_xyxy"], dtype=torch.float32).reshape(1, 4)
        ious = torchvision.ops.box_iou(box, gt)[0]
        best_iou, best_index = ious.max(dim=0)
        if float(best_iou) < 0.5:
            label = "fp"
        elif int(best_index) in claimed:
            label = "dup"
        else:
            label = "tp"
            claimed.add(int(best_index))
        labelled.append((prediction, label))
    return labelled, len(gt) - len(claimed)


def _panel(image_path, gt_boxes, predictions, title, display_score, display_topk):
    image = Image.open(image_path).convert("RGB")
    predictions = [item for item in predictions if item["score"] >= display_score]
    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)[:display_topk]
    labelled, miss = _classify(predictions, gt_boxes)
    banner = 42
    canvas = Image.new("RGB", (image.width, image.height + banner), "black")
    canvas.paste(image, (0, banner))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    counts = {name: sum(label == name for _, label in labelled) for name in ("tp", "fp", "dup")}
    draw.text(
        (8, 8),
        f"{title} pred={len(labelled)} TP={counts['tp']} FP={counts['fp']} DUP={counts['dup']} MISS={miss}",
        fill="white", font=font,
    )
    for box in gt_boxes:
        draw.rectangle([box[0], box[1] + banner, box[2], box[3] + banner], outline=COLORS["gt"], width=2)
    for prediction, label in labelled:
        box = prediction["box_xyxy"]
        x1, x2 = sorted((box[0], box[2]))
        y1, y2 = sorted((box[1], box[3]))
        shifted = [x1, y1 + banner, x2, y2 + banner]
        draw.rectangle(shifted, outline=COLORS[label], width=2)
        draw.text((shifted[0], max(banner, shifted[1] - 10)), f"{prediction['score']:.3f}", fill=COLORS[label], font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selected-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ids", type=int, nargs="+", default=[41, 774, 6774, 2036])
    parser.add_argument("--baseline-budget", type=int, default=512)
    parser.add_argument("--rejected-budget", type=int, default=256)
    parser.add_argument("--selected-budget", type=int, default=512)
    parser.add_argument("--display-score", type=float, default=0.02)
    parser.add_argument("--display-topk", type=int, default=50)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    visual = report["visual_samples"]
    selected_visual = visual
    if args.selected_report is not None:
        selected_visual = json.loads(
            args.selected_report.read_text(encoding="utf-8")
        )["visual_samples"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in args.image_ids:
        key = str(image_id)
        baseline = visual[str(args.baseline_budget)][key]
        rejected = visual[str(args.rejected_budget)][key]
        selected = selected_visual[str(args.selected_budget)][key]
        panels = [
            _panel(
                baseline["image_path"], baseline["gt_xyxy"], baseline["raw"],
                f"baseline logical-q{args.baseline_budget} raw class*quality",
                args.display_score, args.display_topk,
            ),
            _panel(
                rejected["image_path"], rejected["gt_xyxy"], rejected["raw"],
                f"tested logical-q{args.rejected_budget} raw (FAILED recall gate)",
                args.display_score, args.display_topk,
            ),
            _panel(
                selected["image_path"], selected["gt_xyxy"], selected["nms"],
                f"selected logical-q{args.selected_budget} + outdoor NMS IoU{args.nms_iou:.2f}",
                args.display_score, args.display_topk,
            ),
        ]
        combined = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black")
        offset = 0
        for panel in panels:
            combined.paste(panel, (offset, 0))
            offset += panel.width
        path = args.output_dir / f"image_{image_id}_query_budget_comparison.jpg"
        combined.save(path, quality=92)
        rows.append(combined)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "black")
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height
    sheet.save(args.output_dir / "query_budget_comparison_contact_sheet.jpg", quality=90)


if __name__ == "__main__":
    main()
