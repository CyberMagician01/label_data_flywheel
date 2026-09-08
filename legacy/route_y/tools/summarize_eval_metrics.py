#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = [
    "mAP50-95",
    "AP50",
    "AP75",
    "precision_iou50",
    "recall_iou50",
    "NME_bbox_diag",
    "PCK@0.05",
    "PCK@0.10",
    "PCK@0.20",
    "direction_acc_angle_le_90",
    "mean_angle_error_deg",
    "median_angle_error_deg",
    "angle_over_45deg",
    "angle_over_90deg",
    "head_tail_swap_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    parser.add_argument("--title", default="BeePoseTrack-Y eval summary")
    return parser.parse_args()


def fold_name(path: Path) -> str:
    return path.stem.replace("_metrics", "")


def collect(rows: list[dict[str, Any]], section: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        values = [row[section].get(metric) for row in rows if metric in row[section]]
        values = [float(v) for v in values if v is not None]
        if not values:
            continue
        out[f"{metric}_mean"] = round(mean(values), 6)
        if metric in {"mAP50-95", "AP50", "AP75", "PCK@0.10", "direction_acc_angle_le_90"}:
            worst_idx = min(range(len(rows)), key=lambda i: float(rows[i][section].get(metric, 1e9)))
            out[f"{metric}_worst_fold"] = rows[worst_idx]["fold"]
            out[f"{metric}_worst"] = round(float(rows[worst_idx][section].get(metric, 0.0)), 6)
        if metric in {"NME_bbox_diag", "mean_angle_error_deg", "head_tail_swap_rate"}:
            worst_idx = max(range(len(rows)), key=lambda i: float(rows[i][section].get(metric, -1e9)))
            out[f"{metric}_worst_fold"] = rows[worst_idx]["fold"]
            out[f"{metric}_worst"] = round(float(rows[worst_idx][section].get(metric, 0.0)), 6)
    out["frames"] = sum(int(row[section].get("frames", 0)) for row in rows)
    out["gt_instances"] = sum(int(row[section].get("gt_instances", 0)) for row in rows)
    out["pred_instances"] = sum(int(row[section].get("pred_instances", 0)) for row in rows)
    out["pose_matched"] = sum(int(row[section].get("pose_matched", 0)) for row in rows)
    return out


def line_metric(lines: list[str], label: str, data: dict[str, Any], key: str) -> None:
    if key in data:
        lines.append(f"{label}: {data[key]}")


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["fold"] = fold_name(path)
        rows.append(data)
    summary = {
        "title": args.title,
        "folds": rows,
        "mean_all": collect(rows, "all"),
        "mean_rgb": collect(rows, "by_domain") if False else {},
    }
    for domain in ("RGB", "IR"):
        domain_rows = []
        for row in rows:
            domain_rows.append({"fold": row["fold"], "domain": row["by_domain"][domain], "all": row["by_domain"][domain]})
        summary[f"mean_{domain.lower()}"] = collect(domain_rows, "all")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [args.title, ""]
    lines.append("四折平均（ALL）")
    all_s = summary["mean_all"]
    line_metric(lines, "mAP50-95", all_s, "mAP50-95_mean")
    line_metric(lines, "AP50", all_s, "AP50_mean")
    line_metric(lines, "AP75", all_s, "AP75_mean")
    line_metric(lines, "Precision@IoU50", all_s, "precision_iou50_mean")
    line_metric(lines, "Recall@IoU50", all_s, "recall_iou50_mean")
    line_metric(lines, "PCK@0.05", all_s, "PCK@0.05_mean")
    line_metric(lines, "PCK@0.10", all_s, "PCK@0.10_mean")
    line_metric(lines, "PCK@0.20", all_s, "PCK@0.20_mean")
    line_metric(lines, "NME", all_s, "NME_bbox_diag_mean")
    line_metric(lines, "方向正确率(angle<=90)", all_s, "direction_acc_angle_le_90_mean")
    line_metric(lines, "平均角度误差", all_s, "mean_angle_error_deg_mean")
    line_metric(lines, "中位角度误差", all_s, "median_angle_error_deg_mean")
    line_metric(lines, ">45度错误率", all_s, "angle_over_45deg_mean")
    line_metric(lines, ">90度错误率", all_s, "angle_over_90deg_mean")
    line_metric(lines, "头尾交换率", all_s, "head_tail_swap_rate_mean")
    lines.append(f"总帧数: {all_s['frames']}")
    lines.append(f"GT实例数: {all_s['gt_instances']}")
    lines.append(f"预测实例数: {all_s['pred_instances']}")
    lines.append("")

    for domain in ("rgb", "ir"):
        label = domain.upper()
        data = summary[f"mean_{domain}"]
        lines.append(f"{label} 四折平均")
        line_metric(lines, "mAP50-95", data, "mAP50-95_mean")
        line_metric(lines, "AP50", data, "AP50_mean")
        line_metric(lines, "AP75", data, "AP75_mean")
        line_metric(lines, "PCK@0.10", data, "PCK@0.10_mean")
        line_metric(lines, "NME", data, "NME_bbox_diag_mean")
        line_metric(lines, "方向正确率(angle<=90)", data, "direction_acc_angle_le_90_mean")
        line_metric(lines, "头尾交换率", data, "head_tail_swap_rate_mean")
        lines.append("")

    lines.append("逐折 ALL 指标")
    for row in rows:
        data = row["all"]
        lines.append(
            f"{row['fold']}: mAP50-95={data.get('mAP50-95')}, AP50={data.get('AP50')}, AP75={data.get('AP75')}, "
            f"PCK@0.10={data.get('PCK@0.10')}, NME={data.get('NME_bbox_diag')}, "
            f"dir_acc={data.get('direction_acc_angle_le_90')}, swap={data.get('head_tail_swap_rate')}"
        )
    lines.append("")
    lines.append("说明：这些指标基于同一 fold 的 dataset_manifest.jsonl 和 val_images.txt 计算，预测坐标来自 YOLO 原图坐标输出。")
    lines.append("说明：跟踪指标 MOTA/IDF1/HOTA/IDSW/FRAG 需要在 Track-by-Detection 轨迹 JSONL 生成后再补。")
    args.output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_txt)


if __name__ == "__main__":
    main()
