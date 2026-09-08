#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ["MOTA", "IDF1", "HOTA_proxy", "IDSW", "FRAG_proxy", "FP", "FN", "matches"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    parser.add_argument("--title", default="YOLO26 A1 Track-by-Detection tracking eval")
    return parser.parse_args()


def fold_name(path: Path) -> str:
    return path.stem.replace("_tracking_metrics", "")


def collect(rows: list[dict[str, Any]], section: str) -> dict[str, Any]:
    out = {}
    for metric in METRICS:
        vals = [float(r[section][metric]) for r in rows if metric in r[section]]
        if not vals:
            continue
        out[f"{metric}_mean"] = round(mean(vals), 6)
        if metric in {"IDSW", "FRAG_proxy", "FP", "FN"}:
            out[f"{metric}_sum"] = int(sum(vals))
    out["frames"] = sum(int(r[section].get("frames", 0)) for r in rows)
    out["gt_instances"] = sum(int(r[section].get("gt_instances", 0)) for r in rows)
    return out


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["fold"] = fold_name(path)
        rows.append(data)
    summary = {
        "folds": rows,
        "mean_all": collect(rows, "all"),
    }
    for domain in ("RGB", "IR"):
        domain_rows = []
        for row in rows:
            domain_rows.append({"fold": row["fold"], "all": row["by_domain"][domain]})
        summary[f"mean_{domain.lower()}"] = collect(domain_rows, "all")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [args.title, ""]
    for label, key in [("ALL", "mean_all"), ("RGB", "mean_rgb"), ("IR", "mean_ir")]:
        data = summary[key]
        lines.append(f"{label} 四折平均")
        for metric in ["MOTA", "IDF1", "HOTA_proxy", "IDSW", "FRAG_proxy", "FP", "FN"]:
            mean_key = f"{metric}_mean"
            if mean_key in data:
                lines.append(f"{metric}: {data[mean_key]}")
            sum_key = f"{metric}_sum"
            if sum_key in data:
                lines.append(f"{metric}_sum: {data[sum_key]}")
        lines.append(f"frames: {data['frames']}")
        lines.append(f"gt_instances: {data['gt_instances']}")
        lines.append("")
    lines.append("逐折 ALL 指标")
    for row in rows:
        data = row["all"]
        lines.append(
            f"{row['fold']}: MOTA={data.get('MOTA')}, IDF1={data.get('IDF1')}, "
            f"HOTA_proxy={data.get('HOTA_proxy')}, IDSW={data.get('IDSW')}, FRAG_proxy={data.get('FRAG_proxy')}"
        )
    lines.append("")
    lines.append("说明：MOTA/IDF1/IDSW 使用 manifest track_id 与预测轨迹按 IoU=0.5 匹配计算。")
    lines.append("说明：HOTA_proxy/FRAG_proxy 是内部近似口径，最终报告建议接 E 路线同一 TrackEval/官方评估器复核。")
    args.output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_txt)


if __name__ == "__main__":
    main()
