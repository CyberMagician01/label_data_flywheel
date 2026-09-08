"""汇总完整姿态/密度实验的指标表和曲线图。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "scene_A_no_public": "Outdoor / own only",
    "scene_A_with_public": "Outdoor / + public",
    "scene_A_with_public_occlusion_direction": "Outdoor / + public + robust",
    "scene_B_no_public": "Indoor / own only",
    "scene_B_with_public": "Indoor / + public",
    "scene_B_with_public_all_labels": "Indoor / + low-quality labels",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_pose(root):
    rows = []

    def append_metrics(name, subset, metrics):
        rows.append({
            "experiment": name, "display_name": DISPLAY_NAMES.get(name, name),
            "subset": subset, "instances": metrics["instances"],
            "pck@0.05": metrics["pck@0.05"], "pck@0.10": metrics["pck@0.10"],
            "pck@0.20": metrics["pck@0.20"],
            "orientation_accuracy": metrics["orientation_accuracy"],
            "mean_angle_error_deg": metrics["mean_angle_error_deg"],
            "median_angle_error_deg": metrics["median_angle_error_deg"],
            "angle_within_30deg": metrics["angle_within_30deg"],
            "angle_over_45deg": metrics["angle_over_45deg"],
            "nme_bbox_diagonal": metrics["nme_bbox_diagonal"],
        })

    for path in sorted((root / "evaluations").glob("scene_*_*.json")):
        if path.parent.name == "scene_A_postprocessing" or path.name.endswith("predictions.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics")
        if not metrics:
            continue
        suffix = "occlusion_all" if path.stem.endswith("_occlusion") else "clean"
        name = path.stem[: -len("_" + suffix)]
        if suffix == "occlusion_all":
            name = path.stem[: -len("_occlusion")]
        append_metrics(name, suffix, metrics)
        occluded = metrics.get("subgroups", {}).get("synthetic_occlusion", {})
        if occluded.get("instances", 0):
            append_metrics(name, "synthetic_occlusion_only", occluded)
    return rows


def plot_pose(rows, target):
    clean = [row for row in rows if row["subset"] == "clean"]
    if not clean:
        return
    labels = [row["display_name"] for row in clean]
    y = np.arange(len(clean))
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, len(clean) * 0.65)))
    specs = [
        ("pck@0.10", "PCK@0.10", (0, 1)),
        ("orientation_accuracy", "Orientation accuracy", (0, 1)),
        ("mean_angle_error_deg", "Mean angle error (deg, lower better)", None),
    ]
    for axis, (key, title, limits) in zip(axes, specs):
        values = [row[key] for row in clean]
        axis.barh(y, values, color="#3b82f6")
        axis.set_yticks(y, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        if limits:
            axis.set_xlim(*limits)
        axis.grid(axis="x", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(value, index, f" {value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_training_curves(root):
    curves = {}
    for work_dir in sorted((root / "work_dirs").iterdir()):
        if not work_dir.is_dir() or work_dir.name.startswith("smoke") or work_dir.name.startswith("density"):
            continue
        values = defaultdict(list)
        for log_path in work_dir.glob("*.log.json"):
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("mode") == "train" and "loss" in record:
                    values[int(record["epoch"])].append(float(record["loss"]))
        if values:
            curves[work_dir.name] = (
                sorted(values), [float(np.mean(values[epoch])) for epoch in sorted(values)]
            )
    return curves


def plot_curves(curves, target):
    if not curves:
        return
    fig, axis = plt.subplots(figsize=(10, 6))
    for name, (epochs, losses) in curves.items():
        axis.plot(epochs, losses, label=DISPLAY_NAMES.get(name, name), linewidth=1.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_density(root):
    rows = []
    for scene in ("A", "B"):
        path = root / "work_dirs" / f"density_scene_{scene}" / "metrics.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        best = min(records, key=lambda item: item["val_mae"])
        rows.append({"scene": scene, **best})
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_rows = collect_pose(args.experiment_root)
    write_csv(args.output_dir / "pose_metrics.csv", pose_rows)
    plot_pose(pose_rows, args.output_dir / "pose_metrics_comparison.png")
    curves = collect_training_curves(args.experiment_root)
    plot_curves(curves, args.output_dir / "training_loss_curves.png")
    density_rows = collect_density(args.experiment_root)
    write_csv(args.output_dir / "density_metrics.csv", density_rows)
    summary = {"pose_experiments": len(pose_rows), "density_experiments": len(density_rows)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
