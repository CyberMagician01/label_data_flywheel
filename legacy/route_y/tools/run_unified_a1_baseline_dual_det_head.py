#!/usr/bin/env python3
"""Run the isolated RGB/IR dual-detection-head A1 baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, default=Path("/data/bee26/datasets/bee_e_y_unified_20260901"))
    parser.add_argument("--y-root", type=Path, default=Path("/data/bee26/beeposetrack_y_20260827"))
    parser.add_argument("--model", default="yolo26m-pose.pt")
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--det-epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=768)
    parser.add_argument(
        "--dataset-dir-name",
        default="y_unified_20260901_labelme5_a1_strict_v8_annotator01_val_03_test_detposemask",
    )
    parser.add_argument(
        "--eval-name",
        default=None,
        help="Evaluation directory name.",
    )
    parser.add_argument("--supervision-mode", choices=["detposemask", "pose_only"], default="detposemask")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def training_complete(run_dir: Path, expected_epochs: int) -> bool:
    if not (run_dir / "weights" / "best.pt").exists():
        return False
    results = run_dir / "results.csv"
    if not results.exists():
        return False
    rows = [line for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        return False
    try:
        return int(float(rows[-1].split(",", 1)[0])) >= max(expected_epochs - 1, 0)
    except ValueError:
        return False


def write_alignment_report(
    path: Path,
    args: argparse.Namespace,
    stats: dict[str, Any],
    data_yaml: Path,
    det_run_dir: Path,
    pose_run_dir: Path,
    manifest: Path,
) -> None:
    lines = [
        "# Y A1 Unified Dual Detection Head Baseline Alignment Report",
        "",
        "## Data",
        f"- unified_root: `{args.unified_root}`",
        f"- package manifest: `{args.unified_root / 'manifests' / 'dataset_manifest.json'}`",
        f"- derived frame manifest: `{manifest}`",
        f"- data yaml: `{data_yaml}`",
        f"- supervision mode: `{args.supervision_mode}`",
        "- split policy: annotator_004/02/05=train, annotator_01=val/calibration, annotator_03=test/dev_holdout.",
        "- data files are reused as-is; this script does not rewrite images, labels, or manifest.",
        "",
        "## Model",
        "- baseline family: YOLO26m-Pose single-frame A1.",
        "- added isolation: RGB/IR separated detection heads only.",
        "- shared modules: backbone, neck, pose head, keypoint heads, RLE flow.",
        "- RGB routing: image names beginning with `A-` use RGB box/class heads.",
        "- IR routing: image names beginning with `B-` use IR box/class heads.",
        "",
        "## Training",
        f"- model/init: `{args.init_model or args.model}`",
        f"- seed: `{args.seed}`",
        f"- imgsz={args.imgsz}, batch={args.batch}, nbs={args.nbs}",
        "- optimizer: MuSGD; AMP enabled; deterministic seed enabled.",
        "- augmentation matches strict-v8 baseline: Mosaic/MixUp/CutMix/Copy-Paste/Erasing disabled; geometry augmentation retained.",
        f"- det-only run_dir: `{det_run_dir}`",
        f"- pose-from-det run_dir: `{pose_run_dir}`",
        "",
        "## Loss Rules",
        "- det-only stage: all instances update only box/class/DFL losses; pose/kobj/RLE losses are forced to zero.",
        "- pose-from-det stage: complete head+tail instances update detection and pose losses.",
        "- pose-from-det stage: det-only instances keep detection supervision and are masked out of pose/kobj/RLE losses.",
        "",
        "## Compliance",
        "- This is an isolated experiment and does not overwrite the previous staged decoupled baseline.",
        "- It keeps the current aligned dataset and frozen split unchanged.",
        "- It does not enable RouteBest, P2, temporal input, domain adapters, or extra data augmentation.",
        "- Final evaluation uses the same prediction, pose-aware NMS, detection/pose metrics, tracker, and tracking metrics pipeline as the strict-v8 baseline, except prediction routes RGB/IR through separate detection heads.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    package_ready = args.unified_root / "manifests" / "PACKAGE_READY.json"
    sha_list = args.unified_root / "manifests" / "annotations_sha256.txt"
    if not package_ready.exists():
        raise RuntimeError("PACKAGE_READY.json is required before formal aligned training")
    if read_json(package_ready).get("status") != "READY":
        raise RuntimeError(f"Unified package is not READY: {package_ready}")
    if not sha_list.exists():
        raise RuntimeError("annotations_sha256.txt is required before formal aligned training")

    dataset_dir = args.y_root / "datasets" / args.dataset_dir_name
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        run(
            [
                args.python,
                str(args.y_root / "tools" / "prepare_unified_labelme_yolo_pose.py"),
                "--unified-root",
                str(args.unified_root),
                "--out-dir",
                str(dataset_dir),
                "--split-policy",
                "annotator01_val_03_test",
                "--supervision-mode",
                args.supervision_mode,
            ]
        )
    stats = read_json(summary_path)

    eval_name = args.eval_name or f"a1_strict_v8_detposemask_dual_det_head_det100_pose100_annotator01_val03_seed{args.seed}"
    eval_dir = args.y_root / "eval_unified_20260901" / eval_name
    stats_dir = eval_dir / "pretrain_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "data_distribution_before_train.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md = dataset_dir / "dataset_summary.md"
    if summary_md.exists():
        (stats_dir / "data_distribution_before_train.md").write_text(summary_md.read_text(encoding="utf-8"), encoding="utf-8")

    split_lists = {
        "train": dataset_dir / "train_images.txt",
        "val": dataset_dir / "val_images.txt",
        "test": dataset_dir / "test_images.txt",
    }
    data_yaml = dataset_dir / "bee_yolo_pose_strict.yaml"
    manifest = dataset_dir / "dataset_manifest.jsonl"
    project = args.y_root / "runs_unified_20260901" / "a1_aligned"
    tag = Path(str(args.init_model or args.model)).stem.replace("-pose", "").replace("_", "-")
    data_tag = dataset_dir.name.replace("y_unified_20260901_labelme5_a1_", "").replace("_", "-")
    base_name = f"a1_unified_{data_tag}_{tag}_seed{args.seed}_img{args.imgsz}_eb{args.nbs}_dual_det_head"
    det_name = f"{base_name}_det_only_stage"
    pose_name = f"{base_name}_pose_from_det_stage"
    det_run_dir = project / det_name
    pose_run_dir = project / pose_name
    write_alignment_report(eval_dir / "alignment_report.md", args, stats, data_yaml, det_run_dir, pose_run_dir, manifest)

    if not training_complete(det_run_dir, args.det_epochs):
        det_train_model = args.init_model or args.model
        det_extra_args: list[str] = []
        if args.resume:
            det_last = det_run_dir / "weights" / "last.pt"
            if det_last.exists():
                det_train_model = det_last
                det_extra_args.append("--resume")
        run(
            [
                args.python,
                str(args.y_root / "tools" / "train_yolo_pose_dual_det_head.py"),
                "--data",
                str(data_yaml),
                "--model",
                str(det_train_model),
                "--project",
                str(project),
                "--name",
                det_name,
                "--epochs",
                str(args.det_epochs),
                "--batch",
                str(args.batch),
                "--nbs",
                str(args.nbs),
                "--imgsz",
                str(args.imgsz),
                "--device",
                str(args.device),
                "--workers",
                str(args.workers),
                "--seed",
                str(args.seed),
                "--det-only",
                *det_extra_args,
            ]
        )

    det_best = det_run_dir / "weights" / "best.pt"
    if not det_best.exists():
        raise FileNotFoundError(det_best)

    if not training_complete(pose_run_dir, args.epochs):
        pose_train_model = det_best
        pose_extra_args: list[str] = []
        if args.resume:
            pose_last = pose_run_dir / "weights" / "last.pt"
            if pose_last.exists():
                pose_train_model = pose_last
                pose_extra_args.append("--resume")
        run(
            [
                args.python,
                str(args.y_root / "tools" / "train_yolo_pose_dual_det_head.py"),
                "--data",
                str(data_yaml),
                "--model",
                str(pose_train_model),
                "--project",
                str(project),
                "--name",
                pose_name,
                "--epochs",
                str(args.epochs),
                "--batch",
                str(args.batch),
                "--nbs",
                str(args.nbs),
                "--imgsz",
                str(args.imgsz),
                "--device",
                str(args.device),
                "--workers",
                str(args.workers),
                "--seed",
                str(args.seed),
                *pose_extra_args,
            ]
        )

    best = pose_run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)

    raw = eval_dir / "test_pred_raw.jsonl"
    pred = eval_dir / "test_pred_pose_aware.jsonl"
    metrics = eval_dir / "test_metrics.json"
    tracks = eval_dir / "test_tracks_raw.jsonl"
    tracks_fixed = eval_dir / "test_tracks_head_tail_dp.jsonl"
    tracking_metrics = eval_dir / "test_tracking_metrics.json"
    run(
        [
            args.python,
            str(args.y_root / "tools" / "predict_yolo_pose_jsonl_dual_det_head.py"),
            "--model",
            str(best),
            "--source",
            str(split_lists["test"]),
            "--output",
            str(raw),
            "--imgsz",
            str(args.imgsz),
            "--device",
            str(args.device),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--max-det",
            str(args.max_det),
            "--model-name",
            "BeePoseTrack-Y-A1-Unified-DualDetHead",
        ]
    )
    run([args.python, str(args.y_root / "tools" / "pose_aware_nms.py"), "--input", str(raw), "--output", str(pred), "--iou", str(args.iou), "--max-det", str(args.max_det)])
    run([args.python, str(args.y_root / "tools" / "evaluate_manifest_predictions.py"), "--manifest", str(manifest), "--pred-jsonl", str(pred), "--output", str(metrics), "--eval-split", "test", "--pose-iou", "0.3", "--pr-iou", "0.5"])
    run([args.python, str(args.y_root / "tracking" / "pose_motion_tracker.py"), "--detections", str(pred), "--output", str(tracks), "--max-age", "3", "--max-cost", "1.35", "--max-center-dist", "140.0", "--max-candidates-per-frame", "384", "--max-new-tracks-per-frame", "384", "--max-active-tracks", "512", "--hungarian-limit", "384"])
    run([args.python, str(args.y_root / "tracking" / "head_tail_dp.py"), "--input", str(tracks), "--output", str(tracks_fixed)])
    run([args.python, str(args.y_root / "tools" / "evaluate_manifest_tracking.py"), "--manifest", str(manifest), "--track-jsonl", str(tracks_fixed), "--output", str(tracking_metrics), "--eval-split", "test", "--iou", "0.5"])


if __name__ == "__main__":
    main()
