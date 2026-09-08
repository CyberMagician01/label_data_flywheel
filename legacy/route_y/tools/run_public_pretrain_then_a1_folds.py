#!/usr/bin/env python3
"""Run the Y route with E-aligned public pretraining and A1 fold fine-tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-data-root", type=Path, default=Path("/data/bee26/beeposetrack_e_20260827/data"))
    parser.add_argument("--own-image-root", type=Path, default=Path("/data/bee26/vitpose_size_ablation_13_20260827/data/bee_keypoints_13_20260827"))
    parser.add_argument("--y-root", type=Path, default=Path("/data/bee26/beeposetrack_y_20260827"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--public-epochs", type=int, default=24)
    parser.add_argument("--public-imgsz", type=int, default=960)
    parser.add_argument("--a1-epochs", type=int, default=100)
    parser.add_argument("--a1-imgsz", type=int, default=1280)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def training_complete(run_dir: Path, expected_epochs: int) -> bool:
    if not (run_dir / "weights" / "best.pt").exists():
        return False
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return False
    rows = [line.strip() for line in results_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        return False
    try:
        last_epoch = int(float(rows[-1].split(",", 1)[0].strip()))
    except ValueError:
        return False
    return last_epoch >= expected_epochs


def run(cmd: list[str], expected_run_dir: Path | None = None, expected_epochs: int | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        return
    if (
        expected_run_dir is not None
        and expected_epochs is not None
        and training_complete(expected_run_dir, expected_epochs)
    ):
        print(
            f"WARN command returned {result.returncode}, but training outputs are complete: {expected_run_dir}",
            flush=True,
        )
        return
    result.check_returncode()


def model_tag(model_path: str) -> str:
    stem = Path(model_path).stem
    return stem.replace("-pose", "").replace("_", "-")


def best_weight(project: Path, name: str) -> Path:
    return project / name / "weights" / "best.pt"


def validate_yolo_dataset(yaml_path: Path, allow_empty_labels: bool = False) -> dict[str, int]:
    root = yaml_path.parent
    stats = {}
    for split in ("train", "val"):
        list_path = root / f"{split}_images.txt"
        if list_path.exists():
            images = [Path(x) for x in list_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            images = sorted((root / "images" / split).glob("*"))
        missing = 0
        empty = 0
        for img in images:
            label = root / "labels" / split / img.with_suffix(".txt").name
            if not label.exists():
                missing += 1
            elif not label.read_text(encoding="utf-8").strip():
                empty += 1
        stats[f"{split}_images"] = len(images)
        stats[f"{split}_missing_labels"] = missing
        stats[f"{split}_empty_labels"] = empty
    bad_keys = ["missing_labels"] if allow_empty_labels else ["missing_labels", "empty_labels"]
    if any(v for k, v in stats.items() if any(k.endswith(suffix) for suffix in bad_keys)):
        raise RuntimeError(f"invalid dataset {yaml_path}: {stats}")
    return stats


def main() -> None:
    args = parse_args()
    tools = args.y_root / "tools"
    datasets = args.y_root / "datasets"
    public_dir = datasets / "y_public_beepose_mendeley_e_aligned_v2"
    folds_root = datasets / "y_a1_manifest_folds_e_aligned"

    public_yaml = public_dir / "bee_yolo_pose.yaml"
    if not public_yaml.exists():
        run(
            [
                args.python,
                str(tools / "coco_to_yolo_pose.py"),
                "--ann-root",
                str(args.e_data_root / "annotations"),
                "--image-root",
                str(args.e_data_root / "public_images"),
                "--out-dir",
                str(public_dir),
                "--train-json",
                "public_bee_pose_train.json",
                "--val-json",
                "public_bee_pose_val.json",
                "--tag",
                "BeePose_Mendeley_public_E_aligned",
                "--keep-partial-keypoints",
                "--include-empty-images",
            ]
        )
    public_stats = validate_yolo_dataset(public_yaml, allow_empty_labels=True)

    fold_summary = folds_root / "fold_dataset_summary.json"
    if not fold_summary.exists():
        run(
            [
                args.python,
                str(tools / "prepare_manifest_folds_yolo_pose.py"),
                "--manifest",
                str(args.e_data_root / "annotations" / "folds" / "dataset_manifest.jsonl"),
                "--image-root",
                str(args.own_image_root),
                "--out-root",
                str(folds_root),
                "--folds",
                *[str(f) for f in args.folds],
            ]
        )
    fold_stats = {}
    for fold in args.folds:
        yaml_path = folds_root / f"fold{fold}" / "bee_yolo_pose_strict.yaml"
        fold_stats[f"fold{fold}"] = validate_yolo_dataset(yaml_path)

    metadata = {
        "e_data_root": str(args.e_data_root),
        "own_image_root": str(args.own_image_root),
        "models": args.models,
        "folds": args.folds,
        "seeds": args.seeds,
        "device": args.device,
        "batch": args.batch,
        "nbs": args.nbs,
        "public": {"epochs": args.public_epochs, "imgsz": args.public_imgsz, "stats": public_stats},
        "a1": {"epochs": args.a1_epochs, "imgsz": args.a1_imgsz, "stats": fold_stats},
    }
    run_root = args.y_root / "runs_e_aligned"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    public_project = run_root / "public_pretrain"
    a1_project = run_root / "a1_folds"
    for model in args.models:
        tag = model_tag(model)
        public_name = f"public_{tag}_seed2026_img{args.public_imgsz}_eb{args.nbs}"
        public_run_dir = public_project / public_name
        public_best = best_weight(public_project, public_name)
        if not training_complete(public_run_dir, args.public_epochs):
            run(
                [
                    args.python,
                    str(tools / "train_yolo_pose_aligned.py"),
                    "--data",
                    str(public_yaml),
                    "--model",
                    model,
                    "--project",
                    str(public_project),
                    "--name",
                    public_name,
                    "--epochs",
                    str(args.public_epochs),
                    "--batch",
                    str(args.batch),
                    "--nbs",
                    str(args.nbs),
                    "--imgsz",
                    str(args.public_imgsz),
                    "--device",
                    args.device,
                    "--workers",
                    str(args.workers),
                    "--seed",
                    "2026",
                ],
                expected_run_dir=public_run_dir,
                expected_epochs=args.public_epochs,
            )
        if not public_best.exists():
            raise FileNotFoundError(public_best)
        for fold in args.folds:
            fold_yaml = folds_root / f"fold{fold}" / "bee_yolo_pose_strict.yaml"
            for seed in args.seeds:
                name = f"a1_fold{fold}_{tag}_public_seed{seed}_img{args.a1_imgsz}_eb{args.nbs}"
                a1_run_dir = a1_project / name
                if training_complete(a1_run_dir, args.a1_epochs):
                    continue
                run(
                    [
                        args.python,
                        str(tools / "train_yolo_pose_aligned.py"),
                        "--data",
                        str(fold_yaml),
                        "--model",
                        str(public_best),
                        "--project",
                        str(a1_project),
                        "--name",
                        name,
                        "--epochs",
                        str(args.a1_epochs),
                        "--batch",
                        str(args.batch),
                        "--nbs",
                        str(args.nbs),
                        "--imgsz",
                        str(args.a1_imgsz),
                        "--device",
                        args.device,
                        "--workers",
                        str(args.workers),
                        "--seed",
                        str(seed),
                    ],
                    expected_run_dir=a1_run_dir,
                    expected_epochs=args.a1_epochs,
                )


if __name__ == "__main__":
    main()
