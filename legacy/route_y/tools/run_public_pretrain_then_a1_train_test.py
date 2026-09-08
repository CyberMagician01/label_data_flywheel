#!/usr/bin/env python3
"""Run Y A1 single train/test fine-tuning from E schema-v2 trackfix data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema-v2-root",
        type=Path,
        default=Path("/data/bee26/beeposetrack_e_20260827/outputs/bee_e_schema_v2_latest03_detposefix_trackfix_20260830"),
    )
    parser.add_argument(
        "--e-data-root",
        type=Path,
        default=Path("/data/bee26/beeposetrack_e_20260827/data"),
        help="Only used to prepare public BeePose+Mendeley pretraining data if it is missing.",
    )
    parser.add_argument(
        "--own-image-root",
        type=Path,
        default=Path("/data/bee26/vitpose_size_ablation_13_20260827/data/bee_keypoints_13_20260827"),
    )
    parser.add_argument("--y-root", type=Path, default=Path("/data/bee26/beeposetrack_y_20260827"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--public-epochs", type=int, default=24)
    parser.add_argument("--public-imgsz", type=int, default=960)
    parser.add_argument("--a1-epochs", type=int, default=100)
    parser.add_argument("--a1-imgsz", type=int, default=1280)
    parser.add_argument("--dataset-name", default="y_a1_schema_v2_latest03_detposefix_trackfix_train_test")
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
    if expected_run_dir is not None and expected_epochs is not None and training_complete(expected_run_dir, expected_epochs):
        print(f"WARN command returned {result.returncode}, but outputs are complete: {expected_run_dir}", flush=True)
        return
    result.check_returncode()


def model_tag(model_path: str) -> str:
    stem = Path(model_path).stem
    return stem.replace("-pose", "").replace("_", "-")


def best_weight(project: Path, name: str) -> Path:
    return project / name / "weights" / "best.pt"


def validate_yolo_dataset(yaml_path: Path) -> dict[str, int]:
    root = yaml_path.parent
    stats: dict[str, int] = {}
    for split in ("train", "test"):
        list_path = root / f"{split}_images.txt"
        images = [Path(x) for x in list_path.read_text(encoding="utf-8").splitlines() if x.strip()]
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
    if any(value for key, value in stats.items() if key.endswith("missing_labels")):
        raise RuntimeError(f"invalid dataset {yaml_path}: {stats}")
    return stats


def main() -> None:
    args = parse_args()
    tools = args.y_root / "tools"
    datasets = args.y_root / "datasets"
    public_dir = datasets / "y_public_beepose_mendeley_e_aligned_v2"
    own_dir = datasets / args.dataset_name

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

    summary = own_dir / "dataset_summary.json"
    if not summary.exists():
        run(
            [
                args.python,
                str(tools / "prepare_schema_v2_train_test_yolo_pose.py"),
                "--train-json",
                str(args.schema_v2_root / "train_13_rgb_ir_schema_v2.json"),
                "--test-json",
                str(args.schema_v2_root / "val_13_rgb_ir_schema_v2.json"),
                "--image-root",
                str(args.own_image_root),
                "--out-dir",
                str(own_dir),
            ]
        )
    own_stats = validate_yolo_dataset(own_dir / "bee_yolo_pose_strict.yaml")

    run_root = args.y_root / "runs_schema_v2_trackfix"
    run_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_v2_root": str(args.schema_v2_root),
        "own_image_root": str(args.own_image_root),
        "dataset": str(own_dir),
        "models": args.models,
        "seeds": args.seeds,
        "device": args.device,
        "batch": args.batch,
        "nbs": args.nbs,
        "public": {"epochs": args.public_epochs, "imgsz": args.public_imgsz},
        "a1": {"epochs": args.a1_epochs, "imgsz": args.a1_imgsz, "stats": own_stats},
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    public_project = args.y_root / "runs_e_aligned" / "public_pretrain"
    a1_project = run_root / "a1_train_test"
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
        for seed in args.seeds:
            name = f"a1_train_test_{tag}_schema_v2_trackfix_public_seed{seed}_img{args.a1_imgsz}_eb{args.nbs}"
            a1_run_dir = a1_project / name
            if training_complete(a1_run_dir, args.a1_epochs):
                continue
            run(
                [
                    args.python,
                    str(tools / "train_yolo_pose_aligned.py"),
                    "--data",
                    str(own_dir / "bee_yolo_pose_strict.yaml"),
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
