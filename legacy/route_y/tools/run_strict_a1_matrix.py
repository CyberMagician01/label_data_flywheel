#!/usr/bin/env python3
"""Launch strict Y-Aligned A1 training jobs sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 3407, 827])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", type=Path, required=True)
    return parser.parse_args()


def model_tag(model_path: str) -> str:
    stem = Path(model_path).stem
    return stem.replace("-pose", "").replace("_", "-")


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for seed in args.seeds:
            name = f"strict_a1_{model_tag(model)}_seed{seed}_img{args.imgsz}_eb{args.nbs}"
            cmd = [
                args.python,
                str(args.train_script),
                "--data",
                str(args.data),
                "--model",
                model,
                "--project",
                str(args.project),
                "--name",
                name,
                "--epochs",
                str(args.epochs),
                "--batch",
                str(args.batch),
                "--nbs",
                str(args.nbs),
                "--imgsz",
                str(args.imgsz),
                "--device",
                args.device,
                "--workers",
                str(args.workers),
                "--seed",
                str(seed),
            ]
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
