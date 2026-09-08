#!/usr/bin/env python3
"""Train the BeePoseTrack-Y single-frame aligned YOLO pose baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/beeposetrack_y"))
    parser.add_argument("--name", default="y_aligned_single_frame_pose")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16, help="Nominal/effective batch size for gradient accumulation.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        task="pose",
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="MuSGD",
        lr0=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_bias_lr=0.0,
        nbs=args.nbs,
        amp=True,
        seed=args.seed,
        deterministic=True,
        max_det=768,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        degrees=10.0,
        translate=0.10,
        scale=0.30,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.5,
        flipud=0.0,
        auto_augment=None,
        patience=0,
        save=True,
        save_period=10,
        val=not args.no_val,
        exist_ok=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
