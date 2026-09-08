#!/usr/bin/env python3
"""Smoke-test one RouteBest stage without touching the frozen dataset files."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.train_y_route_best import load_config, train_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/y_route_best.yaml"))
    parser.add_argument("--stage", choices=["y2", "y3"], default="y2")
    parser.add_argument("--device", default="1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("/tmp/bee_y_routebest_smoke"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["training"]["device"] = args.device
    cfg["training"]["workers"] = 0
    cfg["model"]["input_size"] = args.imgsz

    if args.stage == "y2":
        train_stage(
            cfg,
            "Y2_structure",
            Path(cfg["model"]["a1_checkpoint"]),
            1,
            args.project,
            "y2_save_smoke_fixed",
            0,
            temporal=False,
            patience=1,
            model_yaml=cfg["model"]["y2_yaml"],
        )
    else:
        source = args.project / "y2_save_smoke_fixed" / "weights" / "best.pt"
        train_stage(
            cfg,
            "Y3_spatiotemporal",
            source,
            1,
            args.project,
            "y3_five_frame_smoke",
            0,
            temporal=True,
            patience=1,
            model_yaml=cfg["model"]["route_yaml"],
        )


if __name__ == "__main__":
    main()
