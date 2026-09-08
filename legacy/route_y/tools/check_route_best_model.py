#!/usr/bin/env python3
"""Build-check the Route-Best YOLO config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo26_bee_pose import register_ultralytics_modules
from tools.train_y_route_best import load_config, load_route_weights, validate_frozen_data
from ultralytics import YOLO


def main() -> None:
    register_ultralytics_modules()
    model = YOLO(str(ROOT / "configs" / "yolo26m_pose_p2_route_best.yaml"))
    print(type(model.model).__name__)
    print("layers", len(model.model.model))
    print("params", sum(p.numel() for p in model.model.parameters()))
    cfg = load_config(ROOT / "configs" / "y_route_best.yaml")
    print("data", validate_frozen_data(cfg))
    ckpt = Path(cfg["model"]["a1_checkpoint"])
    print("a1_ckpt_exists", ckpt.exists())
    load_route_weights(model, ckpt)
    print("a1_partial_load_ok")


if __name__ == "__main__":
    main()
