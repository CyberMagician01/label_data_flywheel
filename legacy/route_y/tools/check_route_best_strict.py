#!/usr/bin/env python3
"""Strict Route-Best implementation checks without training or editing data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import FiveFrameManifestIndex
from models.route_context import begin_route_context, get_route_context
from models.yolo26_bee_pose import register_ultralytics_modules
from tools.train_y_route_best import validate_frozen_data


def main() -> None:
    register_ultralytics_modules()
    cfg = yaml.safe_load((ROOT / "configs" / "y_route_best.yaml").read_text(encoding="utf-8"))
    stats = validate_frozen_data(cfg)
    checks = {"data": stats, "models": {}}
    for name, yaml_key, channels in (
        ("Y1_task_adaptation", "y1_yaml", 3),
        ("Y2_structure", "y2_yaml", 3),
        ("Y3_Y5_full_route_best", "route_yaml", 15),
    ):
        model = YOLO(str(cfg["model"][yaml_key]))
        x = torch.zeros(1, channels, 640, 640)
        if channels == 15:
            begin_route_context(clip=x.view(1, 5, 3, 640, 640), domain_ids=torch.tensor([1]), stats=torch.zeros(1, 6))
        with torch.no_grad():
            out = model.model(x)
        ctx = get_route_context()
        checks["models"][name] = {
            "yaml": str(cfg["model"][yaml_key]),
            "layers": len(model.model.model),
            "params": sum(p.numel() for p in model.model.parameters()),
            "input_channels": channels,
            "forward_type": type(out).__name__,
            "density_maps": 0 if ctx is None else len(ctx.density_maps),
            "route_probs": 0 if ctx is None else len(ctx.route_probs),
        }
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
