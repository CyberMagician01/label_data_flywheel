#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from models.yolo26_bee_pose import register_ultralytics_modules
from tools.train_y_route_best import load_route_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    register_ultralytics_modules()
    model = YOLO(str(args.yaml), task="pose")
    load_route_weights(model, args.source)
    model.model.args = getattr(model.model, "args", {})
    model.model.args["task"] = "pose"
    model.model.kpt_shape = [2, 3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
