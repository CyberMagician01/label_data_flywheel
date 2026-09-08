#!/usr/bin/env python3
"""Read-only gates for strict public preadaptation datasets and configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.core import YAMLConfig
from engine.data.dataset.coco_dataset import CocoDetection


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs=2, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_counts = {
        "bee24_train": (12800, 278509),
        "bee24_test": (10762, 168230),
        "public_pose_train": (620, 6753),
        "public_pose_val": (80, 476),
    }
    for name, (expected_images, expected_annotations) in expected_counts.items():
        entry = manifest["files"][name]
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{name} annotation SHA gate failed")
        dataset = json.loads(path.read_text(encoding="utf-8"))
        CocoDetection.validate_schema_dict(dataset, 2)
        if len(dataset["images"]) != expected_images:
            raise ValueError(f"{name} image count changed")
        if expected_annotations is not None and len(dataset["annotations"]) != expected_annotations:
            raise ValueError(f"{name} annotation count changed")
        if name.startswith("bee24_"):
            geometry = [
                annotation for annotation in dataset["annotations"]
                if annotation.get("track_geometry_mask", False)
            ]
            expected_geometry = manifest["reports"][name]["track_geometry_annotations"]
            if len(geometry) != expected_geometry or not geometry:
                raise ValueError(f"{name} track center/scale supervision count changed")
            if any(annotation.get("track_axis_mask", False) for annotation in geometry):
                raise ValueError(f"{name} must not invent body-axis labels without keypoints")
            if any(
                len(annotation.get("track_geometry", [])) != 6
                or not all(math.isfinite(float(value)) for value in annotation["track_geometry"])
                or any(float(value) != 0.0 for value in annotation["track_geometry"][4:])
                for annotation in geometry
            ):
                raise ValueError(f"{name} has invalid track center/scale geometry")
        root = Path(manifest["unified_root"])
        missing = [
            image["file_name"] for image in dataset["images"]
            if not (root / image["file_name"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{name} misses {len(missing)} images; first={missing[0]}")
    if any(manifest["leakage_checks"].values()):
        raise ValueError("Public train/validation leakage gate failed")
    for config in args.configs:
        YAMLConfig(str(config))
    print("ALL_PUBLIC_PREADAPTATION_PREFLIGHT_GATES_OK")


if __name__ == "__main__":
    main()
