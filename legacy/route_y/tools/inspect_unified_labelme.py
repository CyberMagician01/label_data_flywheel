#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/bee26/datasets/bee_e_y_unified_20260901/views/Y/labelme_5_sections"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = Counter()
    shape_types = Counter()
    grouped = Counter()
    examples = {}
    paths = []
    for dirpath, _, filenames in os.walk(args.root, followlinks=True):
        for filename in filenames:
            if filename.endswith(".json"):
                paths.append(Path(dirpath) / filename)
    for path in paths:
        if ".bak" in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))
            stype = str(shape.get("shape_type", ""))
            labels[label] += 1
            shape_types[stype] += 1
            if shape.get("group_id") is not None:
                grouped[(label, stype)] += 1
            examples.setdefault((label, stype), str(path))
    print("labels")
    for key, value in labels.most_common(80):
        print(f"{key}\t{value}")
    print("shape_types")
    for key, value in shape_types.most_common():
        print(f"{key}\t{value}")
    print("grouped")
    for (label, stype), value in grouped.most_common(80):
        print(f"{label}\t{stype}\t{value}\t{examples.get((label, stype), '')}")


if __name__ == "__main__":
    main()
