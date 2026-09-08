"""Rewrite Count Anything annotation image paths after moving the BEE24 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_dir", type=Path)
    parser.add_argument("new_bee24_root", type=Path)
    args = parser.parse_args()

    annotations_dir = args.annotations_dir.resolve()
    new_root = args.new_bee24_root.resolve().as_posix()
    for name in ("train.json", "val.json"):
        path = annotations_dir / name
        records = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        missing = 0
        for record in records:
            image_path = Path(record["image_path"])
            if "train" not in image_path.parts:
                raise ValueError(f"Unexpected BEE24 image path: {image_path}")
            train_index = image_path.parts.index("train")
            new_path = Path(new_root, *image_path.parts[train_index:])
            record["image_path"] = new_path.as_posix()
            missing += not new_path.is_file()
            changed += 1
        if missing:
            raise FileNotFoundError(f"{name}: {missing} referenced images are missing")
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"{path.name}: rewrote {changed} paths")


if __name__ == "__main__":
    main()
