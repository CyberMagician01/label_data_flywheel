"""为按轨迹自动复核生成首、中、尾代表帧子集。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-track", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in dataset["images"]}
    tracks = defaultdict(list)
    for annotation in dataset["annotations"]:
        image = images[annotation["image_id"]]
        track_id = annotation.get("track_id", f"annotation_{annotation['id']}")
        key = annotation.get("source"), image.get("video"), track_id
        tracks[key].append(annotation)

    selected_ids = set()
    for annotations in tracks.values():
        annotations.sort(
            key=lambda item: (
                images[item["image_id"]].get("frame", 0),
                item["id"],
            )
        )
        count = min(args.samples_per_track, len(annotations))
        for index in np.linspace(0, len(annotations) - 1, count, dtype=int):
            selected_ids.add(annotations[int(index)]["id"])

    selected_annotations = [
        annotation
        for annotation in dataset["annotations"]
        if annotation["id"] in selected_ids
    ]
    selected_image_ids = {
        annotation["image_id"] for annotation in selected_annotations
    }
    result = {
        key: value
        for key, value in dataset.items()
        if key not in {"images", "annotations"}
    }
    result["images"] = [
        image for image in dataset["images"] if image["id"] in selected_image_ids
    ]
    result["annotations"] = selected_annotations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tracks": len(tracks),
                "images": len(result["images"]),
                "annotations": len(selected_annotations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
