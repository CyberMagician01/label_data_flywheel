"""为关键点验证集生成可重复的局部遮挡版本。"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    randomizer = random.Random(args.seed)
    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = 0
    for image_record in dataset["images"]:
        source = Path(image_record["file_name"])
        if not source.is_absolute():
            source = args.image_root / source
        image = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8).copy()
        for annotation in by_image[image_record["id"]]:
            occluded = randomizer.random() < args.fraction
            annotation["synthetic_occlusion"] = occluded
            if not occluded:
                continue
            selected += 1
            keypoint_index = randomizer.choice((0, 3))
            center_x = float(annotation["keypoints"][keypoint_index])
            center_y = float(annotation["keypoints"][keypoint_index + 1])
            box = annotation["bbox"]
            half_width = max(2, int(box[2] * randomizer.uniform(0.12, 0.22)))
            half_height = max(2, int(box[3] * randomizer.uniform(0.12, 0.22)))
            x1, x2 = max(0, int(center_x) - half_width), min(image.shape[1], int(center_x) + half_width)
            y1, y2 = max(0, int(center_y) - half_height), min(image.shape[0], int(center_y) + half_height)
            if x2 > x1 and y2 > y1:
                surrounding = image[
                    max(0, y1 - half_height):min(image.shape[0], y2 + half_height),
                    max(0, x1 - half_width):min(image.shape[1], x2 + half_width),
                ]
                fill = np.median(surrounding.reshape(-1, 3), axis=0).astype(np.uint8)
                image[y1:y2, x1:x2] = fill

        target_name = f"{image_record['id']:06d}_{source.name}"
        Image.fromarray(image).save(image_dir / target_name, quality=92)
        image_record["file_name"] = f"images/{target_name}"

    args.output.mkdir(parents=True, exist_ok=True)
    target_json = args.output / args.annotations.name
    target_json.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    print(f"images={len(dataset['images'])} annotations={len(dataset['annotations'])} occluded={selected}")
    print(target_json)


if __name__ == "__main__":
    main()
