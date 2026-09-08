"""下载并转换 Mendeley 蜜蜂头尾关键点数据集（DOI: 10.17632/8gb9r2yhfc.6）。"""

import argparse
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


DATASET_ID = "8gb9r2yhfc"
VERSION = 6
API_ROOT = "https://data.mendeley.com/public-api"
KEYPOINTS = ["head", "abdomen_tip"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--val-hive",
        default="20230711c",
        help="按蜂箱/视频前缀隔离验证集，避免相邻帧泄漏",
    )
    return parser.parse_args()


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def find_pose_folders():
    folders = get_json(f"{API_ROOT}/datasets/{DATASET_ID}/folders/{VERSION}")
    pose = next(item for item in folders if item["name"] == "pose" and not item.get("parent_id"))
    children = {item["name"]: item for item in folders if item.get("parent_id") == pose["id"]}
    return children["images"]["id"], children["labels"]["id"]


def list_files(folder_id):
    url = f"{API_ROOT}/datasets/{DATASET_ID}/files?version={VERSION}&folder_id={folder_id}"
    return get_json(url)


def download_one(item, destination):
    target = destination / item["filename"]
    expected_size = int(item["content_details"]["size"])
    if target.exists() and target.stat().st_size == expected_size:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                item["content_details"]["download_url"],
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"下载大小不一致: {item['filename']}")
    temporary.replace(target)


def download_folder(files, destination, workers):
    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, item, destination) for item in files]
        for index, future in enumerate(futures, start=1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"下载 {destination.name}: {index}/{len(futures)}", flush=True)


def empty_coco(split):
    return {
        "info": {
            "description": f"Mendeley honey-bee head/stinger pose ({split})",
            "source": "https://data.mendeley.com/datasets/8gb9r2yhfc/6",
            "doi": "10.17632/8gb9r2yhfc.6",
            "license": "CC BY 4.0",
        },
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "bee",
                "supercategory": "insect",
                "keypoints": KEYPOINTS,
                "skeleton": [[1, 2]],
            }
        ],
    }


def convert(raw_root, output_root, val_hive):
    datasets = {"train": empty_coco("train"), "val": empty_coco("val")}
    image_ids = {"train": 1, "val": 1}
    annotation_ids = {"train": 1, "val": 1}
    skipped = 0

    for image_path in sorted((raw_root / "images").glob("*.jpg")):
        label_path = raw_root / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        split = "val" if image_path.stem.startswith(val_hive) else "train"
        dataset = datasets[split]
        image_id = image_ids[split]
        with Image.open(image_path) as image:
            width, height = image.size
        dataset["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{image_path.name}",
                "width": width,
                "height": height,
                "source_hive": image_path.stem[:9],
            }
        )

        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = [float(value) for value in line.split()]
            if len(fields) != 9:
                skipped += 1
                continue
            _, cx, cy, bw, bh, hx, hy, tx, ty = fields
            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height
            w = bw * width
            h = bh * height
            head_x, head_y = hx * width, hy * height
            tail_x, tail_y = tx * width, ty * height
            if not all(0 <= value <= 1 for value in (hx, hy, tx, ty)):
                skipped += 1
                continue
            dataset["annotations"].append(
                {
                    "id": annotation_ids[split],
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "num_keypoints": 2,
                    "keypoints": [head_x, head_y, 2, tail_x, tail_y, 2],
                }
            )
            annotation_ids[split] += 1

        image_ids[split] += 1

    annotation_root = output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    for split, dataset in datasets.items():
        target = annotation_root / f"mendeley_bee_pose_{split}.json"
        target.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
        print(f"{split}: {len(dataset['images'])} 张图, {len(dataset['annotations'])} 个体")
    print(f"跳过非完整/异常标签: {skipped}")


def main():
    args = parse_args()
    raw_root = args.output / "raw"
    image_folder, label_folder = find_pose_folders()
    image_files = list_files(image_folder)
    label_files = list_files(label_folder)
    print(f"官方姿态数据: {len(image_files)} 张图, {len(label_files)} 个标签")
    download_folder(label_files, raw_root / "labels", args.workers)
    download_folder(image_files, raw_root / "images", args.workers)
    convert(raw_root, args.output, args.val_hive)


if __name__ == "__main__":
    main()
