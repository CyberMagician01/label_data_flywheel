import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


SOURCE_ROOT = Path(
    "/root/autodl-tmp/datasets/bee_e_y_unified_20260901/canonical/labelme_5_sections"
)
OUTPUT_ROOT = Path("/root/autodl-tmp/datasets/scene_B_annotators_01_02_03_sliced_20260905")
ALLOWED_ANNOTATORS = {"标注员_01", "标注员_02", "标注员_03"}
ALLOWED_VIDEOS = {"B-5-1", "B-5-2", "B-5-3", "B-5-4"}
HOLDOUT_POSITIONS = (2, 7, 12)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sliding_starts(length, window=640, overlap=0.20):
    if length <= window:
        return [0]
    stride = int(window * (1.0 - overlap))
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return sorted(set(starts))


def get_bbox(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def intersection(box1, box2):
    width = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    height = max(0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    return width * height


def resolve_image(json_path, annotation):
    image_path = annotation.get("imagePath")
    candidates = []
    if image_path:
        candidates.append(json_path.parent / image_path)
    candidates.extend([json_path.with_suffix(".jpg"), json_path.with_suffix(".png")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def split_for_position(video_id, position):
    if position not in HOLDOUT_POSITIONS:
        return "train"
    video_number = int(video_id.rsplit("-", 1)[-1])
    if video_number % 2:
        return "calibration" if position == 7 else "test"
    return "test" if position == 7 else "calibration"


def collect_samples():
    samples = []
    section_counts = Counter()
    for section in sorted(SOURCE_ROOT.iterdir(), key=lambda path: path.name):
        video_id = section.name.split("_区段_", 1)[0]
        if video_id not in ALLOWED_VIDEOS:
            continue
        resolved = section.resolve()
        annotator = next(
            (name for name in ALLOWED_ANNOTATORS if name in resolved.parts), None
        )
        if annotator is None:
            continue

        frame_items = []
        for json_path in sorted(section.glob("*.json"), key=lambda path: path.name):
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
            image_path = resolve_image(json_path, annotation)
            if image_path is not None:
                frame_items.append((json_path, image_path, annotation))
        if len(frame_items) != 15:
            raise RuntimeError(f"{section.name} 预期15帧，实际{len(frame_items)}帧")

        for position, (json_path, image_path, annotation) in enumerate(frame_items):
            split = split_for_position(video_id, position)
            samples.append(
                {
                    "annotator": annotator,
                    "video_id": video_id,
                    "section_id": section.name,
                    "position": position,
                    "split": split,
                    "json_path": json_path,
                    "image_path": image_path,
                    "annotation": annotation,
                }
            )
            section_counts[(annotator, video_id, split)] += 1

    for annotator in sorted(ALLOWED_ANNOTATORS):
        for video_id in sorted(ALLOWED_VIDEOS):
            counts = [
                section_counts[(annotator, video_id, split)]
                for split in ("train", "calibration", "test")
            ]
            expected = [12, 1, 2] if int(video_id[-1]) % 2 else [12, 2, 1]
            if counts != expected:
                raise RuntimeError(
                    f"{annotator}/{video_id} 划分错误: {counts} != {expected}"
                )

    totals = Counter((sample["annotator"], sample["split"]) for sample in samples)
    for annotator in ALLOWED_ANNOTATORS:
        counts = [totals[(annotator, split)] for split in ("train", "calibration", "test")]
        if counts != [48, 6, 6]:
            raise RuntimeError(f"{annotator} 划分不是48/6/6: {counts}")
    if len(samples) != 180:
        raise RuntimeError(f"总样本不是180: {len(samples)}")
    return samples


def extract_boxes(annotation):
    boxes = []
    for shape in annotation.get("shapes", []):
        if str(shape.get("label", "")).lower() not in {"bee", "0"}:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xmin, ymin, xmax, ymax = get_bbox(points)
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def slice_sample(sample, images_dir, labels_dir, slice_size=640, overlap=0.20):
    image = cv2.imread(str(sample["image_path"]))
    if image is None:
        raise RuntimeError(f"无法读取图像: {sample['image_path']}")
    image_height, image_width = image.shape[:2]
    boxes = extract_boxes(sample["annotation"])
    count = 0
    for y_start in sliding_starts(image_height, slice_size, overlap):
        for x_start in sliding_starts(image_width, slice_size, overlap):
            x_end = min(x_start + slice_size, image_width)
            y_end = min(y_start + slice_size, image_height)
            crop = image[y_start:y_end, x_start:x_end]
            if crop.shape[:2] != (slice_size, slice_size):
                crop = cv2.copyMakeBorder(
                    crop,
                    0,
                    slice_size - crop.shape[0],
                    0,
                    slice_size - crop.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            labels = []
            tile = (x_start, y_start, x_end, y_end)
            for box in boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if intersection(box, tile) / area < 0.30:
                    continue
                xmin = max(box[0], x_start) - x_start
                ymin = max(box[1], y_start) - y_start
                xmax = min(box[2], x_end) - x_start
                ymax = min(box[3], y_end) - y_start
                labels.append(
                    (
                        0,
                        (xmin + xmax) / (2 * slice_size),
                        (ymin + ymax) / (2 * slice_size),
                        (xmax - xmin) / slice_size,
                        (ymax - ymin) / slice_size,
                    )
                )

            stem = sample["image_path"].stem
            tile_name = (
                f"{sample['annotator']}_{sample['section_id']}_{stem}_"
                f"{x_start}_{y_start}_{x_end}_{y_end}"
            )
            image_out = images_dir / f"{tile_name}.jpg"
            label_out = labels_dir / f"{tile_name}.txt"
            if not cv2.imwrite(str(image_out), crop):
                raise RuntimeError(f"切片写入失败: {image_out}")
            label_out.write_text(
                "".join(
                    f"{cls} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}\n"
                    for cls, cx, cy, width, height in labels
                ),
                encoding="utf-8",
            )
            count += 1
    return count, len(boxes)


def main():
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空，拒绝覆盖: {OUTPUT_ROOT}")

    samples = collect_samples()
    image_hash_splits = defaultdict(set)
    manifest_rows = []
    for split in ("train", "calibration", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    tile_counts = Counter()
    box_counts = Counter()
    for sample in tqdm(samples, desc="切片 Scene B 标注员01/02/03"):
        split = sample["split"]
        digest = sha256(sample["image_path"])
        image_hash_splits[digest].add(split)
        tiles, boxes = slice_sample(
            sample,
            OUTPUT_ROOT / "images" / split,
            OUTPUT_ROOT / "labels" / split,
        )
        tile_counts[split] += tiles
        box_counts[split] += boxes
        manifest_rows.append(
            {
                "annotator": sample["annotator"],
                "video_id": sample["video_id"],
                "section_id": sample["section_id"],
                "position": sample["position"],
                "split": split,
                "image": str(sample["image_path"]),
                "annotation": str(sample["json_path"]),
                "image_sha256": digest,
                "source_box_count": boxes,
                "tile_count": tiles,
            }
        )

    leaked_hashes = [
        digest for digest, splits in image_hash_splits.items() if len(splits) > 1
    ]
    if leaked_hashes:
        raise RuntimeError(f"发现跨集合精确重复图像: {len(leaked_hashes)}")

    summary = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "annotators": sorted(ALLOWED_ANNOTATORS),
        "videos": sorted(ALLOWED_VIDEOS),
        "full_images": dict(Counter(sample["split"] for sample in samples)),
        "tiles": dict(tile_counts),
        "source_boxes": dict(box_counts),
        "exact_duplicate_cross_split": 0,
        "split_policy": "每区段15帧中12 train；位置2/7/12按B序号奇偶交替分配calibration/test",
        "samples": manifest_rows,
    }
    (OUTPUT_ROOT / "split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("full_images", "tiles", "source_boxes")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
