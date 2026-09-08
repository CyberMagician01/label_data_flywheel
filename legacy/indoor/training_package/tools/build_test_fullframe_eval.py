import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    test_rows = [row for row in manifest["samples"] if row["split"] == "test"]
    wanted = {
        (row["annotator"].replace("标注员_", "annotator_"), Path(row["image"]).name): row
        for row in test_rows
    }
    if len(wanted) != 18:
        raise RuntimeError(f"预期18张测试图，实际{len(wanted)}")

    selected = []
    categories = None
    for annotation_path in args.annotations:
        dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = categories or dataset.get("categories", [])
        anns_by_image = defaultdict(list)
        for ann in dataset["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)
        for image in dataset["images"]:
            key = (image["source"], Path(image["file_name"]).name)
            if key in wanted:
                selected.append((image, anns_by_image[image["id"]], wanted[key]))

    found = {(image["source"], Path(image["file_name"]).name) for image, _, _ in selected}
    missing = sorted(set(wanted) - found)
    if missing or len(selected) != 18:
        raise RuntimeError(f"测试图映射失败: selected={len(selected)}, missing={missing}")

    images_out, annotations_out = [], []
    annotation_id = 1
    for image_id, (image, annotations, row) in enumerate(
        sorted(selected, key=lambda item: (item[0]["source"], item[0]["file_name"])), 1
    ):
        copied_image = dict(image)
        copied_image["id"] = image_id
        images_out.append(copied_image)
        for annotation in annotations:
            copied_annotation = dict(annotation)
            copied_annotation["id"] = annotation_id
            copied_annotation["image_id"] = image_id
            annotations_out.append(copied_annotation)
            annotation_id += 1

        stem = Path(image["file_name"]).stem
        video = stem.split("_frame_", 1)[0]
        frame_number = int(stem.rsplit("_", 1)[-1])
        frame_dir = args.output / "frames" / video
        frame_dir.mkdir(parents=True, exist_ok=True)
        link = frame_dir / f"frame_{frame_number:08d}.jpg"
        os.symlink(Path(row["image"]).resolve(), link)

    annotations_dir = args.output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    output_json = annotations_dir / "test_annotators_01_02_03.json"
    output_json.write_text(
        json.dumps(
            {
                "images": images_out,
                "annotations": annotations_out,
                "categories": categories,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "images": len(images_out),
        "annotations": len(annotations_out),
        "sources": sorted({image["source"] for image in images_out}),
        "output": str(output_json),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
