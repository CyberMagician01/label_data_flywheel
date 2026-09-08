#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_BEE24 = Path("/data/bee26/datasets/bee_e_y_unified_20260901/sources/BEE24")
DEFAULT_OUT = Path("/data/bee26/beeposetrack_y_20260827/datasets/bee24_detection_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bee24-root", type=Path, default=DEFAULT_BEE24)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", default=["test"])
    return parser.parse_args()


def read_seqinfo(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return dict(parser["Sequence"])


def read_gt(path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    if not path.exists():
        return by_frame
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) < 6:
            continue
        frame_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        x, y, w, h = [float(v) for v in parts[2:6]]
        mark = int(float(parts[6])) if len(parts) > 6 else 1
        class_id = int(float(parts[7])) if len(parts) > 7 else 1
        visibility = float(parts[8]) if len(parts) > 8 else 1.0
        if mark == 0 or w <= 1 or h <= 1:
            continue
        by_frame[frame_id].append(
            {
                "bbox_xyxy": [x, y, x + w, y + h],
                "class_id": 0,
                "keypoints": [],
                "visibility": [],
                "track_id": f"{path.parent.parent.name}:{track_id}",
                "source_group_id": str(track_id),
                "pairing_method": "bee24_gt_box",
                "det_mask": 1,
                "pose_mask": 0,
                "track_mask": 1,
                "quality": visibility,
                "source_dataset": "BEE24",
                "source_label": f"class_{class_id}",
            }
        )
    return by_frame


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    split_lists: dict[str, list[str]] = {split: [] for split in args.splits}
    for split in args.splits:
        split_dir = args.bee24_root / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            info_path = seq_dir / "seqinfo.ini"
            if not info_path.exists():
                continue
            info = read_seqinfo(info_path)
            img_dir = seq_dir / info.get("imdir", "img1")
            ext = info.get("imext", ".jpg")
            width = int(info.get("imwidth", 0))
            height = int(info.get("imheight", 0))
            seq_len = int(info.get("seqlength", 0))
            gt_by_frame = read_gt(seq_dir / "gt" / "gt.txt")
            for frame_id in range(1, seq_len + 1):
                image_path = img_dir / f"{frame_id:06d}{ext}"
                if not image_path.exists():
                    continue
                instances = gt_by_frame.get(frame_id, [])
                rows.append(
                    {
                        "source_image_path": str(image_path.resolve()),
                        "image_path": str(image_path.resolve()),
                        "video_id": seq_dir.name,
                        "section_id": seq_dir.name,
                        "frame_id": frame_id,
                        "domain": "RGB",
                        "split": split,
                        "width": width,
                        "height": height,
                        "instances": instances,
                        "pairing_audit": {
                            "det_instances": len(instances),
                            "pose_instances": 0,
                            "bee24_source": str(seq_dir.resolve()),
                        },
                    }
                )
                split_lists[split].append(str(image_path.resolve()))

    with (args.out_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split, images in split_lists.items():
        (args.out_dir / f"{split}_images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    summary = {
        "bee24_root": str(args.bee24_root),
        "out_dir": str(args.out_dir),
        "splits": {
            split: {
                "frames": sum(1 for row in rows if row["split"] == split),
                "det_instances": sum(len(row["instances"]) for row in rows if row["split"] == split),
                "videos": sorted({row["video_id"] for row in rows if row["split"] == split}),
            }
            for split in args.splits
        },
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
