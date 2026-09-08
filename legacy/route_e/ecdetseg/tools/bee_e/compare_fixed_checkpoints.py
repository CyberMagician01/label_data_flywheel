#!/usr/bin/env python3
"""Compare fixed-sample predictions from multiple checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _ordered_box(box: list[float]) -> list[float]:
    return [
        min(box[0], box[2]),
        min(box[1], box[3]),
        max(box[0], box[2]),
        max(box[1], box[3]),
    ]


def _iou(left: list[float], right: list[float]) -> float:
    left, right = _ordered_box(left), _ordered_box(right)
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def _classify(
    predictions: list[dict], ground_truth: list[dict], iou_threshold: float
) -> tuple[list[tuple[dict, str]], set[int], dict]:
    matched_gt: set[int] = set()
    classified: list[tuple[dict, str]] = []
    for prediction in predictions:
        overlaps = [_iou(prediction["box"], item["box"]) for item in ground_truth]
        order = sorted(range(len(overlaps)), key=overlaps.__getitem__, reverse=True)
        unmatched = next(
            (
                index
                for index in order
                if overlaps[index] >= iou_threshold and index not in matched_gt
            ),
            None,
        )
        if unmatched is not None:
            matched_gt.add(unmatched)
            status = "tp"
        elif any(
            overlaps[index] >= iou_threshold and index in matched_gt for index in order
        ):
            status = "duplicate"
        else:
            status = "fp"
        classified.append((prediction, status))
    counts = {
        "predictions": len(predictions),
        "ground_truth": len(ground_truth),
        "tp": len(matched_gt),
        "fp": sum(status == "fp" for _, status in classified),
        "duplicate": sum(status == "duplicate" for _, status in classified),
        "miss": len(ground_truth) - len(matched_gt),
    }
    counts["precision"] = counts["tp"] / len(predictions) if predictions else 0.0
    counts["recall"] = counts["tp"] / len(ground_truth) if ground_truth else 0.0
    return classified, matched_gt, counts


def _resolve_image(sample: dict, image_root: Path | None) -> Path:
    remote = Path(sample["image_path"])
    if remote.is_file():
        return remote
    if image_root is not None:
        candidate = image_root / sample["file_name"]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'image_id={sample["image_id"]} image is unavailable')


def _draw_panel(
    image: Image.Image,
    ground_truth: list[dict],
    classified: list[tuple[dict, str]],
    matched_gt: set[int],
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for index, item in enumerate(ground_truth):
        color = (0, 255, 0) if index in matched_gt else (255, 165, 0)
        draw.rectangle(_ordered_box(item["box"]), outline=color, width=2)
    prediction_colors = {
        "tp": (0, 255, 255),
        "fp": (255, 40, 40),
        "duplicate": (255, 0, 255),
    }
    for prediction, status in classified:
        box = _ordered_box(prediction["box"])
        color = prediction_colors[status]
        draw.rectangle(box, outline=color, width=2)
        draw.text(
            (box[0], max(24, box[1] - 11)),
            f'{float(prediction["score"]):.3f}',
            fill=color,
            font=font,
        )
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 255, 255), font=font)
    return panel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="LABEL=JSON",
        help="Repeat once per checkpoint, in display order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--score", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    sources: list[tuple[str, Path, dict]] = []
    for value in args.predictions:
        label, separator, file_name = value.partition("=")
        if not separator or not label:
            raise ValueError(f"Invalid --predictions value: {value}")
        path = Path(file_name)
        sources.append((label, path, json.loads(path.read_text(encoding="utf-8"))))

    reference_ids = [int(item["image_id"]) for item in sources[0][2]["samples"]]
    for label, _, source in sources[1:]:
        current = [int(item["image_id"]) for item in source["samples"]]
        if current != reference_ids:
            raise ValueError(f"Sample order mismatch for {label}: {current}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_maps = [
        {int(sample["image_id"]): sample for sample in source["samples"]}
        for _, _, source in sources
    ]
    report = {
        "policy": {
            "score_threshold": args.score,
            "display_topk": args.topk,
            "match_iou_threshold": args.iou,
            "ordinary_nms": False,
            "scene_adaptive_postprocess": False,
            "weights": "EMA",
            "legend": {
                "matched_gt": "green",
                "missed_gt": "orange",
                "tp_prediction": "cyan",
                "fp_prediction": "red",
                "duplicate_prediction": "magenta",
            },
        },
        "checkpoints": [],
        "samples": [],
    }
    totals = {
        label: {
            key: 0
            for key in ("predictions", "ground_truth", "tp", "fp", "duplicate", "miss")
        }
        for label, _, _ in sources
    }
    for label, path, source in sources:
        checkpoint = Path(source["checkpoint"])
        report["checkpoints"].append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
                "predictions_json": str(path.resolve()),
                "predictions_sha256": _sha256(path),
            }
        )

    thumbnails: list[Image.Image] = []
    for image_id in reference_ids:
        reference = sample_maps[0][image_id]
        image = Image.open(_resolve_image(reference, args.image_root)).convert("RGB")
        panels: list[Image.Image] = []
        item_report = {
            "image_id": image_id,
            "file_name": reference["file_name"],
            "ground_truth": len(reference["gt"]),
            "checkpoints": {},
        }
        for (label, _, _), sample_map in zip(sources, sample_maps):
            sample = sample_map[image_id]
            if sample["gt"] != reference["gt"]:
                raise ValueError(f"Ground truth mismatch for {label}, image_id={image_id}")
            predictions = sorted(
                (
                    item
                    for item in sample["predictions"]
                    if float(item["score"]) >= args.score
                ),
                key=lambda item: float(item["score"]),
                reverse=True,
            )[: args.topk]
            classified, matched_gt, metrics = _classify(
                predictions, sample["gt"], args.iou
            )
            item_report["checkpoints"][label] = metrics
            for key in totals[label]:
                totals[label][key] += metrics[key]
            title = (
                f"{label} id={image_id} pred={metrics['predictions']} "
                f"TP={metrics['tp']} FP={metrics['fp']} "
                f"DUP={metrics['duplicate']} MISS={metrics['miss']}"
            )
            panels.append(
                _draw_panel(image, sample["gt"], classified, matched_gt, title)
            )
        comparison = Image.new("RGB", (sum(panel.width for panel in panels), image.height))
        offset = 0
        for panel in panels:
            comparison.paste(panel, (offset, 0))
            offset += panel.width
        output_path = args.output_dir / f"image_{image_id}_checkpoint_comparison.jpg"
        comparison.save(output_path, quality=92)
        item_report["output"] = str(output_path.resolve())
        report["samples"].append(item_report)
        thumbnail = comparison.copy()
        thumbnail.thumbnail((1500, 500))
        thumbnails.append(thumbnail)

    for label, total in totals.items():
        total["precision"] = total["tp"] / total["predictions"] if total["predictions"] else 0.0
        total["recall"] = total["tp"] / total["ground_truth"] if total["ground_truth"] else 0.0
    report["totals"] = totals
    report_path = args.output_dir / "checkpoint_comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    width = max(item.width for item in thumbnails)
    height = sum(item.height for item in thumbnails)
    contact_sheet = Image.new("RGB", (width, height), (25, 25, 25))
    offset = 0
    for item in thumbnails:
        contact_sheet.paste(item, (0, offset))
        offset += item.height
    contact_sheet.save(args.output_dir / "checkpoint_comparison_contact_sheet.jpg", quality=90)
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
