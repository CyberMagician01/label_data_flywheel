"""用高质量标注训练出的模型自动筛选并纠正低质量标注。"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq-train", type=Path, required=True)
    parser.add_argument("--lq-train", type=Path, required=True)
    parser.add_argument("--hq-evaluation", type=Path, required=True)
    parser.add_argument("--lq-evaluation", type=Path, required=True)
    parser.add_argument("--output-lq", type=Path, required=True)
    parser.add_argument("--output-combined", type=Path, required=True)
    parser.add_argument("--track-propagation", action="store_true")
    return parser.parse_args()


def angle_error(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-9:
        return 180.0
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(math.degrees(math.acos(cosine)))


def merge_datasets(hq, filtered_lq):
    result = {
        key: value
        for key, value in hq.items()
        if key not in {"images", "annotations"}
    }
    images = {image["id"]: image for image in hq["images"]}
    images.update({image["id"]: image for image in filtered_lq["images"]})
    annotations = {annotation["id"]: annotation for annotation in hq["annotations"]}
    annotations.update(
        {annotation["id"]: annotation for annotation in filtered_lq["annotations"]}
    )
    result["images"] = sorted(images.values(), key=lambda item: item["id"])
    result["annotations"] = sorted(
        annotations.values(), key=lambda item: item["id"]
    )
    return result


def track_key(annotation, image):
    track_id = annotation.get("track_id")
    if track_id is None:
        track_id = f"annotation_{annotation['id']}"
    return annotation.get("source"), image.get("video"), track_id


def measure_annotation(annotation, record):
    pred = record["pred_keypoints"]
    label = annotation["keypoints"]
    bbox = annotation["bbox"]
    diagonal = max(math.hypot(bbox[2], bbox[3]), 1.0)
    pred_head = np.asarray(pred[0:2], dtype=float)
    pred_tail = np.asarray(pred[3:5], dtype=float)
    label_head = np.asarray(label[0:2], dtype=float)
    label_tail = np.asarray(label[3:5], dtype=float)
    pred_vector = pred_tail - pred_head
    label_vector = label_tail - label_head

    direct_nme = (
        np.linalg.norm(pred_head - label_head)
        + np.linalg.norm(pred_tail - label_tail)
    ) / (2 * diagonal)
    swapped_nme = (
        np.linalg.norm(pred_head - label_tail)
        + np.linalg.norm(pred_tail - label_head)
    ) / (2 * diagonal)
    return {
        "direct_nme": float(direct_nme),
        "swapped_nme": float(swapped_nme),
        "direct_angle": angle_error(pred_vector, label_vector),
        "swapped_angle": angle_error(pred_vector, -label_vector),
        "confidence": float(min(pred[2], pred[5])),
    }


def main():
    args = parse_args()
    hq_train = json.loads(args.hq_train.read_text(encoding="utf-8"))
    lq_train = json.loads(args.lq_train.read_text(encoding="utf-8"))
    hq_eval = json.loads(args.hq_evaluation.read_text(encoding="utf-8"))
    lq_eval = json.loads(args.lq_evaluation.read_text(encoding="utf-8"))

    hq_records = [
        record
        for record in hq_eval["predictions"]
        if record.get("source")
        in {"annotator_01", "annotator_02", "annotator_03"}
    ]
    hq_nme = [
        (record["head_error_norm"] + record["tail_error_norm"]) / 2
        for record in hq_records
    ]
    hq_angles = [record["angle_error_deg"] for record in hq_records]
    hq_confidence = [
        min(record["pred_keypoints"][2], record["pred_keypoints"][5])
        for record in hq_records
    ]
    nme_threshold = min(0.25, float(np.quantile(hq_nme, 0.90)))
    angle_threshold = min(60.0, max(30.0, float(np.quantile(hq_angles, 0.90))))
    confidence_threshold = max(0.20, float(np.quantile(hq_confidence, 0.10)))

    predictions = {
        record["annotation_id"]: record for record in lq_eval["predictions"]
    }
    images = {image["id"]: image for image in lq_train["images"]}
    source_annotations = {
        annotation["id"]: annotation for annotation in lq_train["annotations"]
    }
    kept_annotations = []
    counts = {"kept": 0, "swapped": 0, "rejected": 0}
    track_counts = {"kept": 0, "swapped": 0, "rejected": 0}

    track_decisions = {}
    if args.track_propagation:
        track_measurements = defaultdict(list)
        for annotation_id, record in predictions.items():
            annotation = source_annotations[annotation_id]
            key = track_key(annotation, images[annotation["image_id"]])
            track_measurements[key].append(measure_annotation(annotation, record))

        for key, measurements in track_measurements.items():
            direct_nme = float(np.median([item["direct_nme"] for item in measurements]))
            swapped_nme = float(np.median([item["swapped_nme"] for item in measurements]))
            use_swapped = swapped_nme < direct_nme
            angle_name = "swapped_angle" if use_swapped else "direct_angle"
            chosen_angle = float(np.median([item[angle_name] for item in measurements]))
            confidence = float(np.median([item["confidence"] for item in measurements]))
            chosen_nme = min(direct_nme, swapped_nme)
            rejected = (
                confidence < confidence_threshold
                or chosen_nme > nme_threshold
                or chosen_angle > angle_threshold
            )
            action = "rejected" if rejected else ("swapped" if use_swapped else "kept")
            track_counts[action] += 1
            track_decisions[key] = {
                "action": action,
                "nme": chosen_nme,
                "angle": chosen_angle,
                "confidence": confidence,
            }

    for source_annotation in lq_train["annotations"]:
        annotation = dict(source_annotation)
        label = annotation["keypoints"]
        if args.track_propagation:
            key = track_key(annotation, images[annotation["image_id"]])
            decision = track_decisions.get(key, {"action": "rejected"})
            action = decision["action"]
            chosen_nme = decision.get("nme", 1.0)
            chosen_angle = decision.get("angle", 180.0)
            confidence = decision.get("confidence", 0.0)
        else:
            measurement = measure_annotation(annotation, predictions[annotation["id"]])
            use_swapped = measurement["swapped_nme"] < measurement["direct_nme"]
            chosen_nme = min(measurement["direct_nme"], measurement["swapped_nme"])
            chosen_angle = measurement["swapped_angle" if use_swapped else "direct_angle"]
            confidence = measurement["confidence"]
            rejected = (
                confidence < confidence_threshold
                or chosen_nme > nme_threshold
                or chosen_angle > angle_threshold
            )
            action = "rejected" if rejected else ("swapped" if use_swapped else "kept")

        if action == "rejected":
            counts["rejected"] += 1
            continue

        if action == "swapped":
            annotation["keypoints"] = label[3:6] + label[0:3]
        annotation["quality_filter_action"] = action
        annotation["quality_filter_nme"] = round(chosen_nme, 6)
        annotation["quality_filter_angle_deg"] = round(chosen_angle, 4)
        annotation["quality_filter_confidence"] = round(float(confidence), 6)
        kept_annotations.append(annotation)
        counts[action] += 1

    kept_image_ids = {annotation["image_id"] for annotation in kept_annotations}
    filtered_lq = {
        key: value
        for key, value in lq_train.items()
        if key not in {"images", "annotations"}
    }
    filtered_lq["images"] = [
        image for image in lq_train["images"] if image["id"] in kept_image_ids
    ]
    filtered_lq["annotations"] = kept_annotations
    combined = merge_datasets(hq_train, filtered_lq)

    args.output_lq.parent.mkdir(parents=True, exist_ok=True)
    args.output_lq.write_text(
        json.dumps(filtered_lq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_combined.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = {
        "thresholds": {
            "mean_keypoint_nme": round(nme_threshold, 6),
            "angle_deg": round(angle_threshold, 4),
            "min_keypoint_confidence": round(confidence_threshold, 6),
        },
        "low_quality_input_annotations": len(lq_train["annotations"]),
        "reviewed_annotations": len(predictions),
        "track_propagation": args.track_propagation,
        "track_actions": track_counts if args.track_propagation else None,
        **counts,
        "filtered_annotations": len(kept_annotations),
        "combined_annotations": len(combined["annotations"]),
    }
    stats_path = args.output_lq.with_name(args.output_lq.stem + "_stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
