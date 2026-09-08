"""评估 ViTPose++、颜色头尾纠正和 Track ID 时序平滑三组实验。"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["A", "B"], required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def disk_median(channel, point, radius):
    x, y = (int(round(value)) for value in point)
    radius = max(2, int(round(radius)))
    height, width = channel.shape
    x1, x2 = max(0, x - radius), min(width, x + radius + 1)
    y1, y2 = max(0, y - radius), min(height, y + radius + 1)
    yy, xx = np.ogrid[y1 - y : y2 - y, x1 - x : x2 - x]
    mask = xx * xx + yy * yy <= radius * radius
    values = channel[y1:y2, x1:x2][mask]
    return float(np.median(values))


def color_evidence(lightness, bbox, head, tail):
    """正值表示当前 head 比 tail 更暗，符合蜜蜂颜色先验。"""
    x, y, width, height = bbox
    x1, x2 = max(0, int(x)), min(lightness.shape[1], int(x + width) + 1)
    y1, y2 = max(0, int(y)), min(lightness.shape[0], int(y + height) + 1)
    crop = lightness[y1:y2, x1:x2]
    q25, q75 = np.percentile(crop, [25, 75])
    contrast_scale = max(float(q75 - q25), 10.0)
    radius = np.clip(math.hypot(width, height) * 0.07, 2, 6)
    head_lightness = disk_median(lightness, head, radius)
    tail_lightness = disk_median(lightness, tail, radius)
    return (tail_lightness - head_lightness) / contrast_scale


def group_annotations(dataset):
    grouped = defaultdict(list)
    for annotation in dataset["annotations"]:
        grouped[annotation["image_id"]].append(annotation)
    return grouped


def image_path(record, image_root):
    path = Path(record["file_name"])
    return path if path.is_absolute() else image_root / path


def calibrate_color_rule(dataset, image_root):
    images = {item["id"]: item for item in dataset["images"]}
    evidence = []
    for image_id, annotations in group_annotations(dataset).items():
        image = read_image(image_path(images[image_id], image_root))
        lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        for annotation in annotations:
            keypoints = annotation["keypoints"]
            evidence.append(
                color_evidence(
                    lightness,
                    annotation["bbox"],
                    keypoints[0:2],
                    keypoints[3:5],
                )
            )

    evidence = np.asarray(evidence, dtype=float)
    # 用训练集的 5% 分位控制误交换率；证据很弱时至少要求 0.15。
    swap_threshold = max(0.15, -float(np.quantile(evidence, 0.05)))
    return {
        "train_instances": int(evidence.size),
        "tail_brighter_rate": round(float(np.mean(evidence > 0)), 6),
        "median_evidence": round(float(np.median(evidence)), 6),
        "swap_threshold": round(swap_threshold, 6),
    }


def model_predictions(model_results):
    output = {}
    for item in model_results["predictions"]:
        keypoints = item["pred_keypoints"]
        output[item["annotation_id"]] = {
            "points": [keypoints[0], keypoints[1], keypoints[3], keypoints[4]],
            "scores": [keypoints[2], keypoints[5]],
            "color_evidence": None,
            "color_swapped": False,
            "color_swap_rejected_by_track": False,
            "temporal_swapped": False,
            "temporal_smoothed": False,
        }
    return output


def clone_predictions(predictions):
    return {
        annotation_id: {
            key: list(value) if isinstance(value, list) else value
            for key, value in prediction.items()
        }
        for annotation_id, prediction in predictions.items()
    }


def apply_color_correction(dataset, predictions, threshold, image_root):
    images = {item["id"]: item for item in dataset["images"]}
    corrected = clone_predictions(predictions)
    swap_count = 0
    for image_id, annotations in group_annotations(dataset).items():
        image = read_image(image_path(images[image_id], image_root))
        lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        for annotation in annotations:
            prediction = corrected[annotation["id"]]
            points = prediction["points"]
            evidence = color_evidence(
                lightness, annotation["bbox"], points[0:2], points[2:4]
            )
            prediction["color_evidence"] = round(float(evidence), 6)
            if -evidence > threshold:
                points[0:2], points[2:4] = points[2:4], points[0:2]
                prediction["scores"][0], prediction["scores"][1] = (
                    prediction["scores"][1],
                    prediction["scores"][0],
                )
                prediction["color_swapped"] = True
                swap_count += 1
    return corrected, swap_count


def unit_vector(points):
    vector = np.asarray(points[2:4], dtype=float) - np.asarray(points[0:2], dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def relative_points(points, bbox):
    x, y, width, height = bbox
    return np.asarray(
        [
            (points[0] - x) / width,
            (points[1] - y) / height,
            (points[2] - x) / width,
            (points[3] - y) / height,
        ],
        dtype=float,
    )


def absolute_points(points, bbox):
    x, y, width, height = bbox
    return [
        x + points[0] * width,
        y + points[1] * height,
        x + points[2] * width,
        y + points[3] * height,
    ]


def apply_track_postprocessing(dataset, predictions):
    corrected = clone_predictions(predictions)
    images = {item["id"]: item for item in dataset["images"]}
    tracks = defaultdict(list)
    for annotation in dataset["annotations"]:
        image = images[annotation["image_id"]]
        segment = str(Path(image.get("source_json", image["file_name"])).parent)
        key = (image.get("source", ""), segment, str(annotation.get("track_id")))
        tracks[key].append(annotation)

    temporal_swaps = 0
    smoothed_instances = 0
    for annotations in tracks.values():
        annotations.sort(key=lambda item: images[item["image_id"]].get("frame", item["image_id"]))
        if len(annotations) < 3:
            continue

        # 颜色交换先接受 Track ID 的方向一致性复核；若与相邻帧相反则撤销。
        initial_directions = [
            unit_vector(corrected[item["id"]]["points"]) for item in annotations
        ]
        rejected_color_indices = []
        for index, annotation in enumerate(annotations):
            prediction = corrected[annotation["id"]]
            if not prediction["color_swapped"]:
                continue
            neighbors = []
            if index > 0:
                neighbors.append(initial_directions[index - 1])
            if index + 1 < len(annotations):
                neighbors.append(initial_directions[index + 1])
            if neighbors and float(
                np.mean([np.dot(initial_directions[index], item) for item in neighbors])
            ) < 0:
                rejected_color_indices.append(index)

        for index in rejected_color_indices:
            prediction = corrected[annotations[index]["id"]]
            points = prediction["points"]
            points[0:2], points[2:4] = points[2:4], points[0:2]
            prediction["scores"][0], prediction["scores"][1] = (
                prediction["scores"][1],
                prediction["scores"][0],
            )
            prediction["color_swap_rejected_by_track"] = True

        directions = [unit_vector(corrected[item["id"]]["points"]) for item in annotations]
        swap_indices = []
        for index in range(1, len(annotations) - 1):
            previous_direction = directions[index - 1]
            current_direction = directions[index]
            next_direction = directions[index + 1]
            if float(np.dot(previous_direction, next_direction)) <= 0.5:
                continue
            reference = previous_direction + next_direction
            reference /= max(float(np.linalg.norm(reference)), 1e-9)
            if float(np.dot(current_direction, reference)) < -0.15:
                swap_indices.append(index)

        for index in swap_indices:
            prediction = corrected[annotations[index]["id"]]
            points = prediction["points"]
            points[0:2], points[2:4] = points[2:4], points[0:2]
            prediction["scores"][0], prediction["scores"][1] = (
                prediction["scores"][1],
                prediction["scores"][0],
            )
            prediction["temporal_swapped"] = True
            temporal_swaps += 1

        relative = [
            relative_points(corrected[item["id"]]["points"], item["bbox"])
            for item in annotations
        ]
        updates = {}
        for index in range(1, len(annotations) - 1):
            # 在框内相对坐标上做三帧中值，降低检测框平移造成的影响。
            median_points = np.median(np.stack(relative[index - 1 : index + 2]), axis=0)
            # 只修正一半，保留当前帧的真实姿态变化，避免过度平滑。
            blended_points = 0.5 * relative[index] + 0.5 * median_points
            updates[index] = absolute_points(blended_points, annotations[index]["bbox"])
        for index, points in updates.items():
            prediction = corrected[annotations[index]["id"]]
            prediction["points"] = points
            prediction["temporal_smoothed"] = True
            smoothed_instances += 1

    return corrected, temporal_swaps, smoothed_instances


def angle_error(pred_head, pred_tail, gt_head, gt_tail):
    prediction = np.asarray(pred_tail) - np.asarray(pred_head)
    ground_truth = np.asarray(gt_tail) - np.asarray(gt_head)
    denominator = np.linalg.norm(prediction) * np.linalg.norm(ground_truth)
    if denominator <= 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(prediction, ground_truth) / denominator, -1, 1))
    return math.degrees(math.acos(cosine))


def evaluate(dataset, predictions):
    annotations = {item["id"]: item for item in dataset["annotations"]}
    point_errors = []
    normalized_errors = []
    orientations = []
    angles = []
    records = []
    for annotation_id, prediction in predictions.items():
        annotation = annotations[annotation_id]
        keypoints = annotation["keypoints"]
        gt_head = np.asarray(keypoints[0:2], dtype=float)
        gt_tail = np.asarray(keypoints[3:5], dtype=float)
        points = prediction["points"]
        pred_head = np.asarray(points[0:2], dtype=float)
        pred_tail = np.asarray(points[2:4], dtype=float)
        head_error = float(np.linalg.norm(pred_head - gt_head))
        tail_error = float(np.linalg.norm(pred_tail - gt_tail))
        diagonal = max(math.hypot(annotation["bbox"][2], annotation["bbox"][3]), 1.0)
        direct_error = head_error + tail_error
        swapped_error = float(
            np.linalg.norm(pred_head - gt_tail) + np.linalg.norm(pred_tail - gt_head)
        )
        angle = angle_error(pred_head, pred_tail, gt_head, gt_tail)
        point_errors.extend([head_error, tail_error])
        normalized_errors.extend([head_error / diagonal, tail_error / diagonal])
        orientations.append(direct_error <= swapped_error)
        angles.append(angle)
        records.append(
            {
                "annotation_id": annotation_id,
                "image_id": annotation["image_id"],
                "track_id": annotation.get("track_id"),
                "pred_keypoints": [
                    round(float(points[0]), 3),
                    round(float(points[1]), 3),
                    round(float(prediction["scores"][0]), 6),
                    round(float(points[2]), 3),
                    round(float(points[3]), 3),
                    round(float(prediction["scores"][1]), 6),
                ],
                "head_error_px": head_error,
                "tail_error_px": tail_error,
                "angle_error_deg": angle,
                "orientation_correct": bool(direct_error <= swapped_error),
                "color_evidence": prediction["color_evidence"],
                "color_swapped": prediction["color_swapped"],
                "color_swap_rejected_by_track": prediction[
                    "color_swap_rejected_by_track"
                ],
                "temporal_swapped": prediction["temporal_swapped"],
                "temporal_smoothed": prediction["temporal_smoothed"],
            }
        )

    point_errors = np.asarray(point_errors)
    normalized_errors = np.asarray(normalized_errors)
    angles = np.asarray(angles)
    metrics = {
        "instances": len(predictions),
        "mean_error_px": round(float(point_errors.mean()), 4),
        "median_error_px": round(float(np.median(point_errors)), 4),
        "nme_bbox_diagonal": round(float(normalized_errors.mean()), 6),
        "pck@0.05": round(float(np.mean(normalized_errors <= 0.05)), 6),
        "pck@0.10": round(float(np.mean(normalized_errors <= 0.10)), 6),
        "pck@0.20": round(float(np.mean(normalized_errors <= 0.20)), 6),
        "orientation_accuracy": round(float(np.mean(orientations)), 6),
        "mean_angle_error_deg": round(float(angles.mean()), 4),
        "median_angle_error_deg": round(float(np.median(angles)), 4),
        "angle_within_15deg": round(float(np.mean(angles <= 15)), 6),
        "angle_within_30deg": round(float(np.mean(angles <= 30)), 6),
        "angle_within_45deg": round(float(np.mean(angles <= 45)), 6),
        "angle_over_45deg": round(float(np.mean(angles > 45)), 6),
    }
    bins = {}
    for start, end in zip((0, 15, 30, 45, 90, 135), (15, 30, 45, 90, 135, 180.000001)):
        count = int(np.sum((angles >= start) & (angles < end)))
        bins[f"{start:g}-{min(end, 180):g}deg"] = {
            "count": count,
            "ratio": round(count / max(len(angles), 1), 6),
        }
    metrics["angle_error_bins"] = bins
    return metrics, records


def draw_marker(draw, point, color, marker="circle"):
    x, y = point
    radius = 4
    if marker == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    else:
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)


def make_panel(image, annotation, prediction, title, panel_size=190):
    x, y, width, height = annotation["bbox"]
    padding = max(width, height) * 0.35
    left, top = max(0, int(x - padding)), max(0, int(y - padding))
    right = min(image.width, int(x + width + padding))
    bottom = min(image.height, int(y + height + padding))
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    scale = min((panel_size - 4) / crop.width, (panel_size - 26) / crop.height)
    resized = crop.resize(
        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (panel_size, panel_size), "#15171b")
    offset_x = (panel_size - resized.width) // 2
    offset_y = 24 + (panel_size - 24 - resized.height) // 2
    panel.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill="white", font=ImageFont.load_default())

    def local(point):
        return (
            offset_x + (point[0] - left) * scale,
            offset_y + (point[1] - top) * scale,
        )

    gt = annotation["keypoints"]
    gt_head, gt_tail = local(gt[0:2]), local(gt[3:5])
    draw.line((*gt_head, *gt_tail), fill="#ffffff", width=2)
    draw_marker(draw, gt_head, "#ff4d5a")
    draw_marker(draw, gt_tail, "#39d9ff")
    if prediction is not None:
        points = prediction["points"]
        pred_head, pred_tail = local(points[0:2]), local(points[2:4])
        draw.line((*pred_head, *pred_tail), fill="#ffe45c", width=2)
        draw_marker(draw, pred_head, "#ff9b30", marker="cross")
        draw_marker(draw, pred_tail, "#75ff68", marker="cross")
    return panel


def render_comparison(dataset, experiments, output_path, sample_count, image_root):
    annotations = {item["id"]: item for item in dataset["annotations"]}
    images = {item["id"]: item for item in dataset["images"]}
    pure = experiments["experiment_1_model"]
    final = experiments["experiment_3_color_track"]
    improvements = []
    for annotation_id in pure:
        annotation = annotations[annotation_id]
        gt = annotation["keypoints"]

        def total_error(prediction):
            points = prediction["points"]
            return float(
                np.linalg.norm(np.asarray(points[0:2]) - np.asarray(gt[0:2]))
                + np.linalg.norm(np.asarray(points[2:4]) - np.asarray(gt[3:5]))
            )

        improvements.append(
            (total_error(pure[annotation_id]) - total_error(final[annotation_id]), annotation_id)
        )
    improvements.sort(reverse=True)
    selected = [item[1] for item in improvements[:sample_count]]

    titles = ["GT", "1 Model", "2 +Color", "3 +Track"]
    panel_size = 190
    canvas = Image.new(
        "RGB", (len(titles) * panel_size, len(selected) * panel_size), "#0c0e11"
    )
    cache = {}
    for row, annotation_id in enumerate(selected):
        annotation = annotations[annotation_id]
        source_path = image_path(images[annotation["image_id"]], image_root)
        if source_path not in cache:
            cache[source_path] = Image.open(source_path).convert("RGB")
        source = cache[source_path]
        row_predictions = [
            None,
            experiments["experiment_1_model"][annotation_id],
            experiments["experiment_2_color"][annotation_id],
            experiments["experiment_3_color_track"][annotation_id],
        ]
        for column, (title, prediction) in enumerate(zip(titles, row_predictions)):
            panel = make_panel(source, annotation, prediction, title, panel_size)
            canvas.paste(panel, (column * panel_size, row * panel_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def main():
    args = parse_args()
    train_dataset = load_json(args.train_annotations)
    val_dataset = load_json(args.val_annotations)
    raw_results = load_json(args.model_results)
    calibration = calibrate_color_rule(train_dataset, args.image_root)

    pure = model_predictions(raw_results)
    color, color_swaps = apply_color_correction(
        val_dataset, pure, calibration["swap_threshold"], args.image_root
    )
    color_track, temporal_swaps, smoothed_instances = apply_track_postprocessing(
        val_dataset, color
    )
    rejected_color_swaps = sum(
        item["color_swap_rejected_by_track"] for item in color_track.values()
    )
    experiments = {
        "experiment_1_model": pure,
        "experiment_2_color": color,
        "experiment_3_color_track": color_track,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scene": args.scene,
        "calibration_from_train_only": calibration,
        "operations": {
            "color_swaps": color_swaps,
            "color_swaps_rejected_by_track": rejected_color_swaps,
            "temporal_swaps": temporal_swaps,
            "temporally_smoothed_instances": smoothed_instances,
        },
        "experiments": {},
    }
    for name, predictions in experiments.items():
        metrics, records = evaluate(val_dataset, predictions)
        summary["experiments"][name] = metrics
        (args.output_dir / f"{name}_predictions.json").write_text(
            json.dumps({"metrics": metrics, "predictions": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_comparison(
        val_dataset,
        experiments,
        args.output_dir / "three_experiments_comparison.jpg",
        args.samples,
        args.image_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
