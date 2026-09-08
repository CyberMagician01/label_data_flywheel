"""用全部 GT 蜜蜂框评估头尾关键点，不受 COCO 每图 20 个实例上限影响。"""

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VITPOSE_ROOT = Path(
    os.environ.get("VITPOSE_ROOT", REPO_ROOT / "third_party" / "ViTPose")
)
sys.path.insert(0, str(VITPOSE_ROOT))

from mmpose.apis import init_pose_model, inference_top_down_pose_model  # noqa: E402
from mmpose.datasets import DatasetInfo  # noqa: E402
from mmpose.datasets.builder import PIPELINES  # noqa: E402


@PIPELINES.register_module()
class SetBeeDatasetIndex:
    def __init__(self, dataset_idx=0):
        self.dataset_idx = dataset_idx

    def __call__(self, results):
        results["dataset_idx"] = self.dataset_idx
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="相对 file_name 的根目录；省略时使用标注文件所在目录",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angle_error(pred_head, pred_tail, gt_head, gt_tail):
    pred = np.asarray(pred_tail) - np.asarray(pred_head)
    gt = np.asarray(gt_tail) - np.asarray(gt_head)
    denom = np.linalg.norm(pred) * np.linalg.norm(gt)
    if denom <= 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(pred, gt) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def summarize(records, elapsed, include_subgroups=True):
    if not records:
        return {"instances": 0}
    point_errors = np.asarray(
        [[record["head_error_px"], record["tail_error_px"]] for record in records]
    )
    normalized = np.asarray(
        [
            [record["head_error_norm"], record["tail_error_norm"]]
            for record in records
        ]
    )
    angles = np.asarray([record["angle_error_deg"] for record in records])
    summary = {
        "instances": len(records),
        "elapsed_seconds": round(elapsed, 3),
        "instances_per_second": round(len(records) / max(elapsed, 1e-9), 3),
        "mean_error_px": round(float(point_errors.mean()), 4),
        "median_error_px": round(float(np.median(point_errors)), 4),
        "head_mean_error_px": round(float(point_errors[:, 0].mean()), 4),
        "tail_mean_error_px": round(float(point_errors[:, 1].mean()), 4),
        "nme_bbox_diagonal": round(float(normalized.mean()), 6),
        "orientation_accuracy": round(
            float(np.mean([record["orientation_correct"] for record in records])), 6
        ),
        "mean_angle_error_deg": round(float(angles.mean()), 4),
        "median_angle_error_deg": round(float(np.median(angles)), 4),
        "angle_within_15deg": round(float(np.mean(angles <= 15)), 6),
        "angle_within_30deg": round(float(np.mean(angles <= 30)), 6),
        "angle_within_45deg": round(float(np.mean(angles <= 45)), 6),
        "angle_over_45deg": round(float(np.mean(angles > 45)), 6),
    }
    for threshold in (0.05, 0.10, 0.20):
        summary[f"pck@{threshold:.2f}"] = round(float(np.mean(normalized <= threshold)), 6)
    edges = (0, 15, 30, 45, 90, 135, 180.000001)
    summary["angle_error_bins"] = {}
    for start, end in zip(edges[:-1], edges[1:]):
        count = int(np.sum((angles >= start) & (angles < end)))
        label = f"{start:g}-{min(end, 180):g}deg"
        summary["angle_error_bins"][label] = {
            "count": count,
            "ratio": round(count / len(records), 6),
        }
    if include_subgroups:
        sources = sorted({record["source"] for record in records})
        summary["subgroups"] = {
            "high_quality": summarize(
                [
                    record
                    for record in records
                    if record["source"]
                    in {"annotator_01", "annotator_02", "annotator_03"}
                ],
                elapsed,
                False,
            ),
            "low_quality": summarize(
                [
                    record
                    for record in records
                    if record["source"] in {"annotator_04", "annotator_05"}
                ],
                elapsed,
                False,
            ),
            "by_source": {
                source: summarize(
                    [record for record in records if record["source"] == source],
                    elapsed,
                    False,
                )
                for source in sources
            },
            "boundary": summarize(
                [record for record in records if record["boundary"]], elapsed, False
            ),
            "truncated": summarize(
                [record for record in records if record["truncated"]], elapsed, False
            ),
            "synthetic_occlusion": summarize(
                [record for record in records if record["synthetic_occlusion"]],
                elapsed,
                False,
            ),
        }
    return summary


def main():
    args = parse_args()
    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_root = args.image_root or args.annotations.parent
    images = {image["id"]: image for image in dataset["images"]}
    by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    model = init_pose_model(args.config, args.checkpoint, device=args.device)
    # 旧版 ViTPose 配置只在 data.test 中保存推理流水线，API 则读取顶层字段。
    if "test_pipeline" not in model.cfg:
        model.cfg.test_pipeline = copy.deepcopy(model.cfg.data.test.pipeline)
    model.cfg.test_pipeline.insert(0, dict(type="SetBeeDatasetIndex", dataset_idx=0))
    dataset_info = DatasetInfo(model.cfg.data.test.dataset_info)
    records = []
    started = time.perf_counter()

    for image_index, image_id in enumerate(sorted(by_image), start=1):
        image = images[image_id]
        annotations = sorted(by_image[image_id], key=lambda item: item["id"])
        for offset in range(0, len(annotations), args.batch_size):
            chunk = annotations[offset : offset + args.batch_size]
            person_results = [
                {"bbox": np.asarray(annotation["bbox"] + [1.0], dtype=np.float32)}
                for annotation in chunk
            ]
            image_file = Path(image["file_name"])
            if not image_file.is_absolute():
                image_file = image_root / image_file
            predictions, _ = inference_top_down_pose_model(
                model,
                str(image_file),
                person_results=person_results,
                format="xywh",
                dataset_info=dataset_info,
            )
            if len(predictions) != len(chunk):
                raise RuntimeError("预测数量与 GT 框数量不一致")

            for annotation, prediction in zip(chunk, predictions):
                keypoints = np.asarray(prediction["keypoints"], dtype=float)
                pred_head = keypoints[0, :2]
                pred_tail = keypoints[1, :2]
                gt = annotation["keypoints"]
                gt_head = np.asarray(gt[0:2], dtype=float)
                gt_tail = np.asarray(gt[3:5], dtype=float)
                bbox = annotation["bbox"]
                diagonal = max(math.hypot(bbox[2], bbox[3]), 1.0)
                boundary_margin_x = image["width"] * 0.01
                boundary_margin_y = image["height"] * 0.01
                boundary = (
                    bbox[0] <= boundary_margin_x
                    or bbox[1] <= boundary_margin_y
                    or bbox[0] + bbox[2] >= image["width"] - boundary_margin_x
                    or bbox[1] + bbox[3] >= image["height"] - boundary_margin_y
                )
                head_error = distance(pred_head, gt_head)
                tail_error = distance(pred_tail, gt_tail)
                direct = head_error + tail_error
                swapped = distance(pred_head, gt_tail) + distance(pred_tail, gt_head)
                records.append(
                    {
                        "annotation_id": annotation["id"],
                        "image_id": image_id,
                        "source": annotation.get(
                            "source", image.get("source", "unknown")
                        ),
                        "pred_keypoints": [
                            round(float(pred_head[0]), 3),
                            round(float(pred_head[1]), 3),
                            round(float(keypoints[0, 2]), 6),
                            round(float(pred_tail[0]), 3),
                            round(float(pred_tail[1]), 3),
                            round(float(keypoints[1, 2]), 6),
                        ],
                        "head_error_px": head_error,
                        "tail_error_px": tail_error,
                        "head_error_norm": head_error / diagonal,
                        "tail_error_norm": tail_error / diagonal,
                        "orientation_correct": direct <= swapped,
                        "angle_error_deg": angle_error(
                            pred_head, pred_tail, gt_head, gt_tail
                        ),
                        "boundary": bool(boundary),
                        "truncated": bool(annotation.get("truncated", False)),
                        "synthetic_occlusion": bool(
                            annotation.get("synthetic_occlusion", False)
                        ),
                    }
                )
        print(
            f"[{image_index}/{len(by_image)}] {Path(image['file_name']).name}: "
            f"{len(annotations)} instances",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "annotations": str(args.annotations.resolve()),
        "metrics": summarize(records, elapsed),
        "predictions": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
