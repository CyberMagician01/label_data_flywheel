#!/usr/bin/env python3
"""Evaluate logical outdoor query budgets and IoU dedup on a fixed monitor set."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.cocoeval import COCOeval

from engine.core import YAMLConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _supported_kwargs(model, values: dict) -> dict:
    signature = inspect.signature(model.forward)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _move(value, device):
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _greedy_metrics(records: dict[int, list[dict]], coco) -> dict:
    tp = fp = duplicate = miss = gt_total = 0
    matched_by_image = {}
    for image_id in coco.getImgIds():
        annotations = [coco.anns[index] for index in coco.getAnnIds(imgIds=[image_id])]
        gt_boxes = torch.tensor([
            [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
            for ann in annotations
        ], dtype=torch.float32).reshape(-1, 4)
        gt_total += len(gt_boxes)
        claimed = set()
        predictions = sorted(records.get(int(image_id), []), key=lambda item: item["score"], reverse=True)
        for prediction in predictions:
            if not len(gt_boxes):
                fp += 1
                continue
            box = torch.tensor(prediction["box_xyxy"], dtype=torch.float32).reshape(1, 4)
            ious = torchvision.ops.box_iou(box, gt_boxes)[0]
            best_iou, best_index = ious.max(dim=0)
            if float(best_iou) < 0.5:
                fp += 1
            elif int(best_index) in claimed:
                duplicate += 1
            else:
                tp += 1
                claimed.add(int(best_index))
        miss += len(gt_boxes) - len(claimed)
        matched_by_image[int(image_id)] = claimed
    denominator = tp + fp + duplicate
    return {
        "predictions": denominator,
        "gt": gt_total,
        "tp": tp,
        "fp": fp,
        "duplicate": duplicate,
        "miss": miss,
        "precision": tp / denominator if denominator else 0.0,
        "recall": tp / gt_total if gt_total else 0.0,
        "matched_gt_by_image": {str(key): sorted(value) for key, value in matched_by_image.items()},
    }


def _coco_ap50(records: dict[int, list[dict]], coco) -> dict:
    detections = []
    category_id = int(coco.getCatIds()[0])
    for image_id, predictions in records.items():
        for prediction in predictions:
            detections.append({
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": _xyxy_to_xywh(prediction["box_xyxy"]),
                "score": float(prediction["score"]),
            })
    if not detections:
        return {"ap50": 0.0, "ar50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco.loadRes(detections)
        evaluator = COCOeval(coco, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(int(value) for value in coco.getImgIds())
        evaluator.params.catIds = [category_id]
        evaluator.params.iouThrs = np.array([0.5])
        evaluator.params.maxDets = [1, 10, 768]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][0, :, :, 0, 2]
    recall = evaluator.eval["recall"][0, :, 0, 2]
    precision = precision[precision > -1]
    recall = recall[recall > -1]
    return {
        "ap50": float(precision.mean()) if precision.size else 0.0,
        "ar50": float(recall.mean()) if recall.size else 0.0,
    }


def _at_risk_gt(raw_records, nms_records, coco) -> int:
    risk = 0
    for image_id in coco.getImgIds():
        annotations = [coco.anns[index] for index in coco.getAnnIds(imgIds=[image_id])]
        gt_boxes = torch.tensor([
            [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
            for ann in annotations
        ], dtype=torch.float32).reshape(-1, 4)
        if not len(gt_boxes):
            continue
        raw = raw_records.get(int(image_id), [])
        kept = nms_records.get(int(image_id), [])
        kept_queries = {item["query_index"] for item in kept}
        kept_covered = set()
        for item in kept:
            ious = torchvision.ops.box_iou(
                torch.tensor(item["box_xyxy"]).reshape(1, 4), gt_boxes
            )[0]
            if float(ious.max()) >= 0.5:
                kept_covered.add(int(ious.argmax()))
        risky = set()
        for item in raw:
            if item["query_index"] in kept_queries:
                continue
            ious = torchvision.ops.box_iou(
                torch.tensor(item["box_xyxy"]).reshape(1, 4), gt_boxes
            )[0]
            if float(ious.max()) >= 0.5 and int(ious.argmax()) not in kept_covered:
                risky.add(int(ious.argmax()))
        risk += len(risky)
    return risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--visual-image-ids", type=int, nargs="+", default=[41, 774, 6774, 2036])
    parser.add_argument("--image-ids", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    data_loader = cfg.val_dataloader
    dataset = data_loader.dataset
    if args.image_ids:
        id_to_index = {int(image_id): index for index, image_id in enumerate(dataset.ids)}
        missing = sorted(set(args.image_ids) - set(id_to_index))
        if missing:
            raise ValueError(f"Missing requested image ids: {missing}")
        subset = torch.utils.data.Subset(
            dataset, [id_to_index[int(image_id)] for image_id in args.image_ids],
        )
        data_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=data_loader.collate_fn,
        )
    elif len(dataset) != 512:
        raise ValueError(f"Expected fixed monitor512 dataset, got {len(dataset)} images")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = cfg.model
    model.load_state_dict(checkpoint["ema"]["module"], strict=True)
    model.to(device).eval()
    decoder = model.decoder
    if not decoder.enable_monotonic_query_capacity:
        raise RuntimeError("This test requires the real decoder query-valid capacity path")

    postprocessor = cfg.postprocessor.to(device).eval()
    postprocessor.enable_calibrated_filter = True
    postprocessor.score_exponents_by_domain = {
        "rgb": [1.0, 1.0, 0.0], "ir": [1.0, 1.0, 0.0],
    }
    postprocessor.score_thresholds_by_domain = {
        "rgb": args.score_threshold, "ir": args.score_threshold,
    }
    postprocessor.enable_set_dedup = False

    records = {
        budget: {"raw": defaultdict(list), "nms": defaultdict(list)}
        for budget in args.budgets
    }
    active_counts = {budget: [] for budget in args.budgets}
    inference_seconds = {budget: 0.0 for budget in args.budgets}
    visual_ids = set(args.visual_image_ids)
    visual = {str(budget): {} for budget in args.budgets}

    with torch.inference_mode():
        for step, (samples, targets) in enumerate(data_loader):
            samples = samples.to(device)
            targets = [
                {key: _move(value, device) for key, value in target.items()}
                for target in targets
            ]
            route = {
                "domain_id": torch.stack([target["domain_id"].reshape(-1)[0] for target in targets]),
                "scene_id": torch.stack([target["scene_id"].reshape(-1)[0] for target in targets]),
            }
            if all("temporal_valid_mask" in target for target in targets):
                route["temporal_valid_mask"] = torch.stack([
                    target["temporal_valid_mask"] for target in targets
                ])
            if all("stabilization_theta" in target for target in targets):
                route["stabilization_theta"] = torch.stack([
                    target["stabilization_theta"] for target in targets
                ])
            has_letterbox = all("letterbox_scale" in target for target in targets)
            target_sizes = torch.stack([
                target["input_size"] if has_letterbox else target["orig_size"]
                for target in targets
            ])

            for budget in args.budgets:
                decoder.min_active_queries = int(budget)
                decoder.stage_query_limit = int(budget)
                started = time.perf_counter()
                outputs = model(samples, **_supported_kwargs(model, route))
                batch_results = postprocessor(outputs, target_sizes)
                inference_seconds[budget] += time.perf_counter() - started
                active_counts[budget].extend(
                    int(value) for value in outputs["pred_query_valid"].sum(dim=1).cpu()
                )
                if has_letterbox:
                    for result, target in zip(batch_results, targets):
                        scale = target["letterbox_scale"]
                        pad = target["letterbox_pad"]
                        boxes = result["boxes"]
                        boxes[:, 0::2] = (boxes[:, 0::2] - pad[0]) / scale
                        boxes[:, 1::2] = (boxes[:, 1::2] - pad[1]) / scale
                        original_w, original_h = target["orig_size"]
                        boxes[:, 0::2].clamp_(0, original_w)
                        boxes[:, 1::2].clamp_(0, original_h)

                for result, target in zip(batch_results, targets):
                    image_id = int(target["image_id"].item())
                    raw = []
                    for box, score, query_index in zip(
                        result["boxes"], result["scores"], result["query_indices"]
                    ):
                        raw.append({
                            "box_xyxy": [float(value) for value in box.detach().cpu()],
                            "score": float(score.detach().cpu()),
                            "query_index": int(query_index.detach().cpu()),
                        })
                    records[budget]["raw"][image_id].extend(raw)
                    if raw:
                        boxes = torch.tensor([item["box_xyxy"] for item in raw], device=device)
                        scores = torch.tensor([item["score"] for item in raw], device=device)
                        keep = torchvision.ops.nms(boxes, scores, args.nms_iou).cpu().tolist()
                        nms = [raw[index] for index in keep]
                    else:
                        nms = []
                    records[budget]["nms"][image_id].extend(nms)
                    if image_id in visual_ids:
                        image_info = dataset.coco.imgs[image_id]
                        annotations = [
                            dataset.coco.anns[index]
                            for index in dataset.coco.getAnnIds(imgIds=[image_id])
                        ]
                        visual[str(budget)][str(image_id)] = {
                            "image_path": str((Path(dataset.root) / image_info["file_name"]).resolve()),
                            "gt_xyxy": [[
                                float(ann["bbox"][0]), float(ann["bbox"][1]),
                                float(ann["bbox"][0] + ann["bbox"][2]),
                                float(ann["bbox"][1] + ann["bbox"][3]),
                            ] for ann in annotations],
                            "raw": raw,
                            "nms": nms,
                        }
            if step % 25 == 0:
                print(f"step={step}/{len(data_loader)}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_last_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": "ema.module",
        "dataset_images": len(data_loader.dataset),
        "dataset_annotation_file": str(dataset.coco.dataset.get("info", {}).get("source_annotation", "")),
        "query_tensor_count": int(decoder.num_queries),
        "query_budget_semantics": (
            "Budgets set decoder min_active_queries=stage_query_limit and therefore change "
            "pred_query_valid plus decoder key-padding/masking. Tensor shapes remain 768, so "
            "this is logical active-query reduction, not physical tensor/query pruning."
        ),
        "score_formula": "sigmoid(class) * sigmoid(quality); visibility exponent is zero",
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "budgets": {},
        "visual_samples": visual,
    }
    for budget in args.budgets:
        raw_metrics = _greedy_metrics(records[budget]["raw"], dataset.coco)
        nms_metrics = _greedy_metrics(records[budget]["nms"], dataset.coco)
        raw_metrics.update(_coco_ap50(records[budget]["raw"], dataset.coco))
        nms_metrics.update(_coco_ap50(records[budget]["nms"], dataset.coco))
        raw_metrics.pop("matched_gt_by_image", None)
        nms_metrics.pop("matched_gt_by_image", None)
        report["budgets"][str(budget)] = {
            "active_query_min": min(active_counts[budget]),
            "active_query_max": max(active_counts[budget]),
            "active_query_mean": sum(active_counts[budget]) / len(active_counts[budget]),
            "inference_seconds_total": inference_seconds[budget],
            "inference_ms_per_image": 1000.0 * inference_seconds[budget] / len(data_loader.dataset),
            "raw": raw_metrics,
            f"nms_iou_{args.nms_iou:.2f}": nms_metrics,
            "nms_at_risk_independent_gt": _at_risk_gt(
                records[budget]["raw"], records[budget]["nms"], dataset.coco
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
