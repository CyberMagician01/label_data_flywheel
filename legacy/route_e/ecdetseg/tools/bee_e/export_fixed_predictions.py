#!/usr/bin/env python3
"""Export raw predictions for fixed Schema-v2 samples without changing evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch

from engine.core import YAMLConfig


def _supported_kwargs(model, values: dict) -> dict:
    signature = inspect.signature(model.forward)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _to_device(value, device: torch.device):
    return value.to(device) if isinstance(value, torch.Tensor) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-score", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    dataset = cfg.val_dataloader.dataset
    id_to_index = {int(image_id): index for index, image_id in enumerate(dataset.ids)}
    missing = sorted(set(args.image_ids) - set(id_to_index))
    if missing:
        raise ValueError(f"Requested image ids are absent from configured validation dataset: {missing}")

    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    postprocessor = cfg.postprocessor
    if hasattr(postprocessor, "to"):
        postprocessor.to(device)
    postprocessor.eval()

    output = {"checkpoint": str(args.checkpoint.resolve()), "samples": []}
    with torch.inference_mode():
        for image_id in args.image_ids:
            sample, target = dataset[id_to_index[image_id]]
            sample = sample.unsqueeze(0).to(device)
            target_device = {key: _to_device(value, device) for key, value in target.items()}
            route = {}
            for key in ("domain_id", "scene_id"):
                if key in target_device:
                    route[key] = target_device[key].reshape(1)
            if "temporal_valid_mask" in target_device:
                route["temporal_valid_mask"] = target_device["temporal_valid_mask"].unsqueeze(0)
            if "stabilization_theta" in target_device:
                route["stabilization_theta"] = target_device["stabilization_theta"].unsqueeze(0)
            started = time.perf_counter()
            outputs = model(sample, **_supported_kwargs(model, route))
            target_size = target_device.get("input_size", target_device["orig_size"]).reshape(1, 2)
            result = postprocessor(outputs, target_size)[0]
            elapsed = time.perf_counter() - started
            if "letterbox_scale" in target_device:
                boxes = result["boxes"]
                scale = target_device["letterbox_scale"]
                pad = target_device["letterbox_pad"]
                boxes[:, 0::2] = (boxes[:, 0::2] - pad[0]) / scale
                boxes[:, 1::2] = (boxes[:, 1::2] - pad[1]) / scale
                width, height = target_device["orig_size"][1], target_device["orig_size"][0]
                boxes[:, 0::2].clamp_(0, width)
                boxes[:, 1::2].clamp_(0, height)

            image_info = dataset.coco.imgs[image_id]
            annotations = [dataset.coco.anns[index] for index in dataset.coco.getAnnIds(imgIds=[image_id])]
            ground_truth = []
            for annotation in annotations:
                x, y, width, height = (float(value) for value in annotation["bbox"])
                ground_truth.append({"ann_id": int(annotation["id"]), "box": [x, y, x + width, y + height]})
            predictions = []
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                if float(score) < args.min_score:
                    continue
                predictions.append({
                    "box": [float(value) for value in box.detach().cpu()],
                    "score": float(score.detach().cpu()),
                    "label": int(label.detach().cpu()),
                })
            output["samples"].append({
                "role": "stratified_control",
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "image_path": str((Path(dataset.root) / image_info["file_name"]).resolve()),
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "gt": ground_truth,
                "predictions": predictions,
                "inference_seconds": elapsed,
                "route_domain_id": int(target_device["domain_id"].reshape(-1)[0]),
                "route_scene_id": int(target_device["scene_id"].reshape(-1)[0]),
                "raw_query_count": int(outputs["pred_logits"].shape[1]),
                "valid_query_count": int(outputs.get("pred_query_valid", torch.ones(outputs["pred_logits"].shape[1], device=device)).sum()),
            })
            print(f"image_id={image_id} predictions={len(predictions)} seconds={elapsed:.2f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
