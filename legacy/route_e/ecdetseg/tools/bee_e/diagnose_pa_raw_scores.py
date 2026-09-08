#!/usr/bin/env python3
"""Read-only raw-score diagnosis for BeePoseTrack-E PA checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch

from engine.core import YAMLConfig


QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
THRESHOLDS = (0.01, 0.02)


def _supported_kwargs(model, values: dict) -> dict:
    signature = inspect.signature(model.forward)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _to_device(value, device: torch.device):
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: torch.Tensor) -> dict:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {"count": 0, "quantiles": {}, "mean": None, "min": None, "max": None}
    q = torch.tensor(QUANTILES, dtype=torch.float32)
    quantile_values = torch.quantile(values, q)
    return {
        "count": int(values.numel()),
        "quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(QUANTILES, quantile_values)
        },
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _score_counts(score: torch.Tensor, valid: torch.Tensor) -> dict:
    score = score.detach().float().reshape(-1)
    valid = valid.detach().bool().reshape(-1)
    return {
        "all": {f"gt_{threshold:.2f}": int((score > threshold).sum()) for threshold in THRESHOLDS},
        "active": {
            f"gt_{threshold:.2f}": int(((score > threshold) & valid).sum())
            for threshold in THRESHOLDS
        },
    }


def _summarize_outputs(outputs: dict) -> tuple[dict, dict[str, torch.Tensor]]:
    class_score = outputs["pred_logits"].sigmoid()[0].amax(dim=-1)
    quality = outputs.get("pred_quality")
    quality_score = quality.sigmoid()[0] if quality is not None else torch.ones_like(class_score)
    visibility = outputs.get("pred_visibility")
    if visibility is None:
        visibility_score = torch.ones_like(class_score)
        visibility_flat = visibility_score[:, None]
    else:
        visibility_flat = visibility.sigmoid()[0]
        visibility_score = visibility_flat.mean(dim=-1)
    query_valid = outputs.get("pred_query_valid")
    if query_valid is None:
        query_valid = torch.ones_like(class_score, dtype=torch.bool)
    else:
        query_valid = query_valid[0].bool()

    formulas = {
        "class_only": class_score,
        "class_x_quality": class_score * quality_score,
        "class_x_visibility": class_score * visibility_score,
        "class_x_quality_x_visibility": class_score * quality_score * visibility_score,
    }
    sample = {
        "query_count": int(class_score.numel()),
        "active_query_count": int(query_valid.sum()),
        "class_sigmoid": _stats(class_score),
        "quality_sigmoid": _stats(quality_score),
        "visibility_sigmoid_flat": _stats(visibility_flat),
        "visibility_sigmoid_query_mean": _stats(visibility_score),
        "score_counts": {
            name: _score_counts(score, query_valid) for name, score in formulas.items()
        },
        "route_domain_id": int(outputs.get("route_domain_id", torch.tensor([-1]))[0]),
        "route_scene_id": int(outputs.get("route_scene_id", torch.tensor([-1]))[0]),
    }
    tensors = {
        "class_sigmoid": class_score,
        "quality_sigmoid": quality_score,
        "visibility_sigmoid_flat": visibility_flat.reshape(-1),
        "visibility_sigmoid_query_mean": visibility_score,
        "query_valid": query_valid,
        **{f"score::{name}": score for name, score in formulas.items()},
    }
    return sample, tensors


def _aggregate(samples: list[dict], tensors: list[dict[str, torch.Tensor]]) -> dict:
    merged = {
        key: torch.cat([item[key].detach().cpu().reshape(-1) for item in tensors])
        for key in (
            "class_sigmoid", "quality_sigmoid", "visibility_sigmoid_flat",
            "visibility_sigmoid_query_mean",
        )
    }
    valid = torch.cat([item["query_valid"].detach().cpu().reshape(-1) for item in tensors])
    result = {
        "sample_count": len(samples),
        "query_count": int(sum(item["query_count"] for item in samples)),
        "active_query_count": int(valid.sum()),
        "active_query_fraction": float(valid.float().mean()),
        **{key: _stats(value) for key, value in merged.items()},
        "score_counts": {},
    }
    for name in (
        "class_only", "class_x_quality", "class_x_visibility",
        "class_x_quality_x_visibility",
    ):
        score = torch.cat([
            item[f"score::{name}"].detach().cpu().reshape(-1) for item in tensors
        ])
        result["score_counts"][name] = _score_counts(score, valid)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--image-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoints[0]))
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    dataset = cfg.val_dataloader.dataset
    id_to_index = {int(image_id): index for index, image_id in enumerate(dataset.ids)}
    missing = sorted(set(args.image_ids) - set(id_to_index))
    if missing:
        raise ValueError(f"Missing validation image ids: {missing}")

    prepared = []
    for image_id in args.image_ids:
        sample, target = dataset[id_to_index[image_id]]
        target_device = {key: _to_device(value, device) for key, value in target.items()}
        prepared.append((image_id, sample.unsqueeze(0).to(device), target_device))

    model = cfg.model.to(device).eval()
    result = {
        "config": str(args.config.resolve()),
        "image_ids": args.image_ids,
        "scene_contract": {"0": "indoor_B", "1": "outdoor_A"},
        "diagnostic_warning": (
            "Forced scene routes are diagnostic only and are not formal metrics. "
            "No PostProcessor, NMS, deduplication, or checkpoint mutation is used."
        ),
        "postprocessor_config": {
            "enable_calibrated_filter": bool(cfg.postprocessor.enable_calibrated_filter),
            "score_exponents_by_domain": cfg.postprocessor.score_exponents_by_domain,
            "score_thresholds_by_domain": cfg.postprocessor.score_thresholds_by_domain,
        },
        "checkpoints": [],
    }

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_result = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "last_epoch": int(checkpoint.get("last_epoch", -1)),
            "weights": {},
        }
        sources = {"model": checkpoint["model"]}
        if "ema" in checkpoint and "module" in checkpoint["ema"]:
            sources["ema"] = checkpoint["ema"]["module"]
        for source_name, state in sources.items():
            model.load_state_dict(state, strict=True)
            model.eval()
            source_result = {}
            for forced_scene_id in (0, 1):
                samples = []
                raw_tensors = []
                route_started = time.perf_counter()
                with torch.inference_mode():
                    for image_id, sample, target in prepared:
                        route = {
                            "domain_id": target["domain_id"].reshape(1),
                            "scene_id": torch.tensor([forced_scene_id], device=device),
                        }
                        if "temporal_valid_mask" in target:
                            route["temporal_valid_mask"] = target["temporal_valid_mask"].unsqueeze(0)
                        if "stabilization_theta" in target:
                            route["stabilization_theta"] = target["stabilization_theta"].unsqueeze(0)
                        outputs = model(sample, **_supported_kwargs(model, route))
                        sample_result, tensors = _summarize_outputs(outputs)
                        sample_result["image_id"] = int(image_id)
                        sample_result["dataset_scene_id"] = int(target["scene_id"].reshape(-1)[0])
                        sample_result["dataset_domain_id"] = int(target["domain_id"].reshape(-1)[0])
                        samples.append(sample_result)
                        raw_tensors.append(tensors)
                route_name = "indoor_B_forced" if forced_scene_id == 0 else "outdoor_A_forced"
                source_result[route_name] = {
                    "forced_scene_id": forced_scene_id,
                    "elapsed_seconds": time.perf_counter() - route_started,
                    "aggregate": _aggregate(samples, raw_tensors),
                    "samples": samples,
                }
                print(
                    checkpoint_path.name, source_name, route_name,
                    "class_max", source_result[route_name]["aggregate"]["class_sigmoid"]["max"],
                    "combined>0.01", source_result[route_name]["aggregate"]["score_counts"]
                    ["class_x_quality_x_visibility"]["active"]["gt_0.01"],
                    flush=True,
                )
            checkpoint_result["weights"][source_name] = source_result
        result["checkpoints"].append(checkpoint_result)
        del checkpoint
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
