#!/usr/bin/env python3
"""Run the BeePoseTrack-Y Route-Best train/test pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Any

import yaml
import re
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo26_bee_pose import register_ultralytics_modules
from data.five_frame_trainer import FiveFramePoseTrainer, RouteBestPoseTrainer
from quantification.events import extract_events
from quantification.group import quantify_group
from quantification.individual import quantify_individual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "y_route_best.yaml")
    parser.add_argument("--stage", choices=["all", "y1", "y2", "y3", "y5", "test"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def complete(run_dir: Path, epochs: int) -> bool:
    if not (run_dir / "weights" / "best.pt").exists():
        return False
    if (run_dir / "route_stage_metadata.json").exists():
        return True
    results = run_dir / "results.csv"
    if not results.exists():
        return False
    rows = [r for r in results.read_text(encoding="utf-8").splitlines() if r.strip()]
    if len(rows) < 2:
        return False
    try:
        last_epoch = int(float(rows[-1].split(",", 1)[0]))
        return last_epoch >= epochs
    except ValueError:
        return False


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_route_weights(model: YOLO, source: Path) -> None:
    """Load exact matches and layer-shifted A1 weights after inserting frame adapter."""

    model.load(str(source))
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    source_model = ckpt.get("ema") or ckpt.get("model")
    if source_model is None:
        return
    source_state = source_model.float().state_dict()
    target_state = model.model.state_dict()
    remapped = {}
    pattern = re.compile(r"^(model\.)(\d+)(\..+)$")
    for key, value in source_state.items():
        match = pattern.match(key)
        candidates = [key]
        if match:
            candidates.append(f"{match.group(1)}{int(match.group(2)) + 1}{match.group(3)}")
        for target_key in candidates:
            if target_key in target_state and tuple(target_state[target_key].shape) == tuple(value.shape):
                remapped[target_key] = value
                break
    model.model.load_state_dict(remapped, strict=False)
    print(f"Route weight transfer {len(remapped)}/{len(target_state)} tensors from {source}", flush=True)


def validate_frozen_data(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    dataset_dir = Path(data["dataset_dir"])
    manifest = Path(data["manifest"])
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    schema = Path(data["schema_v2_root"])
    train_json = schema / "train_13_rgb_ir_schema_v2.json"
    test_json = schema / "val_13_rgb_ir_schema_v2.json"
    actual_train = sha256(train_json)
    actual_test = sha256(test_json)
    if actual_train != data["train_json_sha256"]:
        raise RuntimeError(f"train_json sha mismatch: {actual_train}")
    if actual_test != data["test_json_sha256"]:
        raise RuntimeError(f"test_json sha mismatch: {actual_test}")

    stats = {"train": {"frames": 0, "instances": 0, "domains": {}}, "test": {"frames": 0, "instances": 0, "domains": {}}}
    videos: dict[str, set[str]] = {"train": set(), "test": set()}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row["split"]
        if split not in stats:
            raise RuntimeError(f"unexpected split in manifest: {split}")
        if not Path(row["image_path"]).exists():
            raise FileNotFoundError(row["image_path"])
        label = dataset_dir / "labels" / split / Path(row["image_path"]).with_suffix(".txt").name
        if not label.exists():
            raise FileNotFoundError(label)
        stats[split]["frames"] += 1
        stats[split]["instances"] += len(row.get("instances", []))
        stats[split]["domains"][row.get("domain", "unknown")] = stats[split]["domains"].get(row.get("domain", "unknown"), 0) + 1
        videos[split].add(str(row["video_id"]))
    if sorted(videos["train"]) != sorted(data["train_videos"]):
        raise RuntimeError(f"train videos mismatch: {sorted(videos['train'])}")
    if sorted(videos["test"]) != sorted(data["test_videos"]):
        raise RuntimeError(f"test videos mismatch: {sorted(videos['test'])}")
    return stats


def train_stage(cfg: dict[str, Any], stage_name: str, model_source: str | Path, epochs: int, project: Path, name: str, close_mosaic: int, temporal: bool = False, lr_scale: float = 1.0, patience: int | None = None, model_yaml: str | Path | None = None) -> Path:
    register_ultralytics_modules()
    run_dir = project / name
    if complete(run_dir, epochs):
        return run_dir / "weights" / "best.pt"
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug = cfg["augmentation"]
    model = YOLO(str(model_yaml or model_cfg["route_yaml"]))
    if Path(str(model_source)).exists():
        load_route_weights(model, Path(model_source))
    old_env = {
        "BEEPOSETRACK_MANIFEST": os.environ.get("BEEPOSETRACK_MANIFEST"),
        "BEEPOSETRACK_CLIP_LEN": os.environ.get("BEEPOSETRACK_CLIP_LEN"),
        "BEEPOSETRACK_TEMPORAL_STRIDE": os.environ.get("BEEPOSETRACK_TEMPORAL_STRIDE"),
        "BEEPOSETRACK_ROUTE_CONFIG": os.environ.get("BEEPOSETRACK_ROUTE_CONFIG"),
        "BEEPOSETRACK_STAGE": os.environ.get("BEEPOSETRACK_STAGE"),
    }
    os.environ["BEEPOSETRACK_MANIFEST"] = str(cfg["data"]["manifest"])
    os.environ["BEEPOSETRACK_CLIP_LEN"] = str(model_cfg["clip_len"])
    os.environ["BEEPOSETRACK_TEMPORAL_STRIDE"] = "1"
    os.environ["BEEPOSETRACK_ROUTE_CONFIG"] = str(args_config_path())
    os.environ["BEEPOSETRACK_STAGE"] = stage_name
    train_kwargs = dict(
        task="pose",
        data=str(cfg["data"]["dataset_yaml"]),
        project=str(project),
        name=name,
        epochs=epochs,
        batch=int(train_cfg["batch"]),
        nbs=int(train_cfg["nbs"]),
        imgsz=int(model_cfg["input_size"]),
        device=str(train_cfg["device"]),
        workers=int(train_cfg["workers"]),
        optimizer=str(train_cfg["optimizer"]),
        lr0=float(train_cfg["lr0"]) * lr_scale,
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
        amp=bool(train_cfg["amp"]),
        seed=int(train_cfg["seed"]),
        deterministic=True,
        max_det=int(model_cfg["max_det"]),
        close_mosaic=close_mosaic,
        mosaic=float(aug["mosaic"]),
        mixup=float(aug["mixup"]),
        cutmix=float(aug["cutmix"]),
        copy_paste=float(aug["copy_paste"]),
        erasing=float(aug["erasing"]),
        degrees=float(aug["degrees"]),
        translate=float(aug["translate"]),
        scale=float(aug["scale"]),
        fliplr=float(aug["fliplr"]),
        flipud=float(aug["flipud"]),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        patience=int(patience if patience is not None else train_cfg.get("patience", 20)),
        save=True,
        save_period=10,
        val=True,
        exist_ok=True,
    )
    trainer = FiveFramePoseTrainer if temporal else RouteBestPoseTrainer
    try:
        model.train(trainer=trainer, **train_kwargs)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    meta = {
        "stage": stage_name,
        "route_config": cfg,
        "data_stats": validate_frozen_data(cfg),
        "source_checkpoint": str(model_source),
        "model_yaml": str(model_yaml or model_cfg["route_yaml"]),
    }
    (run_dir / "route_stage_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir / "weights" / "best.pt"


def args_config_path() -> Path:
    return Path(os.environ.get("BEEPOSETRACK_ACTIVE_CONFIG", ROOT / "configs" / "y_route_best.yaml"))


def write_summary(metrics: Path, tracking: Path, onnx: Path, output: Path) -> None:
    det = json.loads(metrics.read_text(encoding="utf-8"))
    trk = json.loads(tracking.read_text(encoding="utf-8"))
    ox = json.loads(onnx.read_text(encoding="utf-8")) if onnx.exists() else {}
    lines = [
        "# BeePoseTrack-Y Route-Best summary",
        "",
        f"det_pose_metrics: {metrics}",
        f"tracking_metrics: {tracking}",
        f"onnx_validation: {onnx}",
        "",
        "| scope | AP50 | mAP50-95 | NME_bbox_diag | PCK@0.10 | direction_acc | head_tail_swap | IDF1 | MOTA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in ("all", "RGB", "IR"):
        d = det["all"] if scope == "all" else det["by_domain"][scope]
        t = trk["all"] if scope == "all" else trk["by_domain"][scope]
        lines.append(
            f"| {scope} | {d.get('AP50', 0):.6f} | {d.get('mAP50-95', 0):.6f} | {d.get('NME_bbox_diag', 0):.6f} | {d.get('PCK@0.10', 0):.6f} | {d.get('direction_acc_angle_le_90', 0):.6f} | {d.get('head_tail_swap_rate', 0):.6f} | {t.get('IDF1', 0):.6f} | {t.get('MOTA', 0):.6f} |"
        )
    if ox:
        lines.extend(["", f"ONNX size MB: {ox.get('size_mb')}", f"ONNX P99 ms: {ox.get('latency_ms', {}).get('p99')}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    os.environ["BEEPOSETRACK_ACTIVE_CONFIG"] = str(args.config.resolve())
    if args.device is not None:
        cfg["training"]["device"] = args.device
    stats = validate_frozen_data(cfg)
    register_ultralytics_modules()

    seed = int(cfg["training"]["seed"])
    project = ROOT / "runs_route_best" / f"seed{seed}"
    eval_dir = ROOT / "eval_route_best" / f"seed{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "frozen_data_check.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    a1 = Path(cfg["model"]["a1_checkpoint"])
    if not a1.exists():
        raise FileNotFoundError(a1)
    ckpt = a1
    if args.stage in {"all", "y1"}:
        ckpt = train_stage(cfg, "Y1_task_adaptation", ckpt, int(cfg["training"]["y1_epochs"]), project, "strict_y1_task_adaptation_p2_768", int(cfg["augmentation"]["close_mosaic"]), temporal=False, patience=int(cfg["training"].get("y1_patience", cfg["training"].get("patience", 30))), model_yaml=cfg["model"]["y1_yaml"])
    if args.stage in {"all", "y2"}:
        source = ckpt if args.stage == "all" else project / "strict_y1_task_adaptation_p2_768" / "weights" / "best.pt"
        ckpt = train_stage(cfg, "Y2_structure", source, int(cfg["training"]["y2_epochs"]), project, "strict_y2_structure_p2_pose", 0, temporal=False, patience=int(cfg["training"].get("y2_patience", cfg["training"].get("patience", 12))), model_yaml=cfg["model"]["y2_yaml"])
    if args.stage in {"all", "y3"}:
        source = ckpt if args.stage == "all" else project / "strict_y2_structure_p2_pose" / "weights" / "best.pt"
        ckpt = train_stage(cfg, "Y3_spatiotemporal", source, int(cfg["training"]["y3_epochs"]), project, "strict_y3_spatiotemporal_route_best", int(cfg["augmentation"]["close_mosaic"]), temporal=True, patience=int(cfg["training"].get("y3_patience", cfg["training"].get("patience", 16))), model_yaml=cfg["model"]["route_yaml"])
    if args.stage in {"all", "y5"}:
        source = ckpt if args.stage == "all" else project / "strict_y3_spatiotemporal_route_best" / "weights" / "best.pt"
        ckpt = train_stage(cfg, "Y5_low_lr_finetune", source, int(cfg["training"]["y5_epochs"]), project, "strict_y5_low_lr_finetune", int(cfg["augmentation"]["close_mosaic"]), temporal=True, lr_scale=float(cfg["training"].get("y5_lr_scale", 0.1)), patience=int(cfg["training"].get("y5_patience", cfg["training"].get("patience", 8))), model_yaml=cfg["model"]["route_yaml"])

    final_ckpt = project / "strict_y5_low_lr_finetune" / "weights" / "best.pt"
    if not final_ckpt.exists():
        final_ckpt = ckpt
    raw_pred = eval_dir / "test_pred_raw.jsonl"
    pred = eval_dir / "test_pred_pose_aware.jsonl"
    tracks = eval_dir / "test_tracks_raw.jsonl"
    tracks_fixed = eval_dir / "test_tracks_head_tail_dp.jsonl"
    metrics = eval_dir / "test_metrics.json"
    tracking_metrics = eval_dir / "test_tracking_metrics.json"
    onnx_metrics = eval_dir / "onnx_validation.json"

    if args.stage in {"all", "test"}:
        candidate_conf = cfg["eval"].get("candidate_conf", cfg["eval"]["conf"])
        predict_cmd = [args.python, str(ROOT / "tools" / "predict_yolo_pose_jsonl.py"), "--model", str(final_ckpt), "--source", str(Path(cfg["data"]["dataset_dir"]) / "test_images.txt"), "--output", str(raw_pred), "--imgsz", str(cfg["model"]["input_size"]), "--device", str(cfg["training"]["device"]), "--conf", str(candidate_conf), "--iou", str(cfg["eval"]["iou"]), "--max-det", str(cfg["model"]["max_det"]), "--model-name", "BeePoseTrack-Y-Route-Best"]
        if bool(cfg["eval"].get("five_frame", True)):
            predict_cmd.extend(["--manifest", str(cfg["data"]["manifest"]), "--eval-split", "test", "--five-frame"])
        run(predict_cmd)
        run([args.python, str(ROOT / "tools" / "pose_aware_nms.py"), "--input", str(raw_pred), "--output", str(pred), "--iou", str(cfg["eval"]["iou"]), "--max-det", str(cfg["model"]["max_det"])])
        run([args.python, str(ROOT / "tools" / "evaluate_manifest_predictions.py"), "--manifest", str(cfg["data"]["manifest"]), "--pred-jsonl", str(pred), "--output", str(metrics), "--eval-split", "test", "--pose-iou", str(cfg["eval"]["pose_iou"]), "--pr-iou", str(cfg["eval"]["pr_iou"])])
        run([args.python, str(ROOT / "tracking" / "pose_motion_tracker.py"), "--detections", str(pred), "--output", str(tracks), "--max-age", str(cfg["tracking"]["max_age"]), "--max-cost", str(cfg["tracking"]["max_cost"]), "--max-center-dist", str(cfg["tracking"]["max_center_dist"]), "--max-candidates-per-frame", str(cfg["tracking"].get("max_candidates_per_frame", cfg["model"]["max_det"])), "--max-new-tracks-per-frame", str(cfg["tracking"].get("max_new_tracks_per_frame", 540)), "--max-active-tracks", str(cfg["tracking"].get("max_active_tracks", cfg["model"]["max_det"])), "--hungarian-limit", str(cfg["tracking"].get("hungarian_limit", 600))])
        run([args.python, str(ROOT / "tracking" / "head_tail_dp.py"), "--input", str(tracks), "--output", str(tracks_fixed)])
        run([args.python, str(ROOT / "tools" / "evaluate_manifest_tracking.py"), "--manifest", str(cfg["data"]["manifest"]), "--track-jsonl", str(tracks_fixed), "--output", str(tracking_metrics), "--eval-split", "test", "--iou", "0.5"])
        quantify_individual(tracks_fixed, eval_dir / "individual_quantification.csv")
        quantify_group(tracks_fixed, eval_dir / "group_quantification.csv")
        extract_events(eval_dir / "individual_quantification.csv", eval_dir / "events.csv")
        run([args.python, str(ROOT / "export" / "export_onnx.py"), "--weights", str(final_ckpt), "--output-dir", str(eval_dir), "--imgsz", str(cfg["model"]["input_size"]), "--device", str(cfg["training"]["device"])])
        run([args.python, str(ROOT / "export" / "validate_onnx.py"), "--onnx", str(eval_dir / "beeposetrack_one.onnx"), "--output", str(onnx_metrics), "--imgsz", str(cfg["model"]["input_size"])])
        write_summary(metrics, tracking_metrics, onnx_metrics, eval_dir / "route_best_summary.md")


if __name__ == "__main__":
    main()
