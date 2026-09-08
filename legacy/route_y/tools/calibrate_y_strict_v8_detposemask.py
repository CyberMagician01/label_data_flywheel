#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/data/bee26/beeposetrack_y_20260827")
DATASET = ROOT / "datasets" / "y_unified_20260901_labelme5_a1_strict_v8_annotator01_val_03_test_detposemask"
RUN_DIR = ROOT / "runs_unified_20260901/a1_aligned/a1_unified_strict-v8-annotator01-val-03-test-detposemask_best_seed2026_img1280_eb16"
BASE_EVAL = ROOT / "eval_unified_20260901/a1_strict_v8_annotator01_val03_seed2026"
CALIB_DIR = ROOT / "eval_unified_20260901/a1_strict_v8_annotator01_val03_seed2026_calibration"
FROZEN_TEST_DIR = ROOT / "eval_unified_20260901/a1_strict_v8_annotator01_val03_seed2026_calibrated_test"


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_predictions(src: Path, dst: Path, conf: float, max_det: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            detections = [
                det for det in row.get("detections", [])
                if float(det.get("score", 0.0)) >= conf
            ]
            detections.sort(
                key=lambda d: float(d.get("score", 0.0)) * max(float(d.get("pose_score", 0.0)), 0.25),
                reverse=True,
            )
            row["detections"] = detections[:max_det]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def detection_f1(metrics: dict[str, Any]) -> float:
    p = float(metrics["all"].get("precision_iou50", 0.0))
    r = float(metrics["all"].get("recall_iou50", 0.0))
    return 2 * p * r / max(p + r, 1e-9)


def candidate_score(metrics: dict[str, Any]) -> float:
    all_metrics = metrics["all"]
    return (
        0.35 * detection_f1(metrics)
        + 0.25 * float(all_metrics.get("AP50", 0.0))
        + 0.20 * float(all_metrics.get("PCK@0.10", 0.0))
        + 0.20 * float(all_metrics.get("direction_acc_angle_le_90", 0.0))
    )


def tracking_score(metrics: dict[str, Any]) -> float:
    all_metrics = metrics["all"]
    return (
        float(all_metrics.get("MOTA", 0.0))
        + float(all_metrics.get("IDF1", 0.0))
        + 0.25 * float(all_metrics.get("HOTA_proxy", 0.0))
    )


def eval_pred(py: str, manifest: Path, pred: Path, out: Path, split: str) -> dict[str, Any]:
    run([
        py,
        str(ROOT / "tools/evaluate_manifest_predictions.py"),
        "--manifest",
        str(manifest),
        "--pred-jsonl",
        str(pred),
        "--output",
        str(out),
        "--eval-split",
        split,
        "--pose-iou",
        "0.3",
        "--pr-iou",
        "0.5",
    ])
    return read_json(out)


def eval_track(py: str, manifest: Path, pred: Path, out_dir: Path, split: str, params: dict[str, Any]) -> dict[str, Any]:
    raw_tracks = out_dir / "tracks_raw.jsonl"
    fixed_tracks = out_dir / "tracks_head_tail_dp.jsonl"
    metrics = out_dir / "tracking_metrics.json"
    run([
        py,
        str(ROOT / "tracking/pose_motion_tracker.py"),
        "--detections",
        str(pred),
        "--output",
        str(raw_tracks),
        "--max-age",
        str(params["max_age"]),
        "--max-cost",
        str(params["max_cost"]),
        "--max-center-dist",
        str(params["max_center_dist"]),
        "--max-candidates-per-frame",
        str(params["max_candidates"]),
        "--max-new-tracks-per-frame",
        str(params["max_new_tracks"]),
        "--max-active-tracks",
        str(params["max_active_tracks"]),
        "--hungarian-limit",
        str(params["hungarian_limit"]),
    ])
    run([py, str(ROOT / "tracking/head_tail_dp.py"), "--input", str(raw_tracks), "--output", str(fixed_tracks)])
    run([
        py,
        str(ROOT / "tools/evaluate_manifest_tracking.py"),
        "--manifest",
        str(manifest),
        "--track-jsonl",
        str(fixed_tracks),
        "--output",
        str(metrics),
        "--eval-split",
        split,
        "--iou",
        "0.5",
    ])
    result = read_json(metrics)
    result["track_jsonl"] = str(fixed_tracks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/data/bee26/count/envs/count-anything/bin/python3.10")
    parser.add_argument("--device", default="0")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    py = args.python
    best = RUN_DIR / "weights/best.pt"
    manifest = DATASET / "dataset_manifest.jsonl"
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    val_raw = CALIB_DIR / "val_pred_raw_conf0001.jsonl"

    if not val_raw.exists():
        run([
            py,
            str(ROOT / "tools/predict_yolo_pose_jsonl.py"),
            "--model",
            str(best),
            "--source",
            str(DATASET / "val_images.txt"),
            "--output",
            str(val_raw),
            "--imgsz",
            "1280",
            "--device",
            args.device,
            "--conf",
            "0.001",
            "--iou",
            "0.7",
            "--max-det",
            "768",
            "--model-name",
            "BeePoseTrack-Y-A1-Unified-Aligned-Calib",
        ])

    # Local calibration around the already aligned schema-v2 baseline:
    # conf=0.001, nms_iou=0.7, max_det=768.  The grid still probes all three
    # requested knobs, but avoids spending hours on far-off postprocess values.
    pred_param_grid = [
        (0.001, 0.60, 384),
        (0.001, 0.60, 768),
        (0.001, 0.70, 384),
        (0.001, 0.70, 768),
        (0.003, 0.60, 384),
        (0.003, 0.70, 384),
        (0.005, 0.70, 384),
        (0.010, 0.70, 384),
        (0.020, 0.70, 384),
        (0.050, 0.70, 384),
    ]
    pred_candidates = []
    for conf, nms_iou, max_det in pred_param_grid:
                tag = f"conf{conf:g}_nms{nms_iou:g}_max{max_det}".replace(".", "p")
                cand_dir = CALIB_DIR / "pred_grid" / tag
                filtered = cand_dir / "filtered.jsonl"
                pred = cand_dir / "pose_aware.jsonl"
                metrics_path = cand_dir / "metrics.json"
                if not metrics_path.exists():
                    filter_predictions(val_raw, filtered, conf, max_det)
                    run([
                        py,
                        str(ROOT / "tools/pose_aware_nms.py"),
                        "--input",
                        str(filtered),
                        "--output",
                        str(pred),
                        "--iou",
                        str(nms_iou),
                        "--max-det",
                        str(max_det),
                    ])
                    metrics = eval_pred(py, manifest, pred, metrics_path, "val")
                else:
                    metrics = read_json(metrics_path)
                pred_candidates.append(
                    {
                        "tag": tag,
                        "conf": conf,
                        "nms_iou": nms_iou,
                        "max_det": max_det,
                        "pred": str(pred),
                        "metrics": metrics,
                        "candidate_score": candidate_score(metrics),
                        "detection_f1": detection_f1(metrics),
                    }
                )

    pred_candidates.sort(key=lambda x: x["candidate_score"], reverse=True)
    top_pred = pred_candidates[: args.top_k]

    track_param_grid = []
    for max_age in [3]:
        for max_cost in [1.10, 1.35, 1.60]:
            for max_center_dist in [120.0, 140.0]:
                for max_candidates in [384]:
                    track_param_grid.append(
                        {
                            "max_age": max_age,
                            "max_cost": max_cost,
                            "max_center_dist": max_center_dist,
                            "max_candidates": max_candidates,
                            "max_new_tracks": max_candidates,
                            "max_active_tracks": 512,
                            "hungarian_limit": max_candidates,
                        }
                    )

    track_candidates = []
    for pred_item in top_pred:
        pred = Path(pred_item["pred"])
        for params in track_param_grid:
            ptag = pred_item["tag"]
            ttag = (
                f"age{params['max_age']}_cost{params['max_cost']:g}_dist{params['max_center_dist']:g}_cand{params['max_candidates']}"
            ).replace(".", "p")
            out_dir = CALIB_DIR / "track_grid" / ptag / ttag
            metrics_path = out_dir / "tracking_metrics.json"
            if metrics_path.exists():
                metrics = read_json(metrics_path)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics = eval_track(py, manifest, pred, out_dir, "val", params)
            track_candidates.append(
                {
                    "pred": pred_item,
                    "tracking_params": params,
                    "tracking_metrics": metrics,
                    "tracking_score": tracking_score(metrics),
                }
            )

    track_candidates.sort(key=lambda x: x["tracking_score"], reverse=True)
    best_item = track_candidates[0]
    frozen = {
        "selection_split": "val",
        "selection_semantics": "alignment protocol calibration / annotator_01",
        "selection_rule": "maximize MOTA + IDF1 + 0.25 * HOTA_proxy over top prediction candidates from val",
        "best_prediction_params": {
            "conf": best_item["pred"]["conf"],
            "nms_iou": best_item["pred"]["nms_iou"],
            "max_det": best_item["pred"]["max_det"],
        },
        "best_tracking_params": best_item["tracking_params"],
        "best_val_prediction_metrics": best_item["pred"]["metrics"],
        "best_val_tracking_metrics": best_item["tracking_metrics"],
        "top_prediction_candidates": [
            {
                "tag": item["tag"],
                "conf": item["conf"],
                "nms_iou": item["nms_iou"],
                "max_det": item["max_det"],
                "candidate_score": round(item["candidate_score"], 6),
                "detection_f1": round(item["detection_f1"], 6),
                "metrics_all": item["metrics"]["all"],
            }
            for item in top_pred
        ],
        "top_tracking_candidates": [
            {
                "prediction_params": {
                    "conf": item["pred"]["conf"],
                    "nms_iou": item["pred"]["nms_iou"],
                    "max_det": item["pred"]["max_det"],
                },
                "tracking_params": item["tracking_params"],
                "tracking_score": round(item["tracking_score"], 6),
                "tracking_all": item["tracking_metrics"]["all"],
            }
            for item in track_candidates[:20]
        ],
    }
    (CALIB_DIR / "frozen_calibration_config.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    FROZEN_TEST_DIR.mkdir(parents=True, exist_ok=True)
    test_raw = BASE_EVAL / "test_pred_raw.jsonl"
    if not test_raw.exists():
        test_raw = FROZEN_TEST_DIR / "test_pred_raw_conf0001.jsonl"
        run([
            py,
            str(ROOT / "tools/predict_yolo_pose_jsonl.py"),
            "--model",
            str(best),
            "--source",
            str(DATASET / "test_images.txt"),
            "--output",
            str(test_raw),
            "--imgsz",
            "1280",
            "--device",
            args.device,
            "--conf",
            "0.001",
            "--iou",
            "0.7",
            "--max-det",
            "768",
            "--model-name",
            "BeePoseTrack-Y-A1-Unified-Aligned-Calibrated",
        ])
    test_filtered = FROZEN_TEST_DIR / "test_pred_filtered.jsonl"
    test_pred = FROZEN_TEST_DIR / "test_pred_pose_aware_calibrated.jsonl"
    test_metrics = FROZEN_TEST_DIR / "test_metrics_calibrated.json"
    p = frozen["best_prediction_params"]
    filter_predictions(test_raw, test_filtered, float(p["conf"]), int(p["max_det"]))
    run([
        py,
        str(ROOT / "tools/pose_aware_nms.py"),
        "--input",
        str(test_filtered),
        "--output",
        str(test_pred),
        "--iou",
        str(p["nms_iou"]),
        "--max-det",
        str(p["max_det"]),
    ])
    test_pred_metrics = eval_pred(py, manifest, test_pred, test_metrics, "test")
    test_track_metrics = eval_track(py, manifest, test_pred, FROZEN_TEST_DIR, "test", frozen["best_tracking_params"])
    frozen_test = {
        "frozen_config": frozen,
        "test_prediction_metrics": test_pred_metrics,
        "test_tracking_metrics": test_track_metrics,
        "outputs": {
            "test_pred": str(test_pred),
            "test_metrics": str(test_metrics),
            "test_tracks": str(FROZEN_TEST_DIR / "tracks_head_tail_dp.jsonl"),
            "test_tracking_metrics": str(FROZEN_TEST_DIR / "tracking_metrics.json"),
        },
    }
    (FROZEN_TEST_DIR / "frozen_test_summary.json").write_text(
        json.dumps(frozen_test, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(frozen_test, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
