#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/root/autodl-tmp/flywheel_outdoor_pose_20260903/env/bin/python"

cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export EPOCHS="${EPOCHS:-300}"
export BATCH="${BATCH:-16}"
export WORKERS="${WORKERS:-12}"
export CACHE="${CACHE:-ram}"
export SEED="${SEED:-2026}"
export DEVICE="0,1"
export PROJECT="${PROJECT:-$WORK_ROOT/runs/unified}"
export RUN_NAME="${RUN_NAME:-scene_b_annotators_01_02_03_yolov8x_p2_seed2026}"
export PATIENCE="${PATIENCE:-50}"
export EXIST_OK="${EXIST_OK:-false}"
export SAVE_PERIOD="${SAVE_PERIOD:-10}"

exec "$PYTHON_BIN" -u train_scene_b_unified_yolov8x_p2.py
