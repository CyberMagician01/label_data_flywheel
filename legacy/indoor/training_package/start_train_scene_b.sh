#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/root/autodl-tmp/flywheel_outdoor_pose_20260903/env/bin/python"
cd "$SCRIPT_DIR"

# 明确仅暴露健康的 GPU 0，从环境层面完全屏蔽掉不稳定的 GPU 1
export CUDA_VISIBLE_DEVICES=0

# -u 确保 Python stdout/stderr 实时无缓冲追加写入 log
mkdir -p "$WORK_ROOT/logs"
"$PYTHON_BIN" -u resume_train_scene_b.py >> "$WORK_ROOT/logs/train_scene_b_unified.log" 2>&1
