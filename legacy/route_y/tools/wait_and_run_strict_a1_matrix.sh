#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/bee26"
YROOT="$ROOT/beeposetrack_y_20260827"
PY="$ROOT/count/envs/count-anything/bin/python3.10"
DATA="$YROOT/datasets/y_pose_aligned_strict_fold4/bee_yolo_pose_strict.yaml"
PROJECT="$YROOT/runs_strict_a1"
LOG_DIR="$YROOT/logs"
TRAIN_SCRIPT="$YROOT/tools/train_yolo_pose_aligned.py"
MATRIX_SCRIPT="$YROOT/tools/run_strict_a1_matrix.py"

mkdir -p "$PROJECT" "$LOG_DIR"

echo "started_at=$(date -Is)"
echo "data=$DATA"
echo "project=$PROJECT"
echo "models=$ROOT/yolo26s-pose.pt $ROOT/yolo26m-pose.pt $ROOT/yolo26l-pose.pt"
echo "seeds=2026 3407 827"
echo "epochs=100 batch=2 nbs=16 imgsz=1280 optimizer=MuSGD max_det=768 patience=0"

sha256sum \
  "$YROOT/datasets/y_pose_aligned_strict_fold4/train_images.txt" \
  "$YROOT/datasets/y_pose_aligned_strict_fold4/val_images.txt" \
  "$YROOT/datasets/y_pose_aligned_strict_fold4/dataset_manifest.jsonl" \
  "$DATA" \
  "$ROOT/yolo26s-pose.pt" \
  "$ROOT/yolo26m-pose.pt" \
  "$ROOT/yolo26l-pose.pt"

until CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
do
  echo "[$(date -Is)] CUDA not available; sleep 300s"
  sleep 300
done

echo "[$(date -Is)] CUDA available; launching strict A1 matrix"
CUDA_VISIBLE_DEVICES=0 "$PY" "$MATRIX_SCRIPT" \
  --python "$PY" \
  --train-script "$TRAIN_SCRIPT" \
  --data "$DATA" \
  --project "$PROJECT" \
  --models "$ROOT/yolo26s-pose.pt" "$ROOT/yolo26m-pose.pt" "$ROOT/yolo26l-pose.pt" \
  --seeds 2026 3407 827 \
  --device 0 \
  --batch 2 \
  --nbs 16 \
  --workers 4 \
  --imgsz 1280 \
  --epochs 100

echo "finished_at=$(date -Is)"
