#!/usr/bin/env bash
set -euo pipefail

cd /data/bee26/beeposetrack_y_20260827
echo "RESUME_WAIT_START $(date)"

while [ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]; do
  busy="$(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';')"
  echo "WAIT_GPU0_BUSY $(date) ${busy}"
  sleep 300
done

echo "GPU0_FREE_START_TRAIN $(date)"
/data/bee26/count/envs/count-anything/bin/python3.10 tools/run_public_pretrain_then_a1_folds.py \
  --e-data-root /data/bee26/beeposetrack_e_20260827/data \
  --own-image-root /data/bee26/vitpose_size_ablation_13_20260827/data/bee_keypoints_13_20260827 \
  --y-root /data/bee26/beeposetrack_y_20260827 \
  --models /data/bee26/yolo26s-pose.pt /data/bee26/yolo26m-pose.pt /data/bee26/yolo26l-pose.pt \
  --folds 0 1 2 3 \
  --seeds 2026 3407 827 \
  --device 0 \
  --batch 2 \
  --nbs 16 \
  --workers 4 \
  --public-epochs 24 \
  --public-imgsz 960 \
  --a1-epochs 100 \
  --a1-imgsz 1280
