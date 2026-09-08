#!/usr/bin/env bash
# ==============================================================================
# SAM 2.1 Video Bee Tracking & Snapping Pipeline (One-Click Launcher)
# ==============================================================================
set -e

# Detect Python Environment
PYTHON_EXEC="/data/bee26/count/envs/count-anything/bin/python3.10"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC=$(which python3)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_SCRIPT="${SCRIPT_DIR}/sam2_bee_tracker.py"
SNAPPING_SCRIPT="${SCRIPT_DIR}/apply_postprocess_snapping.py"
CONFIG_FILE="configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT_FILE="${SCRIPT_DIR}/checkpoints/sam2.1_hiera_large.pt"

# Parse sequence name and frame numbers
TARGET_SEQ="${1:-a51_0_250}"
USER_START="${2}"
USER_END="${3}"
FPS="10.0"

# Auto-detect sequence from string pattern (e.g. a51_0_50 or a54_101_351)
SCENE_NAME="A-5-1"
if [[ "$TARGET_SEQ" =~ a54|A54|A-5-4 ]]; then
    SCENE_NAME="A-5-4"
elif [[ "$TARGET_SEQ" =~ a52|A52|A-5-2 ]]; then
    SCENE_NAME="A-5-2"
elif [[ "$TARGET_SEQ" =~ a53|A53|A-5-3 ]]; then
    SCENE_NAME="A-5-3"
elif [[ "$TARGET_SEQ" =~ a51|A51|A-5-1 ]]; then
    SCENE_NAME="A-5-1"
fi

FRAMES_DIR="/data/bee26/datasets/SY-202601-比赛数据/逐帧图像/巢外监测/${SCENE_NAME}"
DETS_DIR="/data/bee26/datasets/flywheel_outdoor_pose_labels_postprocessed_20260903/${SCENE_NAME}/frames"
ANNOT_DIR=""

# Try parsing range from sequence name like a51_0_50 or a54_101_351
START_F=0
END_F=250

if [[ "$TARGET_SEQ" =~ _([0-9]+)_([0-9]+)$ ]]; then
    START_F="${BASH_REMATCH[1]}"
    END_F="${BASH_REMATCH[2]}"
elif [[ "$TARGET_SEQ" =~ _([0-9]+)$ ]]; then
    START_F=0
    END_F="${BASH_REMATCH[1]}"
fi

if [ -n "$USER_START" ]; then
    START_F=$USER_START
fi
if [ -n "$USER_END" ]; then
    END_F=$USER_END
fi

SEQ_TAG="${SCENE_NAME}_f$(printf "%05d" $START_F)_f$(printf "%05d" $END_F)"

mkdir -p "${SCRIPT_DIR}/demo_outputs"
RAW_JSON="${SCRIPT_DIR}/demo_outputs/sam21_${SEQ_TAG}_raw.json"
SNAPPED_JSON="${SCRIPT_DIR}/demo_outputs/sam21_${SEQ_TAG}_snapped.json"
OUT_VIDEO="${SCRIPT_DIR}/demo_outputs/sam21_${SEQ_TAG}_SNAPPED_DEMO.mp4"

echo "=========================================================="
echo "🐝 SAM 2.1 Bee Video Tracking Pipeline"
echo "Sequence:    ${TARGET_SEQ}"
echo "Frames:      [${START_F}, ${END_F}] @ ${FPS} FPS"
echo "Frames Dir:  ${FRAMES_DIR}"
echo "Detections:  ${DETS_DIR}"
echo "Output MP4:  ${OUT_VIDEO}"
echo "=========================================================="

T_START=$(date +%s)

# Step 1: Run Native Single-Pass SAM 2.1 Streaming Tracker
echo ""
echo ">>> [Phase 1/2] Running SAM 2.1 Native Single-Pass Streaming Video Tracker..."
TRACK_CMD=("${PYTHON_EXEC}" "${TRACKER_SCRIPT}" \
    --model-type "sam2.1" \
    --model-cfg "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_FILE}" \
    --frames-dir "${FRAMES_DIR}" \
    --f-start "${START_F}" \
    --f-end "${END_F}" \
    --fps "${FPS}" \
    --min-conf 0.0 \
    --target-cls 0 \
    --out-json "${RAW_JSON}")

if [ -n "${ANNOT_DIR}" ] && [ -d "${ANNOT_DIR}" ]; then
    TRACK_CMD+=(--annot-dir "${ANNOT_DIR}")
fi
if [ -n "${DETS_DIR}" ] && [ -d "${DETS_DIR}" ]; then
    TRACK_CMD+=(--existing-dets-dir "${DETS_DIR}")
fi

"${TRACK_CMD[@]}"

# Step 2: Run Microsecond Snapping Post-Processor & Render MP4
echo ""
echo ">>> [Phase 2/2] Running Detection Snapping & Video Rendering..."
"${PYTHON_EXEC}" "${SNAPPING_SCRIPT}" \
    --in-json "${RAW_JSON}" \
    --dets-dir "${DETS_DIR}" \
    --frames-dir "${FRAMES_DIR}" \
    --out-json "${SNAPPED_JSON}" \
    --out-video "${OUT_VIDEO}" \
    --fps "${FPS}"

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo ""
echo "=========================================================="
echo "🎉 [Success] SAM 2.1 Pipeline Finished in ${ELAPSED} seconds!"
echo "📄 Final Snapped JSON:  ${SNAPPED_JSON}"
echo "🎬 Rendered Clean Demo: ${OUT_VIDEO}"
echo "=========================================================="
