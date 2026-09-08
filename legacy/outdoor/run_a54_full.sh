#!/usr/bin/env bash
# ==============================================================================
# Full-Scale Outdoor Tracking for Sequence A-5-4 (Frames 0 to 9007)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_FILE="${SCRIPT_DIR}/demo_outputs/a54_full_run.log"
mkdir -p "${SCRIPT_DIR}/demo_outputs"

echo "=========================================================="
echo "🚀 Starting Full-Scale A-5-4 Video Tracking Pipeline"
echo "Sequence:    A-5-4"
echo "Frames:      [0, 9007] (Total 9008 frames)"
echo "Log File:    ${LOG_FILE}"
echo "Start Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

bash "${SCRIPT_DIR}/run_pipeline.sh" a54_0_9007 > "${LOG_FILE}" 2>&1

echo "🎉 [Finished] Full A-5-4 Tracking Finished at $(date '+%Y-%m-%d %H:%M:%S')!" >> "${LOG_FILE}"
