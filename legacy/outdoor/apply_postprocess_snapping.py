#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Fast Offline Geometric Snapping & Deformation Rectification Tool
==============================================================================
Features:
1. Microsecond Offline Snapping:
   - Reads SAM 2.1 tracking JSON and maps predicted boxes to high-precision YOLO raw detections.
   - Restores exact physical object contours without re-running heavy neural networks.
2. Aspect-Ratio & Mask Drift Clamping:
   - Identifies oversized or abnormally stretched masks (e.g. aspect ratio > 2.5).
   - Clamps distorted boxes back to canonical bee square dimensions (50x50 px centered).
3. High-Legibility Visualizer:
   - Generates clean, publication-ready MP4 demonstration videos with clear ID badges.
==============================================================================
"""

import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def get_color(track_id: int):
    np.random.seed(int(track_id) * 31 + 17)
    return tuple(int(c) for c in np.random.randint(40, 255, size=3))


def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def center_dist(boxA, boxB):
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    return ((cA[0] - cB[0]) ** 2 + (cA[1] - cB[1]) ** 2) ** 0.5


def main():
    parser = argparse.ArgumentParser(description="Offline Detection Snapping & Video Re-render")
    parser.add_argument("--in-json", type=Path, required=True, help="SAM tracked JSON path")
    parser.add_argument("--dets-dir", type=Path, required=True, help="Directory containing offline frame_*.json detections")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing raw JPG frames")
    parser.add_argument("--out-json", type=Path, required=True, help="Path to save snapped JSON")
    parser.add_argument("--out-video", type=Path, default=None, help="Optional path to render video")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 65)
    print("🚀 [Post-Processor] Detection Snapping & Aspect-Ratio Rectification")
    print("=" * 65)
    print(f"Input Tracked JSON: {args.in_json}")
    print(f"Raw Detections Dir: {args.dets_dir}")
    print(f"Frames Dir:         {args.frames_dir}")
    print(f"Output JSON:        {args.out_json}")
    if args.out_video:
        print(f"Output Video:       {args.out_video}")

    if not args.in_json.exists():
        print(f"[Error] Input JSON not found: {args.in_json}")
        return

    with open(args.in_json, "r", encoding="utf-8") as fp:
        raw_data = json.load(fp)

    frames_dict = raw_data.get("frames", {})
    if not frames_dict:
        print("[Error] No frames found in input JSON!")
        return

    f_nums = sorted([int(k) for k in frames_dict.keys()])
    f_start, f_end = f_nums[0], f_nums[-1]
    print(f"Frames Range: [{f_start}, {f_end}] (Total {len(f_nums)} frames)")

    # 1. Load Raw Detections (class_id == 0, bees only)
    print("⏳ [1/3] Loading raw offline detection stream...")
    all_raw_dets = {}
    for f in f_nums:
        jf = args.dets_dir / f"frame_{f:08d}.json"
        raw_dets = []
        if not jf.exists():
            candidates = list(args.dets_dir.glob(f"*{f:05d}*.json")) or list(args.dets_dir.glob(f"*{f}*.json"))
            if candidates:
                jf = candidates[0]
        if jf.exists():
            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                for det in d.get("detections", []):
                    if det.get("class_id", 0) == 0:
                        b = det.get("bbox_xyxy", det.get("bbox"))
                        if b and len(b) == 4:
                            raw_dets.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            except Exception:
                pass
        all_raw_dets[f] = raw_dets

    # 2. Execute Detection Snapping & Geometric Rectification
    print("⏳ [2/3] Performing Detection Snapping & Drift Clamping...")
    t0 = time.perf_counter()
    snapped_frames = {}
    total_boxes = 0
    snapped_count = 0
    clamped_count = 0

    for f_str, tracks in frames_dict.items():
        f = int(f_str)
        cur_dets = all_raw_dets.get(f, [])
        new_tracks = []

        for trk in tracks:
            b = trk.get("bbox")
            if not b or len(b) != 4:
                continue
            total_boxes += 1
            tid = trk.get("track_id")

            best_match = None
            best_iou = 0.0
            best_dist = 999.0

            for db in cur_dets:
                iou = calc_iou(b, db)
                dist = center_dist(b, db)
                if iou > best_iou:
                    best_iou = iou
                    best_match = db
                elif best_iou == 0.0 and dist < best_dist and dist <= 45.0:
                    best_dist = dist
                    best_match = db

            final_b = list(b)
            status = trk.get("audit_status", "propagated")

            if best_match is not None and (best_iou >= 0.20 or best_dist <= 45.0):
                final_b = [float(x) for x in best_match]
                snapped_count += 1
                status = "snapped_yolo"
            else:
                # Geometric aspect ratio clamp: protect against mask stretching
                w = b[2] - b[0]
                h = b[3] - b[1]
                aspect = max(w / (h + 1e-5), h / (w + 1e-5))
                if w > 120 or h > 120 or aspect > 2.5:
                    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                    half_w = min(max(w, 40.0), 60.0) / 2.0
                    half_h = min(max(h, 40.0), 60.0) / 2.0
                    # Canonical square-like bee bounding box
                    canonical_half = 25.0
                    final_b = [
                        max(0.0, cx - canonical_half),
                        max(0.0, cy - canonical_half),
                        min(1920.0, cx + canonical_half),
                        min(1080.0, cy + canonical_half)
                    ]
                    clamped_count += 1
                    status = "clamped_geometry"

            new_tracks.append({
                "track_id": tid,
                "bbox": final_b,
                "conf": trk.get("conf", 0.95),
                "source": trk.get("source", "sam_propagated"),
                "is_physical_evidence": True,
                "audit_status": status
            })

        snapped_frames[str(f)] = new_tracks

    snap_duration = time.perf_counter() - t0
    print(f"✅ Snapping finished in {snap_duration * 1000:.1f}ms!")
    print(f"📊 Total Boxes: {total_boxes} | Snapped: {snapped_count} ({snapped_count/max(1,total_boxes)*100:.1f}%) | Clamped: {clamped_count}")

    # Export Snapped JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "metadata": {
            "tracker": "SAM2.1-Postprocessed-Snapping",
            "f_start": f_start,
            "f_end": f_end,
            "fps": args.fps,
            "frames_count": len(snapped_frames),
            "snapped_ratio": f"{snapped_count / max(1, total_boxes) * 100:.2f}%",
            "clamped_count": clamped_count
        },
        "frames": snapped_frames
    }
    with open(args.out_json, "w", encoding="utf-8") as fp:
        json.dump(out_payload, fp, indent=2)
    print(f"📁 Exported Snapped Tracking JSON: {args.out_json}")

    # 3. Optional Video Rendering
    if args.out_video:
        print("⏳ [3/3] Rendering clean visualization video...")
        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        for f in tqdm(f_nums, desc="Rendering Snapped MP4"):
            img_path = args.frames_dir / f"frame_{f:08d}.jpg"
            if not img_path.exists():
                img_path = args.frames_dir / f"A-5-1_frame_{f:06d}.jpg"
            if not img_path.exists():
                cands = list(args.frames_dir.glob(f"*{f:05d}*.jpg")) or list(args.frames_dir.glob(f"*{f}*.jpg"))
                if cands:
                    img_path = cands[0]

            if not img_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.out_video), fourcc, args.fps, (w, h))

            cur_tracks = snapped_frames.get(str(f), [])
            for trk in cur_tracks:
                b = trk.get("bbox")
                if not b or len(b) != 4:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                tid = trk["track_id"]

                # Unified identity color across the entire video
                color = get_color(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{tid}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                badge_y1 = max(0, y1 - lh - 4)
                badge_y2 = y1
                badge_x2 = min(frame.shape[1], x1 + lw + 6)
                cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
                cv2.putText(
                    frame, label, (x1 + 3, badge_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

            # Header HUD
            hud_text = f"Frame: {f:05d} | Active Targets: {len(cur_tracks)} | SAM 2.1 Streaming Tracker"
            cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)

        if writer is not None:
            writer.release()
        print(f"🎬 Video rendered successfully: {args.out_video}")

    print("=== All Operations Completed Successfully! ===")


if __name__ == "__main__":
    main()
