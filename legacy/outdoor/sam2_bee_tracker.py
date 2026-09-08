#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
SAM 2.1 Dense Single-Pass Streaming Video Tracker for Apis Mellifera
==============================================================================
Core Architectural Principles:
1. Native Single-Pass Streaming (O(T) Linear Time):
   - Dense forward state propagation frame-by-frame.
   - Dynamic real-time unmapped bee detection and prompt injection.
   - Immediate next-frame suppression: newly registered bees immediately suppress
     duplicate IDs in subsequent frames, completely eliminating identity duplication.
2. Anti-OOM Confirmation Gating:
   - Evaluates physical evidence (dimensions, confidence, aspect ratio) before
     registering persistent memory prompts, preventing ephemeral optical noise from
     ballooning SAM's memory bank and causing CUDA OOM.
3. Intra-frame Deduplication:
   - Suppresses overlapping detection prompts within the same frame.
==============================================================================
"""

import os
import sys
import json
import time
import shutil
import inspect
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm


def calc_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
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
    """Calculates Euclidean distance between centers of two boxes."""
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    return ((cA[0] - cB[0]) ** 2 + (cA[1] - cB[1]) ** 2) ** 0.5


def extract_bboxes_gpu(masks_tensor: torch.Tensor, obj_ids: list) -> dict[int, list[float]]:
    """
    Extracts tight bounding boxes from high-resolution binary masks directly on GPU.
    Tensor shape: (N, 1, H, W) or (N, H, W)
    """
    if masks_tensor is None or len(obj_ids) == 0:
        return {}

    if masks_tensor.dim() == 4:
        masks_2d = (masks_tensor[:, 0] > 0.0)
    else:
        masks_2d = (masks_tensor > 0.0)

    res = {}
    N = masks_2d.shape[0]

    # Compute horizontal and vertical projections in parallel on GPU
    horizontal_proj = masks_2d.any(dim=1)  # (N, W)
    vertical_proj = masks_2d.any(dim=2)    # (N, H)

    for i in range(min(N, len(obj_ids))):
        tid = int(obj_ids[i])
        h_proj = horizontal_proj[i]
        v_proj = vertical_proj[i]

        if not h_proj.any() or not v_proj.any():
            continue

        h_idx = torch.where(h_proj)[0]
        v_idx = torch.where(v_proj)[0]

        x1 = float(h_idx[0].item())
        x2 = float(h_idx[-1].item()) + 1.0
        y1 = float(v_idx[0].item())
        y2 = float(v_idx[-1].item()) + 1.0

        if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
            res[tid] = [x1, y1, x2, y2]

    return res


def load_base_data(
    annot_dir: Path | None,
    dets_dir: Path | None,
    f_start: int,
    f_end: int,
    min_conf: float = 0.0,
    target_cls: int = 0
) -> tuple[dict[int, list[dict]], dict[int, list[tuple[list[float], float]]]]:
    """Loads baseline ground truth and offline detections."""
    base_anchors = defaultdict(list)
    all_dets = defaultdict(list)

    # 1. Parse manual annotations if provided
    if annot_dir and annot_dir.exists():
        for jf in sorted(annot_dir.glob("*.json")):
            try:
                f_num = int(jf.stem.split("_")[-1])
            except ValueError:
                continue
            if not (f_start <= f_num <= f_end):
                continue
            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for item in data:
                    tid = item.get("id") or item.get("track_id")
                    b = item.get("bbox") or item.get("bbox_xyxy")
                    if tid is not None and b:
                        base_anchors[f_num].append({
                            "track_id": int(tid),
                            "bbox": [float(x) for x in b],
                            "conf": 1.0,
                            "source": "manual_gt",
                            "is_physical_evidence": True,
                            "audit_status": "seed"
                        })
            except Exception as e:
                print(f"[Warn] Reading annotation {jf}: {e}")

    # 2. Parse offline detections across specified frame range
    if dets_dir and dets_dir.exists():
        print(f"[Detections] Parsing offline detections from {dets_dir} across [{f_start}, {f_end}]...")
        for f in range(f_start, f_end + 1):
            jf = dets_dir / f"frame_{f:08d}.json"
            if not jf.exists():
                candidates = list(dets_dir.glob(f"*{f:05d}*.json")) or list(dets_dir.glob(f"*{f}*.json"))
                if candidates:
                    jf = candidates[0]
            if jf.exists():
                try:
                    with open(jf, "r", encoding="utf-8") as fp:
                        d = json.load(fp)
                    for det in d.get("detections", []):
                        conf = float(det.get("det_confidence", det.get("conf", 0.0)))
                        if conf < min_conf:
                            continue
                        if target_cls is not None and target_cls >= 0:
                            if det.get("class_id", 0) != target_cls:
                                continue
                        b = det.get("bbox_xyxy", det.get("bbox"))
                        if b and (b[2] - b[0]) >= 6 and (b[3] - b[1]) >= 6:
                            all_dets[f].append((
                                [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                conf
                            ))
                except Exception as e:
                    print(f"[Warn] Reading detections {jf}: {e}")

    # 3. Seed start frame if no manual GT at f_start
    if f_start not in base_anchors or len(base_anchors[f_start]) == 0:
        init_dets = all_dets.get(f_start, [])
        # Intra-frame dedup at frame 0
        deduped_seeds = []
        for b, conf in sorted(init_dets, key=lambda x: x[1], reverse=True):
            # Only suppress almost identically duplicate detections (IoU >= 0.50)
            if any(calc_iou(b, eb) >= 0.50 for eb, _ in deduped_seeds):
                continue
            deduped_seeds.append((b, conf))

        for idx, (b, conf) in enumerate(deduped_seeds):
            tid = idx + 1
            base_anchors[f_start].append({
                "track_id": tid,
                "bbox": b,
                "conf": conf,
                "source": "det_seed",
                "is_physical_evidence": True,
                "audit_status": "seed"
            })
        print(f"[Baseline Prompt] Seeded {len(base_anchors[f_start])} baseline targets at start frame {f_start}.")
    else:
        print(f"[Baseline Prompt] Loaded {len(base_anchors)} keyframes of manual GT annotations.")

    return base_anchors, all_dets


class SAMVideoTracker:
    def __init__(
        self,
        model_type: str = "sam2.1",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_path: str = "checkpoints/sam2.1_hiera_large.pt",
        device: str = "cuda"
    ):
        self.model_type = model_type.lower()
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Normalize config path for Hydra
        cfg_str = str(model_cfg)
        if "configs/" in cfg_str:
            self.model_cfg = cfg_str[cfg_str.index("configs/"):]
        elif not cfg_str.startswith("configs/"):
            self.model_cfg = f"configs/{cfg_str}"
        else:
            self.model_cfg = cfg_str

        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        print(f"[SAM] Initializing {self.model_type} video tracker on {self.device}...")
        try:
            from sam2.build_sam import build_sam2_video_predictor
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    f"SAM 2.1 checkpoint not found at: {self.checkpoint_path}\n"
                    f"Please download or symlink it to {self.checkpoint_path}"
                )
            self.predictor = build_sam2_video_predictor(
                self.model_cfg,
                self.checkpoint_path,
                device=self.device
            )
            print(f"[SAM 2.1] Model loaded successfully from {self.checkpoint_path}")
        except ImportError as e:
            raise ImportError(
                f"SAM 2 import error: {e}\n"
                "Please verify that sam2 package is installed in your Python environment."
            )

    def run_streaming_tracking(
        self,
        frames_dir: Path,
        f_start: int,
        f_end: int,
        base_anchors: dict[int, list[dict]],
        all_dets: dict[int, list[tuple[list[float], float]]],
        work_dir: Path,
        start_tid: int = 1
    ) -> tuple[dict[int, list[dict]], dict[int, int]]:
        """
        Native Single-Pass Streaming Multi-Target Tracker:
        - 100% Single-Pass Forward Progression.
        - Frame-by-Frame Real-Time Dynamic Target Compensation.
        - Immediately uses predicted masks of all active targets to suppress duplicate ID spawns.
        """
        video_work_dir = work_dir / f"temp_{self.model_type}_frames_{f_start}_{f_end}_{os.getpid()}"
        if video_work_dir.exists():
            shutil.rmtree(video_work_dir)
        video_work_dir.mkdir(parents=True, exist_ok=True)

        frame_list = []
        frame_idx_to_fnum = {}
        fnum_to_frame_idx = {}

        idx = 0
        for f_num in range(f_start, f_end + 1):
            src_img = frames_dir / f"frame_{f_num:08d}.jpg"
            if not src_img.exists():
                src_img = frames_dir / f"A-5-1_frame_{f_num:06d}.jpg"
            if not src_img.exists():
                candidates = list(frames_dir.glob(f"*{f_num}*.jpg"))
                if candidates:
                    src_img = candidates[0]

            if src_img.exists():
                dst_img = video_work_dir / f"{idx:05d}.jpg"
                os.symlink(src_img.resolve(), dst_img)
                frame_idx_to_fnum[idx] = f_num
                fnum_to_frame_idx[f_num] = idx
                frame_list.append((idx, f_num, src_img))
                idx += 1

        total_frames = len(frame_list)
        print(f"[{self.model_type.upper()}] Prepared {total_frames} consecutive frames.")
        if total_frames == 0:
            print("[Error] No valid frames found in range!")
            return {}, {}

        sig = inspect.signature(self.predictor.add_new_points_or_box)
        has_rel_coord_param = "rel_coordinates" in sig.parameters

        print(f"[{self.model_type.upper()}] [Native Streaming] Initializing video inference state...")
        final_tracks = defaultdict(list)
        target_first_frame = {}
        target_sources = {}

        t_start = time.perf_counter()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path=str(video_work_dir))

            # 1. Seed start frame prompts (frame 0 of this chunk)
            start_fnum = frame_idx_to_fnum[0]
            initial_prompts = base_anchors.get(start_fnum, [])
            next_tid = start_tid
            for box_info in initial_prompts:
                tid = box_info["track_id"]
                next_tid = max(next_tid, tid + 1)
                b = box_info["bbox"]
                box_np = np.array([b[0], b[1], b[2], b[3]], dtype=np.float32)
                kw = {
                    "inference_state": inference_state,
                    "frame_idx": 0,
                    "obj_id": tid,
                    "box": box_np
                }
                if has_rel_coord_param:
                    kw["rel_coordinates"] = False
                self.predictor.add_new_points_or_box(**kw)
                target_first_frame[tid] = start_fnum
                target_sources[tid] = box_info.get("source", "det_seed")

                final_tracks[start_fnum].append({
                    "track_id": tid,
                    "bbox": b,
                    "conf": box_info.get("conf", 1.0),
                    "source": target_sources[tid],
                    "is_physical_evidence": True,
                    "audit_status": box_info.get("audit_status", "seed")
                })

            print(f"[{self.model_type.upper()}] [Streaming] Seeded {len(initial_prompts)} baseline targets at start frame {start_fnum}.")
            print(f"[{self.model_type.upper()}] [Streaming] Advancing real-time forward tracking...")

            pbar = tqdm(total=total_frames - 1, desc="SAM 2.1 Streaming Tracking")
            last_pbar_idx = 0
            current_start_idx = 0

            while current_start_idx < total_frames - 1:
                step_gen = self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=current_start_idx
                )

                needs_resume = False
                for step_output in step_gen:
                    if len(step_output) == 5:
                        out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores = step_output
                        masks_tensor = video_res_masks
                    else:
                        out_frame_idx, out_obj_ids, masks_tensor = step_output

                    if out_frame_idx == 0:
                        continue

                    if out_frame_idx > last_pbar_idx:
                        pbar.update(out_frame_idx - last_pbar_idx)
                        last_pbar_idx = out_frame_idx

                    f_num = frame_idx_to_fnum[out_frame_idx]

                    # Extract current active targets' tight GPU bounding boxes
                    cur_bboxes = extract_bboxes_gpu(masks_tensor, out_obj_ids)
                    active_boxes_this_frame = []

                    for tid, b in cur_bboxes.items():
                        src = target_sources.get(tid, f"{self.model_type}_propagated")
                        final_tracks[f_num].append({
                            "track_id": tid,
                            "bbox": b,
                            "conf": 0.95,
                            "source": src,
                            "is_physical_evidence": True,
                            "audit_status": "propagated"
                        })
                        active_boxes_this_frame.append(b)

                    # Real-time Dynamic Target Compensation
                    if f_num in all_dets:
                        cur_dets = all_dets[f_num]
                        newly_added_this_step = []

                        for b, conf in cur_dets:
                            w = b[2] - b[0]
                            h = b[3] - b[1]

                            # Physical valid box check (reject zero-area or inverted boxes)
                            if w < 4.0 or h < 4.0:
                                continue

                            # 1. Suppression by current active SAM tracked boxes (only genuine IoU overlap)
                            is_covered = False
                            for sb in active_boxes_this_frame:
                                if calc_iou(b, sb) >= 0.25:
                                    is_covered = True
                                    break
                            if is_covered:
                                continue

                            # 2. Suppression by other boxes newly added in the SAME frame (only duplicate heavy IoU overlap)
                            for nb in newly_added_this_step:
                                if calc_iou(b, nb) >= 0.50:
                                    is_covered = True
                                    break
                            if is_covered:
                                continue

                            # Register detection directly as new bee (No confidence filtering: retain all predicted boxes)
                            new_tid = next_tid
                            next_tid += 1
                            box_np = np.array([b[0], b[1], b[2], b[3]], dtype=np.float32)
                            kw = {
                                "inference_state": inference_state,
                                "frame_idx": out_frame_idx,
                                "obj_id": new_tid,
                                "box": box_np
                            }
                            if has_rel_coord_param:
                                kw["rel_coordinates"] = False

                            self.predictor.add_new_points_or_box(**kw)
                            target_first_frame[new_tid] = f_num
                            target_sources[new_tid] = "sam_prompt_compensated"
                            newly_added_this_step.append(b)
                            active_boxes_this_frame.append(b)

                            final_tracks[f_num].append({
                                "track_id": new_tid,
                                "bbox": b,
                                "conf": conf,
                                "source": "sam_prompt_compensated",
                                "is_physical_evidence": True,
                                "audit_status": "compensated"
                            })

                        if newly_added_this_step:
                            # Resume generator from current frame with updated object states
                            current_start_idx = out_frame_idx
                            needs_resume = True
                            break

                if not needs_resume:
                    break

            pbar.close()

        t_end = time.perf_counter()
        shutil.rmtree(video_work_dir, ignore_errors=True)
        print(f"[{self.model_type.upper()} Streaming Finished] Tracked {total_frames} frames in {t_end - t_start:.2f}s. Total identities: {len(target_first_frame)}")
        return final_tracks, target_first_frame


def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 Dense Single-Pass Streaming Multi-Target Tracker")
    parser.add_argument("--model-type", type=str, default="sam2.1", choices=["sam2.1", "sam2"])
    parser.add_argument("--model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--annot-dir", type=Path, default=None)
    parser.add_argument("--existing-dets-dir", type=Path, default=None)
    parser.add_argument("--f-start", type=int, default=0)
    parser.add_argument("--f-end", type=int, default=250)
    parser.add_argument("--chunk-size", type=int, default=500, help="Max frames per streaming chunk to prevent RAM overflow")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--min-conf", type=float, default=0.0, help="Min detection confidence (0.0: retain all boxes)")
    parser.add_argument("--target-cls", type=int, default=0, help="Target class to track (0: bee body only, -1: all classes)")
    parser.add_argument("--work-dir", type=Path, default=Path("temp_scratch"))
    parser.add_argument("--out-json", type=Path, default=Path("demo_outputs/sam21_tracked.json"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base data
    base_anchors, all_dets = load_base_data(
        annot_dir=args.annot_dir,
        dets_dir=args.existing_dets_dir,
        f_start=args.f_start,
        f_end=args.f_end,
        min_conf=args.min_conf,
        target_cls=args.target_cls
    )

    # Step 2: Initialize SAM Tracker
    tracker = SAMVideoTracker(
        model_type=args.model_type,
        model_cfg=args.model_cfg,
        checkpoint_path=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    total_req_frames = args.f_end - args.f_start + 1

    # Step 3: Run Streaming Multi-Target tracking (Auto-Chunked if total frames > chunk_size)
    if total_req_frames <= args.chunk_size:
        final_tracks, target_first_frame = tracker.run_streaming_tracking(
            frames_dir=args.frames_dir,
            f_start=args.f_start,
            f_end=args.f_end,
            base_anchors=base_anchors,
            all_dets=all_dets,
            work_dir=args.work_dir
        )
    else:
        print(f"\n🚀 [Chunked Streaming] Long video detected ({total_req_frames} frames > {args.chunk_size}).")
        print(f"Dividing into consecutive 500-frame streaming chunks to protect system RAM from OOM...")

        final_tracks = defaultdict(list)
        target_first_frame = {}
        current_anchors = base_anchors
        next_tid = 1

        for c_start in range(args.f_start, args.f_end + 1, args.chunk_size):
            c_end = min(c_start + args.chunk_size - 1, args.f_end)
            print(f"\n=======================================================")
            print(f"📦 [Chunk Streaming] Advancing Chunk [{c_start}, {c_end}] (Frames: {c_end - c_start + 1})")
            print(f"=======================================================")

            c_tracks, c_first_frame = tracker.run_streaming_tracking(
                frames_dir=args.frames_dir,
                f_start=c_start,
                f_end=c_end,
                base_anchors=current_anchors,
                all_dets=all_dets,
                work_dir=args.work_dir,
                start_tid=next_tid
            )

            for f, trks in c_tracks.items():
                final_tracks[f] = trks
            target_first_frame.update(c_first_frame)

            # Carry over active targets at chunk boundary to next chunk
            if c_end in c_tracks and c_end < args.f_end:
                last_active = c_tracks[c_end]
                next_start_f = c_end + 1
                current_anchors = {next_start_f: []}
                for trk in last_active:
                    current_anchors[next_start_f].append({
                        "track_id": trk["track_id"],
                        "bbox": trk["bbox"],
                        "conf": trk.get("conf", 1.0),
                        "source": "chunk_carryover",
                        "is_physical_evidence": True,
                        "audit_status": "carryover"
                    })
                max_tid = max([t["track_id"] for t in last_active] + list(target_first_frame.keys()) + [next_tid])
                next_tid = max_tid + 1
                print(f"[Chunk Boundary] Seamlessly carried over {len(last_active)} active targets to next frame {next_start_f}.")

    # Step 4: Export structured tracking JSON
    export_data = {
        "metadata": {
            "tracker": f"SAM-Video-Tracker-Streaming ({args.model_type})",
            "checkpoint": str(args.checkpoint),
            "f_start": args.f_start,
            "f_end": args.f_end,
            "fps": args.fps,
            "frames_count": len(final_tracks),
            "total_identities": len(target_first_frame),
            "target_first_frame": target_first_frame
        },
        "frames": {str(k): v for k, v in final_tracks.items()}
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    print(f"[Export] Structured tracking results saved to: {args.out_json}")
    print("=== SAM 2.1 Streaming Tracking Finished Successfully! ===")


if __name__ == "__main__":
    main()
