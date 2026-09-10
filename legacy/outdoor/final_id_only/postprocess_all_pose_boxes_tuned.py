#!/usr/bin/env python3
"""Apply the selected tuned pose-box filter to all four outdoor videos."""

import json
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sweep_pose_postprocess_hyperparams import deduplicate, valid


VIDEOS = ("A-5-1", "A-5-2", "A-5-3", "A-5-4")
KEYPOINT_DISTANCE_RATIO = 0.20
CONTAINMENT_THRESHOLD = 0.60
KEYPOINT_SCORE_THRESHOLD = 0.05


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def process_video(job):
    video, source_root, target_root, checkpoint_sha = job
    source_root, target_root = Path(source_root), Path(target_root)
    started = time.perf_counter()
    source_files = sorted((source_root / video / "frames").glob("frame_*.json"))
    if not source_files:
        raise FileNotFoundError(source_root / video / "frames")
    counts = {
        "video": video,
        "frames": 0,
        "boxes_before": 0,
        "boxes_after": 0,
        "removed_outside_or_invalid": 0,
        "removed_low_keypoint_score": 0,
        "removed_duplicate": 0,
    }
    for source in source_files:
        record = json.loads(source.read_text(encoding="utf-8"))
        if checkpoint_sha and record.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"checkpoint SHA mismatch: {source}")
        geometry_kept = []
        outside = 0
        low_score = 0
        for detection in record["detections"]:
            if int(detection.get("class_id", 0)) != 0:
                continue
            if valid(detection, None):
                if valid(detection, KEYPOINT_SCORE_THRESHOLD):
                    geometry_kept.append(detection)
                else:
                    low_score += 1
            else:
                outside += 1
        bees = deduplicate(geometry_kept, KEYPOINT_DISTANCE_RATIO, CONTAINMENT_THRESHOLD)
        duplicate_removed = len(geometry_kept) - len(bees)
        bee_ids = {id(d) for d in bees}
        kept = [d for d in record["detections"] if int(d.get("class_id", 0)) != 0 or id(d) in bee_ids]
        output = {
            **record,
            "detections": kept,
            "postprocessing": {
                "method": "both_keypoints_inside_and_min_score_then_pose_overlap_dedup",
                "keypoint_score_threshold": KEYPOINT_SCORE_THRESHOLD,
                "keypoint_distance_ratio": KEYPOINT_DISTANCE_RATIO,
                "containment_metric": "intersection_over_smaller_box_area",
                "containment_threshold": CONTAINMENT_THRESHOLD,
                "contained_duplicate_policy": "keep_larger_box",
                "removed_outside_or_invalid": outside,
                "removed_low_keypoint_score": low_score,
                "removed_duplicate": duplicate_removed,
            },
        }
        atomic_json(target_root / video / "frames" / source.name, output)
        counts["frames"] += 1
        counts["boxes_before"] += len(record["detections"])
        counts["boxes_after"] += len(kept)
        counts["removed_outside_or_invalid"] += outside
        counts["removed_low_keypoint_score"] += low_score
        counts["removed_duplicate"] += duplicate_removed
    counts["removed_total"] = counts["boxes_before"] - counts["boxes_after"]
    counts["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(target_root / video / "COMPLETE.json", counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="各 A-5 视频/frames 的父目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=list(VIDEOS))
    parser.add_argument("--checkpoint-sha", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        videos = list(executor.map(process_video, [
            (video, str(args.source), str(args.output), args.checkpoint_sha) for video in args.videos
        ]))
    fields = (
        "frames", "boxes_before", "boxes_after", "removed_outside_or_invalid",
        "removed_low_keypoint_score", "removed_duplicate", "removed_total",
    )
    summary = {
        "completed": True,
        "checkpoint_sha256": args.checkpoint_sha,
        "method": "both_keypoints_inside_and_min_score_then_pose_overlap_dedup",
        "keypoint_score_threshold": KEYPOINT_SCORE_THRESHOLD,
        "keypoint_distance_ratio": KEYPOINT_DISTANCE_RATIO,
        "containment_threshold": CONTAINMENT_THRESHOLD,
        "videos": videos,
        "total": {field: sum(v[field] for v in videos) for field in fields},
    }
    atomic_json(args.output / "COMPLETE.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
