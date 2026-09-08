#!/usr/bin/env python3
"""Summarize Y-route prediction and tracking JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "unknown")
        detections = row.get("detections") or []
        slot = per_video.setdefault(
            video_id,
            {
                "frames": 0,
                "detections": 0,
                "domain": row.get("domain"),
                "max_detections_per_frame": 0,
            },
        )
        slot["frames"] += 1
        slot["detections"] += len(detections)
        slot["max_detections_per_frame"] = max(slot["max_detections_per_frame"], len(detections))

    for slot in per_video.values():
        frames = max(int(slot["frames"]), 1)
        slot["avg_detections_per_frame"] = slot["detections"] / frames
    return per_video


def summarize_tracks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_video_frames: Counter[str] = Counter()
    per_video_track_frames: dict[str, Counter[int]] = defaultdict(Counter)
    per_video_track_dets: Counter[str] = Counter()

    for row in rows:
        video_id = str(row.get("video_id") or "unknown")
        per_video_frames[video_id] += 1
        for trk in row.get("tracks") or []:
            track_id = int(trk["track_id"])
            per_video_track_frames[video_id][track_id] += 1
            per_video_track_dets[video_id] += 1

    summary: dict[str, Any] = {}
    for video_id, frames in per_video_frames.items():
        lengths = list(per_video_track_frames[video_id].values())
        summary[video_id] = {
            "frames": frames,
            "track_detections": per_video_track_dets[video_id],
            "tracks": len(lengths),
            "mean_track_length": mean(lengths) if lengths else 0.0,
            "median_track_length": median(lengths) if lengths else 0.0,
            "max_track_length": max(lengths) if lengths else 0,
            "single_frame_tracks": sum(1 for x in lengths if x == 1),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--track-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pred_rows = read_jsonl(args.pred_jsonl)
    track_rows = read_jsonl(args.track_jsonl)
    payload = {
        "prediction_frames": len(pred_rows),
        "tracking_frames": len(track_rows),
        "predictions_by_video": summarize_predictions(pred_rows),
        "tracks_by_video": summarize_tracks(track_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
