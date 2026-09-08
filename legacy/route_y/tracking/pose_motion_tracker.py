#!/usr/bin/env python3
"""Pose-aware track-by-detection for BeePoseTrack JSONL detections."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    head: np.ndarray | None
    tail: np.ndarray | None
    frame_id: int
    missed: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=1.35)
    parser.add_argument("--max-center-dist", type=float, default=140.0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=768)
    parser.add_argument("--max-new-tracks-per-frame", type=int, default=540)
    parser.add_argument("--max-active-tracks", type=int, default=768)
    parser.add_argument("--hungarian-limit", type=int, default=600)
    return parser.parse_args()


def center(box):
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def angle(head, tail):
    if head is None or tail is None:
        return None
    vec = tail - head
    if np.linalg.norm(vec) <= 1e-9:
        return None
    return math.atan2(float(vec[1]), float(vec[0]))


def angle_delta(a, b):
    if a is None or b is None:
        return 0.5
    diff = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    return diff / math.pi


def length(head, tail):
    if head is None or tail is None:
        return None
    return float(np.linalg.norm(tail - head))


def as_point(value):
    return None if value is None else np.asarray(value, dtype=float)


def cost(track: Track, det: dict, frame_id: int, max_center_dist: float) -> float:
    box = np.asarray(det["bbox_xyxy"], dtype=float)
    dt = max(frame_id - track.frame_id, 1)
    predicted_center = center(track.bbox) + track.velocity * dt
    center_dist = float(np.linalg.norm(center(box) - predicted_center))
    center_cost = min(center_dist / max(max_center_dist, 1.0), 1.5)
    iou_cost = 1.0 - iou(track.bbox, box)
    head = as_point(det.get("head"))
    tail = as_point(det.get("tail"))
    pose_cost = angle_delta(angle(track.head, track.tail), angle(head, tail))
    old_len = length(track.head, track.tail)
    new_len = length(head, tail)
    len_cost = 0.25 if old_len is None or new_len is None else min(abs(new_len - old_len) / max(old_len, 1.0), 1.0)
    score_penalty = 1.0 - float(det.get("score", 0.0))
    return 0.42 * center_cost + 0.28 * iou_cost + 0.16 * pose_cost + 0.07 * len_cost + 0.07 * score_penalty


def vectorized_costs(tracks: list[Track], detections: list[dict], frame_id: int, max_center_dist: float) -> np.ndarray:
    track_boxes = np.asarray([t.bbox for t in tracks], dtype=float)
    det_boxes = np.asarray([d["bbox_xyxy"] for d in detections], dtype=float)
    track_centers = np.column_stack(((track_boxes[:, 0] + track_boxes[:, 2]) / 2, (track_boxes[:, 1] + track_boxes[:, 3]) / 2))
    det_centers = np.column_stack(((det_boxes[:, 0] + det_boxes[:, 2]) / 2, (det_boxes[:, 1] + det_boxes[:, 3]) / 2))
    dt = np.asarray([max(frame_id - t.frame_id, 1) for t in tracks], dtype=float)[:, None]
    velocities = np.asarray([t.velocity for t in tracks], dtype=float)
    predicted_centers = track_centers + velocities * dt
    center_dist = np.linalg.norm(predicted_centers[:, None, :] - det_centers[None, :, :], axis=2)
    center_cost = np.minimum(center_dist / max(max_center_dist, 1.0), 1.5)

    x1 = np.maximum(track_boxes[:, None, 0], det_boxes[None, :, 0])
    y1 = np.maximum(track_boxes[:, None, 1], det_boxes[None, :, 1])
    x2 = np.minimum(track_boxes[:, None, 2], det_boxes[None, :, 2])
    y2 = np.minimum(track_boxes[:, None, 3], det_boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    track_area = np.maximum(0.0, track_boxes[:, 2] - track_boxes[:, 0]) * np.maximum(0.0, track_boxes[:, 3] - track_boxes[:, 1])
    det_area = np.maximum(0.0, det_boxes[:, 2] - det_boxes[:, 0]) * np.maximum(0.0, det_boxes[:, 3] - det_boxes[:, 1])
    iou_cost = 1.0 - inter / np.maximum(track_area[:, None] + det_area[None, :] - inter, 1e-9)

    track_angles = np.asarray([np.nan if angle(t.head, t.tail) is None else angle(t.head, t.tail) for t in tracks], dtype=float)
    det_angles = np.asarray([np.nan if angle(as_point(d.get("head")), as_point(d.get("tail"))) is None else angle(as_point(d.get("head")), as_point(d.get("tail"))) for d in detections], dtype=float)
    angle_diff = np.abs((track_angles[:, None] - det_angles[None, :] + math.pi) % (2 * math.pi) - math.pi) / math.pi
    pose_cost = np.where(np.isnan(angle_diff), 0.5, angle_diff)

    track_lengths = np.asarray([np.nan if length(t.head, t.tail) is None else length(t.head, t.tail) for t in tracks], dtype=float)
    det_lengths = np.asarray([np.nan if length(as_point(d.get("head")), as_point(d.get("tail"))) is None else length(as_point(d.get("head")), as_point(d.get("tail"))) for d in detections], dtype=float)
    len_delta = np.abs(track_lengths[:, None] - det_lengths[None, :]) / np.maximum(track_lengths[:, None], 1.0)
    len_cost = np.where(np.isnan(len_delta), 0.25, np.minimum(len_delta, 1.0))

    score_penalty = 1.0 - np.asarray([float(d.get("score", 0.0)) for d in detections], dtype=float)[None, :]
    return 0.42 * center_cost + 0.28 * iou_cost + 0.16 * pose_cost + 0.07 * len_cost + 0.07 * score_penalty


def track_score(det: dict) -> float:
    return float(det.get("score", 0.0)) * max(float(det.get("pose_score", 0.0)), 0.25)


def assign_tracks(tracks: list[Track], detections: list[dict], frame_id: int, max_center_dist: float, max_cost: float, hungarian_limit: int) -> list[tuple[int, int]]:
    if not tracks or not detections:
        return []
    costs = vectorized_costs(tracks, detections, frame_id, max_center_dist)
    pairs: list[tuple[int, int]] = []
    if max(costs.shape) <= hungarian_limit:
        rows, cols = linear_sum_assignment(costs)
        for row, col in zip(rows, cols):
            if float(costs[row, col]) <= max_cost:
                pairs.append((int(row), int(col)))
        return pairs

    candidates = []
    for row in range(costs.shape[0]):
        col = int(np.argmin(costs[row]))
        value = float(costs[row, col])
        if value <= max_cost:
            candidates.append((value, row, col))
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for _, row, col in sorted(candidates, key=lambda x: x[0]):
        if row in used_tracks or col in used_dets:
            continue
        used_tracks.add(row)
        used_dets.add(col)
        pairs.append((row, col))
    return pairs


def update_track(track: Track, det: dict, frame_id: int) -> None:
    box = np.asarray(det["bbox_xyxy"], dtype=float)
    dt = max(frame_id - track.frame_id, 1)
    track.velocity = 0.35 * track.velocity + 0.65 * ((center(box) - center(track.bbox)) / dt)
    track.bbox = box
    track.head = as_point(det.get("head"))
    track.tail = as_point(det.get("tail"))
    track.frame_id = frame_id
    track.missed = 0


def main() -> None:
    args = parse_args()
    frames = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines()]
    frames.sort(key=lambda r: (str(r.get("section_id", r["video_id"])), int(r["frame_id"])))
    next_id = 1
    tracks_by_sequence: dict[str, list[Track]] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for frame in frames:
            video_id = str(frame["video_id"])
            section_id = str(frame.get("section_id", video_id))
            frame_id = int(frame["frame_id"])
            tracks = tracks_by_sequence.setdefault(section_id, [])
            tracks[:] = sorted([t for t in tracks if t.missed <= args.max_age], key=lambda t: (t.missed, -t.frame_id))[: args.max_active_tracks]
            detections = sorted(frame.get("detections", []), key=track_score, reverse=True)[: args.max_candidates_per_frame]
            assigned_tracks: set[int] = set()
            assigned_dets: set[int] = set()
            if tracks and detections:
                for row, col in assign_tracks(tracks, detections, frame_id, args.max_center_dist, args.max_cost, args.hungarian_limit):
                    update_track(tracks[row], detections[col], frame_id)
                    assigned_tracks.add(row)
                    assigned_dets.add(col)
            for idx, track in enumerate(tracks):
                if idx not in assigned_tracks:
                    track.missed += 1
            new_track_count = 0
            for det_idx, det in enumerate(detections):
                if det_idx in assigned_dets:
                    continue
                if new_track_count >= args.max_new_tracks_per_frame:
                    break
                box = np.asarray(det["bbox_xyxy"], dtype=float)
                tracks.append(Track(next_id, box, as_point(det.get("head")), as_point(det.get("tail")), frame_id))
                assigned_tracks.add(len(tracks) - 1)
                assigned_dets.add(det_idx)
                new_track_count += 1
                next_id += 1
            tracks[:] = [t for t in tracks if t.missed <= args.max_age]
            det_records = []
            for track in tracks:
                if track.frame_id != frame_id or track.missed:
                    continue
                det_records.append(
                    {
                        "track_id": track.track_id,
                        "bbox_xyxy": [round(float(v), 3) for v in track.bbox.tolist()],
                        "head": None if track.head is None else [round(float(v), 3) for v in track.head.tolist()],
                        "tail": None if track.tail is None else [round(float(v), 3) for v in track.tail.tolist()],
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "video_id": video_id,
                        "section_id": section_id,
                        "frame_id": frame_id,
                        "domain": frame.get("domain", "unknown"),
                        "tracks": det_records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
