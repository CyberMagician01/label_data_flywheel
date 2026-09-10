#!/usr/bin/env python3
"""Conservative second-stage stitching of outdoor bidirectional tracklets."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("BEE_OUTDOOR_CONFIG", HERE / "config.json")).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ROOT = (CONFIG_PATH.parent / CONFIG["workspace_root"]).resolve()
EXTRACTED = (CONFIG_PATH.parent / CONFIG["extracted_root"]).resolve()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def tracked_bees(row: dict) -> list[dict]:
    return [
        det for det in row.get("detections", [])
        if int(det.get("class_id", 0)) == 0
        and "interpolation" not in str(det.get("origin", ""))
    ]


def orientation(det: dict) -> np.ndarray | None:
    points = det.get("keypoints")
    if not isinstance(points, dict):
        return None
    head, tail = points.get("head"), points.get("abdomen_tip")
    if not (isinstance(head, list) and isinstance(tail, list) and len(head) >= 2 and len(tail) >= 2):
        return None
    value = np.asarray(tail[:2], dtype=np.float32) - np.asarray(head[:2], dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-6 else None


@dataclass
class Observation:
    frame: int
    index: int
    box: np.ndarray
    center: np.ndarray
    diagonal: float
    orientation: np.ndarray | None


@dataclass
class Tracklet:
    source_id: int
    observations: list[Observation]
    start_frame: int
    end_frame: int
    start_center: np.ndarray
    end_center: np.ndarray
    start_velocity: np.ndarray | None
    end_velocity: np.ndarray | None
    start_diagonal: float
    end_diagonal: float
    start_orientation: np.ndarray | None
    end_orientation: np.ndarray | None
    start_descriptor: np.ndarray
    end_descriptor: np.ndarray


def estimate_velocity(observations: list[Observation]) -> np.ndarray | None:
    if len(observations) < 2:
        return None
    frames = np.asarray([obs.frame for obs in observations], dtype=np.float64)
    if frames[-1] == frames[0]:
        return None
    centers = np.asarray([obs.center for obs in observations], dtype=np.float64)
    centered = frames - frames.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return None
    return (centered[:, None] * centers).sum(axis=0) / denominator


def read_image(sequence: str, frame: int, cache: collections.OrderedDict[int, np.ndarray]) -> np.ndarray:
    if frame not in cache:
        path = EXTRACTED / "frames" / sequence / f"frame_{frame:08d}.jpg"
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot decode {path}")
        cache[frame] = image
        if len(cache) > 64:
            cache.popitem(last=False)
    else:
        cache.move_to_end(frame)
    return cache[frame]


def patch_descriptor(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x1, y1, x2, y2 = np.round(box).astype(int)
    pad_x, pad_y = max(1, (x2 - x1) // 6), max(1, (y2 - y1) // 6)
    x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(144, dtype=np.float32)
    patch = cv2.resize(image[y1:y2, x1:x2], (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
    vector = patch.ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else vector


def endpoint_descriptor(
    sequence: str,
    observations: list[Observation],
    cache: collections.OrderedDict[int, np.ndarray],
) -> np.ndarray:
    values = [patch_descriptor(read_image(sequence, obs.frame, cache), obs.box) for obs in observations]
    value = np.mean(values, axis=0)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-6 else value


def make_tracklet(
    sequence: str,
    source_id: int,
    observations: list[Observation],
    history: int,
    cache: collections.OrderedDict[int, np.ndarray],
) -> Tracklet:
    observations.sort(key=lambda obs: obs.frame)
    first = observations[:history]
    last = observations[-history:]
    return Tracklet(
        source_id=source_id,
        observations=observations,
        start_frame=observations[0].frame,
        end_frame=observations[-1].frame,
        start_center=observations[0].center,
        end_center=observations[-1].center,
        start_velocity=estimate_velocity(first),
        end_velocity=estimate_velocity(last),
        start_diagonal=observations[0].diagonal,
        end_diagonal=observations[-1].diagonal,
        start_orientation=observations[0].orientation,
        end_orientation=observations[-1].orientation,
        start_descriptor=endpoint_descriptor(sequence, first, cache),
        end_descriptor=endpoint_descriptor(sequence, last, cache),
    )


def candidate_edge(left: Tracklet, right: Tracklet, max_speed: float) -> dict | None:
    gap = right.start_frame - left.end_frame
    if gap <= 0:
        return None
    direct = float(np.linalg.norm(right.start_center - left.end_center))
    speed = direct / gap
    if speed > max_speed:
        return None
    scale = max(10.0, min(left.end_diagonal, right.start_diagonal))
    size_ratio = max(left.end_diagonal, right.start_diagonal) / max(min(left.end_diagonal, right.start_diagonal), 1e-6)
    if size_ratio > 3.2:
        return None
    if left.end_velocity is None:
        forward_error = direct / (1 + 0.35 * gap)
        missing_velocity = 1
    else:
        forward_error = float(np.linalg.norm(left.end_center + left.end_velocity * gap - right.start_center))
        missing_velocity = 0
    if right.start_velocity is None:
        backward_error = direct / (1 + 0.35 * gap)
        missing_velocity += 1
    else:
        backward_error = float(np.linalg.norm(right.start_center - right.start_velocity * gap - left.end_center))
    motion = 0.5 * (forward_error + backward_error) / scale
    appearance_similarity = float(left.end_descriptor @ right.start_descriptor)
    orientation_similarity = None
    if left.end_orientation is not None and right.start_orientation is not None:
        orientation_similarity = float(left.end_orientation @ right.start_orientation)
    cost = (
        motion
        + 0.30 * (1 - appearance_similarity)
        + 0.22 * math.log(max(size_ratio, 1.0))
        + 0.035 * (gap - 1)
        + 0.10 * missing_velocity
    )
    if orientation_similarity is not None:
        cost += 0.10 * (1 - orientation_similarity)
    return {
        "left_id": left.source_id,
        "right_id": right.source_id,
        "left_end": left.end_frame,
        "right_start": right.start_frame,
        "gap": gap,
        "cost": cost,
        "motion_cost": motion,
        "direct_speed": speed,
        "appearance_similarity": appearance_similarity,
        "orientation_similarity": orientation_similarity,
        "size_ratio": size_ratio,
        "missing_velocity": missing_velocity,
    }


class UnionFind:
    def __init__(self, nodes: list[int]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, choices=[f"A-5-{i}" for i in range(1, 5)])
    parser.add_argument("--source-run", default="benchmark_bidir_age8")
    parser.add_argument("--run-name", default="benchmark_stitch_v1")
    parser.add_argument("--max-gap", type=int, default=12)
    parser.add_argument("--max-cost", type=float, default=1.35)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-appearance", type=float, default=-0.25)
    parser.add_argument("--max-speed", type=float, default=85.0)
    parser.add_argument("--history", type=int, default=4)
    args = parser.parse_args()
    cv2.setNumThreads(2)

    source_root = ROOT / "outputs/optimized_v1/02_bidirectional" / args.source_run / "associated/frames" / args.sequence
    paths = sorted(source_root.glob("frame_*.json"))
    if not paths:
        raise RuntimeError(f"no source frames under {source_root}")
    rows = {int(path.stem.rsplit("_", 1)[1]): json.loads(path.read_text()) for path in paths}
    observations_by_id: dict[int, list[Observation]] = collections.defaultdict(list)
    started = time.time()
    for frame, row in rows.items():
        for index, det in enumerate(tracked_bees(row)):
            box = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            center = (box[:2] + box[2:]) / 2
            observations_by_id[int(det["track_id"])].append(
                Observation(frame, index, box, center, float(np.linalg.norm(box[2:] - box[:2])), orientation(det))
            )
    image_cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
    tracklets = {
        source_id: make_tracklet(args.sequence, source_id, observations, args.history, image_cache)
        for source_id, observations in observations_by_id.items()
    }
    starts: dict[int, list[Tracklet]] = collections.defaultdict(list)
    for tracklet in tracklets.values():
        starts[tracklet.start_frame].append(tracklet)
    raw_candidates = []
    for left in tracklets.values():
        for start in range(left.end_frame + 1, left.end_frame + args.max_gap + 1):
            for right in starts.get(start, []):
                edge = candidate_edge(left, right, args.max_speed)
                if edge is not None and edge["cost"] <= args.max_cost and edge["appearance_similarity"] >= args.min_appearance:
                    raw_candidates.append(edge)

    by_left: dict[int, list[dict]] = collections.defaultdict(list)
    by_right: dict[int, list[dict]] = collections.defaultdict(list)
    for edge in raw_candidates:
        by_left[edge["left_id"]].append(edge)
        by_right[edge["right_id"]].append(edge)
    def margin(edges: list[dict]) -> float:
        values = sorted(edge["cost"] for edge in edges)
        return values[1] - values[0] if len(values) > 1 else float("inf")
    left_margin = {key: margin(value) for key, value in by_left.items()}
    right_margin = {key: margin(value) for key, value in by_right.items()}
    ordered = sorted(raw_candidates, key=lambda edge: (edge["cost"], edge["gap"]))
    predecessors, successors, accepted = set(), set(), []
    for edge in ordered:
        left, right = edge["left_id"], edge["right_id"]
        if left in successors or right in predecessors:
            continue
        if left_margin[left] < args.min_margin or right_margin[right] < args.min_margin:
            continue
        successors.add(left); predecessors.add(right); accepted.append(edge)

    union = UnionFind(list(tracklets))
    for edge in accepted:
        union.union(edge["left_id"], edge["right_id"])
    components: dict[int, list[int]] = collections.defaultdict(list)
    for source_id in tracklets:
        components[union.find(source_id)].append(source_id)
    ordered_components = sorted(components.values(), key=lambda members: min(tracklets[value].start_frame for value in members))
    new_ids = {source_id: new_id for new_id, members in enumerate(ordered_components, 1) for source_id in members}
    touched = {value for edge in accepted for value in (edge["left_id"], edge["right_id"])}

    output_root = ROOT / "outputs/optimized_v1/06_tracklet_stitch" / args.run_name
    frame_root = output_root / "associated/frames" / args.sequence
    for frame, row in rows.items():
        output = copy.deepcopy(row)
        for det in tracked_bees(output):
            source_id = int(det["track_id"])
            det["pre_stitch_track_id"] = source_id
            det["track_id"] = new_ids[source_id]
            if source_id in touched:
                det["track_evidence"] = sorted(set(det.get("track_evidence", []) + ["tracklet_stitch_v1"]))
        output["tracklet_stitching"] = {
            "run_name": args.run_name,
            "source_run": args.source_run,
            "candidate_id_only": True,
            "baseline_geometry_preserved": True,
            "parameters": {
                "max_gap": args.max_gap, "max_cost": args.max_cost,
                "min_margin": args.min_margin, "min_appearance": args.min_appearance,
                "max_speed": args.max_speed, "history": args.history,
            },
        }
        atomic_json(frame_root / f"frame_{frame:08d}.json", output)

    lengths = [sum(len(tracklets[value].observations) for value in members) for members in ordered_components]
    report = {
        "completed": True,
        "candidate_only": True,
        "sequence": args.sequence,
        "source_run": args.source_run,
        "run_name": args.run_name,
        "parameters": vars(args),
        "frames": len(rows),
        "source_tracklets": len(tracklets),
        "raw_join_candidates": len(raw_candidates),
        "accepted_joins": len(accepted),
        "stitched_ids": len(components),
        "median_observations": float(statistics.median(lengths)) if lengths else 0.0,
        "elapsed_seconds": time.time() - started,
        "prediction_root": str((output_root / "associated/frames").resolve()),
    }
    atomic_json(output_root / f"{args.sequence}_summary.json", report)
    atomic_json(output_root / f"{args.sequence}_accepted_joins.json", accepted)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
