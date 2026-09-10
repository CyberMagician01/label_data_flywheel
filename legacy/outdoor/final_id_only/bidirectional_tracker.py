#!/usr/bin/env python3
"""Conservative overlapping-window forward/reverse tracker for outdoor bees."""

from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("BEE_OUTDOOR_CONFIG", HERE / "config.json")).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ROOT = (CONFIG_PATH.parent / CONFIG["workspace_root"]).resolve()
EXTRACTED = (CONFIG_PATH.parent / CONFIG["extracted_root"]).resolve()
MANIFEST = ROOT / "data/benchmarks/outdoor_human_420.jsonl"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def valid_box(det: dict) -> bool:
    box = det.get("bbox_xyxy", det.get("bbox"))
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def observed_bees(row: dict) -> list[dict]:
    return [
        det
        for det in row.get("detections", [])
        if int(det.get("class_id", 0)) == 0
        and "interpolation" not in str(det.get("origin", ""))
        and valid_box(det)
    ]


def descriptor(gray: np.ndarray, boxes: np.ndarray, scale: float) -> np.ndarray:
    result = []
    height, width = gray.shape[:2]
    for raw in boxes:
        x1, y1, x2, y2 = np.round(raw * scale).astype(int)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        patch = cv2.resize(gray[y1:y2, x1:x2], (8, 8), interpolation=cv2.INTER_AREA)
        vector = patch.astype(np.float32).ravel()
        vector -= vector.mean()
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            vector /= norm
        result.append(vector)
    return np.asarray(result, dtype=np.float32).reshape(-1, 64)


def orientation(det: dict) -> np.ndarray:
    points = det.get("keypoints")
    if not isinstance(points, dict):
        return np.zeros(2, dtype=np.float32)
    head = points.get("head")
    tail = points.get("abdomen_tip")
    if not (isinstance(head, list) and isinstance(tail, list) and len(head) >= 2 and len(tail) >= 2):
        return np.zeros(2, dtype=np.float32)
    vector = np.asarray(tail[:2], dtype=np.float32) - np.asarray(head[:2], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.zeros(2, dtype=np.float32)


@dataclass
class Observation:
    boxes: np.ndarray
    centers: np.ndarray
    diagonals: np.ndarray
    desc: np.ndarray
    orientations: np.ndarray


@dataclass
class Track:
    local_id: int
    last_frame: int
    last_index: int
    box: np.ndarray
    center: np.ndarray
    desc: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity_ready: bool = False


def make_observation(row: dict, gray: np.ndarray, scale: float) -> Observation:
    detections = observed_bees(row)
    boxes = np.asarray([det["bbox_xyxy"] for det in detections], dtype=np.float32).reshape(-1, 4)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return Observation(
        boxes=boxes,
        centers=centers,
        diagonals=np.linalg.norm(sizes, axis=1),
        desc=descriptor(gray, boxes, scale),
        orientations=np.asarray([orientation(det) for det in detections], dtype=np.float32).reshape(-1, 2),
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    intersection = np.maximum(
        0,
        np.minimum(a[:, None, 2:], b[None, :, 2:])
        - np.maximum(a[:, None, :2], b[None, :, :2]),
    ).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def optical_predictions(
    previous: np.ndarray,
    current: np.ndarray,
    tracks: list[Track],
    scale: float,
) -> dict[int, tuple[np.ndarray, float]]:
    if not tracks:
        return {}
    points, owners = [], []
    for track_index, track in enumerate(tracks):
        width, height = track.box[2:] - track.box[:2]
        offsets = np.asarray(
            [[0, 0], [0.18 * width, 0], [-0.18 * width, 0], [0, 0.18 * height], [0, -0.18 * height]],
            dtype=np.float32,
        )
        for point in track.center + offsets:
            points.append(point * scale)
            owners.append(track_index)
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    lk = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, source, None, **lk)
    if forward is None:
        return {}
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **lk)
    if backward is None:
        return {}
    source, forward, backward = source[:, 0], forward[:, 0], backward[:, 0]
    status = status_forward[:, 0].astype(bool) & status_backward[:, 0].astype(bool)
    fb_error = np.linalg.norm(backward - source, axis=1)
    displacement = (forward - source) / scale
    owners = np.asarray(owners)
    result = {}
    for track_index, track in enumerate(tracks):
        indexes = np.flatnonzero(owners == track_index)
        good = indexes[status[indexes] & (fb_error[indexes] <= 2.0)]
        if len(good) < 2:
            continue
        shifts = displacement[good]
        median = np.median(shifts, axis=0)
        residual = np.linalg.norm(shifts - median, axis=1)
        if float(np.median(residual)) > 5.0 or float(np.linalg.norm(median)) > 220.0:
            continue
        reliability = float(np.exp(-np.median(fb_error[good]))) * min(1.0, len(good) / 4)
        result[track.local_id] = (track.center + median, reliability)
    return result


def read_gray(sequence: str, frame: int, scale: float) -> np.ndarray:
    path = EXTRACTED / "frames" / sequence / f"frame_{frame:08d}.jpg"
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def run_direction(
    sequence: str,
    rows: dict[int, dict],
    frames: list[int],
    reverse: bool,
    scale: float,
    max_age: int,
    unmatched_cost: float,
) -> list[dict]:
    ordered = list(reversed(frames)) if reverse else frames
    images = {frame: read_gray(sequence, frame, scale) for frame in frames}
    observations = {frame: make_observation(rows[frame], images[frame], scale) for frame in frames}
    tracks: list[Track] = []
    next_id = 1
    links = []
    previous_frame = None
    previous_image = None
    direction = "reverse" if reverse else "forward"
    for frame in ordered:
        observation = observations[frame]
        tracks = [track for track in tracks if abs(frame - track.last_frame) <= max_age]
        contiguous = [track for track in tracks if previous_frame is not None and track.last_frame == previous_frame]
        flow = (
            optical_predictions(previous_image, images[frame], contiguous, scale)
            if previous_image is not None and contiguous
            else {}
        )
        matched_detections: set[int] = set()
        if tracks and len(observation.boxes):
            predicted_centers, predicted_boxes, flow_good, flow_reliability, ages = [], [], [], [], []
            for track in tracks:
                age = abs(frame - track.last_frame)
                ages.append(age)
                constant = track.center + track.velocity * age if track.velocity_ready else track.center
                if track.local_id in flow:
                    flow_center, reliability = flow[track.local_id]
                    predicted = 0.8 * flow_center + 0.2 * constant
                    flow_good.append(True)
                    flow_reliability.append(reliability)
                else:
                    predicted = constant
                    flow_good.append(False)
                    flow_reliability.append(0.0)
                predicted_centers.append(predicted)
                shift = predicted - track.center
                predicted_boxes.append(track.box + np.r_[shift, shift])
            predicted_centers = np.asarray(predicted_centers)
            predicted_boxes = np.asarray(predicted_boxes)
            ages_array = np.asarray(ages)[:, None]
            distance = np.linalg.norm(predicted_centers[:, None] - observation.centers[None], axis=2)
            iou = box_iou(predicted_boxes, observation.boxes)
            similarity = np.asarray([track.desc for track in tracks]) @ observation.desc.T
            old_sizes = np.asarray([track.box[2:] - track.box[:2] for track in tracks])
            new_sizes = observation.boxes[:, 2:] - observation.boxes[:, :2]
            ratio = np.maximum(
                old_sizes[:, None] / np.maximum(new_sizes[None], 1),
                new_sizes[None] / np.maximum(old_sizes[:, None], 1),
            ).max(axis=2)
            normalizer = np.maximum(
                10.0,
                np.minimum(np.linalg.norm(old_sizes, axis=1)[:, None], observation.diagonals[None]),
            )
            orientation_cost = np.zeros_like(distance)
            for i, track in enumerate(tracks):
                if np.linalg.norm(track.orientation) <= 0:
                    continue
                valid_orientation = np.linalg.norm(observation.orientations, axis=1) > 0
                orientation_cost[i, valid_orientation] = 0.5 * (
                    1 - observation.orientations[valid_orientation] @ track.orientation
                )
            flow_matrix = np.asarray(flow_good, dtype=bool)[:, None]
            distance_limit = np.where(
                flow_matrix,
                np.maximum(20.0, 0.9 * normalizer),
                np.maximum(24.0, np.minimum(75.0, 1.25 * normalizer + 7.0 * (ages_array - 1))),
            )
            valid = (
                (distance <= distance_limit)
                & (ratio <= 3.5)
                & ((iou >= 0.015) | (distance <= 0.7 * normalizer) | flow_matrix)
                & (similarity >= -0.45)
            )
            real_cost = (
                distance / normalizer
                + 0.65 * (1 - iou)
                + 0.20 * (1 - similarity)
                + 0.12 * np.log(np.maximum(ratio, 1))
                + 0.18 * orientation_cost
                + 0.12 * (ages_array - 1)
                - 0.12 * np.asarray(flow_reliability)[:, None]
            )
            matrix = np.full((len(tracks), len(observation.boxes) + len(tracks)), 1e6, dtype=np.float64)
            matrix[:, : len(observation.boxes)] = np.where(valid, real_cost, 1e6)
            for i in range(len(tracks)):
                matrix[i, len(observation.boxes) + i] = unmatched_cost
            left, right = linear_sum_assignment(matrix)
            for i, j in zip(left, right):
                if j >= len(observation.boxes) or not valid[i, j]:
                    continue
                track = tracks[i]
                age = abs(frame - track.last_frame)
                earlier = (track.last_frame, track.last_index)
                later = (frame, int(j))
                if earlier[0] > later[0]:
                    earlier, later = later, earlier
                links.append(
                    {
                        "earlier": earlier,
                        "later": later,
                        "direction": direction,
                        "cost": float(real_cost[i, j]),
                        "flow": bool(flow_good[i]),
                        "flow_reliability": float(flow_reliability[i]),
                        "gap": age,
                    }
                )
                displacement = observation.centers[j] - track.center
                measured_velocity = displacement / max(age, 1)
                track.velocity = 0.55 * track.velocity + 0.45 * measured_velocity if track.velocity_ready else measured_velocity
                track.velocity_ready = True
                updated_desc = 0.75 * track.desc + 0.25 * observation.desc[j]
                updated_desc /= max(float(np.linalg.norm(updated_desc)), 1e-6)
                track.desc = updated_desc
                if np.linalg.norm(observation.orientations[j]) > 0:
                    track.orientation = observation.orientations[j]
                track.last_frame = frame
                track.last_index = int(j)
                track.box = observation.boxes[j].copy()
                track.center = observation.centers[j].copy()
                matched_detections.add(int(j))
        for index in range(len(observation.boxes)):
            if index in matched_detections:
                continue
            tracks.append(
                Track(
                    local_id=next_id,
                    last_frame=frame,
                    last_index=index,
                    box=observation.boxes[index].copy(),
                    center=observation.centers[index].copy(),
                    desc=observation.desc[index].copy(),
                    orientation=observation.orientations[index].copy(),
                )
            )
            next_id += 1
        previous_frame, previous_image = frame, images[frame]
    return links


def windows_for(start: int, end: int, segment: int, overlap: int) -> list[list[int]]:
    if end - start + 1 <= segment:
        return [list(range(start, end + 1))]
    step = segment - overlap
    starts = list(range(start, end - segment + 2, step))
    last = end - segment + 1
    if starts[-1] != last:
        starts.append(last)
    return [list(range(window_start, window_start + segment)) for window_start in sorted(set(starts))]


def benchmark_ranges(sequence: str, padding: int, expected: int) -> list[tuple[int, int]]:
    records = [
        json.loads(line)
        for line in MANIFEST.read_text().splitlines()
        if line and json.loads(line)["sequence"] == sequence
    ]
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for row in records:
        grouped[row["section"]].append(int(row["frame"]))
    ranges = sorted((max(0, min(v) - padding), min(expected - 1, max(v) + padding)) for v in grouped.values())
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def select_edges(
    pool: dict[tuple, list[dict]],
    single_flow_cost: float,
    require_external_range: tuple[int, int] | None = None,
) -> tuple[list[dict], collections.Counter]:
    candidates = []
    reasons = collections.Counter()
    for key, votes in pool.items():
        directions = {vote["direction"] for vote in votes}
        cost = float(statistics.median(vote["cost"] for vote in votes))
        flow_votes = sum(vote["flow"] for vote in votes)
        has_cotracker = any(direction.startswith("cotracker_") for direction in directions)
        has_local = any(direction in {"forward", "reverse"} for direction in directions)
        external_required = (
            require_external_range is not None
            and require_external_range[0] <= key[0]
            and key[2] <= require_external_range[1]
        )
        if external_required and not has_cotracker:
            reasons["rejected_missing_external"] += 1
            continue
        if len(directions) >= 2 and cost <= 2.25:
            reason = (
                "multi_source_consensus" if has_cotracker and has_local else "forward_reverse"
            )
        elif len(votes) >= 2 and cost <= 1.45:
            reason = "overlap_repeat"
        elif flow_votes and cost <= single_flow_cost:
            reason = "single_direction_flow_strong"
        else:
            reasons["rejected"] += 1
            continue
        candidates.append(
            {
                "earlier": key[:2],
                "later": key[2:],
                "votes": len(votes),
                "directions": sorted(directions),
                "median_cost": cost,
                "flow_votes": flow_votes,
                "reason": reason,
            }
        )
    candidates.sort(
        key=lambda edge: (
            -len(edge["directions"]), -edge["votes"], -edge["flow_votes"],
            edge["median_cost"], edge["later"][0] - edge["earlier"][0],
        )
    )
    predecessor, successor, accepted = {}, {}, []
    for edge in candidates:
        earlier, later = tuple(edge["earlier"]), tuple(edge["later"])
        if earlier in successor or later in predecessor:
            reasons["one_to_one_conflict"] += 1
            continue
        successor[earlier] = later
        predecessor[later] = earlier
        accepted.append(edge)
        reasons[edge["reason"]] += 1
    return accepted, reasons


class UnionFind:
    def __init__(self, nodes: list[tuple[int, int]]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, choices=[f"A-5-{index}" for index in range(1, 5)])
    parser.add_argument("--mode", choices=["benchmark", "full"], default="benchmark")
    parser.add_argument("--segment-frames", type=int, default=600)
    parser.add_argument("--overlap-frames", type=int, default=120)
    parser.add_argument("--benchmark-padding", type=int, default=8)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--max-age", type=int, default=4)
    parser.add_argument("--unmatched-cost", type=float, default=1.85)
    parser.add_argument("--single-flow-cost", type=float, default=0.92)
    parser.add_argument("--external-edges", action="append", type=Path, default=[])
    parser.add_argument("--require-external-start", type=int)
    parser.add_argument("--require-external-end", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if args.overlap_frames >= args.segment_frames:
        raise ValueError("overlap must be smaller than segment")
    cv2.setNumThreads(2)
    expected = int(CONFIG["sequences"][args.sequence]["frames"])
    source = EXTRACTED / "annotations/outdoor" / args.sequence / "frames"
    paths = sorted(source.glob("frame_*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"annotation count {len(paths)} != {expected}")
    rows = {int(path.stem.rsplit("_", 1)[1]): json.loads(path.read_text()) for path in paths}
    ranges = benchmark_ranges(args.sequence, args.benchmark_padding, expected) if args.mode == "benchmark" else [(0, expected - 1)]
    windows = [window for start, end in ranges for window in windows_for(start, end, args.segment_frames, args.overlap_frames)]
    started = time.time()
    pool: dict[tuple, list[dict]] = collections.defaultdict(list)
    covered_frames = sorted({frame for window in windows for frame in window})
    covered_set = set(covered_frames)
    for window_index, frames in enumerate(windows):
        for reverse in (False, True):
            links = run_direction(
                args.sequence, rows, frames, reverse, args.flow_scale, args.max_age, args.unmatched_cost
            )
            for link in links:
                pool[(*link["earlier"], *link["later"])].append(link)
        print(json.dumps({"stage": "directional_windows", "sequence": args.sequence, "completed": window_index + 1, "total": len(windows)}), flush=True)
    external_edge_count = 0
    for path in args.external_edges:
        with gzip.open(path, "rt") as handle:
            external = json.load(handle)
        for link in external:
            if link["earlier"][0] not in covered_set or link["later"][0] not in covered_set:
                continue
            pool[(*link["earlier"], *link["later"])].append(link)
            external_edge_count += 1
    require_external_range = None
    if args.require_external_start is not None or args.require_external_end is not None:
        if args.require_external_start is None or args.require_external_end is None:
            raise ValueError("both external range endpoints are required")
        require_external_range = (args.require_external_start, args.require_external_end)
    accepted, edge_stats = select_edges(pool, args.single_flow_cost, require_external_range)
    nodes = [(frame, index) for frame in covered_frames for index in range(len(observed_bees(rows[frame])))]
    union = UnionFind(nodes)
    evidence_by_node: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for edge in accepted:
        earlier, later = tuple(edge["earlier"]), tuple(edge["later"])
        union.union(earlier, later)
        evidence_by_node[earlier].append(edge["reason"])
        evidence_by_node[later].append(edge["reason"])
    components: dict[tuple[int, int], list[tuple[int, int]]] = collections.defaultdict(list)
    for node in nodes:
        components[union.find(node)].append(node)
    ordered_components = sorted(components.values(), key=lambda members: (min(members), len(members)))
    node_ids = {node: track_id for track_id, members in enumerate(ordered_components, 1) for node in members}
    run_name = args.run_name or (
        f"{args.mode}_flow{args.flow_scale:g}_seg{args.segment_frames}_ov{args.overlap_frames}"
        f"_age{args.max_age}_u{args.unmatched_cost:g}_s{args.single_flow_cost:g}"
    )
    output_root = ROOT / "outputs/optimized_v1/02_bidirectional" / run_name
    frame_root = output_root / "associated/frames" / args.sequence
    track_frames: dict[int, list[int]] = collections.defaultdict(list)
    counts = collections.Counter()
    for frame in covered_frames:
        output = copy.deepcopy(rows[frame])
        bee_index, seen = 0, set()
        for det in output.get("detections", []):
            if int(det.get("class_id", 0)) != 0 or "interpolation" in str(det.get("origin", "")):
                counts["non_bee_or_legacy_preserved"] += 1
                continue
            node = (frame, bee_index)
            track_id = node_ids[node]
            det["pre_bidirectional_track_id"] = det.get("track_id")
            det["track_id"] = track_id
            det["track_evidence"] = sorted(set(evidence_by_node.get(node, []))) or ["isolated_observation"]
            if track_id in seen:
                raise RuntimeError(f"duplicate ID at frame {frame}: {track_id}")
            seen.add(track_id)
            track_frames[track_id].append(frame)
            bee_index += 1
            counts["observed_bees"] += 1
        output["bidirectional_tracking"] = {
            "run_name": run_name,
            "baseline_geometry_preserved": True,
            "candidate_id_only": True,
            "segment_frames": args.segment_frames,
            "overlap_frames": args.overlap_frames,
            "flow_scale": args.flow_scale,
            "max_age": args.max_age,
            "unmatched_cost": args.unmatched_cost,
            "single_flow_cost": args.single_flow_cost,
        }
        atomic_json(frame_root / f"frame_{frame:08d}.json", output)
        counts["frames"] += 1
    gaps = sum(
        sum(right - left > 1 for left, right in zip(sorted(frames), sorted(frames)[1:]))
        for frames in track_frames.values()
    )
    lengths = [len(frames) for frames in track_frames.values()]
    report = {
        "completed": True,
        "candidate_only": True,
        "sequence": args.sequence,
        "mode": args.mode,
        "run_name": run_name,
        "parameters": {
            "segment_frames": args.segment_frames,
            "overlap_frames": args.overlap_frames,
            "flow_scale": args.flow_scale,
            "max_age": args.max_age,
            "unmatched_cost": args.unmatched_cost,
            "single_flow_cost": args.single_flow_cost,
            "benchmark_padding": args.benchmark_padding,
        },
        "ranges": ranges,
        "windows": len(windows),
        "directional_runs": 2 * len(windows),
        "counts": dict(counts),
        "pooled_unique_edges": len(pool),
        "accepted_edges": len(accepted),
        "edge_decisions": dict(edge_stats),
        "external_edge_files": [str(path.resolve()) for path in args.external_edges],
        "external_edge_votes": external_edge_count,
        "require_external_range": require_external_range,
        "unique_ids": len(track_frames),
        "median_track_observations": float(np.median(lengths)) if lengths else 0.0,
        "internal_gaps": gaps,
        "elapsed_seconds": time.time() - started,
        "prediction_root": str((output_root / "associated/frames").resolve()),
    }
    atomic_json(output_root / f"{args.sequence}_summary.json", report)
    edge_path = output_root / f"{args.sequence}_accepted_edges.json.gz"
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(edge_path, "wt", compresslevel=1) as handle:
        json.dump(accepted, handle, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
