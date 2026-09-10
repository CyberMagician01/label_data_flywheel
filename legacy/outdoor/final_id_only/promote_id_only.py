#!/usr/bin/env python3
"""Promote the validated outdoor ID graph into a clean ID-only release.

All non-ID instance fields are copied from the immutable uploaded annotation.
The only per-instance addition is ``track_id`` for class-0 outdoor bees.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("BEE_OUTDOOR_CONFIG", HERE / "config.json")).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ROOT = (CONFIG_PATH.parent / CONFIG["workspace_root"]).resolve()
EXTRACTED = (CONFIG_PATH.parent / CONFIG["extracted_root"]).resolve()
SOURCE_TRACKING = ROOT / "outputs/optimized_v1/06_tracklet_stitch/full_stitch_v1/associated/frames"
RELEASE_ROOT = ROOT / "outputs/optimized_v1/11_id_only_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or aggregate the ID-only automated release.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", choices=[f"A-5-{index}" for index in range(1, 5)])
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any], compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    text = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_digest(digest: Any, path: Path, root: Path) -> None:
    digest.update(str(path.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    digest.update(b"\0")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def build_sequence(sequence: str, force: bool) -> None:
    expected = int(CONFIG["sequences"][sequence]["frames"])
    source_root = EXTRACTED / "annotations" / "outdoor" / sequence / "frames"
    tracking_root = SOURCE_TRACKING / sequence
    output_root = RELEASE_ROOT / "annotations" / "outdoor" / sequence
    summary_path = RELEASE_ROOT / f"{sequence}_summary.json"
    validation_path = RELEASE_ROOT / f"{sequence}_validation.json"
    source_paths = sorted(source_root.glob("frame_*.json"))
    tracking_paths = sorted(tracking_root.glob("frame_*.json"))
    if len(source_paths) != expected or len(tracking_paths) != expected:
        raise RuntimeError(
            f"coverage mismatch for {sequence}: source={len(source_paths)} "
            f"tracking={len(tracking_paths)} expected={expected}"
        )
    if summary_path.exists() and validation_path.exists() and not force:
        print(json.dumps({"sequence": sequence, "status": "already_complete"}), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    counts: collections.Counter[str] = collections.Counter()
    failures: collections.Counter[str] = collections.Counter()
    track_frames: dict[int, list[int]] = collections.defaultdict(list)
    previous_by_id: dict[int, tuple[int, float, float]] = {}
    speeds: list[float] = []

    for ordinal, (source_path, tracking_path) in enumerate(zip(source_paths, tracking_paths), 1):
        if source_path.name != tracking_path.name:
            raise RuntimeError(f"frame alignment mismatch: {source_path.name} != {tracking_path.name}")
        frame = int(source_path.stem.rsplit("_", 1)[1])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        source_detections = source.get("detections", [])
        tracking_detections = tracking.get("detections", [])
        if len(source_detections) != len(tracking_detections):
            raise RuntimeError(f"detection count changed at {sequence}:{frame}")

        output = copy.deepcopy(source)
        output_detections = output.get("detections", [])
        seen_ids: set[int] = set()
        for index, (source_det, tracking_det, output_det) in enumerate(
            zip(source_detections, tracking_detections, output_detections)
        ):
            source_class = int(source_det.get("class_id", 0))
            if source_class != int(tracking_det.get("class_id", 0)):
                failures["class_order_mismatch"] += 1
                continue
            if source_class != 0:
                counts["non_bee_instances_preserved"] += 1
                continue
            track_id = tracking_det.get("track_id")
            if not isinstance(track_id, int) or track_id <= 0:
                failures["invalid_track_id"] += 1
                continue
            if track_id in seen_ids:
                failures["same_frame_duplicate_id"] += 1
                continue
            seen_ids.add(track_id)
            output_det["track_id"] = track_id
            counts["bee_instances_with_id"] += 1
            track_frames[track_id].append(frame)
            box = output_det["bbox_xyxy"]
            center_x = 0.5 * (float(box[0]) + float(box[2]))
            center_y = 0.5 * (float(box[1]) + float(box[3]))
            if track_id in previous_by_id:
                previous_frame, previous_x, previous_y = previous_by_id[track_id]
                gap = frame - previous_frame
                if gap <= 0:
                    failures["non_monotone_track"] += 1
                else:
                    speed = math.hypot(center_x - previous_x, center_y - previous_y) / gap
                    speeds.append(speed)
                    if speed > 220.0:
                        counts["speed_over_220"] += 1
            previous_by_id[track_id] = (frame, center_x, center_y)

            reconstructed = copy.deepcopy(output_det)
            reconstructed.pop("track_id", None)
            if reconstructed != {k: v for k, v in source_det.items() if k != "track_id"}:
                failures["non_id_mutation_before_write"] += 1
            if track_id != tracking_det.get("track_id"):
                failures["tracking_id_mismatch_before_write"] += 1

        output["id_optimization"] = {
            "release": "id_only_final",
            "method": "overlapping_forward_reverse_association_then_global_tracklet_stitching",
            "source_detection_and_pose_preserved": True,
            "automatic_only": True,
            "human_review_required": False,
            "source_tracking_run": "outputs/optimized_v1/06_tracklet_stitch/full_stitch_v1",
        }
        output_path = output_root / source_path.name
        atomic_json(output_path, output, compact=True)
        counts["frames_written"] += 1
        if ordinal % 500 == 0 or ordinal == expected:
            print(
                json.dumps(
                    {"sequence": sequence, "stage": "write", "frames": ordinal, "expected": expected}
                ),
                flush=True,
            )

    if failures:
        raise RuntimeError(f"write-stage failures for {sequence}: {dict(failures)}")

    # Full readback: compare every instance to the immutable source and every ID
    # to the already validated stitched graph.
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    readback_counts: collections.Counter[str] = collections.Counter()
    readback_failures: collections.Counter[str] = collections.Counter()
    output_paths = sorted(output_root.glob("frame_*.json"))
    if len(output_paths) != expected:
        readback_failures["missing_or_extra_frames"] += abs(len(output_paths) - expected) or 1
    for source_path, tracking_path, output_path in zip(source_paths, tracking_paths, output_paths):
        update_digest(source_digest, source_path, source_root)
        update_digest(output_digest, output_path, output_root)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        output = json.loads(output_path.read_text(encoding="utf-8"))
        output_meta = copy.deepcopy(output)
        output_meta.pop("id_optimization", None)
        output_detections = output_meta.pop("detections", [])
        source_meta = copy.deepcopy(source)
        source_detections = source_meta.pop("detections", [])
        tracking_detections = tracking.get("detections", [])
        if output_meta != source_meta:
            readback_failures["frame_metadata_mutation"] += 1
        if not (len(output_detections) == len(source_detections) == len(tracking_detections)):
            readback_failures["detection_count_mismatch"] += 1
            continue
        seen: set[int] = set()
        for source_det, tracking_det, output_det in zip(
            source_detections, tracking_detections, output_detections
        ):
            reconstructed = copy.deepcopy(output_det)
            track_id = reconstructed.pop("track_id", None)
            if reconstructed != {k: v for k, v in source_det.items() if k != "track_id"}:
                readback_failures["non_id_instance_mutation"] += 1
            if int(source_det.get("class_id", 0)) == 0:
                readback_counts["bees"] += 1
                if track_id != tracking_det.get("track_id") or not isinstance(track_id, int):
                    readback_failures["id_lineage_mismatch"] += 1
                elif track_id in seen:
                    readback_failures["same_frame_duplicate_id"] += 1
                else:
                    seen.add(track_id)
            else:
                readback_counts["non_bees"] += 1
                if track_id != source_det.get("track_id"):
                    readback_failures["non_bee_id_mutation"] += 1
        readback_counts["frames"] += 1

    internal_gaps = sum(
        max(0, later - earlier - 1)
        for frames in track_frames.values()
        for earlier, later in zip(frames, frames[1:])
    )
    summary = {
        "completed": True,
        "release_status": "automated_final_id_only",
        "sequence": sequence,
        "source_tracking_run": "full_stitch_v1",
        "policy": {
            "detection_boxes": "copied byte-for-value from uploaded annotation",
            "classes_confidence_pose": "copied byte-for-value from uploaded annotation",
            "track_id": "copied from validated full_stitch_v1",
            "beeposetrack_y": "disabled by scope",
            "sam2_cotracker": "not promoted because frozen ablation did not improve ID proxy",
            "human_intervention": False,
        },
        "counts": dict(counts),
        "track_statistics": {
            "unique_ids": len(track_frames),
            "singleton_ids": sum(len(frames) == 1 for frames in track_frames.values()),
            "median_observations": statistics.median(map(len, track_frames.values())),
            "internal_gap_frames": internal_gaps,
            "speed_pixels_per_frame_p95": percentile(speeds, 0.95),
            "speed_pixels_per_frame_p99": percentile(speeds, 0.99),
        },
        "elapsed_seconds": time.time() - started,
        "output_root": str(output_root),
    }
    validation = {
        "completed": True,
        "sequence": sequence,
        "hard_gate_passed": not readback_failures,
        "failures": dict(readback_failures),
        "counts": dict(readback_counts),
        "source_aggregate_sha256": source_digest.hexdigest(),
        "output_aggregate_sha256": output_digest.hexdigest(),
        "guarantees": {
            "all_frames_present": not readback_failures.get("missing_or_extra_frames"),
            "all_non_id_fields_preserved": not (
                readback_failures.get("frame_metadata_mutation")
                or readback_failures.get("non_id_instance_mutation")
            ),
            "ids_equal_validated_stitched_graph": not readback_failures.get("id_lineage_mismatch"),
            "same_frame_ids_unique": not readback_failures.get("same_frame_duplicate_id"),
            "non_bee_instances_preserved": not readback_failures.get("non_bee_id_mutation"),
        },
    }
    atomic_json(summary_path, summary)
    atomic_json(validation_path, validation)
    print(json.dumps({"summary": summary, "validation": validation}, ensure_ascii=False), flush=True)
    if not validation["hard_gate_passed"]:
        raise RuntimeError(f"readback validation failed for {sequence}: {dict(readback_failures)}")


def aggregate_release() -> None:
    summaries, validations = {}, {}
    for sequence in CONFIG["sequences"]:
        summaries[sequence] = json.loads((RELEASE_ROOT / f"{sequence}_summary.json").read_text(encoding="utf-8"))
        validations[sequence] = json.loads((RELEASE_ROOT / f"{sequence}_validation.json").read_text(encoding="utf-8"))
    if not all(v["hard_gate_passed"] for v in validations.values()):
        raise ValueError("ID-only 回读验证未全部通过")
    release = {
        "schema_version": 1, "release_status": "automated_final_id_only",
        "method": "overlapping_forward_reverse_association_then_global_tracklet_stitching",
        "outdoor_final": str(RELEASE_ROOT / "annotations/outdoor"),
        "source_annotations": str(EXTRACTED / "annotations/outdoor"),
        "sequence_summaries": summaries, "sequence_validations": validations,
        "counts": {
            "outdoor_frames": sum(v["counts"]["frames"] for v in validations.values()),
            "outdoor_bees_with_optimized_id": sum(v["counts"]["bees"] for v in validations.values()),
            "outdoor_non_bees_preserved": sum(v["counts"]["non_bees"] for v in validations.values()),
            "outdoor_unique_ids": sum(v["track_statistics"]["unique_ids"] for v in summaries.values()),
        },
        "all_non_id_fields_preserved": all(v["guarantees"]["all_non_id_fields_preserved"] for v in validations.values()),
    }
    atomic_json(RELEASE_ROOT / "release_manifest.json", release)
    print(json.dumps(release, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate_release()
    else:
        build_sequence(args.sequence, args.force)


if __name__ == "__main__":
    main()
