"""Track-level dynamic programming correction for head/tail swaps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _angle(head: list[float] | None, tail: list[float] | None) -> float | None:
    if head is None or tail is None:
        return None
    v = np.asarray(tail, dtype=float) - np.asarray(head, dtype=float)
    if np.linalg.norm(v) <= 1e-9:
        return None
    return math.atan2(float(v[1]), float(v[0]))


def _delta(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def correct_jsonl(input_jsonl: Path, output_jsonl: Path, swap_penalty: float = 0.15) -> None:
    rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_track: dict[tuple[str, int], list[tuple[int, dict]]] = {}
    for row_idx, row in enumerate(rows):
        sequence_id = str(row.get("section_id", row["video_id"]))
        for det in row.get("tracks", []):
            per_track.setdefault((sequence_id, int(det["track_id"])), []).append((row_idx, det))
    for items in per_track.values():
        items.sort(key=lambda x: int(rows[x[0]]["frame_id"]))
        prev_angle = None
        for _, det in items:
            keep = _angle(det.get("head"), det.get("tail"))
            swap = _angle(det.get("tail"), det.get("head"))
            if prev_angle is not None and _delta(prev_angle, swap) + swap_penalty < _delta(prev_angle, keep):
                det["head"], det["tail"] = det.get("tail"), det.get("head")
                det["head_tail_corrected"] = True
                keep = swap
            else:
                det["head_tail_corrected"] = False
            prev_angle = keep
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swap-penalty", type=float, default=0.15)
    args = parser.parse_args()
    correct_jsonl(args.input, args.output, args.swap_penalty)
