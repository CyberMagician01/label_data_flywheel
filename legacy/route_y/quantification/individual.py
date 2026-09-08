"""Individual trajectory quantification."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def quantify_individual(track_jsonl: Path, output_csv: Path) -> None:
    rows = [json.loads(line) for line in track_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    tracks: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for row in rows:
        for det in row.get("tracks", []):
            tracks[(str(row["video_id"]), int(det["track_id"]))].append((int(row["frame_id"]), det))
    lines = ["video_id,track_id,frames,mean_speed,mean_body_length,path_length,turning_mean"]
    for (video_id, track_id), items in sorted(tracks.items()):
        items.sort(key=lambda x: x[0])
        centers = []
        lengths = []
        angles = []
        for _, det in items:
            x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
            centers.append(np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5]))
            if det.get("head") is not None and det.get("tail") is not None:
                h = np.asarray(det["head"], dtype=float)
                t = np.asarray(det["tail"], dtype=float)
                lengths.append(float(np.linalg.norm(t - h)))
                v = t - h
                angles.append(math.atan2(float(v[1]), float(v[0])))
        steps = [float(np.linalg.norm(centers[i] - centers[i - 1])) for i in range(1, len(centers))]
        turns = [abs((angles[i] - angles[i - 1] + math.pi) % (2 * math.pi) - math.pi) for i in range(1, len(angles))]
        lines.append(
            f"{video_id},{track_id},{len(items)},{np.mean(steps) if steps else 0:.6f},{np.mean(lengths) if lengths else 0:.6f},{sum(steps):.6f},{np.mean(turns) if turns else 0:.6f}"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
