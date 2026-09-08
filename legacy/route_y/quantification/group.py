"""Group-level bee activity quantification."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def quantify_group(track_jsonl: Path, output_csv: Path) -> None:
    rows = [json.loads(line) for line in track_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = ["video_id,frame_id,domain,count,density_per_mpix,mean_score_proxy"]
    for row in rows:
        count = len(row.get("tracks", []))
        areas = []
        for det in row.get("tracks", []):
            x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        density = count
        lines.append(f"{row['video_id']},{row['frame_id']},{row.get('domain','unknown')},{count},{density:.6f},{np.mean(areas) if areas else 0:.6f}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
