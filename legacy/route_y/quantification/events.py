"""Simple interaction and activity-event extraction from tracks."""

from __future__ import annotations

import csv
from pathlib import Path


def extract_events(individual_csv: Path, output_csv: Path, stop_speed: float = 2.0, fast_speed: float = 20.0) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with individual_csv.open("r", encoding="utf-8", newline="") as src, output_csv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["video_id", "track_id", "event", "value"])
        writer.writeheader()
        for row in reader:
            speed = float(row["mean_speed"])
            if speed <= stop_speed:
                writer.writerow({"video_id": row["video_id"], "track_id": row["track_id"], "event": "stay", "value": f"{speed:.6f}"})
            if speed >= fast_speed:
                writer.writerow({"video_id": row["video_id"], "track_id": row["track_id"], "event": "fast_motion", "value": f"{speed:.6f}"})
