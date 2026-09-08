#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    for p in sorted(Path("runs_e_aligned/a1_folds").glob("a1_*/results.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        key_map = {k.strip(): k for k in rows[0].keys()}
        epoch_key = key_map["epoch"]

        def best(metric: str) -> tuple[int, float] | None:
            key = key_map.get(metric)
            if key is None:
                return None
            row = max(rows, key=lambda x: float(x[key]))
            return int(float(row[epoch_key])), float(row[key])

        print(p.parent.name)
        print(f"  last_epoch: {int(float(rows[-1][epoch_key]))}")
        for label, metric in [
            ("box_mAP50-95", "metrics/mAP50-95(B)"),
            ("box_mAP50", "metrics/mAP50(B)"),
            ("pose_mAP50-95", "metrics/mAP50-95(P)"),
            ("pose_mAP50", "metrics/mAP50(P)"),
            ("box_precision", "metrics/precision(B)"),
            ("box_recall", "metrics/recall(B)"),
            ("pose_precision", "metrics/precision(P)"),
            ("pose_recall", "metrics/recall(P)"),
        ]:
            result = best(metric)
            if result is not None:
                epoch, value = result
                print(f"  best_{label}: epoch={epoch} value={value:.6f}")


if __name__ == "__main__":
    main()
