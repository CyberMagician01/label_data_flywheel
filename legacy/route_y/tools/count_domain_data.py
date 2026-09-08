#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def count_manifest(path: Path):
    stats = defaultdict(lambda: defaultdict(int))
    videos = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain", "unknown")
        split = row.get("split", "all")
        instances = row.get("instances", [])
        stats[(split, domain)]["frames"] += 1
        stats[(split, domain)]["instances"] += len(instances)
        stats[(split, domain)]["track_instances"] += sum(1 for x in instances if x.get("track_id") is not None)
        videos[domain].add(row.get("video_id"))
    return stats, videos


def main() -> None:
    root = Path("datasets/y_a1_manifest_folds_e_aligned")
    print(f"A1 fold datasets: {root}")
    for fold in range(4):
        stats, _ = count_manifest(root / f"fold{fold}" / "dataset_manifest.jsonl")
        print(f"\nfold{fold}")
        for split in ("train", "val"):
            for domain in ("RGB", "IR"):
                item = stats[(split, domain)]
                print(
                    f"  {split:5s} {domain}: "
                    f"frames={item['frames']}, "
                    f"instances={item['instances']}, "
                    f"track_instances={item['track_instances']}"
                )

    source = Path("/data/bee26/beeposetrack_e_20260827/data/annotations/folds/dataset_manifest.jsonl")
    total = defaultdict(int)
    videos = defaultdict(set)
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain", "unknown")
        instances = row.get("instances", [])
        total[(domain, "frames")] += 1
        total[(domain, "instances")] += len(instances)
        total[(domain, "track_instances")] += sum(1 for x in instances if x.get("track_id") is not None)
        videos[domain].add(row.get("video_id"))

    print("\nsource shared manifest total")
    for domain in ("RGB", "IR"):
        print(
            f"  {domain}: "
            f"frames={total[(domain, 'frames')]}, "
            f"instances={total[(domain, 'instances')]}, "
            f"track_instances={total[(domain, 'track_instances')]}, "
            f"videos={sorted(videos[domain])}"
        )


if __name__ == "__main__":
    main()
