#!/usr/bin/env python3
"""Inspect public BeePose/Mendeley keypoint visibility."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ann", type=Path)
    args = parser.parse_args()
    data = json.loads(args.ann.read_text(encoding="utf-8"))
    visibility = collections.Counter()
    by_image = collections.Counter()
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
        kpts = ann.get("keypoints", [])
        vis = tuple(kpts[2::3]) if len(kpts) >= 6 else ()
        visibility[(len(kpts), vis)] += 1
    for img in data["images"]:
        total = 0
        full_pose = 0
        partial = 0
        for ann in anns_by_image.get(int(img["id"]), []):
            total += 1
            kpts = ann.get("keypoints", [])
            if len(kpts) >= 6 and kpts[2] > 0 and kpts[5] > 0:
                full_pose += 1
            elif len(kpts) >= 6 and (kpts[2] > 0 or kpts[5] > 0):
                partial += 1
        by_image[(total, full_pose, partial)] += 1
    print("ann_visibility")
    for key, value in visibility.most_common(30):
        print(value, key)
    print("image_total_full_partial")
    for key, value in by_image.most_common(40):
        print(value, key)


if __name__ == "__main__":
    main()
