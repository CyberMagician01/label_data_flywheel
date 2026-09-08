#!/usr/bin/env python3
"""Summarize the frozen BeePoseTrack Y/E unified dataset package."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, default=Path("/data/bee26/datasets/bee_e_y_unified_20260901"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    for key in ("records", "samples", "frames", "images", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return list(data.values())
    raise RuntimeError(f"Cannot find frame records in {path}")


def find_manifest(root: Path) -> Path:
    candidates = [
        root / "manifests" / "dataset_manifest.json",
        root / "manifests" / "dataset_manifest.jsonl",
        root / "views" / "Y" / "dataset_manifest.json",
        root / "views" / "Y" / "dataset_manifest.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"dataset_manifest not found under {root}")


def inst_flag(inst: dict[str, Any], key: str, default: bool = True) -> bool:
    value = inst.get(key, default)
    return bool(value)


def summarize(root: Path) -> dict[str, Any]:
    manifest = find_manifest(root)
    records = load_json_or_jsonl(manifest)
    package_ready = root / "manifests" / "PACKAGE_READY.json"
    sha_list = root / "manifests" / "annotations_sha256.txt"
    stats: dict[str, Any] = {
        "unified_root": str(root),
        "manifest": str(manifest),
        "package_ready_exists": package_ready.exists(),
        "annotations_sha256_exists": sha_list.exists(),
        "total_frames": len(records),
        "splits": {},
        "split_domain_table": {},
    }
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        split_rows[str(row.get("split", "unknown"))].append(row)

    for split, rows in sorted(split_rows.items()):
        domains = Counter(str(r.get("domain", "unknown")).upper() for r in rows)
        videos = Counter(str(r.get("video_id", "unknown")) for r in rows)
        sections = Counter(str(r.get("section_id", r.get("section", "unknown"))) for r in rows)
        det = pose = track = 0
        max_instances = 0
        empty = 0
        for row in rows:
            instances = row.get("instances", []) or []
            max_instances = max(max_instances, len(instances))
            if not instances:
                empty += 1
            for inst in instances:
                det += int(inst_flag(inst, "det_mask", True))
                pose += int(inst_flag(inst, "pose_mask", bool(inst.get("keypoints"))))
                track += int(inst_flag(inst, "track_mask", inst.get("track_id") is not None) and inst.get("track_id") is not None)
        stats["splits"][split] = {
            "frames": len(rows),
            "domains": dict(domains),
            "videos": dict(videos),
            "sections": dict(sections),
            "det_instances": det,
            "pose_instances": pose,
            "track_instances": track,
            "empty_frames": empty,
            "max_instances_per_frame": max_instances,
        }
        for domain, count in sorted(domains.items()):
            drows = [r for r in rows if str(r.get("domain", "unknown")).upper() == domain]
            ddet = dpose = dtrack = 0
            for row in drows:
                for inst in row.get("instances", []) or []:
                    ddet += int(inst_flag(inst, "det_mask", True))
                    dpose += int(inst_flag(inst, "pose_mask", bool(inst.get("keypoints"))))
                    dtrack += int(inst_flag(inst, "track_mask", inst.get("track_id") is not None) and inst.get("track_id") is not None)
            stats["split_domain_table"][f"{split}:{domain}"] = {
                "frames": count,
                "det_instances": ddet,
                "pose_instances": dpose,
                "track_instances": dtrack,
            }
    expected = {"train", "calibration", "dev_holdout"}
    missing = sorted(expected.difference(stats["splits"]))
    if missing:
        stats["missing_required_splits"] = missing
    return stats


def write_md(stats: dict[str, Any], path: Path) -> None:
    lines = [
        "# Unified Dataset Distribution",
        "",
        f"unified_root: `{stats['unified_root']}`",
        f"manifest: `{stats['manifest']}`",
        f"PACKAGE_READY.json: `{stats['package_ready_exists']}`",
        f"annotations_sha256.txt: `{stats['annotations_sha256_exists']}`",
        "",
        "| split | frames | RGB frames | IR frames | det instances | pose instances | track instances | empty frames | max/frame |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in stats["splits"].items():
        domains = row["domains"]
        lines.append(
            f"| {split} | {row['frames']} | {domains.get('RGB', 0)} | {domains.get('IR', 0)} | "
            f"{row['det_instances']} | {row['pose_instances']} | {row['track_instances']} | "
            f"{row['empty_frames']} | {row['max_instances_per_frame']} |"
        )
    lines.extend(["", "## Videos"])
    for split, row in stats["splits"].items():
        lines.append(f"- {split}: {', '.join(sorted(row['videos']))}")
    lines.extend(["", "## Sections"])
    for split, row in stats["splits"].items():
        lines.append(f"- {split}: {', '.join(sorted(row['sections']))}")
    if stats.get("missing_required_splits"):
        lines.extend(["", f"Missing required splits: `{stats['missing_required_splits']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stats = summarize(args.unified_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_md(stats, args.output_md)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
