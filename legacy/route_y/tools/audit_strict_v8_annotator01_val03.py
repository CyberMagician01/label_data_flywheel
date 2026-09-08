#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data/bee26/beeposetrack_y_20260827")
DATASET = ROOT / "datasets" / "y_unified_20260901_labelme5_a1_strict_v8_annotator01_val_03_test_detposemask"
MANIFEST = DATASET / "dataset_manifest.jsonl"


def annotator_id(path: str) -> str | None:
    for part in Path(path).parts:
        normalized = part.replace("标注员", "").replace("_", "")
        if normalized in {"01", "02", "03", "004", "05"}:
            return normalized
    return None


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    violations: list[str] = []
    by_split = Counter()
    by_split_domain = Counter()
    by_split_ann = Counter()
    label_rows = Counter()
    visible_pose_rows = Counter()
    pose_rows = Counter()
    det_rows = Counter()
    track_scope_bad = 0
    section_by_split = defaultdict(set)

    for row in rows:
        split = row["split"]
        ann = annotator_id(row["source_json"])
        by_split[split] += 1
        by_split_domain[(split, row["domain"])] += 1
        by_split_ann[(split, ann)] += 1
        section_by_split[split].add(row["section_id"])
        if split == "val" and ann != "01":
            violations.append(f"val contains non annotator_01: {row['source_json']}")
        if split == "test" and ann != "03":
            violations.append(f"test contains non annotator_03: {row['source_json']}")
        if split == "train" and ann not in {"02", "004", "05"}:
            violations.append(f"train contains forbidden annotator {ann}: {row['source_json']}")
        lab = Path(row["image_path"].replace("/images/", "/labels/")).with_suffix(".txt")
        n_label = len([line for line in lab.read_text(encoding="utf-8").splitlines() if line.strip()])
        n_visible_pose = 0
        for line in lab.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            vals = line.split()
            if len(vals) >= 11 and float(vals[7]) > 0 and float(vals[10]) > 0:
                n_visible_pose += 1
        n_pose = sum(int(inst.get("pose_mask", 0)) for inst in row["instances"])
        n_det = len(row["instances"])
        label_rows[split] += n_label
        visible_pose_rows[split] += n_visible_pose
        pose_rows[split] += n_pose
        det_rows[split] += n_det
        if n_label != n_det:
            violations.append(f"label row mismatch {lab}: labels={n_label}, det={n_det}")
        if n_visible_pose != n_pose:
            violations.append(f"visible pose row mismatch {lab}: visible={n_visible_pose}, pose={n_pose}")
        for inst in row["instances"]:
            tid = inst.get("track_id")
            expected_prefix = row["image_path"].split("/")[-1].split("__frame", 1)[0] + ":"
            if tid and not str(tid).startswith(expected_prefix):
                track_scope_bad += 1

    expected_splits = {"train", "val", "test"}
    if set(by_split) != expected_splits:
        violations.append(f"unexpected split set: {sorted(by_split)}")

    report = {
        "dataset": str(DATASET),
        "frames_by_split": dict(sorted(by_split.items())),
        "frames_by_split_domain": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_split_domain.items())},
        "frames_by_split_annotator": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_split_ann.items())},
        "det_instances_by_split": dict(sorted(det_rows.items())),
        "pose_instances_by_split": dict(sorted(pose_rows.items())),
        "label_rows_by_split": dict(sorted(label_rows.items())),
        "visible_pose_rows_by_split": dict(sorted(visible_pose_rows.items())),
        "sections_by_split": {k: sorted(v) for k, v in sorted(section_by_split.items())},
        "track_scope_bad_instances": track_scope_bad,
        "violation_count": len(violations) + track_scope_bad,
        "violations_preview": violations[:20],
    }
    out = DATASET / "strict_v8_y_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["violation_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
