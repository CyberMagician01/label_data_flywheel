"""Build the frozen BeePoseTrack-E schema-v2 view from the shared manifest.

The source manifest remains immutable.  Split roles are assigned from the
annotator directory in ``source_json`` so one annotated section can never be
partly moved between train, calibration and dev_holdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from migrate_schema_v2 import migrate


ANNOTATORS = (
    "标注员_004",
    "标注员_01",
    "标注员_02",
    "标注员_03",
    "标注员_05",
)
BBOX_CLIP_TOLERANCE_PX = 1.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def annotator_from_source(source_json: str) -> str:
    normalized = str(source_json).replace("\\", "/")
    matches = [name for name in ANNOTATORS if f"/{name}/" in normalized]
    if len(matches) != 1:
        raise ValueError(
            f"source_json must identify exactly one frozen annotator: {source_json}"
        )
    return matches[0]


def role_for_annotator(annotator: str) -> str:
    if annotator == "标注员_01":
        return "calibration"
    if annotator == "标注员_03":
        return "dev_holdout"
    return "train"


def canonical_image(unified_root: Path, row: dict) -> tuple[Path, str]:
    source_name = Path(str(row["source_image_path"])).name
    relative = Path("canonical") / "labelme_5_sections" / row["section_id"] / source_name
    absolute = unified_root / relative
    if not absolute.is_file():
        raise FileNotFoundError(f"canonical image is missing: {absolute}")
    return absolute, relative.as_posix()


def xyxy_to_xywh(
    box: list[float], width: int, height: int
) -> tuple[list[float], dict | None]:
    if len(box) != 4:
        raise ValueError(f"bbox_xyxy must contain four values, got {box!r}")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"bbox is degenerate: {box!r}")
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    overflow = max(abs(before - after) for before, after in zip((x1, y1, x2, y2), clipped))
    if overflow > BBOX_CLIP_TOLERANCE_PX:
        raise ValueError(
            f"bbox exceeds the {BBOX_CLIP_TOLERANCE_PX}px canonicalization tolerance: "
            f"{box!r}, size={width}x{height}, overflow={overflow}"
        )
    cx1, cy1, cx2, cy2 = clipped
    if not (cx1 < cx2 and cy1 < cy2):
        raise ValueError(f"bbox becomes degenerate after clipping: {box!r}")
    audit = None
    if overflow > 0.0:
        audit = {
            "source_bbox_xyxy": [x1, y1, x2, y2],
            "clipped_bbox_xyxy": clipped,
            "maximum_coordinate_delta_px": overflow,
        }
    return [cx1, cy1, cx2 - cx1, cy2 - cy1], audit


def coco_keypoints(instance: dict) -> tuple[list[float], int]:
    if not bool(instance.get("pose_mask", 0)):
        if instance.get("keypoints"):
            raise ValueError("pose_mask=0 instance must not retain keypoints")
        return [], 0
    points = instance.get("keypoints") or []
    visibility = instance.get("visibility") or []
    if len(points) != 2 or len(visibility) != 2:
        raise ValueError("pose_mask=1 instance must contain head, tail and visibility")
    flattened = []
    visible = 0
    for point, state in zip(points, visibility):
        if len(point) != 2:
            raise ValueError("each endpoint must contain x and y")
        state = int(state)
        if state not in (0, 1, 2):
            raise ValueError(f"invalid COCO endpoint visibility: {state}")
        flattened.extend([float(point[0]), float(point[1]), state])
        visible += int(state > 0)
    return flattened, visible


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 840:
        raise ValueError(f"the frozen strict-v7 manifest must contain 840 frames, got {len(rows)}")
    return rows


def split_audit(rows: list[dict], image_hashes: dict[str, str]) -> dict:
    roles = ("train", "calibration", "dev_holdout")
    section_sets = {role: set() for role in roles}
    source_sets = {role: set() for role in roles}
    track_sets = {role: set() for role in roles}
    by_video_role = defaultdict(lambda: defaultdict(list))
    for row in rows:
        role = row["split"]
        section_sets[role].add(row["section_id"])
        source_sets[role].add(row["source_json"])
        by_video_role[(row["video_id"], row["domain"])][role].append(int(row["frame_id"]))
        for instance in row.get("instances", []):
            if bool(instance.get("track_mask", 0)):
                track_sets[role].add(
                    f"{row['section_id']}:{instance.get('source_group_id')}"
                )

    pairwise = {}
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            key = f"{left}__{right}"
            minimum_frame_gap = None
            for role_frames in by_video_role.values():
                for a in role_frames.get(left, []):
                    for b in role_frames.get(right, []):
                        gap = abs(a - b)
                        minimum_frame_gap = gap if minimum_frame_gap is None else min(minimum_frame_gap, gap)
            pairwise[key] = {
                "section_overlap": sorted(section_sets[left] & section_sets[right]),
                "source_json_overlap_count": len(source_sets[left] & source_sets[right]),
                "track_scope_overlap_count": len(track_sets[left] & track_sets[right]),
                "minimum_same_video_frame_gap": minimum_frame_gap,
            }

    digest_roles = defaultdict(set)
    for row in rows:
        digest_roles[image_hashes[row["source_json"]]].add(row["split"])
    cross_split_exact_duplicates = sorted(
        digest for digest, digest_role_set in digest_roles.items() if len(digest_role_set) > 1
    )
    violations = []
    for key, record in pairwise.items():
        for field in ("section_overlap", "source_json_overlap_count", "track_scope_overlap_count"):
            if record[field]:
                violations.append(f"{key}:{field}={record[field]}")
        if record["minimum_same_video_frame_gap"] is not None and record["minimum_same_video_frame_gap"] <= 20:
            violations.append(
                f"{key}:near_temporal_duplicate_gap={record['minimum_same_video_frame_gap']}"
            )
    if cross_split_exact_duplicates:
        violations.append(
            f"cross_split_exact_image_duplicates={len(cross_split_exact_duplicates)}"
        )
    return {
        "pairwise": pairwise,
        "cross_split_exact_image_duplicates": cross_split_exact_duplicates,
        "violation_count": len(violations),
        "violations": violations,
    }


def build(source_manifest: Path, unified_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output}")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "annotations" / "source_v1").mkdir(parents=True)

    rows = load_rows(source_manifest)
    derived_rows = []
    image_hashes = {}
    coco_by_role = {
        role: {
            "info": {
                "description": "BeePoseTrack-E strict-v8 annotator-role view",
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": sha256(source_manifest),
                "split_policy": "annotator_01=calibration; annotator_03=dev_holdout; remaining=train",
            },
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "bee",
                    "supercategory": "insect",
                    "keypoints": ["head", "tail"],
                    "skeleton": [[1, 2]],
                }
            ],
        }
        for role in ("train", "calibration", "dev_holdout")
    }
    role_image_ids = Counter()
    role_annotation_ids = Counter()
    role_annotators = defaultdict(Counter)
    image_count = Counter()
    instance_count = defaultdict(Counter)
    bbox_clip_records = []

    for source_index, original in enumerate(rows, start=1):
        row = copy.deepcopy(original)
        annotator = annotator_from_source(row["source_json"])
        role = role_for_annotator(annotator)
        row["base_split"] = row.get("split")
        row["split"] = role
        row["annotator_id"] = annotator
        row["near_duplicate_cluster"] = row["section_id"]
        row["track_scope"] = row["section_id"]
        absolute_image, relative_image = canonical_image(unified_root, row)
        row["image_path"] = str(absolute_image)
        image_hashes[row["source_json"]] = sha256(absolute_image)

        role_image_ids[role] += 1
        image_id = role_image_ids[role]
        image = {
            "id": image_id,
            "file_name": relative_image,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "domain": str(row["domain"]).upper(),
            "source": annotator,
            "annotator_id": annotator,
            "scene": str(row["video_id"]).split("-", 1)[0],
            "video": row["video_id"],
            "video_id": row["video_id"],
            "section_id": row["section_id"],
            "frame": int(row["frame_id"]),
            "frame_id": int(row["frame_id"]),
            "source_json": row["source_json"],
            "source_image_path": row["source_image_path"],
            "split": role,
            "base_split": row["base_split"],
            "near_duplicate_cluster": row["near_duplicate_cluster"],
            "track_scope": row["track_scope"],
            "canonical_image_sha256": image_hashes[row["source_json"]],
        }
        coco_by_role[role]["images"].append(image)
        image_count[(role, row["domain"])] += 1
        role_annotators[role][annotator] += 1

        for instance in row.get("instances", []):
            if not bool(instance.get("det_mask", 0)):
                raise ValueError("strict-v7 instances must keep every det_mask=1 record")
            role_annotation_ids[role] += 1
            bbox, bbox_clip_audit = xyxy_to_xywh(
                instance["bbox_xyxy"], image["width"], image["height"]
            )
            keypoints, num_keypoints = coco_keypoints(instance)
            track_mask = bool(instance.get("track_mask", 0))
            source_group_id = instance.get("source_group_id")
            if track_mask and source_group_id is None:
                raise ValueError("track_mask=1 instance must retain source_group_id")
            annotation = {
                "id": role_annotation_ids[role],
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "pose_mask": bool(instance.get("pose_mask", 0)),
                "track_id": source_group_id if track_mask else None,
                "track_mask": track_mask,
                "det_mask": True,
                "source": annotator,
                "annotator_id": annotator,
                "source_group_id": source_group_id,
                "source_track_id": instance.get("track_id"),
                "pairing_method": instance.get("pairing_method", "unknown"),
                "source_dataset": instance.get("source_dataset", "labelme_5_sections"),
                "source_label": instance.get("source_label", "bee"),
                "quality": float(instance.get("quality", 1.0)),
            }
            if bbox_clip_audit is not None:
                annotation["source_bbox_xyxy"] = bbox_clip_audit["source_bbox_xyxy"]
                annotation["bbox_clip_delta_xyxy"] = [
                    after - before
                    for before, after in zip(
                        bbox_clip_audit["source_bbox_xyxy"],
                        bbox_clip_audit["clipped_bbox_xyxy"],
                    )
                ]
                bbox_clip_records.append(
                    {
                        "split": role,
                        "source_json": row["source_json"],
                        "annotation_id": annotation["id"],
                        **bbox_clip_audit,
                    }
                )
            coco_by_role[role]["annotations"].append(annotation)
            instance_count[role]["det"] += 1
            instance_count[role]["pose"] += int(annotation["pose_mask"])
            instance_count[role]["track"] += int(track_mask)
            instance_count[role]["box_only"] += int(not annotation["pose_mask"])
        derived_rows.append(row)

    expected_frames = {"train": 600, "calibration": 120, "dev_holdout": 120}
    actual_frames = Counter(row["split"] for row in derived_rows)
    if dict(actual_frames) != expected_frames:
        raise ValueError(f"unexpected frame split: expected={expected_frames}, actual={dict(actual_frames)}")
    for role, domain_expected in {
        "train": {"RGB": 300, "IR": 300},
        "calibration": {"RGB": 60, "IR": 60},
        "dev_holdout": {"RGB": 60, "IR": 60},
    }.items():
        actual = {domain: image_count[(role, domain)] for domain in ("RGB", "IR")}
        if actual != domain_expected:
            raise ValueError(f"unexpected {role} domain split: {actual}")

    audit = split_audit(derived_rows, image_hashes)
    if audit["violation_count"]:
        raise ValueError(f"split isolation audit failed: {audit['violations']}")

    manifest_path = temporary / "dataset_manifest.jsonl"
    atomic_text(
        manifest_path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in derived_rows),
    )

    annotation_shas = {}
    migration_summaries = {}
    for role, base_coco in coco_by_role.items():
        source_path = temporary / "annotations" / "source_v1" / f"instances_{role}.json"
        atomic_text(source_path, json.dumps(base_coco, ensure_ascii=False, separators=(",", ":")))
        migrated, migration_summary = migrate(source_path, unified_root)
        migrated["info"].update(
            {
                "source_annotation": str(
                    output / "annotations" / "source_v1" / f"instances_{role}.json"
                ),
                "derived_manifest": str(output / "dataset_manifest.jsonl"),
                "derived_manifest_sha256": sha256(manifest_path),
                "split": role,
            }
        )
        target_path = temporary / "annotations" / f"instances_{role}_schema_v2.json"
        atomic_text(target_path, json.dumps(migrated, ensure_ascii=False, separators=(",", ":")))
        annotation_shas[role] = sha256(target_path)
        migration_summaries[role] = migration_summary

    summary = {
        "schema_name": "BeePoseTrack-E strict-v8 annotator-role view",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "unified_root": str(unified_root),
        "split_policy": {
            "train": ["标注员_004", "标注员_02", "标注员_05"],
            "calibration": ["标注员_01"],
            "dev_holdout": ["标注员_03"],
        },
        "frames": dict(actual_frames),
        "domains": {
            role: {domain: image_count[(role, domain)] for domain in ("RGB", "IR")}
            for role in expected_frames
        },
        "instances": {role: dict(instance_count[role]) for role in expected_frames},
        "annotators": {role: dict(role_annotators[role]) for role in expected_frames},
        "split_audit": audit,
        "bbox_boundary_canonicalization": {
            "policy": "clip_only_when_maximum_coordinate_delta_px<=1.01; never drop instances",
            "tolerance_px": BBOX_CLIP_TOLERANCE_PX,
            "clipped_annotation_count": len(bbox_clip_records),
            "maximum_coordinate_delta_px": max(
                (record["maximum_coordinate_delta_px"] for record in bbox_clip_records),
                default=0.0,
            ),
            "records": bbox_clip_records,
        },
        "derived_manifest_sha256": sha256(manifest_path),
        "annotation_sha256": annotation_shas,
        "migration": migration_summaries,
    }
    summary_path = temporary / "dataset_summary.json"
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    ready = {
        "status": "READY",
        "source_manifest_sha256": summary["source_manifest_sha256"],
        "derived_manifest_sha256": summary["derived_manifest_sha256"],
        "dataset_summary_sha256": sha256(summary_path),
        "annotation_sha256": annotation_shas,
        "split_violation_count": audit["violation_count"],
        "frames": summary["frames"],
    }
    atomic_text(temporary / "VIEW_READY.json", json.dumps(ready, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args.source_manifest.resolve(), args.unified_root.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
