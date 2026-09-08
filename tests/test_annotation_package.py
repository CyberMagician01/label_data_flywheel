import copy
import json

import numpy as np
import pytest

from label_data_flywheel.io import normalize, read_json, write_frames
from label_data_flywheel.annotation_package import (
    export_annotations,
    validate_annotations,
)
from label_data_flywheel.assets.extract_frames import extract


def test_frozen_benchmark_bytes_match_original_manifest():
    from pathlib import Path
    import hashlib

    root = Path(__file__).resolve().parents[1] / "legacy/indoor/benchmark"
    manifest = read_json(root / "manifest.json")
    assert manifest["train_excluded"] and len(manifest["samples"]) == 36
    for record in manifest["samples"]:
        assert (
            hashlib.sha256((root / record["annotation"]).read_bytes()).hexdigest()
            == record["annotation_sha256"]
        )


def source(tmp_path, name, *, frame=0, group="01", split="train", detections=None):
    f = normalize(
        {
            "frame": frame,
            "image_size": [100, 100],
            "detections": detections
            or [
                {"bbox_xyxy": [10, 10, 30, 30], "track_id": 0, "label_status": "human"}
            ],
        },
        video="B-5-1",
        domain="IR_in",
        group=group,
        split=split,
        source="human",
    )
    path = tmp_path / (name + ".jsonl")
    write_frames(path, [f])
    return f, {
        "path": str(path),
        "video": "B-5-1",
        "domain": "IR_in",
        "group": group,
        "split": split,
        "source": "human",
        "frame_index_base": 0,
    }


def test_package_separates_status_visibility_and_preserves_originals(tmp_path):
    f, spec = source(tmp_path, "source")
    f["detections"] += [
        {
            "entity_id": "clip",
            "bbox_xyxy": [-5, 10, 20, 40],
            "track_id": 8,
            "label_status": "human_confirmed",
            "keypoints": {"head": [4, 12, 0.5], "abdomen_tip": [8, 30, 0.8]},
            "keypoint_visibility": {"head": 0, "abdomen_tip": 1},
        },
        {
            "entity_id": "fill",
            "bbox_xyxy": [10, 10, 30, 30],
            "track_id": 7,
            "origin": "interpolation",
            "label_status": "human",
        },
        {
            "entity_id": "machine",
            "bbox_xyxy": [60, 60, 90, 90],
            "track_id": 9,
            "label_status": "soft_positive",
        },
    ]
    f["temporarily_hidden_detections"] = [
        {"entity_id": "hidden", "bbox_xyxy": [10, 10, 30, 30], "track_id": 10}
    ]
    before = copy.deepcopy(f)
    write_frames(spec["path"], [f])
    out = tmp_path / "package"
    review_log = tmp_path / "review.jsonl"
    review_log.write_text(
        json.dumps(
            {"reviewer": "reviewer_01", "entity_id": "clip", "action": "correct"}
        )
        + "\n"
    )
    manifest = export_annotations(
        {"version": "one", "inputs": [spec], "review_audits": [str(review_log)]}, out
    )
    assert len(manifest["review_audit_sources"]) == 1
    assert (
        out / manifest["review_audit_sources"][0]["packaged"]
    ).read_bytes() == review_log.read_bytes()
    assert f == before and manifest["counts"]["confirmed"] == 2
    assert (
        manifest["counts"]["candidates"] == 2 and manifest["counts"]["audit_only"] == 1
    )
    data = read_json(out / "annotations/B-5-1/01/confirmed.coco.json")
    assert data["annotations"][0]["source_track_id"] == 0
    assert data["images"][0]["source_frame_id"] == 0
    assert data["annotations"][1]["bbox"] == [0, 10, 20, 30]
    assert data["annotations"][1]["keypoints"] == [0, 0, 0, 8, 30, 1]
    assert validate_annotations(out)["passed"]
    with pytest.raises(FileExistsError):
        export_annotations({"version": "one", "inputs": [spec]}, out)


def test_annotation_data_never_uses_inference_top600_or_nms(tmp_path):
    boxes = [
        {"bbox_xyxy": [10, 10, 30, 30], "track_id": i, "label_status": "unconfirmed"}
        for i in range(623)
    ]
    _, spec = source(tmp_path, "dense", detections=boxes)
    out = tmp_path / "dense_package"
    report = export_annotations({"version": "dense", "inputs": [spec]}, out)
    assert report["counts"]["candidates"] == 623
    assert not (out / "annotations/B-5-1/01/confirmed.coco.json").exists()
    assert read_json(out / "annotations/B-5-1/01/candidates.coco.json")["images"]
    assert validate_annotations(out)["passed"]


def test_duplicate_physical_frame_cannot_cross_annotator_splits(tmp_path):
    _, a = source(tmp_path, "a", group="01", split="train")
    _, b = source(tmp_path, "b", group="03", split="val")
    out = tmp_path / "leak"
    with pytest.raises(ValueError, match="跨划分"):
        export_annotations({"version": "leak", "inputs": [a, b]}, out)
    assert read_json(out / "build_status.json")["status"] == "failed"
    assert not (out / "manifest.json").exists()


def test_hash_media_audit_and_unassigned_split(tmp_path):
    _, spec = source(tmp_path, "raw", split="unassigned")
    spec["frame_index_base"] = None
    out = tmp_path / "pending"
    manifest = export_annotations({"version": "pending", "inputs": [spec]}, out)
    assert manifest["pending"] and not (out / "splits/train.txt").read_text()
    target = out / "annotations/B-5-1/01/confirmed.coco.json"
    target.write_text(target.read_text() + " ")
    (out / "unexpected.jpg").write_bytes(b"not-an-image")
    result = validate_annotations(out)
    assert not result["passed"]
    assert "unexpected_image_or_video" in result["errors"]
    assert any(x.startswith("hash_mismatch:") for x in result["errors"])


def test_missing_keypoints_and_negative_frame_are_not_invented(tmp_path):
    f, spec = source(tmp_path, "empty", split="calibration")
    f["detections"] = []
    write_frames(spec["path"], [f])
    out = tmp_path / "empty_unreviewed"
    result = export_annotations({"version": "empty", "inputs": [spec]}, out)
    assert not result["outputs"]
    f["annotation_complete"] = True
    write_frames(spec["path"], [f])
    out = tmp_path / "empty_confirmed"
    result = export_annotations({"version": "empty_confirmed", "inputs": [spec]}, out)
    data = read_json(out / "annotations/B-5-1/01/confirmed.coco.json")
    assert len(data["images"]) == 1 and data["annotations"] == []
    assert (out / "splits/calibration.txt").read_text() and not (
        out / "splits/val.txt"
    ).read_text()


def test_coco_occlusion_and_canonical_domain_are_preserved(tmp_path):
    f, spec = source(tmp_path, "pose")
    f["detections"][0]["keypoints"] = {"head": [12, 12, 1], "abdomen_tip": [15, 25, 2]}
    write_frames(spec["path"], [f])
    spec["keypoint_encoding"] = "coco_visibility"
    spec["domain"] = "RGB_out"
    out = tmp_path / "coco_points"
    export_annotations({"version": "points", "inputs": [spec]}, out)
    annotation = read_json(out / "annotations/B-5-1/01/confirmed.coco.json")[
        "annotations"
    ][0]
    assert annotation["keypoints"][2::3] == [1, 2]
    audit = json.loads((out / "audit/records.jsonl").read_text())
    assert audit["track_scope"][0] == "IR_in"


def test_extract_frames_source_index_and_unicode_path(tmp_path):
    import cv2

    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (16, 16)
    )
    assert writer.isOpened()
    for gray in (40, 100, 180):
        writer.write(np.full((16, 16, 3), gray, np.uint8))
    writer.release()
    manifest = tmp_path / "frames.jsonl"
    rows = [
        {
            "video": "A-5-1",
            "source_frame_id": i,
            "frame_index_base": 1,
            "image_ref": f"A-5-1/frame_{i:08d}.jpg",
            "width": 16,
            "height": 16,
        }
        for i in (1, 3)
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "中文 空格目录"
    result = extract(video, "A-5-1", manifest, out, 1)
    assert result["frames_written"] == 2
    for row, expected in zip(rows, (40, 180)):
        data = np.frombuffer((out / row["image_ref"]).read_bytes(), np.uint8)
        assert abs(cv2.imdecode(data, cv2.IMREAD_COLOR).mean() - expected) < 3
    with pytest.raises(FileExistsError):
        extract(video, "A-5-1", manifest, out, 1)
