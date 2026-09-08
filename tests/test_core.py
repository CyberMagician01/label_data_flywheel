import copy
import numpy as np
import pytest
from label_data_flywheel.io import normalize, write_frames, iter_frames
from label_data_flywheel.postprocess import suppress_interpolation, interpolate
from label_data_flywheel.sampling import (
    calibrate_weights,
    ess,
    balanced_batches,
    split_groups,
    largest_remainder,
)
from label_data_flywheel.quality import q2, route_candidate, head_tail_evidence
from label_data_flywheel.review import apply_reviews, feedback_snapshot
from label_data_flywheel.behavior import analyze
from label_data_flywheel.curriculum import Curriculum
from label_data_flywheel.calibration import fit_isotonic, calibrated


def det(tid, box, origin="observed"):
    return {
        "track_id": tid,
        "bbox_xyxy": box,
        "origin": origin,
        "keypoints": {
            "head": [box[0] + 1, box[1] + 1, 0.9],
            "abdomen_tip": [box[2] - 1, box[3] - 1, 0.9],
        },
    }


def frame(index, ds):
    return normalize(
        {"frame": index, "detections": ds, "image_size": [100, 100]},
        video="B",
        domain="IR_in",
    )


def test_only_fill_suppressed_and_input_immutable():
    f = frame(
        0,
        [
            det(1, [0, 0, 20, 20]),
            det(2, [1, 1, 21, 21]),
            det(3, [10, 10, 25, 25], "linear_interpolation"),
            det(4, [50, 50, 70, 70], "linear_interpolation"),
        ],
    )
    original = copy.deepcopy(f)
    result = suppress_interpolation(f)
    assert [d["track_id"] for d in result["detections"]] == [1, 2, 4]
    assert result["temporarily_hidden_detections"][0]["track_id"] == 3
    assert f == original


def test_interpolation_source_indices_and_id():
    frames = [
        frame(10, [det(3, [0, 0, 10, 10])]),
        frame(11, []),
        frame(12, [det(3, [10, 0, 20, 10])]),
    ]
    result = interpolate(frames)
    assert result[1]["detections"][0]["bbox_xyxy"] == [5, 0, 15, 10]
    assert result[1]["detections"][0]["keypoints"] == {}
    assert result[1]["detections"][0]["track_id"] == 3
    assert not frames[1]["detections"]


def test_balancing_coverage_and_suspect():
    records = [
        {
            "sample_id": str(i),
            "domain": "IR_in" if i < 20 else "RGB_out",
            "video": str(i % 4),
            "quality": 1,
            "status": "suspect" if i == 0 else "valid",
        }
        for i in range(24)
    ]
    w = calibrate_weights(records, {"domain": {"IR_in": 0.5, "RGB_out": 0.5}})
    assert w[0] == 0 and ess(w) >= 0.5 * 23
    batches = balanced_batches(records, w)
    assert set(i for b in batches for i in b) == set(range(1, 24))
    assert all(sum(records[i]["domain"] == "IR_in" for i in b) == 4 for b in batches)


def test_split_components_never_leak():
    records = [
        {
            "sample_id": str(i),
            "domain": "IR_in",
            "video": str(i // 2),
            "segment": "S",
            "track_ids": [i],
            "duplicate_cluster": "shared" if i in (1, 2) else str(i),
        }
        for i in range(8)
    ]
    split = split_groups(records)
    assert split["0"] == split["1"] == split["2"] == split["3"]
    assert sum(largest_remainder({"a": 0.3, "b": 0.7}, 7).values()) == 7


def test_quality_and_independent_evidence():
    rows = [
        {"domain": "IR_in", "layer": "L1", "group": "01", "error": v}
        for v in [0, 0.01, 0.02, 1.0]
    ]
    out = q2(rows)
    assert out[-1]["quality"] < out[0]["quality"]
    assert (
        route_candidate(
            [
                {"name": "a", "family": "same", "probability": 0.9},
                {"name": "b", "family": "same", "probability": 0.95},
            ]
        )["route"]
        == "ignore"
    )
    assert (
        route_candidate(
            [{"name": "a", "probability": 0.9}, {"name": "b", "probability": 0.95}]
        )["route"]
        == "soft_positive"
    )


def test_pose_swap_and_isotonic():
    a = det(1, [0, 0, 10, 10])
    b = copy.deepcopy(a)
    b["keypoints"]["head"], b["keypoints"]["abdomen_tip"] = (
        b["keypoints"]["abdomen_tip"],
        b["keypoints"]["head"],
    )
    assert head_tail_evidence(a, b, [0, 0, 10, 10])["action"] == "swap_candidate"
    model = fit_isotonic([0.1, 0.3, 0.5, 0.8], [0, 1, 0, 1])
    p = calibrated([0.1, 0.3, 0.5, 0.8], model)
    assert np.all(np.diff(p) >= 0)
    with pytest.raises(ValueError):
        fit_isotonic([0.1], [1], split="test")


def test_review_does_not_mutate_and_requires_human(tmp_path):
    frames = [frame(0, [det(1, [0, 0, 10, 10])])]
    d = frames[0]["detections"][0]
    result = apply_reviews(
        frames,
        [{"entity_id": d["entity_id"], "action": "confirm"}],
        "reviewer01",
        tmp_path / "audit.jsonl",
    )
    assert len(feedback_snapshot(result)) == 1 and not feedback_snapshot(frames)
    assert (tmp_path / "audit.jsonl").exists()


def test_behavior_source_frame_units_and_crossing():
    frames = [frame(i, [det(1, [30 + i, 40, 40 + i, 50])]) for i in [0, 5, 10, 15, 20]]
    result = analyze(
        frames, {"entrance_line": [[0.5, 0], [0.5, 1]], "min_duration_source_frames": 0}
    )
    assert result["observations"][1]["speed_px_per_source_frame"] == 1.0
    assert result["behavior_labels_are_candidates"]
    assert len(result["events"]) == 1
    assert all(r["calibrated_count"] is None for r in result["group_windows"])


def test_curriculum_coverage_gate():
    c = Curriculum(min_updates=2, plateau_window=2)
    for _ in range(3):
        c.observe({"A": 0.8, "B": 0.8}, False, True)
    assert c.stage == 0
    c.observe({"A": 0.8, "B": 0.8}, True, True)
    assert c.stage == 1


def test_round_trip_preserves_hidden(tmp_path):
    f = suppress_interpolation(
        frame(
            0, [det(1, [0, 0, 10, 10]), det(2, [0, 0, 10, 10], "linear_interpolation")]
        )
    )
    path = tmp_path / "frames.jsonl.gz"
    write_frames(path, [f])
    result = list(iter_frames(path, video="B", domain="IR_in"))
    assert (
        result[0]["temporarily_hidden_detections"] == f["temporarily_hidden_detections"]
    )
