import copy
import numpy as np
import pytest
from label_data_flywheel.colony import EntranceGate, analyze_colony, temporal_analysis
from label_data_flywheel.apiculture import interpret_colony, literature
from label_data_flywheel.io import normalize, write_json
from label_data_flywheel.colony_runner import run_colony


def frame(i, boxes=None, group="g"):
    return normalize(
        {"frame": i, "fps": 30, "image_size": [100, 100], "detections": boxes or []},
        video="v",
        domain="IR_in",
        group=group,
    )


def bee(tid=1, x=20, y=20, origin="observed"):
    return {
        "bbox_xyxy": [x - 5, y - 5, x + 5, y + 5],
        "track_id": tid,
        "origin": origin,
        "confidence": 0.9,
    }


def gate():
    return EntranceGate(
        {
            "line": [[0.1, 0.5], [0.9, 0.5]],
            "deadband_pixels": 2,
            "confirm_seconds": 0.1,
        },
        [100, 100],
        30,
    )


def test_gate_jitter_and_confirmed_crossing():
    g = gate()
    points = [40] * 4 + [49, 51] * 5 + [60] * 4
    events = [
        e
        for i, y in enumerate(points)
        if (e := g.update("a", i, np.array([50.0, y]), 30))
    ]
    assert len(events) == 1 and events[0]["direction"] == "in"


def test_gate_finite_segment_gap_and_jump():
    for x, gap, length in [(99, 1, 30), (50, 30, 30), (50, 1, 0.1)]:
        g = gate()
        events = []
        for i in range(4):
            events.append(g.update("a", i, np.array([float(x), 40.0]), length))
        for i in range(4):
            events.append(
                g.update("a", 3 + gap + i, np.array([float(x), 60.0]), length)
            )
        assert not any(events)


def test_density_conservation_and_fills_do_not_create_motion():
    fs = [frame(i, [bee(), bee(2, 60, 60, "linear_interpolation")]) for i in range(35)]
    original = copy.deepcopy(fs)
    r = analyze_colony(fs)
    w = r["scopes"][0]["windows"][0]
    assert np.asarray(w["density_mean_objects_per_cell"]).sum() == pytest.approx(
        1, abs=1e-6
    )
    assert w["mean_observed_count"] == 1 and w["mean_visible_interpolations"] == 1
    assert w["active_fraction"] == 0
    assert w["entrance_flux_per_minute"] is None
    assert fs == original


def test_network_requires_duration_and_keeps_group_scope():
    fs = [frame(i, [bee(), bee(2, 28)], group=g) for g in ["a", "b"] for i in range(15)]
    r = analyze_colony(fs)
    assert len(r["scopes"]) == 2
    assert all(len(s["windows"][0]["network"]["edges"]) == 1 for s in r["scopes"])
    short = analyze_colony(fs[:3])["scopes"][0]["windows"][0]
    assert not short["network"]["edges"]


def test_missing_frames_are_unobserved_and_ids_not_merged():
    r = analyze_colony([frame(0, [bee()]), frame(29, [bee(2)])])
    w = r["scopes"][0]["windows"][0]
    assert w["coverage"] == pytest.approx(2 / 30)
    assert w["active_fraction"] is None
    assert r["scopes"][0]["unique_track_ids"] == 2
    with pytest.raises(ValueError):
        analyze_colony([frame(2), frame(1)])


def test_unverified_gate_does_not_claim_entrance_flux():
    cfg = {"entrance": {"line": [[0.1, 0.5], [0.9, 0.5]]}}
    r = analyze_colony([frame(i, [bee(y=40 + i)]) for i in range(30)], cfg)
    s = r["scopes"][0]
    assert s["gate_status"] == "reference_line_only"
    assert s["windows"][0]["entrance_flux_per_minute"] is None
    cfg["entrance"]["calibration_status"] = "verified"
    r = analyze_colony([frame(i, [bee(y=40 + i)]) for i in range(30)], cfg)
    assert r["scopes"][0]["windows"][0]["entrance_flux_per_minute"][
        "in"
    ] == pytest.approx(60)


def test_context_cannot_create_diagnosis_or_cross_video():
    r = analyze_colony([frame(i, [bee()]) for i in range(40)])
    r["scopes"][0]["temporal"]["change_candidates"] = [
        {"window_index": 0, "metric": "mean_observed_count", "value": 8}
    ]
    rows = interpret_colony(r)["interpretations"]
    assert all(
        x["status"] == "insufficient_evidence" and x["diagnosis"] is None for x in rows
    )
    context = {
        "domain": "IR_in",
        "video": "wrong",
        "group": "g",
        "window_index": 0,
        "source": "test_only",
        "measurements": {"pesticide_exposure_record": True, "dead_bee_count": 10},
    }
    assert all(
        x["status"] == "insufficient_evidence"
        for x in interpret_colony(r, [context])["interpretations"]
    )
    context["video"] = "v"
    pesticide = next(
        x
        for x in interpret_colony(r, [context])["interpretations"]
        if x["scenario"] == "pesticide"
    )
    assert pesticide["status"] == "review_candidate" and pesticide["diagnosis"] is None
    assert len(literature()["cards"]) >= 7


def test_forward_baseline_and_short_record_no_period():
    r = analyze_colony([frame(i, [bee()]) for i in range(240)], {"window_seconds": 1})
    ws = r["scopes"][0]["windows"]
    ws[-1]["mean_observed_count"] = 20
    t = temporal_analysis(ws, {"window_seconds": 1})
    assert any(
        c["metric"] == "mean_observed_count" and c["window_index"] == 7
        for c in t["change_candidates"]
    )
    assert t["metrics"]["mean_observed_count"]["period_candidate_seconds"] is None


def test_runner_immutable_output_and_receipt(tmp_path):
    source = tmp_path / "input.json"
    write_json(source, [frame(i, [bee()]) for i in range(12)])
    config = {
        "inputs": [
            {"path": str(source), "video": "v", "domain": "IR_in", "group": "g"}
        ],
        "render": False,
    }
    result = run_colony(config, tmp_path / "out")
    assert result["frames"] == 12 and (tmp_path / "out/colony.json").exists()
    with pytest.raises(FileExistsError):
        run_colony(config, tmp_path / "out")


def test_original_behavior_graph_does_not_mix_annotation_groups():
    from label_data_flywheel.behavior import analyze

    a = [frame(i, [bee(), bee(2, 25)], "a") for i in range(8)]
    b = [frame(i, [bee()], "b") for i in range(8)]
    for f in a + b:
        for d in f["detections"]:
            x = (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2
            d["keypoints"] = {"head": [x, 16, 0.9], "abdomen_tip": [x, 24, 0.9]}
    r = analyze(a + b)
    assert r["interaction_edges"]
    assert all(row["group"] == "a" for row in r["interaction_edges"])
    assert all(
        not row["graph_statistics"]["weighted_degree"]
        for row in r["group_windows"]
        if row["group"] == "b"
    )


def test_auxiliary_association_preserves_low_confidence_observations(tmp_path):
    from label_data_flywheel.io import read_json

    source = tmp_path / "input.json"
    low = bee(9, 80)
    low["confidence"] = 0.01
    fs = [frame(i, [bee(), low]) for i in range(12)]
    write_json(source, fs)
    before = source.read_bytes()
    config = {
        "inputs": [
            {
                "path": str(source),
                "video": "v",
                "domain": "IR_in",
                "group": "g",
                "tracking_mode": "analysis_baseline",
            }
        ],
        "render": False,
    }
    run_colony(config, tmp_path / "out")
    report = read_json(tmp_path / "out/colony.json")
    assert report["scopes"][0]["windows"][0]["mean_observed_count"] == 2
    assert source.read_bytes() == before


def test_colony_round_writes_reviewable_evidence(tmp_path):
    from label_data_flywheel.pipeline import run_round

    source = tmp_path / "input.json"
    write_json(source, [frame(i, [bee()]) for i in range(12)])
    config = {
        "inputs": [
            {"path": str(source), "video": "v", "domain": "IR_in", "group": "g"}
        ],
        "colony": {"enabled": True},
        "review_budget": 8,
    }
    run_round(config, tmp_path / "out")
    assert (tmp_path / "out/colony.json").exists()
    assert (tmp_path / "out/colony_review_queue.json").exists()
    from label_data_flywheel.io import read_json

    assert (
        len(read_json(tmp_path / "out/knowledge_graph.json")["literature_cards"]) == 7
    )


def test_labelme_directory_retains_human_instances(tmp_path):
    from label_data_flywheel.io import read_json

    folder = tmp_path / "labels"
    folder.mkdir()
    write_json(
        folder / "frame_00000100.json",
        {
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "shape_type": "rectangle",
                    "group_id": 7,
                    "label": "bee",
                    "points": [[10, 10], [30, 30]],
                }
            ],
        },
    )
    result = run_colony(
        {
            "inputs": [
                {
                    "path": str(folder),
                    "video": "v",
                    "domain": "IR_in",
                    "source": "human",
                }
            ],
            "render": False,
        },
        tmp_path / "out",
    )
    assert result["frames"] == 1
    r = read_json(tmp_path / "out/colony.json")
    assert r["scopes"][0]["source_frame_range"] == [100, 100]
    assert r["scopes"][0]["windows"][0]["mean_observed_count"] == 1


def test_group_reviews_are_named_idempotent_and_separate_from_instances():
    from label_data_flywheel.apiculture import review_colony

    r = analyze_colony([frame(i, [bee()]) for i in range(40)])
    r["scopes"][0]["temporal"]["change_candidates"] = [
        {"window_index": 0, "metric": "mean_observed_count"}
    ]
    r["apiculture"] = interpret_colony(r)
    eid = r["apiculture"]["review_queue"][0]["event_id"]
    decision = {
        "decision_id": "test-001",
        "event_id": eid,
        "action": "confirm",
        "human_label": "合成测试：计数变化",
    }
    reviewed = review_colony(r, [decision], "synthetic_test_reviewer")
    again = review_colony(reviewed, [decision], "synthetic_test_reviewer")
    assert len(again["colony_review_audit"]) == 1
    assert len(again["confirmed_group_windows"]) == 1
    assert r["apiculture"]["review_queue"][0]["status"] == "unconfirmed"
    with pytest.raises(ValueError):
        review_colony(r, [decision], "")
    with pytest.raises(ValueError):
        review_colony(
            reviewed,
            [{**decision, "human_label": "changed"}],
            "synthetic_test_reviewer",
        )


def test_confirmed_group_event_returns_to_frame_window_only(tmp_path):
    from label_data_flywheel.pipeline import run_round
    from label_data_flywheel.io import read_json, iter_frames

    source = tmp_path / "input.json"
    fs = [
        frame(i, [bee()] if i < 210 else [bee(), bee(2, 60), bee(3, 85)])
        for i in range(240)
    ]
    write_json(source, fs)
    decision = {
        "decision_id": "synthetic-01",
        "event_id": "IR_in/g/v/colony/210-239",
        "action": "confirm",
        "human_label": "合成测试：观测数量上升",
    }
    config = {
        "inputs": [
            {"path": str(source), "video": "v", "domain": "IR_in", "group": "g"}
        ],
        "colony": {
            "enabled": True,
            "analysis": {"window_seconds": 1},
            "review_decisions": [decision],
            "reviewer": "synthetic_test_reviewer",
        },
        "review_budget": 8,
    }
    run_round(config, tmp_path / "out")
    assert len(read_json(tmp_path / "out/confirmed_group_windows.json")) == 1
    result = list(
        iter_frames(tmp_path / "out/annotations.jsonl.gz", video="v", domain="IR_in")
    )
    assert "confirmed_colony_events" not in result[0]
    assert (
        result[-1]["confirmed_colony_events"][0]["human_label"]
        == decision["human_label"]
    )
    assert all(d["label_status"] == "unconfirmed" for d in result[-1]["detections"])
