import copy

from label_data_flywheel.apiculture import interpret_colony
from label_data_flywheel.behavior import analyze
from label_data_flywheel.colony import analyze_colony, temporal_analysis
from label_data_flywheel.io import normalize


def frame(index, detections):
    return normalize(
        {"frame": index, "fps": 30, "image_size": [100, 100], "detections": detections},
        video="v", domain="IR_in", group="g",
    )


def bee(identity, x, origin="observed"):
    return {"track_id": identity, "bbox_xyxy": [x, 10, x + 10, 30],
            "origin": origin, "keypoints": {"head": [x + 5, 12, .9], "abdomen_tip": [x + 5, 28, .9]}}


def test_fill_does_not_create_behavior_count_motion_or_relations():
    original = [frame(i, [bee(1, 10 + i)]) for i in range(12)]
    with_fill = copy.deepcopy(original)
    for f in with_fill:
        f["detections"].append(bee(2, 11 + f["frame"], "linear_interpolation"))
    before = copy.deepcopy(with_fill)
    assert analyze(with_fill) == analyze(original)
    assert with_fill == before


def test_observed_endpoints_keep_real_gap_across_interpolation():
    fs = [frame(0, [bee(1, 10)]), frame(1, [bee(1, 60, "linear_interpolation")]), frame(2, [bee(1, 12)])]
    report = analyze(fs)
    assert [r["frame"] for r in report["observations"]] == [0, 2]
    assert report["observations"][-1]["speed_px_per_source_frame"] == 1
    assert [r["raw_count"] for r in report["group_windows"]] == [1, 0, 1]


def test_rejected_instance_does_not_change_colony_density_motion_or_network():
    original = [frame(i, [bee(1, 10 + i)]) for i in range(12)]
    reviewed = copy.deepcopy(original)
    for f in reviewed:
        f["detections"].append({**bee(2, 11 + f["frame"]), "label_status": "invalid"})
    assert analyze(reviewed) == analyze(original)
    assert analyze_colony(reviewed) == analyze_colony(original)


def test_video_only_count_change_does_not_require_ids_or_external_context():
    detection = {"bbox_xyxy": [10, 10, 20, 30]}
    fs = [frame(i, [detection] if i < 210 else [detection] * 4) for i in range(240)]
    r = analyze_colony(fs, {"window_seconds": 1})
    result = interpret_colony(r)
    count = next(x for x in result["interpretations"] if x["metric"] == "mean_observed_count")
    assert count["status"] == "visual_change"
    assert count["source_frame_range"] == [210, 239]
    assert count["baseline_median"] == 1 and count["value"] == 4
    assert result["mode"] == "video_only"
    assert all("external_measurements" not in x and "scenario" not in x for x in result["interpretations"])
    assert len(result["review_queue"]) == 1


def test_motion_change_with_low_coverage_requests_observability_review():
    r = analyze_colony([frame(i, [bee(1, 10)]) for i in range(240)], {"window_seconds": 1})
    w = r["scopes"][0]["windows"][-1]
    w["median_speed_bl_proxy_s"] = 2
    w["motion_observable_fraction"] = .1
    r["scopes"][0]["temporal"] = temporal_analysis(r["scopes"][0]["windows"], r["config"])
    result = interpret_colony(r)
    motion = next(x for x in result["interpretations"] if x["metric"] == "median_speed_bl_proxy_s")
    assert motion["status"] == "observability_check"
