import copy
from label_data_flywheel.io import normalize
from label_data_flywheel.metrics import tracking_sequence, tracking_metrics
from label_data_flywheel.fusion import attach_pose


def sample(i, tid=1):
    return normalize(
        {"frame": i, "detections": [{"track_id": tid, "bbox_xyxy": [10, 10, 30, 30]}]},
        domain="IR_in",
        video="B",
        source="human",
    )


def test_pose_source_alignment_without_changing_identity():
    f = sample(1)
    p = sample(1, 99)
    p["detections"][0]["keypoints"] = {
        "head": [15, 15, 0.9],
        "abdomen_tip": [25, 25, 0.8],
    }
    result = attach_pose(f, p)
    assert result["detections"][0]["track_id"] == 1
    assert result["detections"][0]["keypoints"] == p["detections"][0]["keypoints"]
    assert result["pose_fusion"]["matched"] == 1


def test_official_trackeval_perfect_track_and_switch():
    gt = [sample(i) for i in range(4)]
    pred = copy.deepcopy(gt)
    good = tracking_metrics(tracking_sequence(gt, pred))
    assert (
        good["Identity"]["IDF1"] == 1.0
        and good["CLEAR"]["MOTA"] == 1.0
        and good["HOTA"]["HOTA"] == 1.0
    )
    pred[2]["detections"][0]["track_id"] = 2
    pred[3]["detections"][0]["track_id"] = 2
    bad = tracking_metrics(tracking_sequence(gt, pred))
    assert bad["CLEAR"]["IDSW"] == 1 and bad["Identity"]["IDF1"] < 1.0
