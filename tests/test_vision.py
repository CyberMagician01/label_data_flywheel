import numpy as np
import pytest
from label_data_flywheel.io import normalize
from label_data_flywheel.density import (
    count_preserving_map,
    select_recheck_region,
    motion_evidence,
)
from label_data_flywheel.transforms import affine_labels, photometric
from label_data_flywheel.tracking import AssociationTracker
from label_data_flywheel.count_calibration import fit_recall, corrected_count


def frame(i, boxes):
    return normalize(
        {
            "frame": i,
            "image_size": [100, 100],
            "detections": [{"bbox_xyxy": b, "confidence": c} for b, c in boxes],
        },
        domain="IR_in",
        video="B",
    )


def test_two_stage_tracker_reuses_low_confidence_and_gap_ids():
    t = AssociationTracker(retention=150)
    a = t.update(frame(0, [([10, 10, 30, 30], 0.9)]))
    b = t.update(frame(1, [([11, 10, 31, 30], 0.2)]))
    t.update(frame(2, []))
    c = t.update(frame(3, [([13, 10, 33, 30], 0.9)]))
    assert (
        a["detections"][0]["track_id"]
        == b["detections"][0]["track_id"]
        == c["detections"][0]["track_id"]
    )


def test_density_boundary_count_conservation_and_residual_roi():
    density = count_preserving_map([[0, 0], [99, 99], [75, 20]], (100, 100))
    assert density.sum() == pytest.approx(3.0, abs=1e-5)
    result = select_recheck_region(
        density,
        [{"bbox_xyxy": [0, 0, 1, 1]}, {"bbox_xyxy": [98, 98, 100, 100]}],
        (100, 100),
    )
    x1, y1, x2, y2 = result["roi_xyxy"]
    assert x1 <= 75 <= x2 and y1 <= 20 <= y2


def test_consistent_affine_and_ir_photometry():
    f = frame(0, [([10, 10, 20, 20], 0.9)])
    f["detections"][0]["keypoints"] = {
        "head": [12, 12, 1.0],
        "abdomen_tip": [18, 18, 1.0],
    }
    result = affine_labels(f, [[1, 0, 5], [0, 1, 3]], (100, 100))
    assert result["detections"][0]["bbox_xyxy"] == [15, 13, 25, 23]
    assert result["detections"][0]["keypoints"]["head"][:2] == [17, 15]
    assert f["detections"][0]["bbox_xyxy"] == [10, 10, 20, 20]
    im = photometric(
        np.zeros((10, 10, 3), np.uint8), "IR_in", np.random.default_rng(0), noise=0
    )
    assert np.array_equal(im[:, :, 0], im[:, :, 1])


def test_static_image_does_not_invent_motion_candidates():
    image = np.zeros((64, 64, 3), np.uint8)
    assert not motion_evidence(image, image)["candidate_mask"].any()


def test_recall_calibration_requires_calibrated_probability():
    model = fit_recall(
        [
            {
                "domain": "IR_in",
                "split": "calibration",
                "human_verified": True,
                "true_positive": 8,
                "ground_truth": 10,
            }
        ]
    )
    assert corrected_count(
        [{"calibrated_probability": 0.8}], model, {"domain": "IR_in"}
    )["calibrated_count"] == pytest.approx(1.0)
    assert (
        corrected_count([{"confidence": 0.8}], model, {"domain": "IR_in"})[
            "calibrated_count"
        ]
        is None
    )
