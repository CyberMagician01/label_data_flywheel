import copy
import numpy as np
import pytest
from label_data_flywheel.io import normalize, write_frames, read_json
from label_data_flywheel.training_data import export_coco
from label_data_flywheel.experts import fit_experts, apply_candidate_evidence
from label_data_flywheel.knowledge import default_graph, update_graph, attach_context
from label_data_flywheel.pipeline import run_round
from label_data_flywheel.metrics import (
    detection_metrics,
    pose_metrics,
    behavior_metrics,
)
from label_data_flywheel.registry import choose_champion
from label_data_flywheel.sampling import sinkhorn_weights
from label_data_flywheel.io import collapse_sam_emissions


def example(frame=0, domain="IR_in", source="human"):
    return normalize(
        {
            "frame": frame,
            "image_size": [100, 100],
            "image_path": "/server/image.jpg",
            "detections": [
                {
                    "bbox_xyxy": [10 + frame, 10, 30 + frame, 30],
                    "track_id": 7,
                    "confidence": 0.9,
                    "keypoints": {
                        "head": [20 + frame, 12, 0.9],
                        "abdomen_tip": [20 + frame, 27, 0.9],
                    },
                }
            ],
        },
        domain=domain,
        video="B" if domain == "IR_in" else "A",
        group="01",
        source=source,
        split="train",
    )


def test_actual_training_schema_masks_geometry_and_ignore(tmp_path):
    a = example()
    b = example(1)
    c = example(2)
    c["detections"][0]["keypoints"] = {}
    c["detections"][0]["supervision_mask"]["pose"] = False
    b["ignore_regions"] = [
        {"entity_id": "ignore", "bbox_xyxy": [60, 60, 90, 90], "label_status": "ignore"}
    ]
    b["detections"].append(
        {
            "entity_id": "not-approved",
            "bbox_xyxy": [0, 0, 4, 4],
            "label_status": "unconfirmed",
        }
    )
    data = export_coco([a, b, c], tmp_path / "data.json", "/server")
    assert data["schema_version"] == 2 and data["images"][0]["domain"] == "IR"
    assert len(data["annotations"]) == 4
    anns = data["annotations"]
    assert not anns[0]["track_geometry_mask"]
    assert anns[1]["track_geometry"][0] == pytest.approx(1 / np.hypot(20, 20))
    assert anns[1]["track_geometry"][4:] == [0.0, 1.0]
    assert anns[2]["ignore_region"] and anns[2]["background_weight"] == 0
    assert anns[3]["pose_state"] == 0 and anns[3]["keypoints"] == []
    assert anns[3]["track_mask"] and not anns[3]["track_axis_mask"]
    assert all(x["source_entity_id"] != "not-approved" for x in anns)


def test_pseudo_labels_never_enter_evaluation(tmp_path):
    f = example()
    f["split"] = "calibration"
    f["detections"][0]["label_status"] = "soft_positive"
    data = export_coco([f], tmp_path / "x.json", "/")
    assert data["annotations"] == []
    f["status"] = "invalid"
    f["split"] = "train"
    data = export_coco([f], tmp_path / "y.json", "/")
    assert data["images"][0]["sampling_weight"] == 0 and not data["annotations"]


def test_sam_replay_keeps_last_state_and_audit():
    a = {"track_id": 1, "bbox": [0, 0, 5, 5]}
    b = {"track_id": 1, "bbox": [1, 0, 6, 5]}
    c = {"track_id": 2, "bbox": [1, 0, 6, 5]}
    rows, audit = collapse_sam_emissions([a, b, c])
    assert rows == [b, c] and audit[0]["replaced"] == a
    f = normalize(
        {"frame": "frame_00006000.jpg", "detections": []}, domain="RGB_out", video="A"
    )
    assert f["frame"] == 6000 and f["source_frame_name"] == "frame_00006000.jpg"


def test_calibrated_experts_route_and_mutation(tmp_path):
    records = [
        {
            "domain": "IR_in",
            "layer": "L1",
            "name": name,
            "score": s,
            "correct": int(s > 0.5),
            "human_verified": True,
            "split": "calibration",
        }
        for name in ("local", "trex")
        for s in (0.1, 0.2, 0.8, 0.9)
    ]
    model = fit_experts(records)
    f = example(source="prediction")
    before = copy.deepcopy(f)
    candidate = {
        "bbox_xyxy": [60, 60, 90, 90],
        "experts": [{"name": "local", "score": 0.9}, {"name": "trex", "score": 0.9}],
    }
    out = apply_candidate_evidence(f, [candidate], model)
    assert out["detections"][-1]["label_status"] == "soft_positive" and f == before
    records[0]["split"] = "test"
    with pytest.raises(ValueError):
        fit_experts(records)


def test_knowledge_audit_idempotent_and_disabled_context():
    graph = default_graph()
    decision = {
        "review_id": "1",
        "knowledge_id": "visual_kinematics",
        "reviewer": "human",
        "action": "confirm",
        "video": "B",
        "domain": "IR_in",
    }
    a = update_graph(graph, [decision])
    b = update_graph(a, [decision])
    assert a == b and a["nodes"][0]["evidence_confidence"] == pytest.approx(2 / 3)
    assert (
        attach_context({}, {"enabled": False, "attributes": {"fake": 1}})[
            "external_context"
        ]["attributes"]
        == {}
    )


def test_round_feedback_is_executable_and_non_overwriting(tmp_path):
    sources = []
    for domain in ("IR_in", "RGB_out"):
        p = tmp_path / (domain + ".jsonl")
        write_frames(p, [example(i, domain) for i in range(8)])
        sources.append(
            {
                "path": str(p),
                "domain": domain,
                "video": domain,
                "group": "01",
                "source": "human",
                "split": "train",
            }
        )
    cfg = {
        "inputs": sources,
        "export_training": True,
        "sampling_targets": {"domain": {"IR_in": 0.5, "RGB_out": 0.5}},
        "review_budget": 3,
    }
    out = tmp_path / "round"
    result = run_round(cfg, out)
    assert result["frames"] == 16 and result["sampling_gate"]["passed"]
    assert result["original_observation_boxes_unchanged"] == 16
    assert (out / "training_snapshot/train.coco.json").exists()
    assert len(read_json(out / "review_queue.json")) == 3
    with pytest.raises(FileExistsError):
        run_round(cfg, out)


def test_metric_definitions_reject_silent_pair_pck():
    f = example()
    p = copy.deepcopy(f)
    p["detections"][0]["keypoints"]["head"] = [30, 12, 0.9]
    result = pose_metrics([(p["detections"][0], f["detections"][0])])
    assert result["PCK_endpoint_0.1"] == 0.5
    assert detection_metrics([f], [f])["mAP50_95"] == pytest.approx(1.0)
    confirmed = [
        {"event_id": "e", "event_type": "crossing", "status": "human_confirmed"}
    ]
    assert (
        behavior_metrics(
            confirmed,
            [
                {
                    "event_id": "e",
                    "event_type": "crossing",
                    "upstream_entities": ["x"],
                    "knowledge_source": "gate",
                }
            ],
        )["MacroF1"]
        == 1.0
    )


def test_champion_comparability_and_transport():
    champion = {
        "dataset_sha256": "1",
        "split": "test",
        "protocol_id": "A",
        "metrics": {"score": 0.5},
    }
    candidate = {**champion, "evaluation_status": "verified", "metrics": {"score": 0.6}}
    assert choose_champion(champion, candidate, {"score": "max"})[1]["promoted"]
    candidate["dataset_sha256"] = "2"
    assert not choose_champion(champion, candidate, {"score": "max"})[1]["promoted"]
    champion.pop("protocol_id")
    candidate.pop("protocol_id")
    candidate["dataset_sha256"] = "1"
    assert (
        choose_champion(champion, candidate, {"score": "max"})[1]["reason"]
        == "missing_protocol_id"
    )
    coupling, _ = sinkhorn_weights([[0.0], [1.0]], [[0.0], [1.0]])
    assert np.allclose(coupling.sum(0), 0.5) and np.allclose(coupling.sum(1), 0.5)
