import pytest
from label_data_flywheel.behavior_learning import train, predict


def test_behavior_learning_requires_confirmation_and_masks_missing_heads(tmp_path):
    records = [
        {
            "status": "human_confirmed",
            "split": "train",
            "speed_bl_per_source_frame": 0.001 * i,
            "labels": {"motion": "stationary"},
        }
        for i in range(4)
    ]
    checkpoint = tmp_path / "behavior.pt"
    result = train(records, checkpoint, epochs=2)
    assert len(result["loss_history"]) == 2 and result["benchmark_score"] is None
    output = predict(records, checkpoint)
    assert len(output) == 4 and all(r["status"] == "candidate" for r in output)
    records[0]["status"] = "candidate"
    with pytest.raises(ValueError):
        train(records, tmp_path / "rejected.pt", epochs=1)
