import json

import pytest

from label_data_flywheel.delivery_package import export_delivery, mot_row
from label_data_flywheel.yolo_package import export_yolo, validate_yolo


def test_mot_origin_and_id_retained():
    row = mot_row(
        0, {"track_id": 743, "confidence": 0.8}, "0 0.1 0.3 0.2 0.4", 200, 100
    )
    assert list(map(float, row.split(","))) == [1, 743, 1, 11, 40, 40, 0.8, -1, -1, -1]


def test_trackless_does_not_invent_id():
    assert mot_row(0, {"track_id": None}, "0 0.1 0.3 0.2 0.4", 200, 100) is None
    with pytest.raises(ValueError, match="重编号"):
        mot_row(0, {"track_id": 0}, "0 0.1 0.3 0.2 0.4", 200, 100)


def test_delivery_layout_and_empty_splits(tmp_path):
    source = tmp_path / "frame_00000000.json"
    source.write_text(
        json.dumps(
            {
                "frame": 0,
                "width": 200,
                "height": 100,
                "detections": [{"track_id": 743, "bbox_xyxy": [0, 10, 40, 50]}],
            }
        )
    )
    converted = tmp_path / "converted"
    export_yolo(
        {
            "version": "test",
            "inputs": [{"path": str(source), "video": "B-5-1", "frame_index_base": 0}],
        },
        converted,
    )
    validate_yolo(converted)
    root = tmp_path / "05_数据标注成果"
    result = export_delivery({"inputs": [str(converted)], "workers": 1}, root)
    assert result["total_frames"] == 1 and result["total_tracking_rows"] == 1
    assert (root / "annotations/B-5-1/frame_00000000.txt").exists()
    pose = (root / "annotations_pose/B-5-1/frame_00000000.txt").read_text().split()
    assert len(pose) == 11 and list(map(float, pose[5:])) == [0] * 6
    assert ",743," in (root / "annotations_tracking/B-5-1/tracks.txt").read_text()
    assert not (root / "splits/train.txt").read_text()
    assert (root / "splits/unassigned.txt").read_text() == "B-5-1/frame_00000000.jpg\n"
