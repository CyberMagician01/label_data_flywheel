import gzip
import json
import zipfile

import pytest

from label_data_flywheel.yolo_package import export_yolo, labelme_records, validate_yolo


def test_full_geometry_id_hidden_and_originals(tmp_path):
    f = {
        "frame": 0,
        "width": 200,
        "height": 100,
        "detections": [
            {
                "class_id": 0,
                "track_id": 743,
                "bbox_xyxy": [-2, 10, 40, 50],
                "keypoints": {"head": [20, 20, 0.01], "abdomen_tip": [21, 30, 0.2]},
            },
            {"class_id": 1, "track_id": 900, "bbox_xyxy": [100, 40, 160, 90]},
        ],
        "temporarily_hidden_detections": [
            {"track_id": 253, "bbox_xyxy": [0, 10, 40, 50], "origin": "interpolated"}
        ],
    }
    source = tmp_path / "frame_00000000.json"
    original = json.dumps(f).encode()
    source.write_bytes(original)
    cfg = {
        "version": "test",
        "names": {"0": "bee", "1": "bee_shadow"},
        "inputs": [{"path": str(source), "video": "B-5-1", "frame_index_base": 0}],
    }
    out = tmp_path / "out"
    result = export_yolo(cfg, out)
    assert source.read_bytes() == original
    assert result["counts"]["candidates"] == 2 and result["counts"]["hidden"] == 1
    lines = (
        (out / "detect/labels/candidates/B-5-1/auto/frame_00000000.txt")
        .read_text()
        .splitlines()
    )
    assert list(map(float, lines[0].split())) == [0, 0.1, 0.3, 0.2, 0.4]
    with gzip.open(out / "frame_ids.jsonl.gz", "rt") as h:
        meta = json.loads(next(h))
    assert [r["track_id"] for r in meta["rows"]["candidates"]] == [743, 900]
    assert meta["rows"]["hidden"][0]["track_id"] == 253
    pose = (out / meta["label_files"]["candidates_pose"]).read_text().splitlines()
    assert list(map(float, pose[0].split()[5:])) == [0.1, 0.2, 2, 0.105, 0.3, 2]
    assert list(map(float, pose[1].split()[5:])) == [0] * 6
    assert validate_yolo(out)["passed"]
    with pytest.raises(FileExistsError):
        export_yolo(cfg, out)


def test_labelme_repeated_ids_do_not_drop_boxes():
    shapes = [
        {
            "shape_type": "rectangle",
            "label": "bee",
            "group_id": 5,
            "points": [[1, 2], [5, 8]],
        },
        {
            "shape_type": "rectangle",
            "label": "bee_shadow",
            "group_id": 5,
            "points": [[7, 8], [9, 10]],
        },
        {"shape_type": "point", "label": "tail", "group_id": 5, "points": [[3, 4]]},
    ]
    ds = labelme_records({"shapes": shapes}, {"0": "bee", "1": "bee_shadow"})
    assert len(ds) == 2 and ds[1]["class_id"] == 1
    assert ds[0]["keypoints"]["abdomen_tip"] == [3, 4, 1.0]


def test_zip_preserves_annotator_and_split(tmp_path):
    src = tmp_path / "labels.zip"
    data = {
        "imageWidth": 200,
        "imageHeight": 100,
        "shapes": [
            {
                "shape_type": "rectangle",
                "label": "bee",
                "group_id": 31,
                "points": [[1, 2], [5, 8]],
            }
        ],
    }
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("标注员_01/B-5-3_区段_01/B-5-3_frame_000605.json", json.dumps(data))
    cfg = {
        "version": "test",
        "inputs": [
            {
                "path": str(src),
                "source": "human",
                "split_map": {"01/B-5-3_frame_000605": "calibration"},
            }
        ],
    }
    out = tmp_path / "out"
    export_yolo(cfg, out)
    assert (out / "detect/splits/calibration.txt").exists()
    assert not (out / "detect/dataset.yaml").exists()
    assert validate_yolo(out)["counts"]["confirmed"] == 1


def test_cross_split_rejected(tmp_path):
    src = tmp_path / "frame_00000000.json"
    src.write_text(
        json.dumps({"frame": 0, "width": 200, "height": 100, "detections": []})
    )
    cfg = {
        "version": "test",
        "inputs": [
            {"path": str(src), "video": "B-5-1", "group": group, "split": split}
            for group, split in [("01", "train"), ("03", "test")]
        ],
    }
    with pytest.raises(ValueError, match="跨划分"):
        export_yolo(cfg, tmp_path / "out")
