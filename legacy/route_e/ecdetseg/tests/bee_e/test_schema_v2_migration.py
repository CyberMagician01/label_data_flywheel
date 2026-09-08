import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "bee_e" / "migrate_schema_v2.py"
SPEC = importlib.util.spec_from_file_location("migrate_schema_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_migration_preserves_persistent_group_ids(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "a.jpg",
                        "domain": "RGB",
                        "source": "annotator_01",
                        "scene": "A",
                        "video": "1",
                        "frame": 10,
                    },
                    {
                        "id": 2,
                        "file_name": "b.jpg",
                        "domain": "RGB",
                        "source": "annotator_01",
                        "scene": "A",
                        "video": "1",
                        "frame": 15,
                    },
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "keypoints": [2, 2, 2, 8, 8, 2],
                        "num_keypoints": 2,
                        "pose_mask": 1,
                        "track_id": "spatial_3",
                        "source": "annotator_01",
                        "pairing_method": "spatial_containment",
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [1, 0, 10, 10],
                        "keypoints": [3, 2, 2, 9, 8, 2],
                        "num_keypoints": 2,
                        "pose_mask": 1,
                        "track_id": "spatial_3",
                        "source": "annotator_01",
                        "pairing_method": "spatial_containment",
                    },
                ],
                "categories": [{"id": 1, "name": "bee"}],
            }
        ),
        encoding="utf-8",
    )
    dataset, summary = MODULE.migrate(source)
    first_image, second_image = dataset["images"]
    first, second = dataset["annotations"]

    assert dataset["schema_version"] == 2
    assert first_image["sequence_id"] == "annotator_01:scene_A:video_1:unknown_segment:rgb"
    assert first_image["track_supervised"] is True
    assert second_image["track_supervised"] is True
    assert first["pose_state"] == 2
    assert first["source_group_id"] == "spatial_3"
    assert first["track_id"] == second["track_id"] == 0
    assert first["track_mask"] is second["track_mask"] is True
    assert first["track_geometry_mask"] is False
    assert second["track_geometry_mask"] is True
    assert first["supervision_mask"] == [True, True, True, True]
    assert first["track_quality"] == 1.0
    assert summary["track_supervised_annotations"] == 2
    assert summary["track_geometry_annotations"] == 1


def test_migration_rejects_pose_mask_without_two_endpoints(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "a.jpg",
                        "domain": "IR",
                        "source": "annotator_03",
                        "scene": "B",
                        "video": "1",
                        "frame": 10,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "keypoints": [2, 2, 2],
                        "pose_mask": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        MODULE.migrate(source)
    except ValueError as error:
        assert "two COCO endpoints" in str(error)
    else:
        raise AssertionError("invalid endpoint data was accepted")
