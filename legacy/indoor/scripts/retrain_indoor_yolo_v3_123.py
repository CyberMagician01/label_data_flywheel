"""用 v3 的01/03和原02标注，在原冻结帧划分上重训室内YOLO。"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path("/root/autodl-tmp")
OLD = ROOT / "indoor_scene_b_yolov8x_p2_framework_20260905"
WORK = ROOT / "indoor_scene_b_yolov8x_p2_v3_123_20260905"
DATA = ROOT / "datasets/scene_B_123_v3_sliced_20260905"
ZIP = ROOT / "datasets/human_annotations_01_03_20260905/human_annotations_01_03_20260905_v3.zip"
OLD_MANIFEST = ROOT / "datasets/scene_B_annotators_01_02_03_sliced_20260905/split_manifest.json"
RUN = "scene_b_123_v3_yolov8x_p2_seed2026"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare():
    assert not DATA.exists(), f"拒绝覆盖已有数据：{DATA}"
    assert sha(OLD_MANIFEST) == "13f370d4da7294bfd703be3a402251b951df62cd14bf51a5e67617341db1c53c"
    spec = importlib.util.spec_from_file_location("original_slicer", OLD / "indoor_training_package/tools/slice_scene_b_unified.py")
    slicer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slicer)
    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    with zipfile.ZipFile(ZIP) as archive:
        # 只读取直属区段，明确排除01/1中的旧副本。
        updates = {}
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) == 3 and parts[0] in {"标注员_01", "标注员_03"} and parts[2].startswith("B-") and parts[2].endswith(".json"):
                key = (parts[0], parts[2])
                assert key not in updates
                updates[key] = (name, archive.read(name))
    assert len(updates) == 120
    samples, split_counts, box_counts, tile_counts = [], Counter(), Counter(), Counter()
    source_counts, hash_splits = Counter(), defaultdict(set)
    for split in ("train", "calibration", "test"):
        (DATA / "images" / split).mkdir(parents=True)
        (DATA / "labels" / split).mkdir(parents=True)
    for row in old["samples"]:
        assert row["annotator"] in {"标注员_01", "标注员_02", "标注员_03"}
        key = (row["annotator"], Path(row["annotation"]).name)
        if row["annotator"] == "标注员_02":
            origin, raw = row["annotation"], Path(row["annotation"]).read_bytes()
            source_kind = "unchanged_02"
        else:
            entry, raw = updates[key]
            origin, source_kind = str(ZIP) + "::" + entry, "v3_direct"
        annotation = json.loads(raw.decode("utf-8-sig"))
        image_path = Path(row["image"])
        image_hash = sha(image_path)
        assert image_hash == row["image_sha256"]
        hash_splits[image_hash].add(row["split"])
        frozen = DATA / "source_annotations" / row["annotator"] / Path(row["annotation"]).name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(raw)
        sample = {"annotation": annotation, "image_path": image_path,
                  "annotator": row["annotator"], "section_id": row["section_id"]}
        tiles, nboxes = slicer.slice_sample(sample, DATA / "images" / row["split"], DATA / "labels" / row["split"], 640, 0.20)
        assert tiles == 8
        split_counts[row["split"]] += 1
        box_counts[row["split"]] += nboxes
        tile_counts[row["split"]] += tiles
        source_counts[row["annotator"]] += nboxes
        samples.append({**row, "annotation": str(frozen), "annotation_sha256": sha(frozen),
                        "source_annotation": origin, "source_kind": source_kind,
                        "old_source_box_count": row["source_box_count"], "source_box_count": nboxes,
                        "tile_count": tiles})
    assert all(len(splits) == 1 for splits in hash_splits.values())
    assert dict(split_counts) == {"train": 144, "test": 18, "calibration": 18}
    assert dict(source_counts) == {"标注员_01": 22907, "标注员_02": 21597, "标注员_03": 21349}
    manifest = {"version": "v3", "zip_path": str(ZIP), "zip_sha256": sha(ZIP),
                "previous_manifest_sha256": sha(OLD_MANIFEST), "full_images": dict(split_counts),
                "source_boxes": dict(box_counts), "tiles": dict(tile_counts),
                "boxes_by_annotator": dict(source_counts), "total_boxes": sum(source_counts.values()),
                "exact_duplicate_cross_split": 0, "samples": samples}
    save(DATA / "split_manifest.json", manifest)
    (DATA / "dataset.yaml").write_text(f"path: {DATA}\ntrain: images/train\nval: images/calibration\ntest: images/test\nnames:\n  0: bee\n", encoding="utf-8")
    WORK.mkdir(exist_ok=True)
    shutil.copy2(OLD / "indoor_training_package/configs/yolov8x-p2.yaml", WORK / "yolov8x-p2.yaml")
    save(DATA / "PACKAGE_READY.json", {"manifest_sha256": sha(DATA / "split_manifest.json"),
                                        "yaml_sha256": sha(DATA / "dataset.yaml"), "total_boxes": 65853})
    print(json.dumps({k:v for k,v in manifest.items() if k != "samples"}, ensure_ascii=False), flush=True)


def train():
    from ultralytics import YOLO
    ready = json.loads((DATA / "PACKAGE_READY.json").read_text())
    assert sha(DATA / "split_manifest.json") == ready["manifest_sha256"]
    assert sha(DATA / "dataset.yaml") == ready["yaml_sha256"]
    assert not (WORK / "runs" / RUN).exists(), "拒绝覆盖或误恢复旧训练"
    coco = OLD / "indoor_training_package/weights/yolov8x.pt"
    save(WORK / "training_provenance.json", {"data_manifest_sha256": ready["manifest_sha256"],
                                              "coco_weights": str(coco), "coco_sha256": sha(coco),
                                              "architecture_sha256": sha(WORK / "yolov8x-p2.yaml"),
                                              "script_sha256": sha(__file__)})
    model = YOLO(str(WORK / "yolov8x-p2.yaml"))
    model.load(str(coco))  # 加载失败立即报错，不回退为随机初始化。
    model.train(data=str(DATA / "dataset.yaml"), epochs=300, batch=32, imgsz=640,
                device="0,1", workers=12, cache="disk", amp=True, seed=2026,
                deterministic=True, project=str(WORK / "runs"), name=RUN,
                exist_ok=False, pretrained=True, optimizer="auto", patience=50,
                save=True, save_period=10, plots=True)
    best = WORK / "runs" / RUN / "weights/best.pt"
    save(WORK / "TRAIN_COMPLETE.json", {"best": str(best), "sha256": sha(best)})
    # 新旧权重都在本次新版测试标签上评估，避免标签版本变化造成虚假增益。
    weights = {"new_v3": best, "old_v1": OLD / "runs/formal/scene_b_annotators_01_02_03_yolov8x_p2_seed2026/weights/best.pt"}
    metrics = {}
    for label, path in weights.items():
        result = YOLO(str(path)).val(data=str(DATA / "dataset.yaml"), split="test", imgsz=640,
                                    batch=16, device="0", project=str(WORK / "eval"),
                                    name=label, exist_ok=False, plots=True, save_json=False)
        metrics[label] = {"weights_sha256": sha(path), "metrics": {k:float(v) for k,v in result.results_dict.items()}}
        save(WORK / "test_comparison.json", metrics)
    save(WORK / "ALL_COMPLETE.json", {"status": "complete", "comparison": str(WORK / "test_comparison.json")})
    print("TRAIN_AND_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else train()
