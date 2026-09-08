"""逐帧 YOLO detect/pose 导出；ID、置信度、隐藏状态由同行号旁文件保留。"""

import gzip
import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from .annotation_package import CONFIRMED, SPLITS, _geometry, _part
from .io import read_json, write_json
from .postprocess import is_fill


def source_records(spec):
    """压缩包流式读取，不解包图片，不修改源文件。"""
    path = Path(spec["path"])
    pattern = re.compile(spec.get("member_pattern", r".*\.json(?:\.gz)?$"))
    if path.is_dir():
        for p in sorted(path.rglob("*.json*")):
            name = p.relative_to(path).as_posix()
            if pattern.fullmatch(name):
                yield name, p.read_bytes()
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if pattern.fullmatch(name):
                    yield name, archive.read(name)
    elif path.name.endswith((".tar.gz", ".tar", ".tar.zst")):
        if path.suffix == ".zst":
            import zstandard

            with (
                path.open("rb") as raw,
                zstandard.ZstdDecompressor().stream_reader(raw) as stream,
            ):
                yield from _tar_records(stream, pattern)
        else:
            with path.open("rb") as stream:
                yield from _tar_records(stream, pattern)
    else:
        yield path.name, path.read_bytes()


def _tar_records(stream, pattern):
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if member.isfile() and pattern.fullmatch(member.name):
                with archive.extractfile(member) as handle:
                    yield member.name, handle.read()


def labelme_records(data, names):
    """每个原始矩形独立保留；同组头尾点关联，不合并重复 group_id 的框。"""
    classes = {v: int(k) for k, v in names.items()}
    points = {}
    for shape in data["shapes"]:
        if shape["shape_type"] == "point" and shape.get("group_id") is not None:
            key = {"tail": "abdomen_tip", "abdomen": "abdomen_tip"}.get(
                shape["label"], shape["label"]
            )
            points.setdefault(shape["group_id"], {})[key] = [*shape["points"][0], 1.0]
    result = []
    for i, shape in enumerate(data["shapes"]):
        if shape["shape_type"] == "point":
            continue
        if shape["shape_type"] != "rectangle":
            raise ValueError("需要显式处理的 LabelMe 图形：" + shape["shape_type"])
        xy = shape["points"]
        result.append(
            {
                "bbox_xyxy": [
                    min(p[0] for p in xy),
                    min(p[1] for p in xy),
                    max(p[0] for p in xy),
                    max(p[1] for p in xy),
                ],
                "class_id": classes[shape["label"]],
                "track_id": shape.get("group_id"),
                "keypoints": points.get(shape.get("group_id"), {}),
                "source_shape_index": i,
                "label_status": "human",
                "origin": "observed",
                "difficult": shape.get("difficult", False),
            }
        )
    return result


def yolo_row(detection, width, height, names, encoding="confidence"):
    d = detection
    if encoding == "coco_visibility":
        d = dict(
            d,
            keypoint_visibility={
                k: p[2] for k, p in d.get("keypoints", {}).items() if len(p) >= 3
            },
        )
    geometry, changes = _geometry(d, width, height)
    if geometry is None:
        return None, None, changes
    cls = d.get("class_id", 0)
    if int(cls) != cls or str(int(cls)) not in names:
        raise ValueError("未声明类别：" + str(cls))
    x, y, w, h = geometry["bbox"]
    fields = [
        str(int(cls)),
        *[
            f"{v:.10f}"
            for v in ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
        ],
    ]
    pose = list(fields)
    for i in (0, 3):
        px, py, v = geometry["keypoints"][i : i + 3]
        pose += [f"{px / width:.10f}", f"{py / height:.10f}", str(int(v))]
    return " ".join(fields), " ".join(pose), changes


def export_yolo(config, destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    names = {str(k): v for k, v in config.get("names", {"0": "bee"}).items()}
    if sorted(map(int, names)) != list(range(len(names))):
        raise ValueError("类别编号须从 0 连续编号")
    counts, classes = Counter(), Counter()
    seen, split_by_frame = set(), {}
    write_json(root / "resolved_config.json", config)
    write_json(root / "status.json", {"completed": False})
    try:
        with gzip.open(
            root / "frame_ids.jsonl.gz", "wt", encoding="utf-8", compresslevel=1
        ) as audit:
            for spec in config["inputs"]:
                for member, raw in source_records(spec):
                    payload = gzip.decompress(raw) if member.endswith(".gz") else raw
                    f = json.loads(payload.decode("utf-8-sig"))
                    if not isinstance(f, dict) or (
                        "detections" not in f and "shapes" not in f
                    ):
                        raise ValueError("不是逐帧标注：" + member)
                    vm = re.search(r"[AB]-5-[1-4]", member)
                    video = _part(
                        spec.get("video") or (vm.group() if vm else f.get("video"))
                    )
                    gm = re.search(r"标注员[_ ]?(\d+)", member)
                    group = _part(spec.get("group", gm.group(1) if gm else "auto"))
                    stem = Path(member.removesuffix(".gz")).stem
                    frame = f.get("frame", f.get("source_frame", stem))
                    frame = int(re.search(r"(\d+)(?:\.\w+)?$", str(frame)).group(1))
                    width, height = f.get(
                        "image_size",
                        [
                            f.get("imageWidth", f.get("width")),
                            f.get("imageHeight", f.get("height")),
                        ],
                    )
                    if not width or not height or width <= 0 or height <= 0:
                        raise ValueError("缺少有效原图尺寸：" + member)
                    if "shapes" in f:
                        f["detections"] = labelme_records(f, names)
                    source_kind = spec.get("source", "prediction")
                    split = spec.get("split_map", {}).get(
                        f"{group}/{stem}", spec.get("split", "unassigned")
                    )
                    if split not in SPLITS:
                        raise ValueError("未知划分：" + split)
                    physical = (video, frame)
                    if (
                        split != "unassigned"
                        and split_by_frame.setdefault(physical, split) != split
                    ):
                        raise ValueError("同一源帧跨划分：" + str(physical))
                    key = (video, group, stem)
                    if key in seen:
                        raise ValueError("重复输出文件：" + str(key))
                    seen.add(key)
                    bins = {
                        tier: {"detect": [], "pose": [], "rows": [], "has_pose": False}
                        for tier in ("confirmed", "candidates", "hidden")
                    }
                    excluded = []
                    for collection in (
                        "detections",
                        "temporarily_hidden_detections",
                        "ignore_regions",
                    ):
                        for i, original in enumerate(f.get(collection, [])):
                            d = dict(original)
                            d.setdefault("bbox_xyxy", d.get("bbox"))
                            status = d.get(
                                "label_status",
                                "human" if source_kind == "human" else "unconfirmed",
                            )
                            if (
                                collection == "ignore_regions"
                                or status in ("invalid", "ignore")
                                or f.get("status") in ("invalid", "suspect")
                            ):
                                excluded.append(
                                    {
                                        "collection": collection,
                                        "source_index": i,
                                        "record": original,
                                    }
                                )
                                continue
                            detect, pose, changes = yolo_row(
                                d,
                                width,
                                height,
                                names,
                                spec.get("keypoint_encoding", "confidence"),
                            )
                            if detect is None:
                                excluded.append(
                                    {
                                        "collection": collection,
                                        "source_index": i,
                                        "record": original,
                                        "changes": changes,
                                    }
                                )
                                counts["degenerate_excluded"] += 1
                                continue
                            tier = (
                                "hidden"
                                if collection != "detections"
                                else "confirmed"
                                if status in CONFIRMED
                                and (not is_fill(d) or status == "human_confirmed")
                                else "candidates"
                            )
                            b = bins[tier]
                            b["detect"].append(detect)
                            b["pose"].append(pose)
                            b["has_pose"] |= bool(d.get("keypoints"))
                            b["rows"].append(
                                {
                                    "line": len(b["detect"]),
                                    "source_collection": collection,
                                    "source_index": i,
                                    "source_shape_index": d.get("source_shape_index"),
                                    "track_id": d.get("track_id"),
                                    "source_track_id": d.get("source_track_id"),
                                    "class_id": d.get("class_id", 0),
                                    "origin": d.get("origin", "observed"),
                                    "confidence": d.get(
                                        "confidence",
                                        d.get("det_confidence", d.get("conf")),
                                    ),
                                    "keypoint_scores": {
                                        k: p[2] if len(p) > 2 else None
                                        for k, p in d.get("keypoints", {}).items()
                                    },
                                    "changes": changes,
                                }
                            )
                            counts[tier] += 1
                            counts["geometry_repairs"] += bool(changes)
                            if tier != "hidden":
                                classes[str(d.get("class_id", 0))] += 1
                    files = {}
                    for tier, b in bins.items():
                        # 无目标机器帧保留空标注；不声称已经过人工完整复核。
                        if not b["rows"] and tier != (
                            "confirmed" if source_kind == "human" else "candidates"
                        ):
                            continue
                        rel = Path(tier) / video / group / (stem + ".txt")
                        for task in ("detect", "pose"):
                            if task == "pose" and not b["has_pose"]:
                                continue
                            p = (
                                root / task / "labels" / rel
                                if tier != "hidden"
                                else root / "audit_hidden" / task / rel
                            )
                            p.parent.mkdir(parents=True, exist_ok=True)
                            p.write_text(
                                "\n".join(b[task]) + ("\n" if b[task] else ""),
                                encoding="utf-8",
                            )
                            files[f"{tier}_{task}"] = p.relative_to(root).as_posix()
                            counts[
                                task + "_files"
                                if tier != "hidden"
                                else "hidden_" + task + "_files"
                            ] += 1
                            if tier != "hidden":
                                listing = root / task / "splits" / (split + ".txt")
                                listing.parent.mkdir(parents=True, exist_ok=True)
                                with listing.open("a", encoding="utf-8") as out:
                                    out.write(
                                        "./../images/"
                                        + rel.with_suffix(".jpg").as_posix()
                                        + "\n"
                                    )
                    record = {
                        "video": video,
                        "group": group,
                        "frame": frame,
                        "frame_index_base": spec.get("frame_index_base"),
                        "split": split,
                        "width": width,
                        "height": height,
                        "source_image_name": f.get(
                            "imagePath", f.get("frame", stem + ".jpg")
                        ),
                        "source_member": member,
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_path": spec["path"],
                        "label_files": files,
                        "rows": {
                            tier: b["rows"] for tier, b in bins.items() if b["rows"]
                        },
                        "excluded": excluded,
                    }
                    audit.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts["frames"] += 1
                    if counts["frames"] % 1000 == 0:
                        write_json(
                            root / "status.json", {"completed": False, "counts": counts}
                        )
        for task in ("detect", "pose"):
            d = root / task
            if not d.exists():
                continue
            # 未指定 train/val 的全集只发布标签规格；不虚构随机划分。
            content = "names:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items())
            if task == "pose":
                content += (
                    "kpt_shape: [2, 3]\nflip_idx: [0, 1]\nkpt_names:\n"
                    + "".join(f"  {k}: [head, abdomen_tip]\n" for k in names)
                )
            (d / "label_schema.yaml").write_text(content, encoding="utf-8")
            if (d / "splits/train.txt").exists() and (d / "splits/val.txt").exists():
                (d / "dataset.yaml").write_text(
                    "path: .\ntrain: splits/train.txt\nval: splits/val.txt\n" + content,
                    encoding="utf-8",
                )
        result = {
            "completed": True,
            "version": config["version"],
            "counts": dict(counts),
            "visible_classes": dict(classes),
            "originals_preserved": True,
            "original_source_paths": [s["path"] for s in config["inputs"]],
            "ids": "frame_ids.jsonl.gz",
            "image_copy": False,
            "format": "Ultralytics YOLO detect (5 columns) / pose (11 columns)",
        }
        write_json(root / "manifest.json", result)
        write_json(root / "status.json", result)
        return result
    except Exception as exc:
        write_json(
            root / "status.json",
            {"completed": False, "counts": dict(counts), "error": str(exc)},
        )
        raise


def validate_yolo(path):
    root = Path(path)
    counts = Counter()
    with gzip.open(root / "frame_ids.jsonl.gz", "rt", encoding="utf-8") as audit:
        for f in map(json.loads, audit):
            counts["frames"] += 1
            for name, rel in f["label_files"].items():
                tier, task = name.split("_")
                rows = f["rows"].get(tier, [])
                lines = (root / rel).read_text().splitlines()
                assert len(lines) == len(rows), rel
                for i, line in enumerate(lines):
                    values = list(map(float, line.split()))
                    assert len(values) == (5 if task == "detect" else 11), rel
                    assert (
                        rows[i]["line"] == i + 1 and values[0] == rows[i]["class_id"]
                    ), rel
                    assert (
                        all(0 <= v <= 1 for v in values[1:5])
                        and values[3] > 0
                        and values[4] > 0
                    ), rel
                    if task == "pose":
                        assert all(0 <= values[j] <= 1 for j in (5, 6, 8, 9)), rel
                        assert all(values[j] in (0, 1, 2) for j in (7, 10)), rel
                    if task == "detect":
                        counts[tier] += 1
                counts["files"] += 1
    manifest = read_json(root / "manifest.json")
    for k in ("frames", "confirmed", "candidates", "hidden"):
        assert counts[k] == manifest["counts"].get(k, 0), k
    result = {
        "passed": True,
        "counts": dict(counts),
        "all_label_rows_and_id_mapping_checked": True,
    }
    write_json(root / "validation.json", result)
    return result
