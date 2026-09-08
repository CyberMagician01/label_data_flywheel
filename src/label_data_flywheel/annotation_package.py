"""数据飞轮交付：流式 COCO、独立候选、质量审计和原始划分；不打包评测程序。"""

from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import tempfile

from .io import iter_frames, read_json, write_json, sha256
from .postprocess import is_fill

KEYPOINTS = ["head", "abdomen_tip"]
SPLITS = {"train", "val", "calibration", "test", "unassigned"}
CONFIRMED = {"human", "human_confirmed"}


def _line(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _part(value):
    value = str(value)
    if not re.fullmatch(r"[\w-]+", value):
        raise ValueError("不安全的目录字段：" + value)
    return value


class CocoWriter:
    """两个临时流避免将全量几千万个实例同时装进内存。"""

    def __init__(self, path, stack):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.images = stack.enter_context(
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8", dir=path.parent)
        )
        self.annotations = stack.enter_context(
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8", dir=path.parent)
        )
        self.n_images = self.n_annotations = 0

    def add(self, image, annotations):
        self.n_images += 1
        _line(self.images, dict(image, id=self.n_images))
        for ann in annotations:
            self.n_annotations += 1
            _line(
                self.annotations,
                dict(ann, id=self.n_annotations, image_id=self.n_images),
            )

    def finish(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as out:
            out.write(
                '{"info":{"description":"Bee annotation data; provenance in audit/"},'
            )
            for name, stream in (
                ("images", self.images),
                ("annotations", self.annotations),
            ):
                out.write(json.dumps(name) + ":[")
                stream.seek(0)
                for i, line in enumerate(stream):
                    out.write(("," if i else "") + line.strip())
                out.write("],")
            out.write(
                '"categories":'
                + json.dumps(
                    [
                        {
                            "id": 1,
                            "name": "bee",
                            "supercategory": "insect",
                            "keypoints": KEYPOINTS,
                            "skeleton": [[1, 2]],
                        }
                    ]
                )
                + "}"
            )
        return {
            "images": self.n_images,
            "annotations": self.n_annotations,
            "sha256": sha256(self.path),
        }


def _geometry(d, width, height):
    source = [float(x) for x in d["bbox_xyxy"]]
    if len(source) != 4 or not all(math.isfinite(x) for x in source):
        raise ValueError("框坐标必须为四个有限数值")
    x1, y1, x2, y2 = source
    clipped = [
        max(0.0, min(width, x1)),
        max(0.0, min(height, y1)),
        max(0.0, min(width, x2)),
        max(0.0, min(height, y2)),
    ]
    changes = []
    if clipped != source:
        changes.append({"action": "clip_to_image", "before": source, "after": clipped})
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        return None, changes + [{"action": "exclude_degenerate_after_clip"}]
    points = d.get("keypoints", {})
    visibility = d.get("keypoint_visibility", {})
    output_points = []
    for name in KEYPOINTS:
        point = points.get(name)
        if point is None:
            output_points += [0, 0, 0]
            continue
        px, py = float(point[0]), float(point[1])
        # 原第三列通常为置信度，不将它当作 COCO 可见性编码。
        v = visibility.get(name, 2 if len(point) < 3 or point[2] > 0 else 0)
        if v not in (0, 1, 2):
            raise ValueError("COCO关键点可见性必须为0、1或2")
        if (
            not math.isfinite(px)
            or not math.isfinite(py)
            or not (0 <= px < width and 0 <= py < height)
        ):
            changes.append(
                {"action": "missing_outside_keypoint", "name": name, "before": point}
            )
            v = 0
        output_points += [px, py, v] if v else [0, 0, 0]
    return {
        "category_id": 1,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "area": (x2 - x1) * (y2 - y1),
        "iscrowd": 0,
        "keypoints": output_points,
        "num_keypoints": sum(v > 0 for v in output_points[2::3]),
    }, changes


def _fingerprint(path):
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "sha256": sha256(path)}
    members = [
        {"name": p.name, "sha256": sha256(p)} for p in sorted(path.glob("*.json*"))
    ]
    value = json.dumps(members, sort_keys=True).encode()
    return {
        "path": str(path),
        "members": members,
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def export_annotations(config, destination):
    root = Path(destination)
    if root.exists():
        raise FileExistsError("标注交付必须使用新的版本目录")
    if not config.get("version"):
        raise ValueError("必须显式指定数据版本")
    root.mkdir(parents=True)
    write_json(root / "build_status.json", {"status": "building"})
    write_json(root / "resolved_config.json", config)
    try:
        result = _export(config, root)
    except Exception as exc:
        write_json(
            root / "build_status.json", {"status": "failed", "message": str(exc)}
        )
        raise
    write_json(root / "build_status.json", {"status": "complete"})
    return result


def _export(config, root):
    counts = Counter()
    partitions = defaultdict(set)
    physical_splits, samples, physical_frames, source_records = {}, set(), {}, []
    writers = {}
    groups = defaultdict(set)
    for folder in ("annotations", "splits", "audit"):
        (root / folder).mkdir()
    with ExitStack() as stack:
        audit = stack.enter_context(
            (root / "audit/records.jsonl").open("w", encoding="utf-8")
        )
        for source in config["inputs"]:
            source_records.append(
                {
                    **_fingerprint(source["path"]),
                    "version": source.get("version"),
                    "kind": source.get("source", "prediction"),
                }
            )
            kwargs = {
                k: source[k]
                for k in ("domain", "video", "group", "split", "source")
                if k in source
            }
            frame_base = source.get("frame_index_base")
            point_encoding = source.get("keypoint_encoding", "confidence")
            if point_encoding not in ("confidence", "coco_visibility"):
                raise ValueError(
                    "keypoint_encoding 必须为 confidence 或 coco_visibility"
                )
            if frame_base not in (None, 0, 1):
                raise ValueError("frame_index_base 必须为0、1或未确认的null")
            start, end = source.get("source_frame_range", [None, None])
            for f in iter_frames(source["path"], **kwargs):
                if (start is not None and f["frame"] < start) or (
                    end is not None and f["frame"] > end
                ):
                    continue
                if f["sample_id"] in samples:
                    raise ValueError("重复的源样本：" + f["sample_id"])
                samples.add(f["sample_id"])
                video, group = _part(f["video"]), _part(f["group"])
                groups[(f["domain"], video)].add(group)
                split = f["split"]
                if split not in SPLITS:
                    raise ValueError("未声明的划分：" + split)
                physical = (f["domain"], video, f["frame"])
                if split != "unassigned":
                    old = physical_splits.setdefault(physical, split)
                    if old != split:
                        raise ValueError(
                            f"同一源帧跨划分泄漏：{physical}: {old}/{split}"
                        )
                width, height = map(int, f["image_size"])
                if width <= 0 or height <= 0 or f["frame"] < 0:
                    raise ValueError("图像尺寸和源帧号非法")
                image_ref = f"{video}/frame_{f['frame']:08d}.jpg"
                frame_metadata = {
                    "video": video,
                    "source_frame_id": f["frame"],
                    "image_ref": image_ref,
                    "width": width,
                    "height": height,
                    "fps": f["fps"],
                    "frame_index_base": frame_base,
                    "decode_index": None
                    if frame_base is None
                    else f["frame"] - frame_base,
                }
                if (
                    physical in physical_frames
                    and physical_frames[physical] != frame_metadata
                ):
                    raise ValueError("同一源帧尺寸或抽帧定义冲突")
                physical_frames[physical] = frame_metadata
                partitions[split].add(image_ref)
                known_negative = (
                    bool(f.get("annotation_complete"))
                    and f.get("source_kind") == "human"
                )
                image = {
                    "file_name": image_ref,
                    "width": width,
                    "height": height,
                    "source_sample_id": f["sample_id"],
                    "source_frame_id": f["frame"],
                    "video_id": video,
                    "annotator_group": group,
                    "split": split,
                    "annotation_complete": bool(f.get("annotation_complete", False)),
                }
                bins = {"confirmed": [], "candidates": []}
                for collection in (
                    "detections",
                    "temporarily_hidden_detections",
                    "ignore_regions",
                ):
                    for d in f.get(collection, []):
                        status = d.get("label_status", "unconfirmed")
                        excluded = (
                            collection != "detections"
                            or status in ("invalid", "ignore")
                            or f.get("status") in ("invalid", "suspect")
                        )
                        geometry_source = d
                        if point_encoding == "coco_visibility":
                            geometry_source = {
                                **d,
                                "keypoint_visibility": {
                                    **{
                                        name: point[2]
                                        for name, point in d.get(
                                            "keypoints", {}
                                        ).items()
                                        if len(point) >= 3
                                    },
                                    **d.get("keypoint_visibility", {}),
                                },
                            }
                        geometry, changes = _geometry(geometry_source, width, height)
                        confirmed = status in CONFIRMED and (
                            not is_fill(d) or status == "human_confirmed"
                        )
                        tier = (
                            "audit_only"
                            if excluded or geometry is None
                            else "confirmed"
                            if confirmed
                            else "candidates"
                        )
                        evidence = {
                            "sample_id": f["sample_id"],
                            "entity_id": d.get("entity_id"),
                            "collection": collection,
                            "label_status": status,
                            "tier": tier,
                            "track_scope": [
                                f["domain"],
                                video,
                                group,
                                f.get("segment", group),
                            ],
                            "source_track_id": d.get("track_id"),
                            "origin": d.get("origin"),
                            "original_bbox_xyxy": d["bbox_xyxy"],
                            "original_keypoints": d.get("keypoints", {}),
                            "geometry_changes": changes,
                            "source_keypoint_encoding": point_encoding,
                            "source_record": d,
                        }
                        _line(audit, evidence)
                        counts[tier] += 1
                        counts["geometry_repairs"] += bool(changes)
                        if tier != "audit_only":
                            bins[tier].append(
                                {
                                    **geometry,
                                    "source_entity_id": d.get("entity_id"),
                                    "source_track_id": d.get("track_id"),
                                    "label_status": status,
                                    "is_pseudo": tier == "candidates",
                                }
                            )
                for tier, rows in bins.items():
                    include = bool(rows) or (tier == "confirmed" and known_negative)
                    if not include:
                        continue
                    key = (video, group, tier)
                    if key not in writers:
                        writers[key] = CocoWriter(
                            root / "annotations" / video / group / f"{tier}.coco.json",
                            stack,
                        )
                    writers[key].add(image, rows)
                counts["frames"] += 1
        outputs = {
            writer.path.relative_to(root).as_posix(): writer.finish()
            for writer in writers.values()
        }
    with (root / "frame_manifest.jsonl").open("w", encoding="utf-8") as out:
        for key in sorted(physical_frames):
            _line(out, physical_frames[key])
    for split in sorted(SPLITS):
        values = sorted(partitions[split])
        (root / "splits" / f"{split}.txt").write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )
    assets = Path(__file__).parent / "assets"
    for name in ("extract_frames.py", "标注说明.md"):
        shutil.copy2(assets / name, root / name)
    pending = []
    if not config.get("external_data_declaration_complete", False):
        pending.append(
            "外部数据/预训练来源声明尚未核实完整，未将空清单解释为未使用外部数据"
        )
    if partitions["unassigned"]:
        pending.append("unassigned 帧未自动加入训练或验证")
    if not partitions["train"] or not partitions["val"]:
        pending.append(
            "train/val 尚未同时提供；保留原有 calibration/test 语义，不自动重划分"
        )
    if any(f["frame_index_base"] is None for f in physical_frames.values()):
        pending.append("部分源视频帧编号基准尚未确认；抽帧时须明确提供0或1")
    if any(len(v) > 1 for v in groups.values()):
        pending.append(
            "同视频含多个标注群组，分别保存；合并训练前须人工仲裁，不自动叠加"
        )
    review_sources = []
    for index, source_path in enumerate(config.get("review_audits", [])):
        path = Path(source_path)
        target = root / "audit" / "reviews" / f"{index:03d}{''.join(path.suffixes)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        review_sources.append(
            {
                "source": str(path),
                "packaged": target.relative_to(root).as_posix(),
                "sha256": sha256(target),
            }
        )
    manifest = {
        "schema_version": 1,
        "component": "data_flywheel_only",
        "version": config["version"],
        "coordinate_convention": "zero-origin pixels; COCO bbox xywh; source IDs unchanged",
        "keypoints": KEYPOINTS,
        "skeleton": [[1, 2]],
        "source_inputs": source_records,
        "counts": dict(counts),
        "physical_frames": len(physical_frames),
        "outputs": outputs,
        "splits": {k: len(partitions[k]) for k in sorted(SPLITS)},
        "pending": pending,
        "formal_competition_submission": False,
        "image_frames_included": False,
        "prediction_count_cap": None,
        "review_policy": "只有已记录的人工确认才视为确认；未虚构多人复核完成状态",
        "review_audit_sources": review_sources,
        "external_data": config.get("external_data", []),
        "external_data_declaration_complete": config.get(
            "external_data_declaration_complete", False
        ),
    }
    manifest["files"] = {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in ("manifest.json", "build_status.json")
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _coco_items(path, key):
    import ijson

    with path.open("rb") as source:
        yield from ijson.items(source, key + ".item", use_float=True)


def validate_annotations(root):
    """独立读取导出文件与哈希，检查 COCO 基础契约及物理帧划分。"""
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    errors = []
    if read_json(root / "build_status.json")["status"] != "complete":
        errors.append("build_not_complete")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            errors.append("hash_mismatch:" + name)
    media = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".avi")
    ]
    if media:
        errors.append("unexpected_image_or_video")
    for name, expected in manifest["outputs"].items():
        images = {}
        for im in _coco_items(root / name, "images"):
            if im["id"] in images:
                errors.append("duplicate_image_id:" + name)
            images[im["id"]] = im
        if len(images) != expected["images"]:
            errors.append("count_or_image_id:" + name)
        annotation_count = 0
        for annotation_count, a in enumerate(
            _coco_items(root / name, "annotations"), 1
        ):
            # 本导出器分配连续正整数索引，逐项检查即可避免为数百万ID驻留集合。
            if (
                a["id"] != annotation_count
                or a["image_id"] not in images
                or a["category_id"] != 1
            ):
                errors.append("annotation_reference:" + name)
                continue
            im = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            if (
                not all(math.isfinite(v) for v in (x, y, w, h))
                or min(x, y) < 0
                or min(w, h) <= 0
                or x + w > im["width"] + 1e-6
                or y + h > im["height"] + 1e-6
            ):
                errors.append("bbox:" + name)
            k = a["keypoints"]
            if len(k) != 6 or a["num_keypoints"] != sum(v > 0 for v in k[2::3]):
                errors.append("keypoints:" + name)
                continue
            for i in (0, 3):
                if k[i + 2] not in (0, 1, 2) or (
                    k[i + 2] == 0 and k[i : i + 2] != [0, 0]
                ):
                    errors.append("visibility:" + name)
            if "confirmed.coco.json" in name and a["is_pseudo"]:
                errors.append("pseudo_in_confirmed:" + name)
        if annotation_count != expected["annotations"]:
            errors.append("annotation_count:" + name)
    seen = {}
    for split in sorted(SPLITS - {"unassigned"}):
        for image in (
            (root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        ):
            if image in seen and seen[image] != split:
                errors.append("split_leakage:" + image)
            seen[image] = split
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "pending": manifest["pending"],
        "frames": manifest["physical_frames"],
        "counts": manifest["counts"],
        "scope": "annotation_data_contract_only",
        "formal_competition_submission": False,
    }
