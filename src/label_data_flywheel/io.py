"""统一四级实体引用；保留源字段、人工身份和隐藏插值。"""

import copy
import gzip
import hashlib
import json
import re
from pathlib import Path
from .semantics import entity_rules


def read_json(path):
    path = Path(path)
    with (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8-sig")
    ) as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def normalize(
    frame, *, video=None, domain=None, group="unknown", split="unassigned", source="prediction"
):
    f = copy.deepcopy(frame)
    # 已标准化记录的域与其 sample_id 一致，不被调用参数静默改写。
    domain = f.get("domain", domain)
    if not f.get("video", video):
        raise ValueError("未标准化输入必须提供 video")
    if domain not in ("IR_in", "RGB_out"):
        raise ValueError("domain必须为IR_in或RGB_out")
    f.update(
        schema_version="1.0",
        video=f.get("video", video),
        domain=domain,
        group=f.get("group", group),
        split=f.get("split", split),
        source_kind=f.get("source_kind", source),
    )
    frame_value = f.get("frame", f.get("source_frame", 0))
    if isinstance(frame_value, str) and not frame_value.isdigit():
        match = re.search(r"(\d+)(?:\.[^.]+)?$", frame_value)
        if not match:
            raise ValueError(f"无法解析源帧编号：{frame_value}")
        f.setdefault("source_frame_name", frame_value)
        frame_value = match.group(1)
    f["frame"] = int(frame_value)
    f["fps"] = float(f.get("fps", 30))
    f["image_size"] = f.get("image_size", [f.get("width", 1920), f.get("height", 1080)])
    f["sample_id"] = f.get(
        "sample_id", f"{domain}/{f['group']}/{f['video']}/{f['frame']:08d}"
    )
    f.setdefault("context_id", "default")
    f.setdefault("hive_id", "default")
    for collection in ("detections", "temporarily_hidden_detections", "ignore_regions"):
        for i, original in enumerate(f.setdefault(collection, [])):
            d = entity_rules(original)
            f[collection][i] = d
            if "bbox_xyxy" not in d:
                d["bbox_xyxy"] = d["bbox"]
            d["bbox_xyxy"] = [float(x) for x in d["bbox_xyxy"]]
            x1, y1, x2, y2 = d["bbox_xyxy"]
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"退化框：{f['sample_id']}:{i}")
            d.setdefault("entity_id", f"{f['sample_id']}/{collection}/{i}")
            d.setdefault("origin", "observed")
            d.setdefault("keypoints", {})
            d.setdefault("confidence", d.get("det_confidence", d.get("conf")))
            d.setdefault(
                "supervision_mask",
                {
                    "detection": True,
                    "pose": bool(d["keypoints"]),
                    "tracking": d.get("track_id") is not None,
                    "behavior": bool(d.get("confirmed_events")),
                },
            )
            d.setdefault(
                "label_status", "ignore" if collection == "ignore_regions" else
                "human" if f["source_kind"] == "human" else "unconfirmed"
            )
    ids = [d["track_id"] for d in f["detections"] if d.get("track_id") is not None
           and d.get("label_status") != "invalid"]
    if len(ids) != len(set(ids)):
        raise ValueError(f"单帧ID重复：{f['sample_id']}")
    return f


def labelme_frame(data, index):
    points_by_group = {}
    for i, s in enumerate(data["shapes"]):
        if s["shape_type"] == "point" and s.get("group_id") is not None:
            name = {"tail": "abdomen_tip", "abdomen": "abdomen_tip"}.get(
                s["label"], s["label"]
            )
            points_by_group.setdefault(s["group_id"], {})[name] = [*s["points"][0], 1.0]
    entities = []
    for i, s in enumerate(data["shapes"]):
        if s["shape_type"] != "rectangle":
            continue
        if s["label"] not in ("bee", "bee_shadow"):
            raise ValueError("未声明的标注类别：" + s["label"])
        points, group = s["points"], s.get("group_id")
        entities.append(entity_rules({
            "class_id": 0 if s["label"] == "bee" else 1,
            "class_name": s["label"], "track_id": group,
            "keypoints": points_by_group.get(group, {}),
            "source_shape_index": i,
            "bbox_xyxy": [min(p[0] for p in points), min(p[1] for p in points),
                          max(p[0] for p in points), max(p[1] for p in points)],
        }))
    return {
        "frame": index,
        "width": data["imageWidth"],
        "height": data["imageHeight"],
        "annotation_complete": bool(data.get("flags", {}).get("annotation_complete", False)),
        "detections": entities,
    }


def collapse_sam_emissions(detections):
    """SAM动态补提示会重放当前帧；同ID取最后状态，保留先前输出审计。"""
    last = {}
    audit = []
    independent = []
    for d in detections:
        tid = d.get("track_id")
        if tid is None:
            independent.append(d)
            continue
        if tid in last:
            audit.append(
                {
                    "reason": "sam_replayed_same_id",
                    "replaced": last[tid],
                    "replacement": d,
                }
            )
        last[tid] = d
    return [*last.values(), *independent], audit


def iter_frames(path, **kwargs):
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.glob("*.json*")):
            f = read_json(p)
            if "shapes" in f:
                f = labelme_frame(
                    f, int(re.search(r"(\d+)(?:\.json)(?:\.gz)?$", p.name).group(1))
                )
            yield normalize(f, **kwargs)
    elif path.name.endswith((".jsonl", ".jsonl.gz")):
        with (
            gzip.open(path, "rt", encoding="utf-8")
            if path.suffix == ".gz"
            else path.open(encoding="utf-8")
        ) as h:
            for line in h:
                if line.strip():
                    yield normalize(json.loads(line), **kwargs)
    else:
        data = read_json(path)
        if isinstance(data, dict) and "shapes" in data:
            frame = int(re.search(r"(\d+)(?:\.json)(?:\.gz)?$", path.name).group(1))
            yield normalize(labelme_frame(data, frame), **kwargs)
            return
        if isinstance(data, dict) and isinstance(data.get("frames"), dict):
            for k, ds in sorted(data["frames"].items(), key=lambda kv: int(kv[0])):
                meta = data.get("metadata", {})
                ds, audit = collapse_sam_emissions(ds)
                yield normalize(
                    {
                        "frame": int(k),
                        "detections": ds,
                        "fps": meta.get("fps", 30),
                        "width": meta.get("width", 1920),
                        "height": meta.get("height", 1080),
                        "upstream_emission_audit": audit,
                    },
                    **kwargs,
                )
        else:
            for f in data if isinstance(data, list) else [data]:
                yield normalize(f, **kwargs)


def write_frames(path, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        gzip.open(path, "wt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("w", encoding="utf-8")
    ) as out:
        for frame in frames:
            out.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
