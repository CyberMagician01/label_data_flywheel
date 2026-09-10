"""LabelMe 人工任务包：与团队框点/ID 工具共用 JSON，回传差异生成复核决定。"""

from pathlib import Path
import copy
from .io import write_json, read_json, labelme_frame


def export_tasks(frames, queue, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    selected = {r.get("frame_id") for r in queue}
    selected |= {r.get("entity_id", r["sample_id"]).rsplit("/", 1)[0] for r in queue}
    entities = {r.get("entity_id", r["sample_id"]): r for r in queue}
    manifest = []
    for f in frames:
        ds = [d for name in ("detections", "ignore_regions", "temporarily_hidden_detections")
              for d in f.get(name, []) if d.get("label_status") != "invalid"]
        if f["sample_id"] not in selected and not any(d["entity_id"] in entities for d in ds):
            continue
        shapes = []
        for i, d in enumerate(ds):
            key = d["entity_id"]
            group = d.get("track_id")
            if group is None:
                group = -(i + 1)
            flags = {"entity_id": key, "source_track_id": d.get("track_id"),
                     "knowledge_id": entities.get(key, {}).get("knowledge_id")}
            shapes.append({"label": "bee_shadow" if d.get("class_id", 0) == 1 else "bee",
                           "shape_type": "rectangle", "points": [d["bbox_xyxy"][:2], d["bbox_xyxy"][2:]],
                           "group_id": group, "flags": flags})
            for name, point in d.get("keypoints", {}).items():
                if len(point) > 2 and point[2] <= 0:
                    continue
                shapes.append({"label": "tail" if name == "abdomen_tip" else name,
                               "shape_type": "point", "points": [point[:2]], "group_id": group,
                               "flags": {"entity_id": key}})
        name = f"{f['domain']}_{f['group']}_{f['video']}_frame_{f['frame']:08d}.json"
        data = {"version": "5.5.0", "imagePath": f.get("image_path"),
                "imageData": None, "imageWidth": f["image_size"][0], "imageHeight": f["image_size"][1],
                "flags": {"source_frame_id": f["sample_id"], "review_complete": False},
                "shapes": shapes}
        write_json(root / "editable" / name, data)
        write_json(root / "original" / name, data)
        manifest.append({"file": name, "frame_id": f["sample_id"]})
    write_json(root / "tasks.json", manifest)
    return manifest


def import_tasks(directory):
    root = Path(directory)
    decisions = []
    for entry in read_json(root / "tasks.json"):
        original = read_json(root / "original" / entry["file"])
        edited = read_json(root / "editable" / entry["file"])
        before = [s for s in original["shapes"] if s["shape_type"] == "rectangle"]
        records = labelme_frame(edited, 0)["detections"]
        seen = set()
        for d in records:
            shape = edited["shapes"][d["source_shape_index"]]
            flags = shape.get("flags", {})
            key = flags.get("entity_id")
            if not key:
                matches = [s for s in before if s["points"] == shape["points"] and s["label"] == shape["label"]]
                if len(matches) == 1:
                    key = matches[0]["flags"]["entity_id"]
            base = next((s for s in before if s["flags"]["entity_id"] == key), None)
            seen.add(key)
            tid = d.get("track_id")
            if tid is not None and tid < 0:
                tid = None
            changes = {"bbox_xyxy": d["bbox_xyxy"], "keypoints": d["keypoints"],
                       "track_id": tid, "class_id": d["class_id"]}
            if base is None:
                decisions.append({"action": "add", "entity_id": f"{entry['frame_id']}/added/{d['source_shape_index']}",
                                  "frame_id": entry["frame_id"], **changes})
                continue
            # 按原结构比较框、点与 ID，未修改且未明确确认的对象不冒充复核。
            old_points = {s["label"]: s["points"] for s in original["shapes"]
                          if s["shape_type"] == "point" and s.get("group_id") == base.get("group_id")}
            new_points = {s["label"]: s["points"] for s in edited["shapes"]
                          if s["shape_type"] == "point" and s.get("group_id") == shape.get("group_id")}
            changed = any(base.get(k) != shape.get(k) for k in ("points", "group_id", "label")) or old_points != new_points
            action = flags.get("review_action")
            if not action:
                action = "correct" if changed else "confirm" if edited.get("flags", {}).get("review_complete") else None
            if action:
                decisions.append({"action": action, "entity_id": key,
                                  "knowledge_id": base["flags"].get("knowledge_id"), **changes})
        for base in before:
            key = base["flags"]["entity_id"]
            if key not in seen:
                decisions.append({"action": "reject", "entity_id": key,
                                  "knowledge_id": base["flags"].get("knowledge_id")})
    return decisions
