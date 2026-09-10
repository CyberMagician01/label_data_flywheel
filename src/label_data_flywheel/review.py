"""预算约束复核、追加式审计、ErrorCube与下一轮训练数据。"""

from collections import defaultdict
import copy
import json
from pathlib import Path
import numpy as np
from .sampling import representative_indices
from .semantics import refresh_supervision


def prioritize(records, budget, diversity=0.3):
    if not records:
        return []
    values = [
        r.get("uncertainty", 0)
        + r.get("disagreement", 0)
        + r.get("rarity", 0)
        + r.get("knowledge_gain", 0)
        + r.get("cross_layer_value", 0)
        + r.get("behavior_value", 0)
        - 0.2 * r.get("review_cost", 1)
        for r in records
    ]
    features = [
        r.get("features", [values[i], r.get("density", 0), r.get("scale", 0)])
        for i, r in enumerate(records)
    ]
    chosen = representative_indices(
        features, values, min(budget, len(records)), diversity=diversity
    )
    return [
        dict(records[i], review_priority=values[i], review_status="pending")
        for i in chosen
    ]


def apply_reviews(frames, decisions, reviewer, audit_path):
    if not reviewer.strip():
        raise ValueError("人工复核必须记录reviewer")
    result = copy.deepcopy(frames)
    collections = ("detections", "ignore_regions", "temporarily_hidden_detections")
    lookup = {d["entity_id"]: (f, name, d)
              for f in result for name in collections for d in f.get(name, [])}
    frame_lookup = {f["sample_id"]: f for f in result}
    logs = []
    for item in decisions:
        if item["action"] == "add":
            if item["entity_id"] in lookup:
                raise ValueError("补标的 entity_id 已存在")
            frame = frame_lookup[item["frame_id"]]
            target = {"entity_id": item["entity_id"], "bbox_xyxy": item["bbox_xyxy"],
                      "keypoints": {}, "track_id": None, "origin": "human_review"}
            frame["detections"].append(target)
            collection = "detections"
            lookup[item["entity_id"]] = (frame, collection, target)
        else:
            frame, collection, target = lookup[item["entity_id"]]
        before = copy.deepcopy(target)
        if item["action"] == "reject":
            target["label_status"] = "invalid"
            frame.setdefault("verified_background_regions", []).append({
                "bbox_xyxy": list(target["bbox_xyxy"]), "entity_id": target["entity_id"],
                "reviewer": reviewer,
            })
        elif item["action"] in ("confirm", "correct", "add"):
            frame["verified_background_regions"] = [
                r for r in frame.get("verified_background_regions", [])
                if r.get("entity_id") != target["entity_id"]]
            for key in ("bbox_xyxy", "keypoints", "track_id", "confirmed_events",
                        "behavior_labels", "class_id", "class_name"):
                if key in item:
                    target[key] = item[key]
            if "class_id" in item and "class_name" not in item:
                target["class_name"] = "bee" if int(item["class_id"]) == 0 else "bee_shadow"
            target["label_status"] = "human_confirmed"
            if "interpolation" in target.get("origin", ""):
                target["review_source_origin"] = target["origin"]
                target["origin"] = "human_review"
            if collection != "detections":
                frame[collection].remove(target)
                frame["detections"].append(target)
                target["review_source_origin"] = target.get("origin")
                target["origin"] = "human_review"
                lookup[item["entity_id"]] = (frame, "detections", target)
        else:
            raise ValueError("复核操作必须为confirm/correct/reject/add")
        target["reviewer"] = reviewer
        refresh_supervision(target)
        logs.append(
            {
                "entity_id": item["entity_id"],
                "reviewer": reviewer,
                "action": item["action"],
                "before": before,
                "after": copy.deepcopy(target),
                "frame_id": frame["sample_id"],
            }
        )
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in logs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def error_cube(records):
    groups = defaultdict(list)
    dimensions = (
        "domain",
        "video",
        "split",
        "layer",
        "group",
        "scale_bin",
        "density_bin",
        "occlusion_bin",
        "orientation_bin",
        "quality_bin",
        "observability_level",
        "behavior_state",
        "root_cause",
    )
    for r in records:
        groups[tuple(str(r.get(k, "unknown")) for k in dimensions)].append(r)
    return [
        {
            **dict(zip(dimensions, key)),
            "count": len(rows),
            "mean_error": float(np.mean([r.get("error", 0) for r in rows])),
            "next_action": {
                "overlap_candidate": "review_overlap",
                "pose_swap": "review_pose",
                "identity_switch": "sample_association_pairs",
            }.get(key[-1], "review_evidence"),
        }
        for key, rows in groups.items()
    ]


def feedback_snapshot(frames):
    # 无确认数据不自动进入硬标签集；模型软伪正与ignore独立存放。
    result = []
    for frame in frames:
        f = copy.deepcopy(frame)
        f["detections"] = [
            d
            for d in f["detections"]
            if d.get("label_status") in ("human", "human_confirmed")
        ]
        if (
            f["detections"]
            or f.get("annotation_complete")
            or f.get("source_kind") == "human"
        ):
            result.append(f)
    return result
