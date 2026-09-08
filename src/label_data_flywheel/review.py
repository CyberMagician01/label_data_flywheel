"""预算约束复核、追加式审计、ErrorCube与下一轮训练数据。"""

from collections import defaultdict
import copy
import json
from pathlib import Path
import numpy as np
from .sampling import representative_indices


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
    lookup = {d["entity_id"]: d for f in result for d in f["detections"]}
    logs = []
    for item in decisions:
        target = lookup[item["entity_id"]]
        before = copy.deepcopy(target)
        if item["action"] == "reject":
            target["label_status"] = "invalid"
        elif item["action"] in ("confirm", "correct"):
            for key in ("bbox_xyxy", "keypoints", "track_id", "confirmed_events"):
                if key in item:
                    target[key] = item[key]
            target["label_status"] = "human_confirmed"
        else:
            raise ValueError("复核操作必须为confirm/correct/reject")
        logs.append(
            {
                "entity_id": item["entity_id"],
                "reviewer": reviewer,
                "action": item["action"],
                "before": before,
                "after": target,
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
        "layer",
        "group",
        "scale_bin",
        "density_bin",
        "occlusion_bin",
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
