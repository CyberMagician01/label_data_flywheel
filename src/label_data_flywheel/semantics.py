"""蜂体与影子的共享语义；原始记录保留，监督只使用适用的字段。"""

import copy


def is_bee(detection):
    return detection.get("class_name", detection.get("label")) != "bee_shadow" and int(
        detection.get("class_id", 0)
    ) == 0


def entity_rules(detection):
    d = copy.deepcopy(detection)
    d.setdefault("class_id", 1 if d.get("class_name", d.get("label")) == "bee_shadow" else 0)
    if not is_bee(d):
        if d.get("keypoints"):
            d.setdefault("source_keypoints", copy.deepcopy(d["keypoints"]))
        if d.get("track_id") is not None:
            d.setdefault("source_track_id", d["track_id"])
        d["keypoints"], d["track_id"] = {}, None
        d["supervision_mask"] = {
            "detection": True, "pose": False, "tracking": False, "behavior": False
        }
    return d


def refresh_supervision(d):
    """人工补点、改 ID 后同步任务可观测性，不沿用修改前的空任务掩码。"""
    d.update(entity_rules(d))
    d["supervision_mask"] = {
        "detection": d.get("label_status") != "invalid",
        "pose": is_bee(d) and any(
            len(p) >= 2 and (len(p) < 3 or p[2] > 0)
            for p in d.get("keypoints", {}).values()
        ),
        "tracking": is_bee(d) and d.get("track_id") is not None,
        "behavior": is_bee(d) and bool(d.get("confirmed_events") or d.get("behavior_labels")),
    }
