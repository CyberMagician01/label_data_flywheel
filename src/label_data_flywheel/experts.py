"""可校准Q-MoE与漏标路由：校准集拟合，按域/层/密度选择专家。"""

from collections import defaultdict
import copy
import numpy as np
from .calibration import fit_isotonic, calibrated
from .quality import fuse_experts, route_candidate, head_tail_evidence
from .geometry import overlap, pose, center, wrap
from .semantics import is_bee
from .postprocess import is_fill


def context_key(record):
    return "|".join(
        str(record.get(k, "all")) for k in ("domain", "layer", "density_bin")
    )


def fit_experts(reference_records):
    groups = defaultdict(list)
    for r in reference_records:
        if r["split"] != "calibration" or not r.get("human_verified"):
            raise ValueError("专家概率必须由calibration人工参考集校准")
        groups[(context_key(r), r["name"])].append(r)
    models = {}
    for (key, name), rows in groups.items():
        model = fit_isotonic([r["score"] for r in rows], [r["correct"] for r in rows])
        prob = calibrated([r["score"] for r in rows], model)
        brier = float(np.mean((prob - [r["correct"] for r in rows]) ** 2))
        models.setdefault(key, {})[name] = {
            "calibration": model,
            "gate_logit": -4 * brier,
            "fit_brier": brier,
            "family": rows[0].get("family", name),
        }
    return {
        "contexts": models,
        "fit_split": "calibration",
        "reliability_is_in_sample": True,
    }


def infer_experts(record, model):
    profiles = model["contexts"]
    key = context_key(record)
    profile = profiles.get(
        key, profiles.get("|".join((record["domain"], record["layer"], "all")), {})
    )
    evidence = []
    reliability = {}
    for raw in record.get("experts", []):
        entry = profile.get(raw["name"])
        if entry is None:
            continue  # 未校准输出不能冒充后验概率。
        probability = float(calibrated([raw["score"]], entry["calibration"])[0])
        evidence.append({**raw, "probability": probability, "family": entry["family"]})
        reliability[raw["name"]] = entry["gate_logit"]
    return {**fuse_experts(evidence, reliability), "evidence": evidence, "context": key}


def apply_candidate_evidence(frame, candidates, model, thresholds=None):
    """候选必须已按空间实体聚合；观察框不因伪标签融合而被覆盖。"""
    f = copy.deepcopy(frame)
    f.setdefault("ignore_regions", [])
    f.setdefault("candidate_audit", [])
    thresholds = thresholds or {}
    boxes = [d["bbox_xyxy"] for d in f["detections"]]
    for i, candidate in enumerate(candidates):
        row = {**candidate, "domain": f["domain"], "layer": "L1"}
        fused = infer_experts(row, model)
        routing = route_candidate(
            fused["evidence"],
            thresholds.get("positive", 0.85),
            thresholds.get("ignore", 0.55),
        )
        om, _, _ = overlap([candidate["bbox_xyxy"]], boxes)
        if np.any(om >= 0.5):
            routing = {"route": "already_observed"}
        d = {
            **candidate,
            "entity_id": candidate.get("entity_id", f"{f['sample_id']}/candidate/{i}"),
            "expert_fusion": fused,
            "quality": fused["probability"] or 0.0,
            "label_status": routing["route"],
            "origin": "multi_expert_candidate",
            "keypoints": candidate.get("keypoints", {}),
            "track_id": None,
            "supervision_mask": {
                "detection": True,
                "pose": bool(candidate.get("keypoints")),
                "tracking": False,
                "behavior": False,
            },
        }
        if routing["route"] == "soft_positive":
            f["detections"].append(d)
            boxes.append(d["bbox_xyxy"])
        elif routing["route"] == "ignore":
            f["ignore_regions"].append(d)
        f["candidate_audit"].append(
            {"entity_id": d["entity_id"], "routing": routing, "fusion": fused}
        )
    return f


def cross_layer_evidence(frames, behavior=None):
    """无GT时仅报告跨层一致性代理，不伪造NME、IDSW或行为准确率。"""
    rows = []
    previous = {}
    for f in sorted(
        frames, key=lambda f: (f["domain"], f["video"], f["group"], f["frame"])
    ):
        for d in f["detections"]:
            if not is_bee(d) or is_fill(d) or d.get("label_status") == "invalid":
                continue
            base = {
                "sample_id": d["entity_id"],
                "frame_id": f["sample_id"],
                "domain": f["domain"],
                "group": f["group"],
                "video": f["video"],
                "is_gt_error": False,
                "observability_level": "O1",
            }
            p = pose(d)
            b = np.array(d["bbox_xyxy"])
            tid = d.get("track_id")
            scope = (f["domain"], f["video"], f["group"], tid)
            if p:
                points = np.array([p["head"][:2], p["tail"][:2]])
                outside = np.maximum(b[:2] - points, 0) + np.maximum(points - b[2:], 0)
                error = float(
                    np.linalg.norm(outside, axis=1).mean()
                    / max(np.linalg.norm(b[2:] - b[:2]), 1.0)
                )
                rows.append(
                    {
                        **base,
                        "layer": "L2",
                        "error": error,
                        "root_cause": "pose_outside_box" if error > 0 else "none",
                        "evidence_type": "topology_proxy",
                    }
                )
            old = previous.get(scope) if tid is not None else None
            if old:
                of, od = old
                gap = f["frame"] - of["frame"]
                op = pose(od)
                if gap > 0:
                    unit = max(float(np.linalg.norm(b[2:] - b[:2])), 1.0)
                    displacement = float(
                        np.linalg.norm(center(d) - center(od)) / gap / unit
                    )
                    angle = (
                        float(abs(wrap(p["angle"] - op["angle"])) / np.pi)
                        if p and op
                        else 0.0
                    )
                    rows.append(
                        {
                            **base,
                            "layer": "L3",
                            "error": displacement + 0.1 * angle,
                            "root_cause": "motion_jump"
                            if displacement > 0.2
                            else "none",
                            "evidence_type": "motion_pose_consistency",
                            "source_frame_gap": gap,
                            "observability_level": "O2",
                        }
                    )
            if tid is not None:
                previous[scope] = (f, d)
    if behavior:
        for event in behavior.get("events", []):
            confidence = event.get("evidence_confidence")
            rows.append(
                {
                    "sample_id": event["event_id"],
                    "domain": event["domain"],
                    "layer": "L4",
                    "group": event.get("group", "unknown"),
                    "error": 1 - confidence if confidence is not None else 1.0,
                    "is_gt_error": False,
                    "root_cause": "unconfirmed_behavior",
                    "observability_level": "O3",
                    "evidence_type": "event_support_proxy",
                }
            )
    return rows


def calibrated_pose_decision(pred, reference, thresholds):
    result = head_tail_evidence(
        pred, reference, reference["bbox_xyxy"], thresholds["nme"]
    )
    p = pose(pred)
    if not p or p["confidence"] is None or p["confidence"] < thresholds["confidence"]:
        return {
            **result,
            "action": "review",
            "reason": "low_or_missing_pose_confidence",
        }
    angle = result.get("angle_error_deg", 180.0)
    if (
        result["action"] == "swap_candidate"
        and abs(180 - angle) > thresholds["angle_degrees"]
    ):
        result["action"] = "review"
    if result["action"] == "keep" and angle > thresholds["angle_degrees"]:
        result["action"] = "review"
    return result


def track_pose_vote(decisions):
    """IR轨迹级正常/交换中位证据；只返回建议，原人工关键点保持可审计。"""
    usable = [d for d in decisions if "nme_normal" in d]
    if not usable:
        return {"action": "review", "observations": 0}
    normal = float(np.median([d["nme_normal"] for d in usable]))
    swapped = float(np.median([d["nme_swap"] for d in usable]))
    votes = sum(d["action"] == "swap_candidate" for d in usable)
    return {
        "action": "swap_candidate"
        if votes > len(usable) / 2 and swapped < normal
        else "keep_or_review",
        "nme_normal_median": normal,
        "nme_swap_median": swapped,
        "swap_votes": votes,
        "observations": len(usable),
    }
