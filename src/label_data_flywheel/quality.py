"""Q²：群间经验贝叶斯收缩、群内MAD；Q-MoE：可校准专家证据融合。"""

from collections import defaultdict
import math
import numpy as np
from .geometry import pose, overlap, wrap


def q2(records, tau=20, rho=0.5):
    groups = defaultdict(list)
    domains = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r["domain"], r["layer"], r["group"])].append(i)
        domains[(r["domain"], r["layer"])].append(float(r["error"]))
    out = [dict(r) for r in records]
    for (d, l, g), ids in groups.items():
        values = np.array([records[i]["error"] for i in ids], float)
        n = len(ids)
        group_error = (n * values.mean() + tau * np.mean(domains[(d, l)])) / (n + tau)
        median = np.median(values)
        scale = max(1.4826 * np.median(np.abs(values - median)), 0.02)
        for i, v in zip(ids, values):
            z = float((v - median) / scale)
            inter = float(np.exp(-max(group_error, 0)))
            intra = float(np.exp(-max(v, 0)) / (1 + max(z, 0)))
            out[i].update(
                q_inter=inter,
                q_intra=intra,
                robust_z=z,
                quality=rho * inter + (1 - rho) * intra,
            )
    return out


def fuse_experts(evidence, reliability=None):
    """输入必须为层内校准概率；缺失专家不充当零分，不同模型家族保留独立性。"""
    valid = [e for e in evidence if e.get("probability") is not None]
    if not valid:
        return {"probability": None, "disagreement": None, "weights": {}}
    rel = reliability or {}
    logits = np.array([rel.get(e["name"], 0.0) for e in valid])
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    probs = np.array([e["probability"] for e in valid], float)
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("专家证据必须是[0,1]概率")
    mean = float(weights @ probs)
    return {
        "probability": mean,
        "disagreement": float(weights @ ((probs - mean) ** 2)),
        "weights": {e["name"]: float(w) for e, w in zip(valid, weights)},
    }


def route_candidate(evidence, positive=0.85, ignore=0.55):
    # 同一模型的多个增强预测只算同一来源家族，避免伪造独立证据。
    families = {
        e.get("family", e["name"])
        for e in evidence
        if e.get("probability") is not None and e["probability"] >= positive
    }
    probs = [e["probability"] for e in evidence if e.get("probability") is not None]
    if len(families) >= 2:
        return {
            "route": "soft_positive",
            "weight": min(probs),
            "human_confirmed": False,
        }
    if any(p >= ignore for p in probs):
        return {"route": "ignore", "background_weight": 0.0, "human_confirmed": False}
    return {"route": "background_candidate", "weight": 1.0, "human_confirmed": False}


def head_tail_evidence(pred, reference, bbox, threshold=0.2):
    p = pose(pred)
    r = pose(reference)
    if p is None or r is None:
        return {"action": "review", "reason": "missing_pose"}
    diagonal = max(float(np.linalg.norm(np.asarray(bbox)[2:] - bbox[:2])), 1e-6)
    a = np.array([p["head"][:2], p["tail"][:2]])
    b = np.array([r["head"][:2], r["tail"][:2]])
    normal = float(np.linalg.norm(a - b, axis=1).mean() / diagonal)
    swapped = float(np.linalg.norm(a - b[::-1], axis=1).mean() / diagonal)
    action = (
        "swap_candidate"
        if swapped < normal and swapped <= threshold
        else ("keep" if normal <= threshold else "review")
    )
    return {
        "action": action,
        "nme_normal": normal,
        "nme_swap": swapped,
        "angle_error_deg": float(abs(wrap(p["angle"] - r["angle"])) * 180 / math.pi),
    }


def frame_evidence(frame):
    ds = frame["detections"]
    boxes = [d["bbox_xyxy"] for d in ds]
    om, _, _ = overlap(boxes, boxes)
    if len(ds):
        np.fill_diagonal(om, 0)
    records = []
    for i, d in enumerate(ds):
        error = float(om[i].max(initial=0))
        records.append(
            {
                "sample_id": d["entity_id"],
                "frame_id": frame["sample_id"],
                "domain": frame["domain"],
                "layer": "L1",
                "group": frame["group"],
                "error": error,
                "evidence_type": "geometric_overlap_proxy",
                "is_gt_error": False,
                "root_cause": "overlap_candidate" if error > 0.2 else "none",
            }
        )
    return records
