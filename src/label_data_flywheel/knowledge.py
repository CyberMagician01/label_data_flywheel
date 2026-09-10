"""BEEG知识节点、确认反馈与外部上下文；规则强度不是行为真值。"""

from collections import defaultdict
import copy
import numpy as np


def default_graph():
    rules = [
        ("visual_kinematics", "O2"),
        ("configured_entrance_line", "O3"),
        ("directed_interaction", "O3"),
        ("axis_oscillation", "O3"),
        ("pose_topology", "O1"),
        ("density_consistency", "O1"),
        ("colony_temporal_change", "O3"),
    ]
    return {
        "nodes": [
            {
                "id": name,
                "knowledge_source": "configured_visual_rule",
                "observability_level": level,
                "domains": ["RGB_out", "IR_in"],
                "parameters": {},
                "alpha": 1.0,
                "beta": 1.0,
                "evidence_confidence": 0.5,
                "validation_status": "prior_only",
            }
            for name, level in rules
        ],
        "edges": [
            {
                "source": "visual_kinematics",
                "target": "directed_interaction",
                "relation": "supports",
            },
            {
                "source": "pose_topology",
                "target": "axis_oscillation",
                "relation": "supports",
            },
        ],
    }


def update_graph(graph, reviews):
    result = copy.deepcopy(graph)
    by_rule = defaultdict(list)
    for r in reviews:
        if r.get("split") == "test":
            raise ValueError("测试集判读不能更新训练知识")
        if not r.get("reviewer") or r.get("action") not in (
            "confirm",
            "reject",
            "correct",
        ):
            raise ValueError("知识更新只接受具名人工复核")
        by_rule[r["knowledge_id"]].append(r)
    for node in result["nodes"]:
        rows = by_rule.get(node["id"], [])
        if not rows:
            continue
        used = set(node.get("review_ids", []))
        fresh = list({r["review_id"]: r for r in rows if r["review_id"] not in used}.values())
        node["alpha"] += sum(r["action"] == "confirm" for r in fresh)
        node["beta"] += sum(r["action"] != "confirm" for r in fresh)
        node["review_ids"] = sorted(used | {r["review_id"] for r in fresh})
        node["validated_videos"] = sorted(
            set(node.get("validated_videos", [])) | {r["video"] for r in fresh}
        )
        node["validated_domains"] = sorted(
            set(node.get("validated_domains", [])) | {r["domain"] for r in fresh}
        )
        node["evidence_confidence"] = node["alpha"] / (node["alpha"] + node["beta"])
        gains = [
            r["downstream_gain"]
            for r in fresh
            if r.get("matched_protocol_verified") and "downstream_gain" in r
        ]
        node["downstream_gains"] = [*node.get("downstream_gains", []), *gains]
        node["validation_status"] = "human_reviewed"
    return result


def rule_evidence(graph, knowledge_id, domain=None, video=None):
    """规则支持度决定解释强度；一次复核的预期方差缩减衡量知识获取价值。"""
    node = next((n for n in graph["nodes"] if n["id"] == knowledge_id), None)
    if node is None or (domain and domain not in node.get("domains", [domain])):
        return {"knowledge_id": knowledge_id, "knowledge_support": 0.5, "knowledge_gain": 1.0}
    a, b = float(node["alpha"]), float(node["beta"])
    total = a + b
    gain = 36 * a * b / (total * total * (total + 1) ** 2)
    if video and video not in node.get("validated_videos", []):
        gain *= 1.25
    return {"knowledge_id": knowledge_id, "knowledge_support": a / total,
            "knowledge_gain": float(min(gain, 1.0))}


def apply_behavior_knowledge(report, graph):
    out = copy.deepcopy(report)
    for event in out.get("events", []):
        evidence = rule_evidence(graph, event["knowledge_source"], event["domain"], event["video"])
        event.update(evidence)
        visual = event.get("evidence_confidence")
        event["visual_evidence_confidence"] = visual
        if visual is not None:
            event["evidence_confidence"] = visual * evidence["knowledge_support"]
    return out


def attach_context(report, context=None):
    ctx = context or {"enabled": False}
    out = copy.deepcopy(report)
    if not ctx.get("enabled"):
        out["external_context"] = {
            "enabled": False,
            "context_id": "default",
            "hive_id": "default",
            "attributes": {},
        }
        return out
    out["external_context"] = {
        "enabled": True,
        "context_id": ctx["context_id"],
        "hive_id": ctx["hive_id"],
        "attributes": ctx.get("attributes", {}),
    }
    for name in (
        "observations",
        "individual_windows",
        "group_windows",
        "interaction_edges",
        "events",
    ):
        for row in out.get(name, []):
            row.update(
                context_id=ctx["context_id"],
                hive_id=ctx["hive_id"],
                context_attributes=ctx.get("attributes", {}),
            )
    return out


def context_correlations(records, feature, outcome):
    """仅在真实记录配对后计算相关性，不解释为因果关系。"""
    pairs = [
        (r["context_attributes"][feature], r[outcome])
        for r in records
        if isinstance(r.get("context_attributes", {}).get(feature), (int, float))
        and isinstance(r.get(outcome), (int, float))
    ]
    x = np.asarray(pairs, float)
    if len(x) < 3 or np.any(x.std(0) == 0):
        return {"samples": len(x), "pearson": None}
    return {
        "samples": len(x),
        "pearson": float(np.corrcoef(x.T)[0, 1]),
        "causal_claim": False,
    }
