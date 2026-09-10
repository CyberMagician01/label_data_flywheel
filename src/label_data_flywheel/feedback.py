"""把误差证据、层级监督、知识价值和下一轮采样连接到同一实体。"""

from collections import Counter, defaultdict
import numpy as np
from .diagnostics import sample_attributes
from .geometry import overlap
from .quality import q2, frame_evidence
from .experts import cross_layer_evidence, calibrated_pose_decision, track_pose_vote
from .knowledge import rule_evidence
from .postprocess import is_fill
from .semantics import is_bee
from .sampling import target_distribution

CATEGORIES = {
    "overlap_candidate": ("box", "pose_topology"),
    "pose_outside_box": ("pose", "pose_topology"),
    "pose_swap": ("pose", "pose_topology"),
    "pose_reference_error": ("pose", "pose_topology"),
    "motion_jump": ("identity_visibility", "visual_kinematics"),
    "identity_switch": ("identity_visibility", "visual_kinematics"),
    "missing_candidate": ("coverage", "density_consistency"),
    "reference_miss": ("coverage", "density_consistency"),
    "unconfirmed_behavior": ("supervision", "visual_kinematics"),
}


def evidence_quality(frames, behavior, references=()):
    usable = [{**f, "detections": [d for d in f["detections"]
               if is_bee(d) and not is_fill(d) and d.get("label_status") != "invalid"]}
              for f in frames]
    records = [r for f in usable for r in frame_evidence(f)]
    records += cross_layer_evidence(usable, behavior)
    attributes = {r["sample_id"]: r for r in sample_attributes(usable)}
    for event in behavior.get("events", []):
        first = next((attributes[e] for e in event.get("upstream_entities", []) if e in attributes), {})
        attributes[event["event_id"]] = {**first, "video": event["video"]}
    pose_tracks = defaultdict(list)
    for f in frames:
        for d in f.get("ignore_regions", []):
            if d.get("label_status") == "invalid" or not is_bee(d):
                continue
            probability = d.get("expert_fusion", {}).get("probability")
            records.append({"sample_id": d["entity_id"], "frame_id": f["sample_id"],
                            "domain": f["domain"], "video": f["video"], "group": f["group"],
                            "layer": "L1", "error": 1 - (probability or 0),
                            "root_cause": "missing_candidate", "is_gt_error": False,
                            "evidence_type": "calibrated_candidate_support"})
            attributes.update({r["sample_id"]: r for r in sample_attributes(
                [{**f, "detections": [d]}])})
    reference_by_frame = {(f["domain"], f["video"], f["frame"]): f for f in references}
    previous_reference_id = {}
    by_entity_layer = {(r["sample_id"], r["layer"]): r for r in records}
    for f in sorted(usable, key=lambda f: (f["domain"], f["video"], f["group"], f["frame"])):
        ref = reference_by_frame.get((f["domain"], f["video"], f["frame"]))
        if ref is None:
            continue
        if f["split"] == "train" and ref["split"] == "test":
            raise ValueError("测试参考不能为训练样本提供质量监督")
        gt = [d for d in ref["detections"] if is_bee(d)]
        ds = f["detections"]
        iou = overlap([d["bbox_xyxy"] for d in ds], [d["bbox_xyxy"] for d in gt])[2]
        from scipy.optimize import linear_sum_assignment
        a, b = linear_sum_assignment(np.where(iou >= .5, 1-iou, 1e6))
        matched_truth = set()
        for i, j in zip(a, b):
            if iou[i,j] < .5:
                continue
            matched_truth.add(j)
            d, truth = ds[i], gt[j]
            row = by_entity_layer[(d["entity_id"], "L1")]
            row.update(error=float(1-iou[i,j]), is_gt_error=True,
                       evidence_type="reference_localization", reference_entity_id=truth["entity_id"])
            decision = calibrated_pose_decision(d, truth, {"nme":.2,"confidence":.05,"angle_degrees":35})
            if "nme_normal" in decision:
                pose_row = by_entity_layer.get((d["entity_id"], "L2"))
                if pose_row is not None:
                    pose_row.update(error=decision["nme_normal"], is_gt_error=True,
                                    evidence_type="reference_pose", pose_decision=decision,
                                    root_cause="pose_swap" if decision["action"]=="swap_candidate"
                                    else "pose_reference_error" if decision["action"]=="review" else "none")
                    if f["domain"] == "IR_in" and d.get("track_id") is not None:
                        pose_tracks[(f["domain"],f["video"],f["group"],d["track_id"])].append(pose_row)
            if d.get("track_id") is not None and truth.get("track_id") is not None:
                key = (f["domain"],f["video"],f["group"],d["track_id"])
                old = previous_reference_id.get(key)
                previous_reference_id[key] = truth["track_id"]
                track_row = by_entity_layer.get((d["entity_id"],"L3"))
                if track_row is not None and old is not None and old != truth["track_id"]:
                    track_row.update(error=1.0, is_gt_error=True, root_cause="identity_switch",
                                     evidence_type="reference_identity_correspondence")
        for j, truth in enumerate(gt):
            if j in matched_truth:
                continue
            key = f"{f['sample_id']}/reference_miss/{j}"
            candidate = {**truth, "entity_id": key}
            attributes[key] = sample_attributes([{**f, "detections": [candidate]}])[0]
            records.append({"sample_id": key, "frame_id": f["sample_id"], "domain": f["domain"],
                            "group": f["group"], "layer": "L1", "error": 1.,
                            "root_cause": "reference_miss", "is_gt_error": True,
                            "reference_bbox": truth["bbox_xyxy"],
                            "evidence_type": "unmatched_reference"})
    for rows in pose_tracks.values():
        vote = track_pose_vote([r["pose_decision"] for r in rows])
        for row in rows:
            row["track_pose_vote"] = vote
    for row in records:
        attr = attributes.get(row["sample_id"], {})
        scale, density = attr.get("scale"), attr.get("density")
        direction, overlap_proxy = attr.get("direction"), attr.get("occlusion_proxy")
        row.update({
            k: attr[k] for k in ("video","frame","split","segment","scale","density","direction")
            if k in attr
        })
        row.update(scale_bin="small" if scale is not None and scale<.02 else
                   "medium" if scale is not None and scale<.05 else "large" if scale is not None else "not_applicable",
                   density_bin="low" if density is not None and density<50 else
                   "medium" if density is not None and density<200 else "high" if density is not None else "not_applicable",
                   occlusion_bin="overlap_high" if overlap_proxy is not None and overlap_proxy>=.5 else
                   "overlap_low" if overlap_proxy is not None else "not_observed",
                   occlusion_source="box_overlap_proxy",
                   orientation_bin=str(int((direction % (2*np.pi))/(np.pi/4))) if direction is not None else "not_observed")
        category, knowledge = CATEGORIES.get(row["root_cause"], ("none","pose_topology"))
        row.update(error_category=category, knowledge_id=knowledge)
    split_records = defaultdict(list)
    for row in records:
        split_records[row.get("split", "unassigned")].append(row)
    result = [row for rows in split_records.values() for row in q2(rows)]
    for row in result:
        row["quality_bin"] = "low" if row["quality"] < .5 else "high"
    return result


def bind_quality(frames, quality):
    evidence = defaultdict(dict)
    for row in quality:
        evidence[row["sample_id"]][row["layer"]] = row
    for f in frames:
        for d in f["detections"] + f.get("ignore_regions", []):
            layers = evidence.get(d["entity_id"], {})
            if not layers:
                continue
            d["quality_by_layer"] = {k: r["quality"] for k,r in layers.items()}
            first = layers.get("L1", next(iter(layers.values())))
            d.update(quality=first["quality"], q_inter=first["q_inter"], q_intra=first["q_intra"])
            for layer, name in (("L2","pose_quality"),("L3","track_quality")):
                if layer in layers:
                    d[name] = layers[layer]["quality"]


def review_candidates(frames, quality, graph, behavior):
    entities = {d["entity_id"]:d for f in frames
                for name in ("detections","ignore_regions") for d in f.get(name, [])}
    behavior_entities = Counter(e for event in behavior.get("events", [])
                                for e in event.get("upstream_entities", []))
    counts = Counter((r["domain"],r["scale_bin"],r["density_bin"],r["error_category"]) for r in quality)
    result = []
    for row in quality:
        entity = entities.get(row["sample_id"], {})
        fusion = entity.get("expert_fusion", {})
        rarity = 1 / np.sqrt(counts[(row["domain"],row["scale_bin"],row["density_bin"],row["error_category"])])
        knowledge = rule_evidence(graph,row["knowledge_id"],row["domain"],row.get("video"))
        result.append({**row, **knowledge, "entity_id":row["sample_id"],
                       "uncertainty":1-row["quality"],
                       "disagreement":float(fusion.get("disagreement") or 0),
                       "rarity":float(rarity), "review_cost":entity.get("review_cost",1),
                       "cross_layer_value":.3 if row["layer"]!="L1" else 0,
                       "behavior_value":float(min(1,behavior_entities[row["sample_id"]]/3)),
                       "features":[row["error"],row["quality"],row.get("scale",0),
                                   min(row.get("density",0)/200,3)],
                       "bbox_xyxy":entity.get("bbox_xyxy", row.get("reference_bbox"))})
    return result


def feedback_targets(records, error_slices, distribution_config=None):
    """应用分布、评价覆盖与误差共同形成下一轮边际目标，交给现有 IPF/ESS。"""
    cfg = distribution_config or {}
    application, benchmark = cfg.get("application",{}), cfg.get("benchmark",{})
    targets = {}
    axes = cfg.get("axes",["domain","video","scale_bin","density_bin","occlusion_bin","orientation_bin","quality_bin"])
    for axis in axes:
        freq = Counter(str(r[axis]) for r in records if axis in r)
        if not freq:
            continue
        empirical = {k:v/sum(freq.values()) for k,v in freq.items()}
        errors = defaultdict(list)
        for row in error_slices:
            if str(row.get(axis)) in freq:
                errors[str(row[axis])].append(row["mean_error"])
        targets[axis] = target_distribution(
            application.get(axis,empirical),benchmark.get(axis,empirical),
            {k:float(np.mean(v)) for k,v in errors.items()},
            cfg.get("alpha",1),cfg.get("beta",.3),cfg.get("gamma",.5))
    return targets
