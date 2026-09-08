"""分层评价与分组报告；预测不能作为GT自动评分。"""

from collections import defaultdict
import numpy as np
from .geometry import overlap
from .quality import head_tail_evidence


def detection_metrics(gt_frames, pred_frames, thresholds=None):
    thresholds = np.arange(0.5, 0.96, 0.05) if thresholds is None else thresholds
    # 用视频/源帧对应，群组可不同；一次评价限定同一GT群组。
    pred = {(f["domain"], f["video"], f["frame"]): f for f in pred_frames}
    total = sum(len(f["detections"]) for f in gt_frames)
    aps = []
    summary = {}
    for threshold in thresholds:
        scored = []
        for f in gt_frames:
            candidates = pred.get((f["domain"], f["video"], f["frame"]), {}).get(
                "detections", []
            )
            candidates = sorted(
                candidates,
                key=lambda d: (
                    d.get("confidence") if d.get("confidence") is not None else 1.0
                ),
                reverse=True,
            )
            _, _, ious = overlap(
                [d["bbox_xyxy"] for d in candidates],
                [d["bbox_xyxy"] for d in f["detections"]],
            )
            matched = set()
            for i, d in enumerate(candidates):
                eligible = [
                    j
                    for j in np.argsort(-ious[i])
                    if j not in matched and ious[i, j] >= threshold
                ]
                hit = bool(eligible)
                if hit:
                    matched.add(int(eligible[0]))
                scored.append(
                    (
                        d.get("confidence") if d.get("confidence") is not None else 1.0,
                        hit,
                    )
                )
        scored.sort(key=lambda x: -x[0])
        tp = np.cumsum([s[1] for s in scored])
        fp = np.arange(1, len(tp) + 1) - tp
        recall = tp / max(total, 1)
        precision = tp / np.maximum(tp + fp, 1)
        ap = (
            float(
                np.mean(
                    [
                        precision[recall >= r].max(initial=0)
                        for r in np.linspace(0, 1, 101)
                    ]
                )
            )
            if len(tp)
            else 0.0
        )
        aps.append(ap)
        if abs(threshold - 0.5) < 1e-5:
            summary = {
                "AP50": ap,
                "Recall50": float(recall[-1]) if len(tp) else 0.0,
                "FP50": int(fp[-1]) if len(fp) else 0,
                "GT": total,
            }
    return {
        **summary,
        "mAP50_95": float(np.mean(aps)),
        "definition": "single-class 101-point interpolated AP, no COCO crowd/area/maxDet filters",
        "independent_test": False,
    }


def pose_metrics(pairs):
    evidence = [head_tail_evidence(a, b, b["bbox_xyxy"]) for a, b in pairs]
    good = [r for r in evidence if "nme_normal" in r]
    if not good:
        return {"instances": 0}
    nme = np.array([r["nme_normal"] for r in good])
    angle = np.array([r["angle_error_deg"] for r in good])
    endpoint_errors = []
    for pred, reference in pairs:
        diagonal = max(
            float(
                np.linalg.norm(
                    np.asarray(reference["bbox_xyxy"])[2:] - reference["bbox_xyxy"][:2]
                )
            ),
            1e-6,
        )
        for name in ("head", "abdomen_tip"):
            p = pred.get("keypoints", {}).get(name)
            r = reference.get("keypoints", {}).get(name)
            if r is not None and (len(r) < 3 or r[2] > 0):
                endpoint_errors.append(
                    float(np.linalg.norm(np.asarray(p[:2]) - r[:2]) / diagonal)
                    if p is not None
                    else float("inf")
                )
    return {
        "instances": len(good),
        "NME_bbox_diagonal": float(nme.mean()),
        "PCK_pair_mean_0.1": float((nme < 0.1).mean()),
        "PCK_endpoint_0.1": float(np.mean(np.asarray(endpoint_errors) < 0.1))
        if endpoint_errors
        else None,
        "PCK_endpoint_0.2": float(np.mean(np.asarray(endpoint_errors) < 0.2))
        if endpoint_errors
        else None,
        "angle_MAE_degrees": float(angle.mean()),
        "head_tail_swap_candidate_rate": sum(
            r["action"] == "swap_candidate" for r in good
        )
        / len(good),
    }


def tracking_metrics(data):
    # 调用官方TrackEval，避免把几何变化误称IDSW或重新定义IDF1。
    from trackeval.metrics import HOTA, CLEAR, Identity
    from trackeval.metrics import hota, identity

    class LegacyNumpy:
        # 上游使用被 NumPy 1.24 移除的类型别名。仅替换两个指标模块的
        # np 引用，不修改全局 numpy、匹配逻辑、阈值或指标公式。
        float = float
        int = int

        def __getattr__(self, name):
            return getattr(np, name)

    hota.np = identity.np = LegacyNumpy()

    result = {}
    for metric in (
        HOTA(),
        CLEAR({"PRINT_CONFIG": False}),
        Identity({"PRINT_CONFIG": False}),
    ):
        values = metric.eval_sequence(data)
        result[metric.get_name()] = {
            k: float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)
            for k, v in values.items()
        }
    return result


def count_metrics(gt, pred):
    error = np.asarray(pred) - gt
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
    }


def tracking_sequence(gt_frames, pred_frames):
    """TrackEval输入，ID按区段映射，评测只覆盖人工标注源帧。"""
    gt_ids = {}
    pd_ids = {}
    gi = []
    pi = []
    similarity = []
    pred = {(f["domain"], f["video"], f["frame"]): f for f in pred_frames}
    for f in sorted(gt_frames, key=lambda f: f["frame"]):
        g = [d for d in f["detections"] if d.get("track_id") is not None]
        p = [
            d
            for d in pred.get((f["domain"], f["video"], f["frame"]), {}).get(
                "detections", []
            )
            if d.get("track_id") is not None
        ]
        for d in g:
            gt_ids.setdefault(str(d["track_id"]), len(gt_ids))
        for d in p:
            pd_ids.setdefault(str(d["track_id"]), len(pd_ids))
        gi.append(np.array([gt_ids[str(d["track_id"])] for d in g], dtype=int))
        pi.append(np.array([pd_ids[str(d["track_id"])] for d in p], dtype=int))
        similarity.append(
            overlap([d["bbox_xyxy"] for d in g], [d["bbox_xyxy"] for d in p])[2]
        )
    return {
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(pd_ids),
        "num_gt_dets": sum(map(len, gi)),
        "num_tracker_dets": sum(map(len, pi)),
        "num_timesteps": len(gi),
        "gt_ids": gi,
        "tracker_ids": pi,
        "similarity_scores": similarity,
    }


def behavior_metrics(confirmed, predictions):
    """按共同event_id评价已确认事件；未人工确认的候选没有准确率。"""
    if any(r.get("status") != "human_confirmed" for r in confirmed):
        raise ValueError("行为评价需要人工确认参考")
    lookup = {r["event_id"]: r["event_type"] for r in predictions}
    labels = sorted({r["event_type"] for r in confirmed} | set(lookup.values()))
    results = {}
    known = {r["event_id"] for r in confirmed}
    for label in labels:
        tp = sum(
            r["event_type"] == label and lookup.get(r["event_id"]) == label
            for r in confirmed
        )
        fp = sum(
            r["event_type"] != label and lookup.get(r["event_id"]) == label
            for r in confirmed
        ) + sum(k not in known and v == label for k, v in lookup.items())
        fn = sum(
            r["event_type"] == label and lookup.get(r["event_id"]) != label
            for r in confirmed
        )
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
    return {
        "per_class": results,
        "MacroF1": float(np.mean([v["f1"] for v in results.values()]))
        if results
        else None,
        "evidence_chain_completeness": sum(
            bool(p.get("upstream_entities")) and bool(p.get("knowledge_source"))
            for p in predictions
        )
        / max(len(predictions), 1),
    }


def evaluate(gt_frames, pred_frames, with_tracking=False):
    from .calibration import bootstrap_interval
    from scipy.optimize import linear_sum_assignment

    if any(
        d.get("label_status") not in ("human", "human_confirmed")
        for f in gt_frames
        for d in f["detections"]
    ):
        raise ValueError("GT含未确认标签，不能用预测给自身评分")
    groups = defaultdict(list)
    for f in gt_frames:
        groups[
            (f["domain"], f["video"], f["group"], f.get("segment", f["group"]))
        ].append(f)
    reports = {}
    pred_map = {(f["domain"], f["video"], f["frame"]): f for f in pred_frames}
    for key, frames in groups.items():
        result = detection_metrics(frames, pred_frames)
        pairs = []
        gt_counts = []
        pd_counts = []
        for f in frames:
            g = f["detections"]
            p = pred_map.get((f["domain"], f["video"], f["frame"]), {}).get(
                "detections", []
            )
            iou = overlap([d["bbox_xyxy"] for d in g], [d["bbox_xyxy"] for d in p])[2]
            if iou.size:
                a, b = linear_sum_assignment(1 - iou)
                pairs.extend((p[j], g[i]) for i, j in zip(a, b) if iou[i, j] >= 0.5)
            gt_counts.append(len(g))
            pd_counts.append(len(p))
        result["pose"] = pose_metrics(pairs)
        result["count"] = count_metrics(gt_counts, pd_counts)
        if with_tracking:
            result["tracking"] = tracking_metrics(
                tracking_sequence(frames, pred_frames)
            )
        reports["|".join(map(str, key))] = result
    values = [r["Recall50"] for r in reports.values()]
    return {
        "groups": reports,
        "worst_group_recall": min(values) if values else None,
        "group_recall_bootstrap_interval": bootstrap_interval(values),
        "independent_test": False,
        "protocol": "source-frame-aligned; group/segment-scoped IDs; IoU0.5 tracking; bbox-diagonal pose normalization",
    }
