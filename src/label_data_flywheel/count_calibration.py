"""按域、密度、遮挡、尺度校准召回；保留原始数与校正数。"""

from collections import defaultdict


def slice_key(row):
    return "|".join(
        str(row.get(k, "all"))
        for k in ("domain", "density_bin", "occlusion_bin", "scale_bin")
    )


def fit_recall(rows, prior_strength=10.0):
    groups = defaultdict(list)
    domains = defaultdict(list)
    for r in rows:
        if r["split"] != "calibration" or not r.get("human_verified"):
            raise ValueError("召回校准必须使用人工calibration参考")
        groups[slice_key(r)].append(r)
        domains[r["domain"]].append(r)
    result = {}
    for key, records in groups.items():
        domain = records[0]["domain"]
        population = domains[domain]
        prior = sum(r["true_positive"] for r in population) / max(
            sum(r["ground_truth"] for r in population), 1
        )
        tp = sum(r["true_positive"] for r in records)
        gt = sum(r["ground_truth"] for r in records)
        recall = (tp + prior_strength * prior) / (gt + prior_strength)
        result[key] = {
            "recall": float(recall),
            "gt": int(gt),
            "true_positive": int(tp),
            "prior": float(prior),
        }
    return {"slices": result, "fit_split": "calibration"}


def corrected_count(detections, model, context):
    estimate = 0.0
    missing = 0
    for d in detections:
        row = {
            **context,
            **{
                k: d[k] for k in ("density_bin", "occlusion_bin", "scale_bin") if k in d
            },
        }
        entry = model["slices"].get(slice_key(row))
        p = d.get("calibrated_probability")
        if entry is None or p is None or entry["recall"] <= 0:
            missing += 1
            continue
        estimate += p / entry["recall"]
    return {
        "raw_count": len(detections),
        "calibrated_count": float(estimate) if missing == 0 else None,
        "missing_calibration_instances": missing,
        "calibration_complete": missing == 0,
    }
