"""保留既有最优版本；只有相同评测协议的候选能够被提升。"""

import copy
from .io import write_json


def choose_champion(champion, candidate, directions, tolerances=None):
    if candidate.get("evaluation_status") != "verified":
        return champion, {"promoted": False, "reason": "candidate_not_verified"}
    for field in ("dataset_sha256", "split", "protocol_id"):
        if not candidate.get(field) or not champion.get(field):
            return champion, {"promoted": False, "reason": "missing_" + field}
        if candidate.get(field) != champion.get(field):
            return champion, {"promoted": False, "reason": "incomparable_" + field}
    tolerances = tolerances or {}
    changes = {}
    for metric, direction in directions.items():
        if metric not in candidate["metrics"] or metric not in champion["metrics"]:
            return champion, {"promoted": False, "reason": "missing_metric_" + metric}
        gain = (candidate["metrics"][metric] - champion["metrics"][metric]) * (
            1 if direction == "max" else -1
        )
        changes[metric] = gain
        if gain < -tolerances.get(metric, 0):
            return champion, {
                "promoted": False,
                "reason": "regression_" + metric,
                "gains": changes,
            }
    improved = any(v > tolerances.get(k, 0) for k, v in changes.items())
    return (copy.deepcopy(candidate) if improved else champion), {
        "promoted": improved,
        "reason": "pareto_improvement" if improved else "no_material_gain",
        "gains": changes,
    }


def create_release(path, manifest):
    # 发布使用新目录；不覆盖原release或冠军指针。
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError("发布目录已存在，请使用新的版本名称")
    path.mkdir(parents=True)
    write_json(path / "manifest.json", manifest)
