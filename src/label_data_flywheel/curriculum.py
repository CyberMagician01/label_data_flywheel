"""六阶段单一权重谱系；平台判定、召回债务和覆盖门共同控制阶段切换。"""

from dataclasses import dataclass, field, asdict
import numpy as np
from .calibration import bootstrap_interval

STAGES = (
    "semantic_calibration",
    "spatial_foundation",
    "structure_joint",
    "temporal_density",
    "recall_reinforcement",
    "distribution_return",
)


def theil_sen(values):
    x = np.asarray(values, float)
    return (
        float(
            np.median(
                [
                    (x[j] - x[i]) / (j - i)
                    for i in range(len(x))
                    for j in range(i + 1, len(x))
                ]
            )
        )
        if len(x) > 1
        else float("inf")
    )


@dataclass
class Curriculum:
    stage: int = 0
    updates: int = 0
    history: list = field(default_factory=list)
    stage_updates: int = 0
    min_updates: int = 5
    plateau_window: int = 5
    tolerance: float = 0.002

    def controls(self, route="Y"):
        s = self.stage
        ramp = min(1.0, self.stage_updates / max(self.min_updates, 1))
        return {
            "stage": STAGES[s],
            "route": route,
            "resolution_scale": [0.5, 0.75, 1, 1, 1, 1][s],
            "pose_gate": min(1.0, ramp) if s == 0 else 1.0,
            "density_gate": ramp if s == 3 else float(s > 3),
            "temporal_gate": ramp if s == 3 else float(s > 3),
            "btca": s in (3, 4),
            "ema_teacher": s >= 3,
            "soft_positive_weight": ramp if s == 4 else 0.0,
            "public_data_weight": 0.0 if s == 5 else 1.0,
            "strong_augmentation": 0.0 if s == 5 else (1.0 if route == "Y" else 0.25),
            "query_fraction": [0.25, 0.5, 0.75, 1, 1, 1][s] if route == "E" else None,
            "loss_weights": {
                "detection": 1.0,
                "pose": 0.25,
                "density": 0.1 if s >= 3 else 0.0,
                "knowledge": 0.05 if s >= 2 else 0.0,
                "cross_layer": 0.05 if s >= 3 else 0.0,
            },
            "freeze_postprocess": s == 5,
            "domain_balance": "equal_per_optimizer_update",
        }

    def observe(self, by_video, coverage_complete, valid_ess, split="calibration"):
        if split != "calibration":
            raise ValueError("阶段调度只能使用calibration聚合统计")
        self.updates += 1
        self.stage_updates += 1
        self.history.append(
            {
                "stage": self.stage,
                "values": dict(by_video),
                "worst": min(by_video.values()),
                "mean": float(np.mean(list(by_video.values()))),
            }
        )
        hist = [h for h in self.history if h["stage"] == self.stage][
            -self.plateau_window :
        ]
        plateau = (
            len(hist) == self.plateau_window
            and abs(theil_sen([h["mean"] for h in hist])) <= self.tolerance
            and abs(theil_sen([h["worst"] for h in hist])) <= self.tolerance
        )
        interval = bootstrap_interval(list(by_video.values()))
        ready = (
            plateau
            and coverage_complete
            and valid_ess
            and self.stage_updates >= self.min_updates
        )
        finished = ready and self.stage == 5
        if ready and self.stage < 5:
            self.stage += 1
            self.stage_updates = 0
        debt = {k: float(max(by_video.values()) - v) for k, v in by_video.items()}
        return {
            "advance": ready and not finished,
            "stop": finished,
            "recall_debt": debt,
            "video_bootstrap_interval": interval,
            "state": asdict(self),
        }


def input_gate(records, batches, weights, min_ess_ratio=0.5):
    from .sampling import ess

    exposed = {i for batch in batches for i in batch}
    valid = {i for i, w in enumerate(weights) if w > 0}
    balanced = all(
        sum(records[i]["domain"] == "RGB_out" for i in b)
        == sum(records[i]["domain"] == "IR_in" for i in b)
        for b in batches
    )
    bad = [
        i
        for b in batches
        for i in b
        if records[i].get("status") in ("suspect", "invalid")
    ]
    return {
        "base_coverage": valid <= exposed,
        "domain_balanced": balanced,
        "invalid_exposures": len(bad),
        "ess_pass": ess(weights) >= min_ess_ratio * len(valid),
        "passed": valid <= exposed
        and balanced
        and not bad
        and ess(weights) >= min_ess_ratio * len(valid),
    }


def meta_policy(error_slices, remaining_review_budget):
    roles = {
        "distribution": [],
        "quality": [],
        "ethogram": [],
        "knowledge": [],
        "model": [],
        "attribution": [],
    }
    for row in error_slices:
        reason = row.get("root_cause", "unknown")
        sample = {
            "slice": row,
            "action": "review" if remaining_review_budget > 0 else "defer_review",
        }
        roles["attribution"].append(sample)
        if row.get("mean_error", 0) > 0.2:
            roles["distribution"].append({"action": "raise_slice_quota", "slice": row})
        roles["quality"].append(sample)
        if row.get("layer") == "L4":
            roles["ethogram"].append(
                {"action": "request_behavior_confirmation", "slice": row}
            )
        if "knowledge" in reason:
            roles["knowledge"].append({"action": "recalibrate_rule", "slice": row})
        if row.get("layer") in ("L1", "L2", "L3"):
            roles["model"].append(
                {"action": "evaluate_candidate_on_calibration", "slice": row}
            )
    return {
        "policy_type": "deterministic_six_role_controller",
        "roles": roles,
        "automatic_human_confirmation": False,
    }
