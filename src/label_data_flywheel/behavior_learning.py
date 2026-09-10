"""人工确认后才启用的L4监督学习；四个并行状态头分别掩码。"""

from pathlib import Path
import numpy as np

STATES = {
    "motion": ["stationary", "walking", "fast_moving", "unknown"],
    "region": ["image", "entrance_inside", "entrance_outside", "region_unknown"],
    "interaction": ["none", "approach", "directed_interaction", "following", "unknown"],
    "rhythm": ["no_significant_rhythm", "axis_oscillation", "unknown"],
}
FEATURES = (
    "speed_bl_per_source_frame",
    "angular_speed_rad_per_source_frame",
    "body_length",
    "normalized_distance",
    "co_motion",
    "spectral_concentration",
)


def feature_vector(record, features=FEATURES):
    values = [record.get(k) for k in features]
    # 缺失指示与数值分离，不把未观测速度当成静止。
    return [0.0 if x is None else float(x) for x in values] + [
        float(x is not None) for x in values
    ]


def build_model(states=None, features=FEATURES):
    from torch import nn
    states = states or STATES

    class BehaviorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(len(features) * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            self.heads = nn.ModuleDict(
                {k: nn.Linear(32, len(v)) for k, v in states.items()}
            )

        def forward(self, x):
            features = self.shared(x)
            return {k: head(features) for k, head in self.heads.items()}

    return BehaviorModel()


def train(records, output, epochs=100, device="cpu", seed=3407, states=None, features=FEATURES):
    import torch
    from torch.nn import functional as F
    states = states or STATES

    if not records or any(
        r.get("status") != "human_confirmed" or r.get("split") != "train"
        for r in records
    ):
        raise ValueError("行为训练只接受人工确认的train事件；当前未确认事件不能训练")
    path = Path(output)
    if path.exists():
        raise FileExistsError("行为模型使用新版本路径")
    torch.manual_seed(seed)
    raw = np.asarray([feature_vector(r, features) for r in records], np.float32)
    mean = raw.mean(0)
    std = np.maximum(raw.std(0), 1e-3)
    x = torch.tensor((raw - mean) / std, device=device)
    labels = {
        k: torch.tensor(
            [
                states[k].index(r["labels"][k]) if k in r.get("labels", {}) else -1
                for r in records
            ],
            device=device,
        )
        for k in states
    }
    if not any((v >= 0).any() for v in labels.values()):
        raise ValueError("没有可用行为监督")
    model = build_model(states, features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = x.sum() * 0
        for k, target in labels.items():
            mask = target >= 0
            if mask.any():
                loss = loss + F.cross_entropy(logits[k][mask], target[mask])
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "states": states,
            "features": features,
            "training_samples": len(records),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return {
        "samples": len(records),
        "epochs": epochs,
        "loss_history": history,
        "checkpoint": str(path),
        "benchmark_score": None,
    }


def predict(records, checkpoint, device="cpu"):
    import torch

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    states, features = state["states"], state["features"]
    model = build_model(states, features).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    raw = np.asarray([feature_vector(r, features) for r in records], np.float32)
    x = torch.tensor(
        (raw - state["feature_mean"]) / np.asarray(state["feature_std"]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prob = {k: torch.softmax(v, 1).cpu().numpy() for k, v in model(x).items()}
    return [
        {
            **r,
            "predicted_states": {
                k: states[k][int(p[i].argmax())] for k, p in prob.items()
            },
            "state_probabilities": {k: p[i].tolist() for k, p in prob.items()},
            "status": "candidate",
        }
        for i, r in enumerate(records)
    ]


def train_colony(records, output, epochs=100, device="cpu"):
    from .behavior_supervision import COLONY_FEATURES
    if any(r.get("supervision_unit") != "group_window" for r in records):
        raise ValueError("群体训练只接受群体窗口监督")
    labels = sorted({r["labels"]["colony"] for r in records})
    if len(labels) < 2:
        raise ValueError("群体分类至少需要两类已确认训练窗口")
    return train(records, output, epochs, device, states={"colony": labels}, features=COLONY_FEATURES)
