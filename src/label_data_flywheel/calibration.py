"""仅用calibration集合拟合；返回可序列化校准参数。"""

import numpy as np


def fit_isotonic(scores, correct, split="calibration"):
    if split != "calibration":
        raise ValueError("禁止用test集拟合校准器")
    x = np.asarray(scores, float)
    y = np.asarray(correct, float)
    order = np.argsort(x)
    blocks = []
    for i in order:
        blocks.append([float(x[i]), float(x[i]), float(y[i]), 1])
        while len(blocks) > 1 and (
            blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]
            or blocks[-2][1] == blocks[-1][0]
        ):
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    return {
        "upper": [b[1] for b in blocks],
        "probability": [b[2] / b[3] for b in blocks],
        "samples": len(x),
        "fit_split": split,
    }


def calibrated(scores, model):
    if not model["upper"]:
        raise ValueError("空校准器")
    ix = np.searchsorted(model["upper"], scores, side="left").clip(
        0, len(model["upper"]) - 1
    )
    return np.asarray(model["probability"])[ix]


def bootstrap_interval(values, repeats=500, seed=3407):
    x = np.asarray(values, float)
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(repeats)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def pose_thresholds(nme, angle, confidence):
    return {
        "nme": min(float(np.quantile(nme, 0.9)), 0.25),
        "angle_degrees": float(np.clip(np.quantile(angle, 0.9), 30, 60)),
        "confidence": max(float(np.quantile(confidence, 0.1)), 0.2),
    }
