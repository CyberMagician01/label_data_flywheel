import numpy as np


def overlap(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    inter = np.maximum(
        np.minimum(a[:, None, 2:], b[None, :, 2:])
        - np.maximum(a[:, None, :2], b[None, :, :2]),
        0,
    ).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return (
        inter / np.maximum(np.minimum(aa[:, None], bb[None, :]), 1e-9),
        inter,
        inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9),
    )


def center(d):
    b = np.asarray(d["bbox_xyxy"])
    return (b[:2] + b[2:]) / 2


def pose(d):
    k = d.get("keypoints", {})
    h = k.get("head")
    t = k.get("abdomen_tip", k.get("tail"))
    if h is None or t is None:
        return None
    delta = np.asarray(h[:2]) - t[:2]
    length = float(np.linalg.norm(delta))
    return {
        "head": h,
        "tail": t,
        "length": length,
        "angle": float(np.arctan2(delta[1], delta[0])),
        "confidence": float(min(h[2], t[2])) if len(h) > 2 and len(t) > 2 else None,
    }


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
