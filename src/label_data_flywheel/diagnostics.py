import numpy as np
from .geometry import pose, overlap


def wasserstein_1d(a, b):
    a = np.sort(a)
    b = np.sort(b)
    if not len(a) or not len(b):
        return None
    grid = np.sort(np.r_[a, b])
    delta = np.diff(grid)
    return float(
        np.sum(
            delta
            * np.abs(
                np.searchsorted(a, grid[:-1], side="right") / len(a)
                - np.searchsorted(b, grid[:-1], side="right") / len(b)
            )
        )
    )


def kuiper(a, b):
    a = np.sort(np.mod(a, 2 * np.pi))
    b = np.sort(np.mod(b, 2 * np.pi))
    if not len(a) or not len(b):
        return None
    grid = np.sort(np.r_[a, b])
    d = np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(
        b, grid, side="right"
    ) / len(b)
    return float(d.max() - d.min())


def circular_wasserstein(a, b, bins=72):
    if not len(a) or not len(b):
        return None
    p = np.histogram(np.mod(a, 2 * np.pi), bins=bins, range=(0, 2 * np.pi))[0] / len(a)
    q = np.histogram(np.mod(b, 2 * np.pi), bins=bins, range=(0, 2 * np.pi))[0] / len(b)
    c = np.cumsum(p - q)
    return float(np.abs(c - np.median(c)).sum() * 2 * np.pi / bins)


def mmd(a, b, bandwidth=1.0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    def kernel(x, y):
        return np.exp(-((x[:, None] - y[None, :]) ** 2).sum(2) / (2 * bandwidth**2))

    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean())


def sample_attributes(frames):
    rows = []
    for f in frames:
        boxes = [d["bbox_xyxy"] for d in f["detections"]]
        om, _, _ = overlap(boxes, boxes)
        if len(boxes):
            np.fill_diagonal(om, 0)
        for i, d in enumerate(f["detections"]):
            b = np.asarray(d["bbox_xyxy"])
            p = pose(d)
            scale = float(np.sqrt(np.prod(b[2:] - b[:2]) / np.prod(f["image_size"])))
            rows.append(
                {
                    "sample_id": d["entity_id"],
                    "domain": f["domain"],
                    "video": f["video"],
                    "group": f["group"],
                    "frame": f["frame"],
                    "layer": "L2" if p else "L1",
                    "scale": scale,
                    "density": len(boxes),
                    "occlusion_proxy": float(om[i].max(initial=0)),
                    "direction": p["angle"] if p else None,
                    "quality": d.get("quality", 1.0),
                    "status": d.get("label_status", "unconfirmed"),
                    "track_ids": [d["track_id"]]
                    if d.get("track_id") is not None
                    else [],
                    "segment": f.get("segment", f["video"]),
                    "split": f["split"],
                    "supervision_mask": d.get("supervision_mask", {}),
                }
            )
    return rows
