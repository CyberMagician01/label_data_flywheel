"""可见性去重与身份关联分离；所有删除决策有来源。"""

import copy
from collections import defaultdict
import numpy as np
from .geometry import overlap, center
from .semantics import is_bee


def is_fill(d):
    return "interpolation" in d.get("origin", "")


def priority(d):
    p = d.get("interpolation", {})
    gap = p.get("right_source_frame", 0) - p.get("left_source_frame", 0)
    if not gap:
        gap = p.get("missing_frames", 0) + 1
    return (
        gap,
        -p.get("observed_length", 0),
        p.get("endpoint_displacement", 0),
        d.get("track_id", 0),
    )


def suppress_interpolation(frame, threshold=0.2, min_pixels=16):
    f = copy.deepcopy(frame)
    obs = [d for d in f["detections"] if not is_fill(d)]
    ins = sorted([d for d in f["detections"] if is_fill(d)], key=priority)
    kept = list(obs)
    hidden = f.setdefault("temporarily_hidden_detections", [])
    for d in ins:
        blockers = [x for x in kept if is_bee(x) and x.get("label_status") != "invalid"]
        om, area, _ = overlap([d["bbox_xyxy"]], [x["bbox_xyxy"] for x in blockers])
        conflict = np.flatnonzero((om[0] >= threshold) & (area[0] >= min_pixels))
        if len(conflict):
            k = int(conflict[np.argmax(om[0, conflict])])
            d["suppression"] = {
                "reason": "overlap",
                "kept_id": blockers[k].get("track_id"),
                "iomin": float(om[0, k]),
                "threshold": threshold,
            }
            hidden.append(d)
        else:
            kept.append(d)
    f["detections"] = kept
    f["counts"] = {
        "observed": len(obs),
        "visible": len(kept),
        "interpolated_kept": len(kept) - len(obs),
        "interpolated_hidden": len(hidden),
    }
    return f


def interpolate(frames, max_gap=90):
    """源帧坐标线性插值；不外推、不给插值框伪造关键点。"""
    result = copy.deepcopy(frames)
    by_frame = {f["frame"]: f for f in result}
    tracks = defaultdict(list)
    scopes = {(f["domain"], f["video"], f["group"]) for f in result}
    if len(scopes) > 1:
        raise ValueError("一次插值只能处理同域同视频同群组")
    for f in result:
        for d in f["detections"]:
            if (d.get("track_id") is not None and not is_fill(d) and is_bee(d)
                    and d.get("label_status") != "invalid"):
                tracks[d["track_id"]].append((f["frame"], d))
    occupied = {
        (f["frame"], d.get("track_id"))
        for f in result
        for d in f["detections"] + f.get("temporarily_hidden_detections", [])
    }
    for tid, rows in tracks.items():
        rows.sort(key=lambda x: x[0])
        for (a, da), (b, db) in zip(rows, rows[1:]):
            if not 1 < b - a <= max_gap:
                continue
            for fr in range(a + 1, b):
                if fr not in by_frame or (fr, tid) in occupied:
                    continue
                t = (fr - a) / (b - a)
                by_frame[fr]["detections"].append(
                    {
                        "track_id": tid,
                        "entity_id": f"{by_frame[fr]['sample_id']}/fill/{tid}",
                        "bbox_xyxy": (
                            (1 - t) * np.asarray(da["bbox_xyxy"])
                            + t * np.asarray(db["bbox_xyxy"])
                        ).tolist(),
                        "keypoints": {},
                        "origin": "linear_interpolation",
                        "label_status": "unconfirmed",
                        "interpolation": {
                            "left_source_frame": a,
                            "right_source_frame": b,
                            "alpha": t,
                            "observed_length": len(rows),
                            "endpoint_displacement": float(
                                np.linalg.norm(center(db) - center(da))
                            ),
                        },
                    }
                )
    return result
