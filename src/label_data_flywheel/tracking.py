"""备选两阶段关联器：观测速度、Kalman距离、IoU和姿态共同门控。

该可消融实现不替换已发布的appearance20，也不冒称官方ByteTrack/OC-SORT。
"""

import copy
import numpy as np
from .geometry import center, overlap, pose, wrap


class AssociationTracker:
    def __init__(self, retention=150, high=0.5, low=0.1, max_mahalanobis=16.0):
        self.retention = retention
        self.high = high
        self.low = low
        self.gate = max_mahalanobis
        self.tracks = {}
        self.next_id = 1

    def update(self, frame):
        from scipy.optimize import linear_sum_assignment

        f = copy.deepcopy(frame)
        now = f["frame"]
        ds = f["detections"]
        active = {}
        assigned = {}
        for tid, t in self.tracks.items():
            dt = now - t["frame"]
            if dt <= 0:
                raise ValueError("跟踪输入必须按源帧严格递增")
            if dt > self.retention:
                continue
            transition = np.eye(4)
            transition[0, 2] = transition[1, 3] = dt
            x = transition @ t["state"]
            p = transition @ t["cov"] @ transition.T + np.diag(
                [dt, dt, 0.2 * dt, 0.2 * dt]
            )
            active[tid] = {**t, "pred": x, "pred_cov": p}
        unmatched = set(active)
        scores = np.array(
            [
                d.get("confidence") if d.get("confidence") is not None else 1.0
                for d in ds
            ]
        )
        for indices in (
            np.flatnonzero(scores >= self.high),
            np.flatnonzero((scores >= self.low) & (scores < self.high)),
        ):
            tids = sorted(unmatched)
            if not tids or not len(indices):
                continue
            costs = np.full((len(tids), len(indices)), 1e6)
            for a, tid in enumerate(tids):
                t = active[tid]
                p0 = pose(t["detection"])
                inv = np.linalg.inv(t["pred_cov"][:2, :2] + np.eye(2) * 4)
                for b, i in enumerate(indices):
                    residual = center(ds[i]) - t["pred"][:2]
                    mahal = float(residual @ inv @ residual)
                    if mahal > self.gate:
                        continue
                    _, _, iou = overlap(
                        [t["detection"]["bbox_xyxy"]], [ds[i]["bbox_xyxy"]]
                    )
                    p1 = pose(ds[i])
                    pose_cost = 0.0
                    if p0 and p1 and p0["length"] > 0 and p1["length"] > 0:
                        pose_cost = abs(
                            float(wrap(p0["angle"] - p1["angle"]))
                        ) / np.pi + abs(np.log(p1["length"] / p0["length"]))
                    costs[a, b] = (
                        mahal / self.gate + 1 - float(iou[0, 0]) + 0.2 * pose_cost
                    )
            ai, bi = linear_sum_assignment(costs)
            for a, b in zip(ai, bi):
                if costs[a, b] < 1e5:
                    assigned[int(indices[b])] = tids[a]
                    unmatched.remove(tids[a])
        for i, d in enumerate(ds):
            if i not in assigned:
                if scores[i] < self.high:
                    continue
                tid = self.next_id
                self.next_id += 1
                state = np.r_[center(d), 0.0, 0.0]
                cov = np.eye(4) * 10
            else:
                tid = assigned[i]
                t = active[tid]
                dt = now - t["frame"]
                observed_velocity = (center(d) - center(t["detection"])) / dt
                h = np.eye(4)[:2]
                gain = (
                    t["pred_cov"]
                    @ h.T
                    @ np.linalg.inv(h @ t["pred_cov"] @ h.T + np.eye(2) * 4)
                )
                state = t["pred"] + gain @ (center(d) - h @ t["pred"])
                state[2:] = 0.5 * state[2:] + 0.5 * observed_velocity
                cov = (np.eye(4) - gain @ h) @ t["pred_cov"]
            d["source_track_id"] = d.get("track_id")
            d["track_id"] = tid
            self.tracks[tid] = {
                "state": state,
                "cov": cov,
                "frame": now,
                "detection": copy.deepcopy(d),
            }
        self.tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if now - t["frame"] <= self.retention
        }
        f["detections"] = [
            d for i, d in enumerate(ds) if i in assigned or scores[i] >= self.high
        ]
        return f
