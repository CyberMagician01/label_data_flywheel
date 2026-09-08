"""流式群体统计：观测、推断与蜂学解释分层，绝不修改输入标注。"""

from collections import defaultdict, deque
import math
import numpy as np
from .geometry import center, pose
from .postprocess import is_fill
from .density import count_preserving_map


DEFAULTS = {
    "window_seconds": 10,
    "grid_size": [64, 36],
    "kde_sigma_cells": 1.2,
    "contact_body_lengths": 1.5,
    "contact_min_seconds": 0.3,
    "motion_lag_seconds": 0.3,
    "max_gap_seconds": 0.15,
    "active_speed_bl_s": 0.2,
    "max_speed_bl_s": 20,
    "pose_min_confidence": 0.05,
    "baseline_windows": 6,
    "change_mad_z": 4,
    "min_window_coverage": 0.9,
}


class EntranceGate:
    """有限有向线段 + 死区 + 连续确认；缺帧、跳变不跨越补计。"""

    def __init__(self, config, size, fps):
        self.config, self.fps = config, fps
        self.a, self.b = np.asarray(config["line"], float) * size
        self.v = self.b - self.a
        self.length = float(np.linalg.norm(self.v))
        if self.length == 0:
            raise ValueError("巢口线的两个端点不能相同")
        self.deadband = config.get("deadband_pixels", 5)
        self.confirm = max(1, math.ceil(config.get("confirm_seconds", 0.1) * fps))
        self.memory = {}

    def update(self, track, frame, point, body_length):
        offset = point - self.a
        distance = float((self.v[0] * offset[1] - self.v[1] * offset[0]) / self.length)
        side = 0 if abs(distance) <= self.deadband else (1 if distance > 0 else -1)
        state = self.memory.get(track)
        if state is not None:
            dt = (frame - state["last_frame"]) / self.fps
            jump = np.linalg.norm(point - state["last_point"]) / max(body_length, 1)
            if (
                dt > self.config.get("max_gap_seconds", 0.15)
                or jump > self.config.get("max_speed_bl_s", 20) * dt
            ):
                state = None
        if state is None:
            state = {"side": 0, "pending": side, "count": 0, "last_event": -1e12}
            self.memory[track] = state
        state.update(last_frame=frame, last_point=point.copy())
        if not side:
            state["pending"], state["count"] = 0, 0
            return None
        if state["pending"] != side:
            state["pending"], state["count"] = side, 0
        state["count"] += 1
        if state["count"] < self.confirm:
            return None
        event = None
        if state["side"] and side != state["side"]:
            anchor = state["anchor"]
            offset = anchor - self.a
            da = float((self.v[0] * offset[1] - self.v[1] * offset[0]) / self.length)
            crossing = anchor + (point - anchor) * da / (da - distance)
            along = float(np.dot(crossing - self.a, self.v) / self.length**2)
            if 0 <= along <= 1 and (
                frame - state["last_event"]
            ) / self.fps >= self.config.get("cooldown_seconds", 0.5):
                direction = "in" if side == self.config.get("inside_sign", 1) else "out"
                event = {
                    "frame": frame,
                    "track_id": track,
                    "direction": direction,
                    "crossing_xy": crossing.tolist(),
                    "confirmed_after_frames": self.confirm,
                }
                state["last_event"] = frame
        state["side"], state["anchor"] = side, point.copy()
        return event


def _new_window(index, cfg):
    gw, gh = cfg["grid_size"]
    return {
        "index": index,
        "n": 0,
        "counts": [],
        "fills": 0,
        "hidden": 0,
        "exposure": 0.0,
        "motion_n": 0,
        "tracked_n": 0,
        "objects": 0,
        "active": 0,
        "speeds": [],
        "jumps": 0,
        "new_ids": 0,
        "heat": np.zeros((gh, gw)),
        "edges": defaultdict(float),
        "nodes": {},
        "flow": {"in": 0, "out": 0},
        "first_frame": None,
        "last_frame": None,
    }


def _network(window):
    nodes = window["nodes"]
    neighbors = {k: set() for k in nodes}
    strength = defaultdict(float)
    edges = []
    for (a, b), seconds in sorted(window["edges"].items()):
        neighbors[a].add(b)
        neighbors[b].add(a)
        strength[a] += seconds
        strength[b] += seconds
        edges.append({"source": a, "target": b, "qualified_proximity_seconds": seconds})
    seen, components = set(), []
    for n in nodes:
        if n in seen:
            continue
        pending, count = [n], 0
        seen.add(n)
        while pending:
            current = pending.pop()
            count += 1
            for other in neighbors[current] - seen:
                seen.add(other)
                pending.append(other)
        components.append(count)
    return {
        "nodes": [
            {
                "id": n,
                "position_normalized": (row[0] / row[1]).tolist(),
                "strength_seconds": strength[n],
            }
            for n, row in nodes.items()
        ],
        "edges": edges,
        "edge_density": 2 * len(edges) / max(len(nodes) * (len(nodes) - 1), 1),
        "largest_component_fraction": max(components, default=0) / max(len(nodes), 1),
        "meaning": "持续空间近邻候选网络；不等价于饲喂、真实接触或信息传递网络",
    }


def _finish_window(w, cfg, fps, gate_status):
    graph = _network(w)
    heat = w["heat"] / max(w["n"], 1)
    mean = float(heat.mean())
    start = w["first_frame"] / fps
    end = (w["last_frame"] + 1) / fps
    # 首尾部分窗只按实际覆盖的时间段计算；内部缺帧不被补成零流量。
    coverage = w["exposure"] / (end - start)
    duration = w["exposure"]
    return {
        "index": w["index"],
        "start_seconds": start,
        "end_seconds": end,
        "first_frame": w["first_frame"],
        "last_frame": w["last_frame"],
        "frames": w["n"],
        "observed_exposure_seconds": duration,
        "coverage": coverage,
        "mean_observed_count": float(np.mean(w["counts"])),
        "mean_visible_interpolations": w["fills"] / w["n"],
        "mean_hidden_instances": w["hidden"] / w["n"],
        "tracked_fraction": w["tracked_n"] / max(w["objects"], 1),
        "motion_observable_fraction": w["motion_n"] / max(w["objects"], 1),
        "active_fraction": w["active"] / w["motion_n"] if w["motion_n"] else None,
        "median_speed_bl_proxy_s": float(np.median(w["speeds"]))
        if w["speeds"]
        else None,
        "rejected_motion_jumps": w["jumps"],
        "new_ids_after_first_frame": w["new_ids"],
        "density_cv": float(heat.std() / mean) if mean else 0,
        "density_mean_objects_per_cell": heat.tolist(),
        "network": graph,
        "line_crossings": dict(w["flow"]) if gate_status != "unconfigured" else None,
        "entrance_flux_per_minute": {
            "in": w["flow"]["in"] * 60 / duration,
            "out": w["flow"]["out"] * 60 / duration,
            "net_in": (w["flow"]["in"] - w["flow"]["out"]) * 60 / duration,
        }
        if gate_status == "verified_entrance" and coverage >= cfg["min_window_coverage"]
        else None,
    }


def temporal_analysis(windows, config):
    """前向历史基线，不使用未来窗校准当前异常；短片段只报告短时变化。"""
    cfg = {**DEFAULTS, **config}
    names = [
        "mean_observed_count",
        "active_fraction",
        "median_speed_bl_proxy_s",
        "density_cv",
    ]
    changes, summaries = [], {}
    for metric in names:
        rows = [
            (i, w[metric])
            for i, w in enumerate(windows)
            if w[metric] is not None and w["coverage"] >= cfg["min_window_coverage"]
        ]
        for i, w in enumerate(windows):
            value = w[metric]
            prior = [
                v
                for j, v in rows
                if j < i
                and w["index"] - cfg["baseline_windows"]
                <= windows[j]["index"]
                < w["index"]
            ]
            if (
                value is None
                or len(prior) < cfg["baseline_windows"]
                or w["coverage"] < cfg["min_window_coverage"]
            ):
                continue
            median = float(np.median(prior))
            scale = max(
                1.4826 * float(np.median(np.abs(np.asarray(prior) - median))),
                abs(median) * 0.05,
                0.01,
            )
            z = (value - median) / scale
            if abs(z) >= cfg["change_mad_z"]:
                changes.append(
                    {
                        "window_index": w["index"],
                        "metric": metric,
                        "robust_z": z,
                        "baseline_median": median,
                        "value": value,
                        "status": "candidate",
                        "meaning": "同片段历史偏离；阈值是工程复核参数，不是蜂学诊断阈值",
                    }
                )
        values = np.asarray([v for _, v in rows], float)
        result = {
            "valid_windows": len(rows),
            "trend_per_minute": None,
            "autocorrelation": [],
            "period_candidate_seconds": None,
        }
        if len(values) >= 3:
            times = np.asarray([windows[i]["start_seconds"] / 60 for i, _ in rows])
            result["trend_per_minute"] = float(np.polyfit(times, values, 1)[0])
        # 只有完整、等间隔且至少12窗时分析自相关；不跨缺帧缝隙拼出周期。
        regular = (
            len(rows) == len(windows)
            and len(rows) >= 12
            and np.allclose(
                np.diff([w["start_seconds"] for w in windows]), cfg["window_seconds"]
            )
        )
        if regular:
            x = np.arange(len(values))
            residual = values - np.polyval(np.polyfit(x, values, 1), x)
            denom = float(residual @ residual)
            if denom > 1e-10:
                acf = (
                    np.correlate(residual, residual, mode="full")[len(values) - 1 :]
                    / denom
                )
                result["autocorrelation"] = acf[: len(values) // 3 + 1].tolist()
                peaks = [
                    lag
                    for lag in range(2, len(values) // 3)
                    if acf[lag] > 0.5
                    and acf[lag] > acf[lag - 1]
                    and acf[lag] >= acf[lag + 1]
                ]
                if peaks:
                    result["period_candidate_seconds"] = (
                        max(peaks, key=lambda lag: acf[lag]) * cfg["window_seconds"]
                    )
        summaries[metric] = result
    return {
        "metrics": summaries,
        "change_candidates": changes,
        "inference_scope": "片段内短时变化；昼夜节律、季节规律及生产预警性能须另有长时记录与事件真值",
    }


def analyze_colony(frames, config=None):
    """输入可为生成器；内存随当前窗和活跃近邻增长，不保存逐帧框副本。"""
    from scipy.spatial import cKDTree

    cfg = {**DEFAULTS, **(config or {})}
    if cfg["window_seconds"] <= 0:
        raise ValueError("时间窗必须大于零")
    scopes = {}
    for frame in frames:
        scope = (frame["domain"], frame["video"], frame["group"])
        f, fps = frame["frame"], frame["fps"]
        size = np.asarray(frame["image_size"], float)
        if fps <= 0 or np.any(size <= 0):
            raise ValueError("帧率与图像尺寸必须为正")
        if scope not in scopes:
            gate_config = cfg.get("entrance")
            scopes[scope] = {
                "fps": fps,
                "size": size,
                "first": f,
                "last": f - 1,
                "window": None,
                "windows": [],
                "history": {},
                "seen": set(),
                "pairs": {},
                "events": [],
                "gate": EntranceGate(gate_config, size, fps) if gate_config else None,
                "gate_status": "verified_entrance"
                if gate_config and gate_config.get("calibration_status") == "verified"
                else "reference_line_only"
                if gate_config
                else "unconfigured",
            }
        s = scopes[scope]
        if f <= s["last"] or fps != s["fps"] or not np.array_equal(size, s["size"]):
            raise ValueError("每个域/视频/标注组须按源帧严格递增，帧率与尺寸一致")
        s["last"] = f
        wi = int((f - s["first"]) / fps / cfg["window_seconds"])
        if s["window"] is None or wi != s["window"]["index"]:
            if s["window"] is not None:
                s["windows"].append(
                    _finish_window(s["window"], cfg, fps, s["gate_status"])
                )
            s["window"] = _new_window(wi, cfg)
        w = s["window"]
        if w["first_frame"] is None:
            w["first_frame"] = f
        w["last_frame"], w["n"] = f, w["n"] + 1
        w["exposure"] += 1 / fps
        ds = [d for d in frame["detections"] if not is_fill(d)]
        w["counts"].append(len(ds))
        w["objects"] += len(ds)
        w["fills"] += len(frame["detections"]) - len(ds)
        w["hidden"] += len(frame.get("temporarily_hidden_detections", []))
        points, lengths, tracks = [], [], []
        all_points = [np.clip(center(d) / size, 0, 1) for d in ds]
        gw, gh = cfg["grid_size"]
        w["heat"] += count_preserving_map(
            [p * [gw - 1, gh - 1] for p in all_points], (gw, gh), cfg["kde_sigma_cells"]
        )
        for d in ds:
            tid = d.get("track_id")
            if tid is None:
                continue
            tid = str(tid)
            p = center(d)
            body = pose(d)
            b = d["bbox_xyxy"]
            length = (
                body["length"]
                if body
                and (body["confidence"] or 0) >= cfg["pose_min_confidence"]
                and body["length"] > 1
                else max(b[2] - b[0], b[3] - b[1])
            )
            w["tracked_n"] += 1
            if tid not in s["seen"]:
                if f > s["first"]:
                    w["new_ids"] += 1
                s["seen"].add(tid)
            if tid not in w["nodes"]:
                w["nodes"][tid] = [np.zeros(2), 0]
            w["nodes"][tid][0] += p / size
            w["nodes"][tid][1] += 1
            history = s["history"].setdefault(tid, deque())
            if history and (f - history[-1][0]) / fps > cfg["max_gap_seconds"]:
                history.clear()
            history.append((f, p, length))
            lag = max(1, round(cfg["motion_lag_seconds"] * fps))
            while len(history) > 1 and history[1][0] <= f - lag:
                history.popleft()
            if f - history[0][0] >= lag:
                previous, old_point, old_length = history[0]
                speed = float(
                    np.linalg.norm(p - old_point)
                    / max((length + old_length) / 2, 1)
                    / ((f - previous) / fps)
                )
                if speed <= cfg["max_speed_bl_s"]:
                    w["speeds"].append(speed)
                    w["motion_n"] += 1
                    w["active"] += speed >= cfg["active_speed_bl_s"]
                else:
                    w["jumps"] += 1
            if s["gate"]:
                event = s["gate"].update(tid, f, p, length)
                if event:
                    event.update(sample_id=frame["sample_id"], status=s["gate_status"])
                    s["events"].append(event)
                    w["flow"][event["direction"]] += 1
            points.append(p)
            lengths.append(length)
            tracks.append(tid)
        pairs = {}
        if len(points) > 1:
            points, lengths = np.asarray(points), np.asarray(lengths)
            tree = cKDTree(points)
            for i, j in tree.query_pairs(
                float(max(lengths) * cfg["contact_body_lengths"])
            ):
                if (
                    np.linalg.norm(points[i] - points[j])
                    > cfg["contact_body_lengths"] * (lengths[i] + lengths[j]) / 2
                ):
                    continue
                key = tuple(sorted((tracks[i], tracks[j])))
                old = s["pairs"].get(key)
                continuous = old and (f - old[1]) / fps <= cfg["max_gap_seconds"]
                start = old[0] if continuous else f
                pairs[key] = (start, f)
                if (f - start + 1) / fps >= cfg["contact_min_seconds"]:
                    w["edges"][key] += 1 / fps
        s["pairs"] = pairs
        if f % max(1, int(fps)) == 0:
            s["history"] = {
                k: h
                for k, h in s["history"].items()
                if (f - h[-1][0]) / fps <= cfg["max_gap_seconds"]
            }
            if s["gate"]:
                s["gate"].memory = {
                    k: v
                    for k, v in s["gate"].memory.items()
                    if (f - v["last_frame"]) / fps <= cfg["max_gap_seconds"]
                }
    results = []
    for scope, s in scopes.items():
        s["windows"].append(
            _finish_window(s["window"], cfg, s["fps"], s["gate_status"])
        )
        windows = s["windows"]
        heat = sum(
            np.asarray(w["density_mean_objects_per_cell"]) * w["frames"]
            for w in windows
        ) / sum(w["frames"] for w in windows)
        results.append(
            {
                "domain": scope[0],
                "video": scope[1],
                "group": scope[2],
                "fps": s["fps"],
                "image_size": s["size"].tolist(),
                "source_frame_range": [s["first"], s["last"]],
                "unique_track_ids": len(s["seen"]),
                "gate_status": s["gate_status"],
                "windows": windows,
                "crossing_events": s["events"],
                "mean_density_map": heat.tolist(),
                "temporal": temporal_analysis(windows, cfg),
            }
        )
    if not results:
        raise ValueError("没有可分析帧")
    return {
        "schema_version": "colony-1.0",
        "config": cfg,
        "scopes": results,
        "units": {
            "density": "每格平均观测目标数；积分等于平均观测数",
            "speed": "身体长度代理/秒；低可信姿态时用框长边",
            "flow": "已核验巢口的观测穿越次数/分钟，非采集成功率",
        },
        "input_policy": "只用可见原观测计算行为；插值/隐藏独立统计；原框、ID和最优后处理不改写",
    }
