"""BEEG：只输出可观测状态与待复核事件，不把蜂学解释当成人工标签。"""

from collections import defaultdict
import math
import numpy as np
from .geometry import center, pose, wrap
from .postprocess import is_fill
from .semantics import is_bee


def calibrate_motion(speeds):
    x = np.asarray(speeds, float)
    x = x[np.isfinite(x)]
    if not len(x):
        raise ValueError("运动阈值校准缺少有效参考轨迹")
    return {
        "stationary": float(np.quantile(x, 0.2)),
        "fast": float(np.quantile(x, 0.9)),
        "units": "body_length/source_frame",
        "reference_samples": len(x),
    }


def oscillation(rows, min_frames=30):
    valid = [r for r in rows if r.get("angle") is not None]
    if len(valid) < 8 or valid[-1]["frame"] - valid[0]["frame"] < min_frames:
        return {"state": "unknown", "frequency_hz": None}
    fs = np.array([r["frame"] for r in valid])
    angles = np.unwrap([r["angle"] for r in valid])
    if np.diff(fs).max() > 10:
        return {"state": "unknown", "frequency_hz": None}
    grid = np.arange(fs[0], fs[-1] + 1)
    a = np.interp(grid, fs, angles)
    a -= np.polyval(np.polyfit(grid, a, 1), grid)
    power = np.abs(np.fft.rfft(a)) ** 2
    freq = np.fft.rfftfreq(len(a), 1 / valid[0]["fps"])
    power[0] = 0
    peak = int(np.argmax(power))
    ratio = float(power[peak] / max(power.sum(), 1e-9))
    return {
        "state": "axis_oscillation"
        if ratio > 0.45 and a.std() > 0.1
        else "no_significant_rhythm",
        "frequency_hz": float(freq[peak]),
        "spectral_concentration": ratio,
    }


def analyze(frames, config=None):
    cfg = config or {}
    # 插值维持身份联系；运动、计数与关系使用直接观测，保留输入原样。
    frames = [
        {**f, "detections": [
            d for d in f["detections"]
            if is_bee(d) and not is_fill(d) and d.get("label_status") != "invalid"
        ]}
        for f in frames
    ]
    thresholds = cfg.get("motion", {"stationary": 0.01, "fast": 0.2})
    min_duration = cfg.get("min_duration_source_frames", 5)
    max_gap = cfg.get("max_observation_gap", 10)
    tracks = defaultdict(list)
    observations = []
    group_rows = []
    edges = []
    events = []
    prev = {}
    state = {}
    pair_state = {}
    gate = cfg.get("entrance_line")
    recall = cfg.get("recall_by_domain", {})
    for f in sorted(frames, key=lambda x: (x["video"], x["group"], x["frame"])):
        domain = f["domain"]
        video = f["video"]
        scope = (domain, video, f["group"])
        current = []
        thresholds = cfg.get("motion_by_domain", {}).get(
            domain, cfg.get("motion", {"stationary": 0.01, "fast": 0.2})
        )
        for d in f["detections"]:
            tid = d.get("track_id")
            if tid is None:
                continue
            key = (*scope, tid)
            p = pose(d)
            c = center(d)
            b = d["bbox_xyxy"]
            pose_usable = p and (p["confidence"] or 0) >= cfg.get("pose_min_confidence", 0.05)
            length = p["length"] if pose_usable and p["length"] > 0 else max(b[2]-b[0], b[3]-b[1])
            old = prev.get(key)
            df = f["frame"] - old["frame"] if old else 0
            continuous = old is not None and 0 < df <= max_gap
            speed = (
                float(np.linalg.norm(c - old["center"]) / df) if continuous else None
            )
            bl = speed / length if speed is not None and length else None
            desired = (
                "unknown"
                if bl is None
                else (
                    "stationary"
                    if bl <= thresholds["stationary"]
                    else "fast_moving"
                    if bl >= thresholds["fast"]
                    else "walking"
                )
            )
            previous = state.get(
                key, {"state": "unknown", "candidate": desired, "since": f["frame"]}
            )
            # 迟滞：已确定状态在阈值附近保留，候选持续足够源帧才切换。
            if (
                bl is not None
                and previous["state"] == "stationary"
                and bl < thresholds["stationary"] * 1.2
            ):
                desired = "stationary"
            if desired != previous["candidate"]:
                previous.update(candidate=desired, since=f["frame"])
            if desired == "unknown" or f["frame"] - previous["since"] >= min_duration:
                previous["state"] = desired
            state[key] = previous
            norm = c / np.asarray(f["image_size"])
            region = "image"
            cross = 0
            if gate:
                a, z = np.asarray(gate)
                u = z - a
                v = norm - a
                cross = float(u[0] * v[1] - u[1] * v[0])
                if abs(cross) < 1e-12 and old is not None:
                    cross = old.get("gate_side", cross)
                region = "entrance_inside" if cross >= 0 else "entrance_outside"
                if continuous and old.get("gate_side", 0) * cross < 0:
                    events.append(
                        {
                            "event_type": "entrance_crossing",
                            "involved_tracks": [tid],
                            "direction": region,
                            "representative_frames": [old["frame"], f["frame"]],
                            "knowledge_source": "configured_entrance_line",
                            "observability_level": "O3",
                            "evidence_confidence": 0.5,
                            "evidence_metrics": {"line": gate, "source_frame_gap": df},
                            "upstream_entities": [old["entity_id"], d["entity_id"]],
                            "status": "candidate",
                            "context_id": f["context_id"],
                            "hive_id": f["hive_id"],
                            "domain": domain,
                            "video": video,
                            "group": f["group"],
                        }
                    )
            row = {
                "entity_id": d["entity_id"],
                "track_id": tid,
                "domain": domain,
                "video": video,
                "group": f["group"],
                "frame": f["frame"],
                "fps": f["fps"],
                "center": c.tolist(),
                "normalized_center": norm.tolist(),
                "body_length": length,
                "body_length_source": "head_tail" if pose_usable else "bbox_long_side",
                "angle": p["angle"] if pose_usable else None,
                "speed_px_per_source_frame": speed,
                "speed_bl_per_source_frame": bl,
                "speed_bl_per_second": bl * f["fps"] if bl is not None else None,
                "velocity_px_per_source_frame": (
                    (c - np.asarray(old["center"])) / df
                ).tolist()
                if continuous
                else None,
                "angular_speed_rad_per_source_frame": float(
                    wrap(p["angle"] - old["angle"]) / df
                )
                if continuous and pose_usable and old["angle"] is not None
                else None,
                "motion_state": previous["state"],
                "region_state": region,
                "interaction_state": "unknown" if p is None else "none",
                "observability_level": "O2" if continuous else "O1",
                "knowledge_source": "visual_kinematics",
                "evidence_confidence": p["confidence"] if p else None,
                "gate_side": cross,
                "context_id": f["context_id"],
                "hive_id": f["hive_id"],
            }
            angular = row["angular_speed_rad_per_source_frame"]
            row["angular_speed_rad_per_second"] = angular * f["fps"] if angular is not None else None
            observations.append(row)
            tracks[key].append(row)
            prev[key] = row
            current.append(row)
        adjacency = defaultdict(set)
        nearest = []
        if current:
            centers = np.array([r["center"] for r in current])
            dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
            np.fill_diagonal(dist, np.inf)
            if len(current) > 1:
                nearest = dist.min(1).tolist()
            for i, j in zip(
                *np.where(np.triu(dist < cfg.get("interaction_max_pixels", 120), 1))
            ):
                a, b = current[i], current[j]
                lengths = [v["body_length"] for v in (a, b) if v["body_length"]]
                if len(lengths) != 2:
                    continue
                normalized = float(dist[i, j] / np.mean(lengths))
                if normalized > cfg.get("near_body_lengths", 2):
                    continue
                pair = (*scope, *sorted((a["track_id"], b["track_id"])))
                last = pair_state.get(pair)
                start = (
                    last["start"]
                    if last and f["frame"] - last["last"] <= max_gap
                    else f["frame"]
                )
                duration = f["frame"] - start
                pair_state[pair] = {"start": start, "last": f["frame"]}
                vec = np.asarray(b["center"]) - a["center"]
                direction = math.atan2(vec[1], vec[0])
                facing = max(
                    abs(wrap(a["angle"] - direction)),
                    abs(wrap(b["angle"] - direction - math.pi)),
                ) if a["angle"] is not None and b["angle"] is not None else None
                same_direction = abs(wrap(a["angle"] - b["angle"])) if facing is not None else None
                av = a.get("velocity_px_per_source_frame")
                bv = b.get("velocity_px_per_source_frame")
                co_motion = (
                    float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))
                    if av is not None
                    and bv is not None
                    and np.linalg.norm(av) * np.linalg.norm(bv) > 1e-6
                    else None
                )
                interaction = (
                    "directed_interaction"
                    if facing is not None and facing < math.pi / 3 and duration >= min_duration
                    else (
                        "following"
                        if same_direction is not None and same_direction < math.pi / 6
                        and co_motion is not None
                        and co_motion > 0.7
                        and duration >= min_duration
                        and a["motion_state"] == "walking"
                        and b["motion_state"] == "walking"
                        else "approach"
                    )
                )
                priority = {
                    "unknown": 0,
                    "none": 1,
                    "approach": 2,
                    "following": 3,
                    "directed_interaction": 4,
                }
                for x in (a, b):
                    if priority[interaction] > priority[x["interaction_state"]]:
                        x["interaction_state"] = interaction
                adjacency[a["track_id"]].add(b["track_id"])
                adjacency[b["track_id"]].add(a["track_id"])
                edges.append(
                    {
                        "domain": domain,
                        "video": video,
                        "group": f["group"],
                        "frame": f["frame"],
                        "involved_tracks": [a["track_id"], b["track_id"]],
                        "interaction_state": interaction,
                        "duration_source_frames": duration,
                        "normalized_distance": normalized,
                        "relative_orientation": float(same_direction) if same_direction is not None else None,
                        "co_motion": co_motion,
                        "evidence_components": {
                            "distance": normalized,
                            "facing_error": float(facing) if facing is not None else None,
                            "duration": duration,
                        },
                        "upstream_entities": [a["entity_id"], b["entity_id"]],
                        "observability_level": "O3",
                        "status": "candidate",
                    }
                )
        components = []
        unseen = {r["track_id"] for r in current}
        while unseen:
            stack = [unseen.pop()]
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                fresh = adjacency[node] & unseen
                unseen -= fresh
                stack.extend(fresh)
            components.append(component)
        raw = len(f["detections"])
        r = recall.get(domain)
        probs = [d.get("confidence") for d in f["detections"]]
        calibrated = (
            sum(probs) / r
            if r
            and 0 < r <= 1
            and all(p is not None for p in probs)
            and cfg.get("scores_are_calibrated", False)
            else None
        )
        if cfg.get("recall_calibration"):
            from .count_calibration import corrected_count

            calibrated = corrected_count(
                f["detections"],
                cfg["recall_calibration"],
                {"domain": domain, **f.get("calibration_slice", {})},
            )["calibrated_count"]
        group_rows.append(
            {
                "domain": domain,
                "video": video,
                "frame": f["frame"],
                "raw_count": raw,
                "group": f["group"],
                "calibrated_count": calibrated,
                "density": raw / (f["image_size"][0] * f["image_size"][1]),
                "density_units": "detections/pixel²",
                "activity": sum(
                    x["motion_state"] in ("walking", "fast_moving") for x in current
                )
                / max(len(current), 1),
                "graph_statistics": {
                    "components": len(components),
                    "largest_component": max(map(len, components), default=0),
                    "mean_nearest_neighbor_pixels": float(np.mean(nearest))
                    if nearest
                    else None,
                },
                "context_id": f["context_id"],
                "hive_id": f["hive_id"],
            }
        )
    windows = []
    for key, rows in tracks.items():
        cells = {
            (
                min(19, max(0, int(r["normalized_center"][0] * 20))),
                min(19, max(0, int(r["normalized_center"][1] * 20))),
            )
            for r in rows
        }
        first = rows[0]
        length = first["body_length"]
        passage = next(
            (
                r["frame"] - first["frame"]
                for r in rows
                if length
                and np.linalg.norm(np.asarray(r["center"]) - first["center"]) >= length
            ),
            None,
        )
        rhythm = oscillation(rows)
        windows.append(
            {
                "domain": key[0],
                "video": key[1],
                "group": key[2],
                "track_id": key[3],
                "motion_state": rows[-1]["motion_state"],
                "oscillation": rhythm,
                "visited_coverage": len(cells) / 400,
                "coverage_definition": "20x20 normalized grid visit ratio",
                "first_passage_source_frames": passage,
                "valid_observations": len(rows),
                "observability_level": "O2",
                "context_id": first["context_id"],
                "hive_id": first["hive_id"],
            }
        )
    from .quantification import enrich
    from .knowledge import attach_context

    report = {
        "observations": observations,
        "individual_windows": windows,
        "group_windows": group_rows,
        "interaction_edges": edges,
        "events": events,
        "behavior_labels_are_candidates": True,
        "motion_thresholds": thresholds,
        "motion_thresholds_are_calibrated": bool(
            cfg.get("motion_calibration_reference")
        ),
    }
    return attach_context(enrich(report, frames, cfg), cfg.get("external_context"))
