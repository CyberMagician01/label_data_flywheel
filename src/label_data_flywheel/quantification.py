"""源帧量纲的窗口量化：圆盘覆盖、停留、曲折度、事件索引和证据图。"""

from collections import Counter, defaultdict
import numpy as np


def point_in_polygon(point, vertices):
    x, y = point
    inside = False
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            inside = not inside
    return inside


def disk_coverage(rows, image_size, kappa=0.5, resolution=256):
    width, height = image_size
    mask = np.zeros((resolution, resolution), bool)
    for r in rows:
        if not r["body_length"]:
            continue
        x, y = r["normalized_center"]
        rx = kappa * r["body_length"] / width
        ry = kappa * r["body_length"] / height
        x1 = max(0, int((x - rx) * resolution))
        x2 = min(resolution, int(np.ceil((x + rx) * resolution)))
        y1 = max(0, int((y - ry) * resolution))
        y2 = min(resolution, int(np.ceil((y + ry) * resolution)))
        yy, xx = np.mgrid[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] |= (((xx + 0.5) / resolution - x) / rx) ** 2 + (
            ((yy + 0.5) / resolution - y) / ry
        ) ** 2 <= 1
    return float(mask.mean())


def enrich(report, frames, config=None):
    cfg = config or {}
    by_track = defaultdict(list)
    image_sizes = {
        (f["domain"], f["video"], f["group"]): f["image_size"] for f in frames
    }
    for r in report["observations"]:
        key = (r["domain"], r["video"], r["group"], r["track_id"])
        by_track[key].append(r)
        polygons = cfg.get("regions", {})
        if polygons:
            r["region_state"] = next(
                (
                    name
                    for name, polygon in polygons.items()
                    if point_in_polygon(r["normalized_center"], polygon)
                ),
                "region_unknown",
            )
    for window in report["individual_windows"]:
        key = tuple(window[k] for k in ("domain", "video", "group", "track_id"))
        rows = by_track[key]
        f = np.array([r["frame"] for r in rows])
        c = np.array([r["center"] for r in rows])
        df = np.diff(f)
        gaps = df <= cfg.get("max_observation_gap", 10)
        lengths = np.linalg.norm(np.diff(c, axis=0), axis=1)
        path = float(lengths[gaps].sum())
        net = float(np.linalg.norm(c[-1] - c[0]))
        dwell = Counter()
        for r, d, valid in zip(rows[:-1], df, gaps):
            if valid:
                dwell[r["region_state"]] += int(d)
        duration = sum(dwell.values())
        norm = np.array([r["normalized_center"] for r in rows])
        window.update(
            path_length_pixels=path,
            net_displacement_pixels=net,
            tortuosity=path / net if net > 1e-6 else None,
            region_dwell_source_frames=dict(dwell),
            region_dwell_fraction={k: v / max(duration, 1) for k, v in dwell.items()},
            spatial_radius_of_gyration=float(
                np.sqrt(((norm - norm.mean(0)) ** 2).sum(1).mean())
            ),
            disk_union_coverage=disk_coverage(
                rows, image_sizes[key[:3]], cfg.get("coverage_kappa", 0.5)
            ),
            disk_coverage_raster_resolution=256,
            missing_source_frames=int(sum(df[~gaps])),
            knowledge_source="visual_kinematics",
            upstream_entities=[r["entity_id"] for r in rows],
        )
        for row in rows:
            row["rhythm_state"] = window["oscillation"]["state"]
        if window["oscillation"]["state"] == "axis_oscillation":
            report["events"].append(
                {
                    "event_type": "axis_oscillation",
                    "domain": key[0],
                    "video": key[1],
                    "group": key[2],
                    "involved_tracks": [key[3]],
                    "representative_frames": [int(f[0]), int(f[-1])],
                    "knowledge_source": "axis_oscillation",
                    "observability_level": "O3",
                    "evidence_confidence": window["oscillation"][
                        "spectral_concentration"
                    ],
                    "upstream_entities": [rows[0]["entity_id"], rows[-1]["entity_id"]],
                    "status": "candidate",
                }
            )
    by_frame = defaultdict(list)
    for edge in report["interaction_edges"]:
        weight = float(
            np.exp(-edge["normalized_distance"])
            * min(
                1.0,
                edge["duration_source_frames"]
                / max(cfg.get("min_duration_source_frames", 5), 1),
            )
        )
        edge.update(
            weight=weight,
            evidence_confidence=weight,
            knowledge_source="directed_interaction",
        )
        by_frame[(edge["domain"], edge["video"], edge["frame"])].append(edge)
    for row in report["group_windows"]:
        degree = Counter()
        for edge in by_frame[(row["domain"], row["video"], row["frame"])]:
            for tid in edge["involved_tracks"]:
                degree[str(tid)] += edge["weight"]
        row["graph_statistics"]["weighted_degree"] = dict(degree)
        row["graph_statistics"]["aggregation_ratio"] = row["graph_statistics"][
            "largest_component"
        ] / max(row["raw_count"], 1)
    for i, event in enumerate(report["events"]):
        event["event_id"] = f"{event['domain']}/{event['video']}/event/{i}"
        event.setdefault("context_id", "default")
        event.setdefault("hive_id", "default")
    # 每类状态单一激活，类间共现；unknown保留在分母内。
    report["state_coverage"] = {
        key: dict(Counter(r[key] for r in report["observations"]))
        for key in ("motion_state", "region_state", "interaction_state", "rhythm_state")
    }
    return report
