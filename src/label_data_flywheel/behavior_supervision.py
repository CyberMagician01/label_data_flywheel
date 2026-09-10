"""将具名人工行为复核连接到训练数据，个体与群体保持各自监督粒度。"""

from collections import defaultdict
from pathlib import Path
from .io import write_json
from .semantics import is_bee

COLONY_FEATURES = (
    "mean_observed_count", "active_fraction", "median_speed_bl_proxy_s",
    "density_cv", "coverage", "motion_observable_fraction", "tracked_fraction",
)


def individual_records(frames, behavior):
    observations = {r["entity_id"]: r for r in behavior.get("observations", [])}
    edges = defaultdict(list)
    for row in behavior.get("interaction_edges", []):
        for entity in row.get("upstream_entities", []):
            edges[entity].append(row)
    rhythms = {(r["domain"], r["video"], r["group"], r["track_id"]): r.get("oscillation", {})
               for r in behavior.get("individual_windows", [])}
    result = []
    for f in frames:
        for d in f["detections"]:
            labels = dict(d.get("behavior_labels", {}))
            for event in d.get("confirmed_events", []):
                if isinstance(event, dict):
                    labels.update(event.get("labels", {}))
            if not is_bee(d) or d.get("label_status") != "human_confirmed" or not labels:
                continue
            observation = observations.get(d["entity_id"], {})
            nearby = sorted(edges[d["entity_id"]], key=lambda r: r["normalized_distance"])
            interaction = nearby[0] if nearby else {}
            rhythm = rhythms.get((f["domain"], f["video"], f["group"], d.get("track_id")), {})
            result.append({
                **observation, "entity_id": d["entity_id"], "frame": f["frame"],
                "domain": f["domain"], "video": f["video"], "group": f["group"],
                "split": f["split"], "status": "human_confirmed", "labels": labels,
                "reviewer": d.get("reviewer"), "source_frame_id": f["sample_id"],
                "normalized_distance": interaction.get("normalized_distance"),
                "co_motion": interaction.get("co_motion"),
                "spectral_concentration": rhythm.get("spectral_concentration"),
            })
    return result


def colony_records(frames, report):
    result = []
    scopes = {(s["domain"], s["video"], s["group"]): s for s in report.get("scopes", [])}
    for event in report.get("confirmed_group_windows", []):
        scope_key = (event["domain"], event["video"], event["group"])
        window = next(w for w in scopes[scope_key]["windows"] if w["index"] == event["window_index"])
        splits = {f["split"] for f in frames
                  if (f["domain"], f["video"], f["group"]) == scope_key
                  and event["source_frame_range"][0] <= f["frame"] <= event["source_frame_range"][1]}
        split = next(iter(splits)) if len(splits) == 1 else event.get("split", "unassigned")
        if len(splits) > 1:
            raise ValueError("群体监督窗口横跨不同数据划分")
        result.append({**event, **{name: window.get(name) for name in COLONY_FEATURES},
                       "split": split, "labels": {"colony": event["human_label"]},
                       "supervision_unit": "group_window"})
    return result


def export_behavior_supervision(frames, behavior, colony, output):
    root = Path(output)
    manifest = {}
    for name, records in (("individual", individual_records(frames, behavior)),
                          ("colony", colony_records(frames, colony))):
        splits = sorted({r["split"] for r in records} | {"train"})
        manifest[name] = {}
        for split in splits:
            path = root / f"{name}.{split}.json"
            rows = [r for r in records if r["split"] == split]
            write_json(path, rows)
            manifest[name][split] = {"path": str(path), "samples": len(rows)}
    write_json(root / "manifest.json", manifest)
    return manifest
