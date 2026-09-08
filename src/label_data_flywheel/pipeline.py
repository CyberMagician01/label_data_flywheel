"""一轮可重放飞轮：统一输入→证据→复核→采样→训练快照。"""

from pathlib import Path
import copy
import numpy as np
from .io import iter_frames, read_json, write_json, write_frames, sha256
from .postprocess import suppress_interpolation, interpolate, is_fill
from .quality import q2, frame_evidence
from .experts import cross_layer_evidence, apply_candidate_evidence
from .behavior import analyze
from .review import apply_reviews, prioritize, error_cube
from .sampling import (
    calibrate_weights,
    balanced_batches,
    distribution_report,
    exposure_report,
)
from .curriculum import input_gate, meta_policy
from .knowledge import default_graph, update_graph


def frame_record(f, quality):
    ds = f["detections"]
    w, h = f["image_size"]
    sizes = [
        np.sqrt(
            (d["bbox_xyxy"][2] - d["bbox_xyxy"][0])
            * (d["bbox_xyxy"][3] - d["bbox_xyxy"][1])
            / (w * h)
        )
        for d in ds
    ]
    q = [quality.get(d["entity_id"], 1.0) for d in ds]
    size = float(np.median(sizes)) if sizes else 0.0
    return {
        "sample_id": f["sample_id"],
        "domain": f["domain"],
        "video": f["video"],
        "group": f["group"],
        "segment": f.get("segment", f["group"]),
        "split": f["split"],
        "status": f.get("status", "valid"),
        "quality": float(np.mean(q)) if q else 1.0,
        "scale_bin": "small" if size < 0.02 else "medium" if size < 0.05 else "large",
        "density_bin": "low" if len(ds) < 50 else "medium" if len(ds) < 200 else "high",
        "track_ids": [d["track_id"] for d in ds if d.get("track_id") is not None],
        "duplicate_cluster": f.get("duplicate_cluster"),
    }


def run_round(config, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("输出轮次已存在；请创建新轮次以保留旧版本")
    out.mkdir(parents=True)
    write_json(out / "resolved_config.json", config)
    frames = []
    source_fingerprints = []
    for source in config["inputs"]:
        kwargs = {
            k: source[k]
            for k in ("video", "domain", "group", "split", "source")
            if k in source
        }
        start, end = source.get("source_frame_range", [None, None])
        rows = [
            f
            for f in iter_frames(source["path"], **kwargs)
            if (start is None or f["frame"] >= start)
            and (end is None or f["frame"] <= end)
        ]
        frames.extend(rows)
        p = Path(source["path"])
        source_fingerprints.append(
            {
                "path": str(p),
                "sha256": sha256(p) if p.is_file() else None,
                "selected_frames": len(rows),
            }
        )
    if not frames:
        raise ValueError("输入范围内没有帧")
    identities = [f["sample_id"] for f in frames]
    if len(identities) != len(set(identities)):
        raise ValueError("输入源包含重复sample_id")
    if config.get("pose_sources"):
        from .fusion import attach_pose

        refs = {}
        for source in config["pose_sources"]:
            p = Path(source["path"])
            source_fingerprints.append(
                {
                    "role": "pose",
                    "path": str(p),
                    "sha256": sha256(p) if p.is_file() else None,
                }
            )
            kwargs = {
                k: source[k]
                for k in ("video", "domain", "group", "split", "source")
                if k in source
            }
            for f in iter_frames(source["path"], **kwargs):
                refs[(f["domain"], f["video"], f["frame"])] = f
        frames = [
            attach_pose(
                f,
                refs[(f["domain"], f["video"], f["frame"])],
                config.get("pose_match_iou", 0.5),
            )
            if (f["domain"], f["video"], f["frame"]) in refs
            else f
            for f in frames
        ]
    observed_before = {
        d["entity_id"]: copy.deepcopy(d)
        for f in frames
        for d in f["detections"]
        if not is_fill(d)
    }
    pp = config.get("postprocess", {})
    if pp.get("interpolate", False):
        groups = {}
        for f in frames:
            groups.setdefault((f["domain"], f["video"], f["group"]), []).append(f)
        frames = [
            f
            for rows in groups.values()
            for f in interpolate(rows, max_gap=pp.get("max_gap", 90))
        ]
    if pp.get("enabled", True):
        frames = [
            suppress_interpolation(
                f, pp.get("iomin", 0.2), pp.get("min_intersection", 16)
            )
            for f in frames
        ]
    if config.get("candidate_evidence"):
        evidence = read_json(config["candidate_evidence"])
        model = read_json(config["expert_calibration"])
        frames = [
            apply_candidate_evidence(
                f,
                evidence.get(f["sample_id"], []),
                model,
                config.get("candidate_thresholds"),
            )
            for f in frames
        ]
    if config.get("review_decisions"):
        frames = apply_reviews(
            frames,
            read_json(config["review_decisions"]),
            config["reviewer"],
            out / "review_audit.jsonl",
        )
    behavior = analyze(frames, config.get("behavior", {}))
    quality = q2(
        [r for f in frames for r in frame_evidence(f)]
        + cross_layer_evidence(frames, behavior)
    )
    cube = error_cube(quality)
    q_lookup = {r["sample_id"]: r["quality"] for r in quality if r["layer"] == "L1"}
    review_rows = [
        {
            **r,
            "uncertainty": 1 - r["quality"],
            "disagreement": r["error"],
            "cross_layer_value": 0.3 if r["layer"] != "L1" else 0.0,
            "features": [r["error"], r["quality"]],
        }
        for r in quality
    ]
    queue = prioritize(review_rows, config.get("review_budget", 50))
    if config.get("colony", {}).get("enabled"):
        from .colony import analyze_colony
        from .apiculture import interpret_colony

        colony_config = config["colony"]
        colony_report = analyze_colony(
            sorted(
                frames, key=lambda f: (f["domain"], f["video"], f["group"], f["frame"])
            ),
            colony_config.get("analysis", {}),
        )
        colony_report["apiculture"] = interpret_colony(
            colony_report, colony_config.get("context_records", [])
        )
        if colony_config.get("review_decisions"):
            from .apiculture import review_colony

            colony_report = review_colony(
                colony_report,
                colony_config["review_decisions"],
                colony_config["reviewer"],
            )
            write_json(
                out / "confirmed_group_windows.json",
                colony_report["confirmed_group_windows"],
            )
            write_json(
                out / "colony_review_audit.json", colony_report["colony_review_audit"]
            )
            for event in colony_report["confirmed_group_windows"]:
                for f in frames:
                    if (f["domain"], f["video"], f["group"]) == (
                        event["domain"],
                        event["video"],
                        event["group"],
                    ) and event["source_frame_range"][0] <= f["frame"] <= event[
                        "source_frame_range"
                    ][1]:
                        f.setdefault("confirmed_colony_events", []).append(
                            {
                                "event_id": event["event_id"],
                                "human_label": event["human_label"],
                                "reviewer": event["reviewer"],
                            }
                        )
        colony_queue = [
            row
            for row in colony_report["apiculture"]["review_queue"]
            if row["status"] == "unconfirmed"
        ]
        write_json(out / "colony.json", colony_report)
        write_json(out / "colony_review_queue.json", colony_queue)
        budget = config.get("review_budget", 50)
        reserved = min(len(colony_queue), max(0, budget // 4))
        if reserved:
            queue = queue[: budget - reserved] + colony_queue[:reserved]
    graph = (
        read_json(config["knowledge_graph"])
        if config.get("knowledge_graph")
        else default_graph()
    )
    if config.get("knowledge_reviews"):
        graph = update_graph(graph, read_json(config["knowledge_reviews"]))
    if config.get("colony", {}).get("enabled"):
        graph["literature_cards"] = colony_report["apiculture"]["literature"]["cards"]
        graph["colony_evidence_path"] = "colony.json"
    policy = meta_policy(cube, len(queue))
    sampling = None
    train_records = [frame_record(f, q_lookup) for f in frames if f["split"] == "train"]
    if config.get("sampling_targets") and train_records:
        weights = calibrate_weights(
            train_records, config["sampling_targets"], config.get("min_ess_ratio", 0.5)
        )
        batches = balanced_batches(
            train_records, weights, config.get("effective_batch_size", 8)
        )
        gate = input_gate(
            train_records, batches, weights, config.get("min_ess_ratio", 0.5)
        )
        if not gate["passed"]:
            raise ValueError("训练输入可观测性门未通过")
        sampling = {
            "records": train_records,
            "weights": weights.tolist(),
            "batches": batches,
            "gate": gate,
            "diagnostics": distribution_report(
                train_records, weights, config["sampling_targets"]
            ),
            "actual_planned_exposure": exposure_report(
                train_records, batches, config["sampling_targets"]
            ),
        }
        by_id = {r["sample_id"]: float(w) for r, w in zip(train_records, weights)}
        for f in frames:
            if f["sample_id"] in by_id:
                f["sampling_weight"] = by_id[f["sample_id"]]
        write_json(out / "sampling_plan.json", sampling)
    for name, value in [
        ("quality", quality),
        ("error_cube", cube),
        ("behavior", behavior),
        ("review_queue", queue),
        ("knowledge_graph", graph),
        ("next_round_policy", policy),
    ]:
        write_json(out / f"{name}.json", value)
    write_frames(out / "annotations.jsonl.gz", frames)
    snapshot = None
    if config.get("export_training"):
        from .training_data import export_round

        snapshot = export_round(
            frames, out / "training_snapshot", config.get("image_root", "")
        )
    unchanged = sum(
        d["entity_id"] in observed_before
        and observed_before[d["entity_id"]]["bbox_xyxy"] == d["bbox_xyxy"]
        for f in frames
        for d in f["detections"]
    )
    summary = {
        "config_sha256": sha256(out / "resolved_config.json"),
        "frames": len(frames),
        "visible_boxes": sum(len(f["detections"]) for f in frames),
        "hidden_boxes": sum(
            len(f.get("temporarily_hidden_detections", [])) for f in frames
        ),
        "original_observation_boxes_unchanged": unchanged,
        "quality_layers": sorted({r["layer"] for r in quality}),
        "quality_is_consistency_evidence_not_gt_accuracy": True,
        "events_are_candidates": True,
        "review_candidates": len(queue),
        "source_fingerprints": source_fingerprints,
        "auxiliary_fingerprints": {
            name: {"path": config[name], "sha256": sha256(config[name])}
            for name in (
                "candidate_evidence",
                "expert_calibration",
                "review_decisions",
                "knowledge_graph",
                "knowledge_reviews",
            )
            if config.get(name)
        },
        "training_snapshot": snapshot,
        "sampling_gate": sampling["gate"] if sampling else None,
        "output_sha256": sha256(out / "annotations.jsonl.gz"),
    }
    write_json(out / "run_summary.json", summary)
    return summary
