"""独立群体分析入口，复用已生成标注；不重训、不替换最优路线。"""

import hashlib
import re
import time
from pathlib import Path
from .io import iter_frames, read_json, write_json, normalize, labelme_frame
from .colony import analyze_colony
from .apiculture import interpret_colony


def run_colony(config, output):
    started = time.perf_counter()
    out = Path(output)
    if out.exists():
        raise FileExistsError("群体分析输出目录已存在，请使用新版本目录")
    out.mkdir(parents=True)
    receipts = []

    def inputs():
        for source in config["inputs"]:
            kw = {
                k: source[k]
                for k in ("video", "domain", "group", "split", "source")
                if k in source
            }
            start, end = source.get("source_frame_range", [0, float("inf")])
            p = Path(source["path"])
            digest = hashlib.sha256()
            if p.is_dir():

                def directory_rows():
                    for file in sorted(p.glob("*.json*")):
                        m = re.search(r"(\d+)\.json(?:\.gz)?$", file.name)
                        if m and start <= int(m[1]) <= end:
                            data = file.read_bytes()
                            digest.update(file.name.encode() + b"\0" + data)
                            value = read_json(file)
                            if "shapes" in value:
                                value = labelme_frame(value, int(m[1]))
                            yield normalize(value, **kw)

                frames = directory_rows()
            else:
                with p.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                frames = (f for f in iter_frames(p, **kw) if start <= f["frame"] <= end)
            tracker = None
            if source.get("tracking_mode") == "analysis_baseline":
                from .tracking import AssociationTracker

                tracker = AssociationTracker(
                    retention=source.get("analysis_retention_frames", 15)
                )
            n = 0
            for f in frames:
                if tracker:
                    # 只为分析副本补身份，所有原检测（含未关联者）仍保留。
                    assigned = {
                        d["entity_id"]: d.get("track_id")
                        for d in tracker.update(f)["detections"]
                    }
                    for d in f["detections"]:
                        d["track_id"] = assigned.get(d["entity_id"])
                    f["analysis_identity_source"] = (
                        "auxiliary_association_not_published_sam"
                    )
                n += 1
                yield f
            receipts.append(
                {
                    "input": source,
                    "selected_frames": n,
                    "sha256": digest.hexdigest(),
                    "hash_definition": "有序文件名、NUL和文件字节的串联"
                    if p.is_dir()
                    else "完整源文件字节",
                }
            )

    report = analyze_colony(inputs(), config.get("analysis"))
    report["source_receipts"] = receipts
    report["implementation_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "colony.py",
            "apiculture.py",
            "colony_runner.py",
            "colony_report.py",
            "tracking.py",
            "io.py",
        )
    }
    report["apiculture"] = interpret_colony(report)
    write_json(out / "colony.json", report)
    write_json(out / "review_queue.json", report["apiculture"]["review_queue"])
    write_json(out / "resolved_config.json", config)
    if config.get("render", True):
        from .colony_report import render_report

        render_report(report, out, config.get("background"), config.get("font_path"))
    summary = {
        "scopes": len(report["scopes"]),
        "frames": sum(r["selected_frames"] for r in receipts),
        "windows": sum(len(s["windows"]) for s in report["scopes"]),
        "review_candidates": len(report["apiculture"]["review_queue"]),
        "output": str(out),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "summary.json", summary)
    return summary
