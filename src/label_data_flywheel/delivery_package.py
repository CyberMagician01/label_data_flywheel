"""将已核验的 YOLO 副本整理为检测、姿态、MOT 三个平行目录。"""

import gzip
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .io import read_json, write_json
from .semantics import is_bee


def mot_row(frame, row, label, width, height):
    """MOT Challenge 十列：帧、ID、左上角坐标从 1 起算，尺寸不变。"""
    tid = row.get("track_id")
    if tid is None or not is_bee(row) or int(float(label.split()[0])) != 0:
        return None
    if int(tid) != tid or tid < 1:
        raise ValueError("MOT ID 须为正整数；不自动重编号")
    _, cx, cy, w, h = map(float, label.split())
    x, y = (cx - w / 2) * width + 1, (cy - h / 2) * height + 1
    conf = row.get("confidence")
    conf = 1.0 if conf is None else float(conf)
    return f"{frame + 1},{int(tid)},{x:.6f},{y:.6f},{w * width:.6f},{h * height:.6f},{conf:.8f},-1,-1,-1\n"


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def arrange_dataset(source, destination, auxiliary=None):
    """输入只选一份已定稿的全量机器标注；人工/旧版继续独立保留。"""
    src, root = Path(source), Path(destination)
    extra = Path(auxiliary) if auxiliary else root.with_name(root.name + "_附加信息")
    (extra / "metadata").mkdir(parents=True, exist_ok=True)
    assert read_json(src / "validation.json")["passed"]
    handles, counters, sequence = {}, {}, {}
    try:
        with gzip.open(src / "frame_ids.jsonl.gz", "rt", encoding="utf-8") as stream:
            for f in map(json.loads, stream):
                video = f["video"]
                if video not in handles:
                    (root / "annotations_tracking" / video).mkdir(
                        parents=True, exist_ok=False
                    )
                    handles[video] = (
                        (root / "annotations_tracking" / video / "tracks.txt").open(
                            "w", encoding="utf-8"
                        ),
                        gzip.open(
                            extra / "metadata" / (video + ".jsonl.gz"),
                            "wt",
                            encoding="utf-8",
                            compresslevel=1,
                        ),  # noqa: SIM115 -- 视频流在 finally 一并关闭
                        (extra / "metadata" / (video + ".frames.jsonl")).open(
                            "w", encoding="utf-8"
                        ),
                    )
                    counters[video] = Counter()
                    sequence[video] = {
                        "width": f["width"],
                        "height": f["height"],
                        "max_frame": -1,
                    }
                count = counters[video]
                mot, meta, frames = handles[video]
                assert f["frame_index_base"] == 0
                assert f["frame"] > sequence[video]["max_frame"]
                sequence[video]["max_frame"] = f["frame"]
                rel = Path(video) / Path(f["label_files"]["candidates_detect"]).name
                det_source = src / f["label_files"]["candidates_detect"]
                det_target = root / "annotations" / rel
                _link(det_source, det_target)
                lines = det_source.read_text().splitlines()
                pose_target = root / "annotations_pose" / rel
                pose_rel = f["label_files"].get("candidates_pose")
                if pose_rel:
                    _link(src / pose_rel, pose_target)
                else:
                    pose_target.parent.mkdir(parents=True, exist_ok=True)
                    pose_target.write_text(
                        "".join(line + " 0 0 0 0 0 0\n" for line in lines),
                        encoding="utf-8",
                    )
                assert len(lines) == len(f["rows"].get("candidates", []))
                ids = set()
                for label, row in zip(lines, f["rows"].get("candidates", [])):
                    entry = mot_row(f["frame"], row, label, f["width"], f["height"])
                    if entry is None:
                        count["without_track_id"] += 1
                        row["mot_line"] = None
                        continue
                    assert row["track_id"] not in ids, (
                        video,
                        f["frame"],
                        row["track_id"],
                    )
                    ids.add(row["track_id"])
                    mot.write(entry)
                    count["tracking_rows"] += 1
                    row["mot_line"] = count["tracking_rows"]
                count["frames"] += 1
                count["detection_rows"] += len(lines)
                count["pose_rows"] += len(lines)
                old_paths = f["label_files"]
                f["label_files"] = {
                    "detection": "annotations/" + rel.as_posix(),
                    "pose": "annotations_pose/" + rel.as_posix(),
                    "tracking": f"annotations_tracking/{video}/tracks.txt",
                }
                if "hidden_detect" in old_paths:
                    hidden = extra / "audit" / "hidden" / rel
                    _link(src / old_paths["hidden_detect"], hidden)
                    f["label_files"]["hidden_detect"] = hidden.relative_to(
                        extra
                    ).as_posix()
                    count["hidden_rows"] += len(f["rows"].get("hidden", []))
                meta.write(
                    json.dumps(f, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                frames.write(
                    json.dumps(
                        {
                            "video": video,
                            "source_frame_id": f["frame"],
                            "frame_index_base": 0,
                            "decode_index": f["frame"],
                            "image_ref": rel.with_suffix(".jpg").as_posix(),
                            "width": f["width"],
                            "height": f["height"],
                        }
                    )
                    + "\n"
                )
    finally:
        for streams in handles.values():
            for h in streams:
                h.close()
    for video, data in sequence.items():
        (extra / "metadata" / (video + ".seqinfo.ini")).write_text(
            f"[Sequence]\nname={video}\nimDir=images/{video}\nframeRate=30\nseqLength={data['max_frame'] + 1}\nimWidth={data['width']}\nimHeight={data['height']}\nimExt=.jpg\n",
            encoding="utf-8",
        )
        count = counters[video]
        # 从实际 MOT 文件复读，验证十列、正 ID、坐标起点与实例总数。
        n = 0
        with (root / "annotations_tracking" / video / "tracks.txt").open() as h:
            for line in h:
                v = list(map(float, line.split(",")))
                assert len(v) == 10 and v[0] >= 1 and v[1] >= 1
                assert v[2] >= 1 - 1e-5 and v[3] >= 1 - 1e-5 and v[4] > 0 and v[5] > 0
                n += 1
        assert n == count["tracking_rows"]
        write_json(
            extra / "metadata" / (video + ".verified.json"),
            dict(count, mot_readback_verified=True),
        )
    return {video: dict(c) for video, c in counters.items()}


def _arrange_job(args):
    return arrange_dataset(*args)


def export_delivery(config, destination):
    """完整交付目录，输出路径应命名为 05_数据标注成果。"""
    root = Path(destination)
    extra = Path(config.get("auxiliary_output", root.with_name(root.name + "_附加信息")))
    if extra.resolve() == root.resolve() or root.resolve() in extra.resolve().parents:
        raise ValueError("附加信息须位于正式提交目录之外")
    root.mkdir(parents=True, exist_ok=False)
    extra.mkdir(parents=True, exist_ok=False)
    videos = {}
    with ProcessPoolExecutor(max_workers=config.get("workers", 4)) as pool:
        for result in pool.map(
            _arrange_job, [(p, str(root), str(extra)) for p in config["inputs"]]
        ):
            videos.update(result)
    with (extra / "frame_manifest.jsonl").open("wb") as out:
        for path in sorted((extra / "metadata").glob("*.frames.jsonl")):
            with path.open("rb") as h:
                shutil.copyfileobj(h, out)
    (root / "splits").mkdir()
    for split in ("train", "val"):
        (root / "splits" / (split + ".txt")).write_text("")
    with (extra / "unassigned.txt").open("w", encoding="utf-8") as out:
        for line in (extra / "frame_manifest.jsonl").read_text().splitlines():
            out.write(json.loads(line)["image_ref"] + "\n")
    if config.get("human_split_source"):
        shutil.copytree(
            config["human_split_source"], extra / "metadata/original_human_splits"
        )
    shutil.copy2(
        Path(__file__).parent / "assets/extract_frames.py", root / "extract_frames.py"
    )
    bundled = Path(__file__).parent / "assets/数据标注说明-595335.docx"
    description = Path(config.get("description_docx", bundled))
    team_id = config.get("team_id", "595335" if description == bundled else "")
    if description.is_file():
        name = f"数据标注说明-{team_id}.docx" if team_id else "数据标注说明.docx"
        shutil.copy2(description, root / name)
    result = {
        "completed": True,
        "videos": videos,
        "total_frames": sum(v["frames"] for v in videos.values()),
        "total_detection_rows": sum(v["detection_rows"] for v in videos.values()),
        "total_tracking_rows": sum(v["tracking_rows"] for v in videos.values()),
        "contains_images": False,
        "originals_preserved": True,
        "docx_pending": not description.is_file(),
        "auxiliary_output": str(extra),
        "split_status": "unassigned; original human splits preserved separately",
    }
    write_json(extra / "manifest.json", result)
    write_json(extra / "resolved_config.json", config)
    return result
