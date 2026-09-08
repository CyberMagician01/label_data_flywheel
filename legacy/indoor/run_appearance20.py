"""在独立 Linux 工作目录运行已验证的外观关联版本，不改原始检测和旧版本。"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def stage(source, runtime, frames, detections, assets):
    runtime.mkdir(parents=True, exist_ok=True)
    identity = {"frames": str(frames), "detections": str(detections), "assets": str(assets)}
    identity["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source.glob("*.py")}
    marker = runtime / "input_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("工作目录属于不同输入或代码版本，请使用新的 --work-dir")
    for p in source.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        if p.name == "full_common.py":
            text = text.replace("R=Path('/root/bee_tracking_pilot_20260907')", f"R=Path({str(runtime)!r})")
            text = text.replace("SOURCE=Path('/root/autodl-tmp/indoor_ids_full_latest_20260906/final/frames')", f"SOURCE=Path({str(detections)!r})")
            text = text.replace("IMAGES=Path('/root/autodl-tmp/flywheel_indoor_scene_b_20260904/frames')", f"IMAGES=Path({str(frames)!r})")
        (runtime / p.name).write_text(text, encoding="utf-8")
    for name in ("repo", "data"):
        target = assets / name
        if not target.exists():
            raise FileNotFoundError(f"缺少外观模型资源：{target}")
        link = runtime / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    (runtime / "gt_eval_01_03_20260907").mkdir(exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True, help="包含 bee_tracking repo/ 和 data/ 权重")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable, help="含 TensorFlow 2.15 的 Python")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--stage", choices=["prepare", "embed", "track", "export", "all"], default="all")
    args = ap.parse_args()
    r = args.work_dir.resolve()
    stage(Path(__file__).parent / "pipelines/appearance20", r, args.frames.resolve(), args.detections.resolve(), args.assets.resolve())
    def call(script, *extra):
        subprocess.run([args.python, str(r / script), *map(str, extra)], check=True, cwd=r)
    if args.stage in ("prepare", "all"):
        call("full_common.py")
    videos = ["B-5-1", "B-5-2", "B-5-3", "B-5-4"]
    if args.stage in ("embed", "all"):
        with ThreadPoolExecutor(len(args.gpus.split(","))) as pool:
            list(pool.map(lambda g: call("full_embed_worker.py", g), args.gpus.split(",")))
    if args.stage in ("track", "all"):
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda v: call("full_track_worker.py", v), videos))
    if args.stage in ("export", "all"):
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda v: call("full_export_worker.py", v), videos))


if __name__ == "__main__":
    main()
