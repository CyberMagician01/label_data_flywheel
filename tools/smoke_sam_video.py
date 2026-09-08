"""从真实源视频解码连续帧，直接调用保留的SAM2.1上游代码。"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=6000)
    ap.add_argument("--count", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("请使用新的测试目录")
    frames = out / "frames"
    frames.mkdir(parents=True)
    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    size = None
    for index in range(args.start, args.start + args.count):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"无法解码源帧{index}")
        size = [image.shape[1], image.shape[0]]
        if not cv2.imwrite(str(frames / f"frame_{index:08d}.jpg"), image):
            raise IOError("写帧失败")
    cap.release()
    t = time.time()
    command = [
        sys.executable,
        str(ROOT / "legacy/outdoor/sam2_bee_tracker.py"),
        "--checkpoint",
        args.checkpoint,
        "--frames-dir",
        str(frames),
        "--existing-dets-dir",
        args.detections,
        "--f-start",
        str(args.start),
        "--f-end",
        str(args.start + args.count - 1),
        "--fps",
        str(fps),
        "--work-dir",
        str(out / "scratch"),
        "--out-json",
        str(out / "raw.json"),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "legacy/outdoor/apply_postprocess_snapping.py"),
            "--in-json",
            str(out / "raw.json"),
            "--dets-dir",
            args.detections,
            "--frames-dir",
            str(frames),
            "--out-json",
            str(out / "snapped.json"),
            "--fps",
            str(fps),
        ],
        check=True,
    )
    result = json.loads((out / "snapped.json").read_text())
    result.setdefault("metadata", {}).update(
        fps=fps, width=size[0], height=size[1], source_frame_start=args.start
    )
    (out / "snapped.json").write_text(json.dumps(result))
    report = {
        "passed": True,
        "frames": len(result["frames"]),
        "source_video": args.video,
        "source_frame_range": [args.start, args.start + args.count - 1],
        "fps": fps,
        "elapsed_seconds": time.time() - t,
        "method": "unmodified_upstream_sam21_then_snapping",
        "accuracy_measurement": False,
    }
    (out / "smoke_status.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
