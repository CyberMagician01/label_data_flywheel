"""按数据飞轮 frame_manifest 复现源帧选择；独立运行，无需安装飞轮包。"""

import argparse
import json
from pathlib import Path


def extract(video, video_id, manifest, output, index_base, jpeg_quality=95):
    import cv2

    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["video"] == video_id]
    if not rows:
        raise ValueError("清单中没有该视频")
    if any(r.get("frame_index_base") not in (None, index_base) for r in rows):
        raise ValueError("显式帧编号基准与清单不一致")
    requested = {r["source_frame_id"] - index_base: r for r in rows}
    if min(requested) < 0:
        raise ValueError("帧编号早于指定起始基准")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = {}
    for index, row in requested.items():
        target = (root / row["image_ref"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("图像路径超出输出目录")
        if target.exists():
            raise FileExistsError("不覆盖已有图像：" + str(target))
        targets[index] = target
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("无法打开源视频")
    count = 0
    try:
        # 顺序解码避免长 GOP 视频 seek 引入帧偏差。
        for index in range(max(requested) + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"视频在解码索引{index}提前结束")
            if index not in requested:
                continue
            row = requested[index]
            if image.shape[:2] != (row["height"], row["width"]):
                raise ValueError("源视频尺寸与标注清单不一致")
            success, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not success:
                raise ValueError("JPEG编码失败")
            target = targets[index]
            target.parent.mkdir(parents=True, exist_ok=True)
            # 字节写入支持中文/空格路径；不通过有编码差异的 imwrite。
            target.write_bytes(encoded.tobytes())
            count += 1
    finally:
        cap.release()
    report = {
        "video_id": video_id,
        "frames_written": count,
        "index_base": index_base,
        "jpeg_quality": jpeg_quality,
        "opencv_version": cv2.__version__,
        "reproduction_scope": "same decoded source frames; JPEG byte identity requires original encoder settings",
    }
    (root / f"extraction_{video_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-base", type=int, choices=(0, 1), required=True)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    print(
        json.dumps(
            extract(
                args.video,
                args.video_id,
                args.frame_manifest,
                args.output,
                args.index_base,
                args.jpeg_quality,
            ),
            ensure_ascii=False,
        )
    )
