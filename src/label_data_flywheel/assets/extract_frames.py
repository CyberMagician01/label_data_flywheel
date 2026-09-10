"""默认将官方原视频逐帧导出为 JPG；保留原始帧序、尺寸和标注对应编号。"""

import argparse
import json
import re
from pathlib import Path


def extract(video, video_id, manifest, output, index_base, jpeg_quality=95, annotations=None):
    import cv2

    video_id = video_id or Path(video).stem
    if manifest and annotations:
        raise ValueError('按清单选帧与按标注选帧只能选择一种')
    rows = None
    if manifest:
        # 兼容历史调用；只有显式提供筛选输入时才抽取部分帧。
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in rows if r["video"] == video_id]
    elif annotations:
        label_root = Path(annotations)
        rows = []
        for label in sorted((label_root / video_id).glob('*.txt')):
            match = re.search(r'(\d+)$', label.stem)
            if not match:
                raise ValueError('标注文件名末尾没有源帧编号：' + label.name)
            rows.append({'source_frame_id':int(match.group(1)), 'image_ref':f'{video_id}/{label.stem}.jpg', 'frame_index_base':index_base})
    requested = None
    if rows is not None:
        if not rows:
            raise ValueError("没有找到该视频对应的标注帧")
        if any(r.get("frame_index_base") not in (None, index_base) for r in rows):
            raise ValueError("显式帧编号基准与清单不一致")
        requested = {r["source_frame_id"] - index_base: r for r in rows}
        if len(requested) != len(rows):
            raise ValueError('同一源帧对应多个标注文件名')
        if min(requested) < 0:
            raise ValueError("帧编号早于指定起始基准")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    def target_path(image_ref):
        target = (root / image_ref).resolve()
        target.relative_to(root)
        if target.exists():
            raise FileExistsError("不覆盖已有图像：" + str(target))
        return target

    targets = {i: target_path(r["image_ref"]) for i, r in (requested or {}).items()}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("无法打开源视频")
    count = 0
    index = 0
    last_requested = max(requested) if requested is not None else None
    try:
        # 从首帧顺序解码到视频末尾；不按目标 FPS 重采样，也不默认五抽一。
        while last_requested is None or index <= last_requested:
            ok, image = cap.read()
            if not ok:
                if requested is not None:
                    raise ValueError(f"视频在解码索引{index}提前结束")
                break
            if requested is not None and index not in requested:
                index += 1
                continue
            row = requested[index] if requested is not None else {}
            if 'height' in row and 'width' in row and image.shape[:2] != (row["height"], row["width"]):
                raise ValueError("源视频尺寸与标注清单不一致")
            success, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not success:
                raise ValueError("JPEG编码失败")
            target = targets[index] if requested is not None else target_path(
                f'{video_id}/frame_{index + index_base:08d}.jpg'
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            # 字节写入支持中文/空格路径；不通过有编码差异的 imwrite。
            target.write_bytes(encoded.tobytes())
            count += 1
            index += 1
    finally:
        cap.release()
    if count == 0:
        raise ValueError('原视频未解码出任何帧')
    report = {
        "video_id": video_id,
        "frames_written": count,
        "mode": "all_frames" if requested is None else "selected_frames",
        "index_base": index_base,
        "jpeg_quality": jpeg_quality,
        "opencv_version": cv2.__version__,
        "reproduction_scope": "same decoded source frames; JPEG byte identity requires original encoder settings",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", help='视频目录名，默认使用输入视频文件名（不含扩展名）')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--annotations", help='可选：仅提取该检测标注根目录下已有标注对应的帧')
    selection.add_argument("--frame-manifest", help='可选：仅提取历史清单指定的帧')
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=0)
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
                args.annotations,
            ),
            ensure_ascii=False,
        )
    )
