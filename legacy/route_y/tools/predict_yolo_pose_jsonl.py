#!/usr/bin/env python3
"""Export YOLO pose predictions in the shared BeePoseTrack JSONL format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo26_bee_pose import register_ultralytics_modules
from data.dataset import FiveFrameManifestIndex, frame_statistics, read_rgb_or_ir
from models.route_context import begin_route_context


FRAME_RE = re.compile(r"(?P<video>[AB]-5-\d+)(?:_section(?P<section>\d+))?.*?frame_?0*(?P<frame>\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="Image directory, image file, or txt list.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="1")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=768)
    parser.add_argument("--model-name", default="BeePoseTrack-Y-Aligned")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--five-frame", action="store_true")
    return parser.parse_args()


def parse_ids(path: Path) -> tuple[str, str, int, str]:
    match = FRAME_RE.search(path.stem)
    if match:
        video_id = match.group("video").upper()
        section = match.group("section")
        section_id = f"{video_id}_区段_{int(section):02d}" if section else video_id
        return video_id, section_id, int(match.group("frame")), "RGB" if video_id.startswith("A") else "IR"
    fallback = path.parent.name
    return fallback, fallback, -1, "unknown"


def point_or_null(xy, conf):
    if xy is None or conf is None or float(conf) <= 0:
        return None
    return [round(float(xy[0]), 3), round(float(xy[1]), 3)]


def frame_to_tensor(frame, index: FiveFrameManifestIndex, imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    images = []
    for ref in index.causal_clip(frame):
        img = read_rgb_or_ir(ref.image_path, ref.domain)
        images.append(img)
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensors.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    clip = torch.cat(tensors, dim=0).unsqueeze(0)
    motion = 0.0
    if len(images) > 1:
        a = cv2.cvtColor(images[-1], cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
        b = cv2.cvtColor(images[-2], cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        motion = float(abs(a - b).mean())
    stats = torch.from_numpy(frame_statistics(images[-1], frame.domain, len(frame.instances), motion)).float().unsqueeze(0)
    return clip, stats


def write_record(handle, result, image_path: Path, video_id: str, section_id: str, frame_id: int, domain: str, model_name: str, checkpoint: str, scale_xy: tuple[float, float] = (1.0, 1.0)) -> None:
    boxes = result.boxes
    keypoints = result.keypoints
    detections = []
    n = 0 if boxes is None else len(boxes)
    xyxy = boxes.xyxy.cpu().numpy() if n else []
    scores = boxes.conf.cpu().numpy() if n else []
    kxy = keypoints.xy.cpu().numpy() if keypoints is not None else None
    kconf = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else None
    sx, sy = scale_xy
    for i in range(n):
        head_score = float(kconf[i, 0]) if kconf is not None else 0.0
        tail_score = float(kconf[i, 1]) if kconf is not None else 0.0
        pose_score = (head_score + tail_score) / 2.0 if kconf is not None else 0.0
        box = xyxy[i].copy()
        box[[0, 2]] *= sx
        box[[1, 3]] *= sy
        pts = None if kxy is None else kxy[i].copy()
        if pts is not None:
            pts[:, 0] *= sx
            pts[:, 1] *= sy
        detections.append(
            {
                "bbox_xyxy": [round(float(v), 3) for v in box.tolist()],
                "score": round(float(scores[i]), 6),
                "head": point_or_null(pts[0] if pts is not None else None, head_score),
                "tail": point_or_null(pts[1] if pts is not None else None, tail_score),
                "head_score": round(head_score, 6),
                "tail_score": round(tail_score, 6),
                "pose_score": round(pose_score, 6),
            }
        )
    record = {
        "video_id": video_id,
        "section_id": section_id,
        "frame_id": frame_id,
        "domain": domain,
        "image_path": str(image_path),
        "model_name": model_name,
        "checkpoint": checkpoint,
        "detections": detections,
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    register_ultralytics_modules()
    model = YOLO(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        if args.five_frame:
            if args.manifest is None:
                raise ValueError("--five-frame requires --manifest")
            index = FiveFrameManifestIndex(args.manifest, args.eval_split, 5, 1)
            for frame in index.frames:
                clip, stats = frame_to_tensor(frame, index, args.imgsz)
                domain_ids = torch.tensor([1 if frame.domain.upper() == "IR" else 0], dtype=torch.long)
                begin_route_context(clip=clip.view(1, 5, 3, args.imgsz, args.imgsz), domain_ids=domain_ids, stats=stats, source="five_frame_infer")
                result = model.predict(
                    source=clip,
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    stream=False,
                    verbose=False,
                )[0]
                scale_xy = (float(frame.width or args.imgsz) / float(args.imgsz), float(frame.height or args.imgsz) / float(args.imgsz))
                write_record(handle, result, frame.image_path, frame.video_id, frame.section_id, frame.frame_id, frame.domain, args.model_name, str(Path(args.model).resolve()), scale_xy=scale_xy)
            return
        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            stream=True,
            verbose=False,
        )
        for result in results:
            image_path = Path(result.path)
            video_id, section_id, frame_id, domain = parse_ids(image_path)
            write_record(handle, result, image_path, video_id, section_id, frame_id, domain, args.model_name, str(Path(args.model).resolve()))


if __name__ == "__main__":
    main()
