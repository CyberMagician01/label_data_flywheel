import os
import sys

# 必须在导入 torch 之前设置 PCI_BUS_ID 屏蔽掉卡设备 81:00.0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0000:02:00.0"

import time
import glob
import json
import argparse
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def slice_image_coords(img_w, img_h, slice_w=640, slice_h=640, overlap_w=0.2, overlap_h=0.2):
    stride_w = int(slice_w * (1 - overlap_w))
    stride_h = int(slice_h * (1 - overlap_h))
    
    if img_w <= slice_w:
        x_starts = [0]
    else:
        x_starts = list(range(0, img_w - slice_w + 1, stride_w))
        if x_starts[-1] + slice_w < img_w:
            x_starts.append(img_w - slice_w)
            
    if img_h <= slice_h:
        y_starts = [0]
    else:
        y_starts = list(range(0, img_h - slice_h + 1, stride_h))
        if y_starts[-1] + slice_h < img_h:
            y_starts.append(img_h - slice_h)
            
    coords = []
    for y in sorted(list(set(y_starts))):
        for x in sorted(list(set(x_starts))):
            coords.append((x, y, x + slice_w, y + slice_h))
    return coords

def main():
    parser = argparse.ArgumentParser(description="Indoor Scene B Flywheel Inference (Tile Slicing & Merge Back)")
    parser.add_argument("--image-root", type=str, default="/root/autodl-tmp/flywheel_indoor_scene_b_20260904/frames", help="Root folder of indoor raw frames")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/indoor_scene_b_yolov8x_p2_framework_20260905/indoor_training_package/weights/scene_b_best_yolov8x_p2.pt", help="Unified YOLOv8x-P2 model weight")
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/indoor_scene_b_yolov8x_p2_framework_20260905/flywheel_outputs/pseudo_labels", help="Output directory for flywheel pseudo labels")
    parser.add_argument("--vis-dir", type=str, default="/root/autodl-tmp/indoor_scene_b_yolov8x_p2_framework_20260905/flywheel_outputs/inspection_samples", help="Output directory for visual inspection samples")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.25, help="Merge NMS IoU threshold (user specified 0.25)")
    parser.add_argument("--max-vis", type=int, default=20, help="Number of visualization samples to save")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    print("=" * 85, flush=True)
    print("🐝 [室内 Scene B 巢内监测全量数据飞轮标注启动]", flush=True)
    print(f"📦 图像根目录: {args.image_root}", flush=True)
    print(f"🎯 模型权重: {args.model_path}", flush=True)
    print(f"⚙️ 切片尺寸: 640x640 (重叠率 20%) | 拼接 NMS IoU 阈值: {args.iou} (用户指定)", flush=True)
    print(f"💾 伪标签输出: {labels_dir}", flush=True)
    print(f"🖼️ 抽样画框保存: {args.vis_dir}", flush=True)
    print(f"🖥️ 运行设备: {device}", flush=True)
    print("=" * 85, flush=True)

    all_images = sorted(glob.glob(f"{args.image_root}/**/*.jpg", recursive=True))
    if not all_images:
        all_images = sorted(glob.glob(f"{args.image_root}/**/*.png", recursive=True))

    if args.limit > 0:
        all_images = all_images[:args.limit]

    total_imgs = len(all_images)
    print(f"📊 待处理总帧数: {total_imgs} 张大图", flush=True)

    print("🚀 正在加载 YOLOv8x-P2 室内最佳模型 (FP16)...", flush=True)
    yolo = YOLO(args.model_path)
    model = yolo.model.to(device).half().eval()
    print("✅ 模型就绪！开始 Batched SAHI 切片前向推理与拼接...", flush=True)

    pbar = tqdm(all_images, desc="Scene B Flywheel Progress")
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    vis_saved = 0

    for img_path in pbar:
        rel_path = os.path.relpath(img_path, args.image_root)
        txt_out_path = os.path.join(labels_dir, os.path.splitext(rel_path)[0] + ".txt")

        # 支持断点续跑：如果标签已存在且非空，直接跳过
        if os.path.exists(txt_out_path) and os.path.getsize(txt_out_path) > 0:
            skipped_count += 1
            continue

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        H, W = img_cv.shape[:2]

        # 计算切片网格
        slice_coords = slice_image_coords(W, H, 640, 640, 0.2, 0.2)

        # 1. 批处理切片组织
        slice_tensors = []
        valid_coords = []
        for (x1, y1, x2, y2) in slice_coords:
            crop = img_cv[y1:y2, x1:x2]
            if crop.shape[0] != 640 or crop.shape[1] != 640:
                crop = cv2.resize(crop, (640, 640))
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_t = torch.from_numpy(crop_rgb).permute(2, 0, 1).to(device, non_blocking=True).half() / 255.0
            slice_tensors.append(crop_t)
            valid_coords.append((x1, y1, x2, y2))

        batch_tensor = torch.stack(slice_tensors, dim=0)

        # 2. 前向推理 (FP16 Batched)
        with torch.no_grad():
            preds = model(batch_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

        # YOLOv8x 输出格式: (batch, 4 + nc, num_anchors) -> 转置为 (batch, num_anchors, 4 + nc)
        preds = preds.transpose(1, 2)
        
        all_global_boxes = []
        all_global_scores = []

        for s_idx, (sx1, sy1, sx2, sy2) in enumerate(valid_coords):
            slice_pred = preds[s_idx]
            # 单类别 (nc=1): 第 4 维是 bee 的置信度
            cls_probs = slice_pred[:, 4]
            mask = cls_probs > args.conf
            if not mask.any():
                continue

            valid_preds = slice_pred[mask]
            valid_scores = cls_probs[mask]

            cx = valid_preds[:, 0]
            cy = valid_preds[:, 1]
            bw = valid_preds[:, 2]
            bh = valid_preds[:, 3]

            # 拼回全图坐标 (Global XYXY)
            px1 = (cx - bw / 2) + sx1
            py1 = (cy - bh / 2) + sy1
            px2 = (cx + bw / 2) + sx1
            py2 = (cy + bh / 2) + sy1

            # 截断到大图边界
            px1 = torch.clamp(px1, 0, W)
            py1 = torch.clamp(py1, 0, H)
            px2 = torch.clamp(px2, 0, W)
            py2 = torch.clamp(py2, 0, H)

            boxes = torch.stack([px1, py1, px2, py2], dim=1)
            all_global_boxes.append(boxes)
            all_global_scores.append(valid_scores)

        final_detections = []
        if len(all_global_boxes) > 0:
            all_boxes_t = torch.cat(all_global_boxes, dim=0)
            all_scores_t = torch.cat(all_global_scores, dim=0)

            # 3. 跨切片拼回大图 NMS (按照用户指定 iou=0.25)
            keep_indices = torchvision.ops.nms(all_boxes_t, all_scores_t, args.iou)
            
            b_kept = all_boxes_t[keep_indices].cpu().numpy()
            s_kept = all_scores_t[keep_indices].cpu().numpy()

            for box, score in zip(b_kept, s_kept):
                final_detections.append({
                    "class_id": 0,
                    "class_name": "bee",
                    "bbox": [round(float(x), 1) for x in box],
                    "conf": round(float(score), 3)
                })

        # 4. 保存为标准 YOLO 格式 TXT 伪标签
        os.makedirs(os.path.dirname(txt_out_path), exist_ok=True)
        with open(txt_out_path, 'w', encoding='utf-8') as f:
            for det in final_detections:
                x1, y1, x2, y2 = det["bbox"]
                norm_cx = ((x1 + x2) / 2) / W
                norm_cy = ((y1 + y2) / 2) / H
                norm_w = (x2 - x1) / W
                norm_h = (y2 - y1) / H
                f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {det['conf']:.3f}\n")

        # 5. 抽样可视化保存 (每个序列均匀抽取 5 张有效清晰帧)
        if args.vis_dir and (vis_saved < args.max_vis):
            frame_num_str = os.path.splitext(os.path.basename(img_path))[0].replace("frame_", "")
            try:
                frame_idx = int(frame_num_str)
            except ValueError:
                frame_idx = 0
            
            # 在有效光照帧区间 (如 500, 1500, 3000, 5000, 7000 等) 均匀抽样保存
            if frame_idx in [200, 1000, 2500, 5000, 7500]:
                vis_img = img_cv.copy()
                for det in final_detections:
                    bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
                    cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                
                seq_name = os.path.basename(os.path.dirname(img_path))
                frame_name = os.path.splitext(os.path.basename(img_path))[0]
                count_text = f"{seq_name} {frame_name} | Bee Count: {len(final_detections)} | IoU: {args.iou}"
                cv2.putText(vis_img, count_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                vis_file = os.path.join(args.vis_dir, f"{seq_name}_{frame_name}_count_{len(final_detections)}.jpg")
                cv2.imwrite(vis_file, vis_img)
                vis_saved += 1

        processed_count += 1

    total_time = time.time() - start_time
    print(f"\n🎉 飞轮推理完成！共处理 {processed_count} 张 (跳过已有 {skipped_count} 张)，耗时: {total_time/60:.2f} 分钟。", flush=True)

if __name__ == '__main__':
    main()
