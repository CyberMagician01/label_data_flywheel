import os
import sys
import time
import json
import pickle
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 1. Add CountGDPlusPlus to sys.path
BASE_PATH = Path(os.environ.get(
    "COUNTGD_TRAIN_ROOT",
    "/root/autodl-tmp/third_party/CountGDPlusPlus/training/countgd_plusplus_training",
))
sys.path.insert(0, str(BASE_PATH))

import datasets.transforms as T
from models import build_model
from util.misc import clean_state_dict, NestedTensor
from util.slconfig import SLConfig

def compute_iou_xyxy(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    w = max(0.0, ix2 - ix1)
    h = max(0.0, iy2 - iy1)
    inter = w * h
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def slice_image(img, slice_w=640, slice_h=640, overlap_w=0.2, overlap_h=0.2):
    h, w = img.shape[:2]
    stride_w = int(slice_w * (1 - overlap_w))
    stride_h = int(slice_h * (1 - overlap_h))
    x_starts = list(range(0, w - slice_w + 1, stride_w))
    if x_starts[-1] + slice_w < w:
        x_starts.append(w - slice_w)
    y_starts = list(range(0, h - slice_h + 1, stride_h))
    if y_starts[-1] + slice_h < h:
        y_starts.append(h - slice_h)
    slices = []
    for y in y_starts:
        for x in x_starts:
            crop = img[y:y + slice_h, x:x + slice_w]
            slices.append((crop, x, y))
    return slices

def nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return [boxes[i].tolist() for i in keep], [scores[i].item() for i in keep]

def load_countgdpp(device="cuda:0"):
    config_path = BASE_PATH / "config/countgd_box_a.py"
    if not config_path.exists():
        config_path = BASE_PATH / "config/cfg_odvg.py"
    checkpoint_path = os.environ.get(
        "COUNTGD_CHECKPOINT",
        "/root/autodl-tmp/third_party/CountGDPlusPlus/checkpoints/countgd_plusplus.pth",
    )
    args = SLConfig.fromfile(str(config_path))
    args.device = device
    args.text_encoder_type = os.environ.get(
        "COUNTGD_BERT_ROOT",
        "/root/autodl-tmp/third_party/CountGDPlusPlus/training/countgd_plusplus_training/checkpoints/bert-base-uncased",
    )
    model, _, _ = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    return model

def run_countgdpp_sahi(model, img_cv, exemplar_boxes, device="cuda:0", slice_size=640, overlap=0.2, conf_thresh=0.20):
    H, W = img_cv.shape[:2]
    slices = slice_image(img_cv, slice_size, slice_size, overlap, overlap)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ex_tensors = [[eb[0]/W, eb[1]/H, eb[2]/W, eb[3]/H] for eb in exemplar_boxes]
    ex_tensor = torch.tensor(ex_tensors, dtype=torch.float32, device=device)

    all_boxes = []
    all_scores = []

    for crop, sx, sy in slices:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ch, cw = crop.shape[:2]
        img_t, _ = transform(crop_pil, None)
        img_t = img_t.to(device)

        mask = torch.zeros((1, img_t.shape[1], img_t.shape[2]), dtype=torch.bool, device=device)
        samples = NestedTensor(img_t.unsqueeze(0), mask)
        exemplars = [ex_tensor]
        exemplar_labels = [torch.zeros(len(ex_tensors), dtype=torch.long, device=device)]

        with torch.no_grad():
            outputs = model(samples, exemplars, exemplar_labels, captions=["bee ."])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        mask_f = logits.max(dim=1)[0] > conf_thresh
        sub_logits = logits[mask_f]
        sub_boxes = boxes[mask_f]

        for box, logit in zip(sub_boxes, sub_logits):
            cx, cy, bw, bh = box.numpy()
            px1 = (cx - bw / 2) * cw + sx
            py1 = (cy - bh / 2) * ch + sy
            px2 = (cx + bw / 2) * cw + sx
            py2 = (cy + bh / 2) * ch + sy
            score = logit.max().item()
            all_boxes.append([px1, py1, px2, py2])
            all_scores.append(score)

    merged_boxes, merged_scores = nms(all_boxes, all_scores, iou_thresh=0.45)
    return merged_boxes, merged_scores

def evaluate_predictions(preds_xyxy, gt_boxes_xywh, iou_thresh=0.20):
    gt_xyxy = [[g[0], g[1], g[0]+g[2], g[1]+g[3]] for g in gt_boxes_xywh]
    matched_gt = set()
    matched_pred = set()
    
    for p_idx, p in enumerate(preds_xyxy):
        best_iou = 0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_xyxy):
            if g_idx in matched_gt:
                continue
            iou = compute_iou_xyxy(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_iou >= iou_thresh:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            
    fp = len(preds_xyxy) - len(matched_pred)
    fn = len(gt_xyxy) - len(matched_gt)
    return fp, fn

def get_intersection_boxes(boxes1, boxes2, iou_thresh):
    matched = []
    for b1 in boxes1:
        for b2 in boxes2:
            if compute_iou_xyxy(b1, b2) >= iou_thresh:
                matched.append(b1)
                break
    return matched

def main():
    print("=" * 95)
    print("🐝 [Scene B 全量 420 张大图 双模型微小目标自适应交集评测 (多 IoU 阈值寻优)]")
    print("=" * 95)

    cache_file = os.environ.get(
        "SCENE_B_CACHE",
        "/root/autodl-tmp/indoor_scene_b_yolov8x_p2_framework_20260905/scene_B_cache.pkl",
    )
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)

    all_items = list(cache_data.items())
    total_imgs = len(all_items)
    print(f"📊 载入 Scene B 全量大图: 共 {total_imgs} 张。")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📦 正在加载 Count-GD++ 视觉模型 (Device: {device})...")
    count_model = load_countgdpp(device=device)
    print("✅ Count-GD++ 加载就绪！\n")

    IOU_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]

    stats_list = []
    start_time = time.time()
    pbar = tqdm(all_items, desc="Scene-B 全量 420 张多IoU交集进度", total=total_imgs)

    for idx, (img_path, data) in enumerate(pbar):
        gt_boxes_xywh = data['gt_boxes']
        group = data['group']

        # 1. YOLO 预测框 (Conf >= 0.25)
        yolo_preds = data['predictions']
        yolo_boxes = []
        for p in yolo_preds:
            if p['conf'] >= 0.25:
                b = p.get('bbox', p.get('box'))
                yolo_boxes.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])

        # 2. 读取大图并挑选 3 个参考蜂
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue

        step = max(1, len(gt_boxes_xywh) // 3)
        exemplars = []
        for ex_idx in [0, step, min(len(gt_boxes_xywh)-1, 2*step)]:
            g = gt_boxes_xywh[ex_idx]
            exemplars.append([g[0], g[1], g[0]+g[2], g[1]+g[3]])

        # 3. Count-GD++ SAHI
        count_boxes, _ = run_countgdpp_sahi(count_model, img_cv, exemplars, device=device, conf_thresh=0.20)

        # 4. 单模型评估
        y_fp, y_fn = evaluate_predictions(yolo_boxes, gt_boxes_xywh, iou_thresh=0.20)
        c_fp, c_fn = evaluate_predictions(count_boxes, gt_boxes_xywh, iou_thresh=0.20)

        # 5. 多 IoU 门槛交集评估
        inter_results = {}
        for th in IOU_THRESHOLDS:
            inter_b = get_intersection_boxes(yolo_boxes, count_boxes, iou_thresh=th)
            i_fp, i_fn = evaluate_predictions(inter_b, gt_boxes_xywh, iou_thresh=0.20)
            inter_results[th] = {"fp": i_fp, "fn": i_fn, "count": len(inter_b)}

        stats_list.append({
            "img_path": img_path,
            "group": group,
            "gt_count": len(gt_boxes_xywh),
            "yolo_fp": y_fp, "yolo_fn": y_fn,
            "count_fp": c_fp, "count_fn": c_fn,
            "inter": inter_results
        })

        curr_n = len(stats_list)
        pbar.set_postfix({
            "YOLO_FP": f"{sum(s['yolo_fp'] for s in stats_list)/curr_n:.1f}",
            "Count_FP": f"{sum(s['count_fp'] for s in stats_list)/curr_n:.1f}",
            "Inter0.15_FP": f"{sum(s['inter'][0.15]['fp'] for s in stats_list)/curr_n:.1f}"
        })

    total_time = time.time() - start_time
    n = len(stats_list)
    if n == 0:
        return

    y_fp_tot = sum(s["yolo_fp"] for s in stats_list)
    y_fn_tot = sum(s["yolo_fn"] for s in stats_list)
    c_fp_tot = sum(s["count_fp"] for s in stats_list)
    c_fn_tot = sum(s["count_fn"] for s in stats_list)

    print("\n" + "=" * 105)
    print("📈 [Scene B 全量 420 张大图最终多 IoU 门槛交集对比汇总表 (138,987 只标注蜜蜂)]")
    print("=" * 105)
    print(f"{'评测方案':<32} | {'平均每图 FP (误检)':<18} | {'平均每图 FN (漏检)':<18} | {'全图总误检数':<12} | {'全图总漏检数':<12}")
    print("-" * 105)
    print(f"{'1. 原生 YOLOv8x-P2':<32} | {y_fp_tot/n:<18.2f} | {y_fn_tot/n:<18.2f} | {y_fp_tot:<12} | {y_fn_tot:<12}")
    print(f"{'2. Count-GD++ (3示例)':<32} | {c_fp_tot/n:<18.2f} | {c_fn_tot/n:<18.2f} | {c_fp_tot:<12} | {c_fn_tot:<12}")
    print("-" * 105)
    for th in IOU_THRESHOLDS:
        th_fp = sum(s["inter"][th]["fp"] for s in stats_list)
        th_fn = sum(s["inter"][th]["fn"] for s in stats_list)
        fp_impr = (y_fp_tot - th_fp) / y_fp_tot * 100 if y_fp_tot > 0 else 0
        label = f"3. 🤝 双模型交集 (IoU >= {th:.2f})"
        print(f"{label:<32} | {th_fp/n:<18.2f} | {th_fn/n:<18.2f} | {th_fp:<12} | {th_fn:<12} (FP 压降 {fp_impr:+.1f}%)")
    print("=" * 105)

if __name__ == '__main__':
    main()
