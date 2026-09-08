import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PoseVerifier:
    def __init__(self, vitpose_root, config, checkpoint, device):
        sys.path.insert(0, str(vitpose_root))
        from mmcv.parallel import collate, scatter
        from mmpose.apis import init_pose_model
        from mmpose.apis.inference import _box2cs
        from mmpose.datasets import DatasetInfo
        from mmpose.datasets.builder import PIPELINES
        from mmpose.datasets.pipelines import Compose

        @PIPELINES.register_module(force=True)
        class SetBeeDatasetIndex:
            def __init__(self, dataset_idx=0):
                self.dataset_idx = dataset_idx

            def __call__(self, results):
                results["dataset_idx"] = self.dataset_idx
                return results

        self.collate = collate
        self.scatter = scatter
        self.box2cs = _box2cs
        self.model = init_pose_model(str(config), checkpoint=None, device=f"cuda:{device}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        incompatible = self.model.load_state_dict(state["state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"姿态权重结构不匹配: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model.eval()
        if "test_pipeline" not in self.model.cfg:
            self.model.cfg.test_pipeline = copy.deepcopy(self.model.cfg.data.test.pipeline)
        self.model.cfg.test_pipeline.insert(0, dict(type="SetBeeDatasetIndex", dataset_idx=0))
        self.dataset_info = DatasetInfo(self.model.cfg.data.test.dataset_info)
        self.pipeline = Compose(self.model.cfg.test_pipeline)

    def __call__(self, image, boxes, batch_size=128):
        if not boxes:
            return np.empty((0, 2, 3), dtype=np.float32)
        prepared = []
        for bbox_id, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            bbox = np.asarray([x1, y1, x2 - x1 + 1, y2 - y1 + 1, box[4]], dtype=np.float32)
            center, scale = self.box2cs(self.model.cfg, bbox)
            data = {
                "center": center,
                "scale": scale,
                "bbox_score": float(box[4]),
                "bbox_id": bbox_id,
                "dataset": self.dataset_info.dataset_name,
                "joints_3d": np.zeros((self.model.cfg.data_cfg.num_joints, 3), dtype=np.float32),
                "joints_3d_visible": np.zeros((self.model.cfg.data_cfg.num_joints, 3), dtype=np.float32),
                "rotation": 0,
                "ann_info": {
                    "image_size": np.asarray(self.model.cfg.data_cfg.image_size),
                    "num_joints": self.model.cfg.data_cfg.num_joints,
                    "flip_pairs": self.dataset_info.flip_pairs,
                },
                "img": image,
            }
            prepared.append(self.pipeline(data))

        outputs = []
        for start in range(0, len(prepared), batch_size):
            batch = self.collate(prepared[start:start + batch_size], samples_per_gpu=batch_size)
            batch = self.scatter(batch, [next(self.model.parameters()).device])[0]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                result = self.model(
                    img=batch["img"], img_metas=batch["img_metas"],
                    return_loss=False, return_heatmap=False,
                )
            outputs.append(np.asarray(result["preds"], dtype=np.float32))
        predictions = np.concatenate(outputs, axis=0)
        if predictions.shape != (len(boxes), 2, 3) or not np.isfinite(predictions).all():
            raise RuntimeError(f"非法姿态输出: {predictions.shape}")
        return predictions


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match_counts(predictions, ground_truth, threshold=0.5):
    order = sorted(range(len(predictions)), key=lambda i: predictions[i][4], reverse=True)
    unmatched = set(range(len(ground_truth)))
    tp = 0
    for pred_index in order:
        candidates = [(iou(predictions[pred_index][:4], ground_truth[j]), j) for j in unmatched]
        if candidates:
            overlap, gt_index = max(candidates)
            if overlap >= threshold:
                unmatched.remove(gt_index)
                tp += 1
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(unmatched)}


def nms(predictions, threshold=0.45):
    if not predictions:
        return []
    order = sorted(range(len(predictions)), key=lambda i: predictions[i][4], reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [i for i in order if iou(predictions[current][:4], predictions[i][:4]) < threshold]
    return [predictions[i] for i in keep]


def read_yolo_labels(path, width, height, confidence_threshold):
    predictions = []
    if not path.exists():
        return predictions
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(v) for v in line.split()]
        if len(values) < 5:
            continue
        _, cx, cy, bw, bh, *tail = values
        confidence = tail[0] if tail else 1.0
        if confidence < confidence_threshold:
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        predictions.append([x1, y1, x2, y2, confidence])
    return predictions


def dynamic_low_density_centers(boxes, width, height, cell_w, cell_h,
                                neighbor_threshold, max_candidates=12, scale=4):
    """在连续密度图上找局部低谷，不把候选位置锁死在固定网格交点。"""
    map_w = int(np.ceil(width / scale))
    map_h = int(np.ceil(height / scale))
    occupancy = np.zeros((map_h, map_w), dtype=np.float32)
    for box in boxes:
        cx = int(np.clip(round((box[0] + box[2]) / (2 * scale)), 0, map_w - 1))
        cy = int(np.clip(round((box[1] + box[3]) / (2 * scale)), 0, map_h - 1))
        occupancy[cy, cx] += 1.0

    central = cv2.boxFilter(
        occupancy, -1,
        (max(1, round(cell_w / scale)), max(1, round(cell_h / scale))),
        normalize=False, borderType=cv2.BORDER_CONSTANT,
    )
    neighborhood = cv2.boxFilter(
        occupancy, -1,
        (max(1, round(3 * cell_w / scale)), max(1, round(3 * cell_h / scale))),
        normalize=False, borderType=cv2.BORDER_CONSTANT,
    )
    ring = neighborhood - central
    valid = (central < 0.5) & (ring >= float(neighbor_threshold))

    # 局部最大值给出随检测分布变化的位置；随后用距离抑制去掉同一空洞的重复点。
    local_peak = ring >= cv2.dilate(ring, np.ones((9, 9), np.uint8))
    ys, xs = np.where(valid & local_peak)
    candidates = sorted(
        [(float(ring[y, x]), int(x * scale), int(y * scale)) for y, x in zip(ys, xs)],
        reverse=True,
    )
    selected = []
    min_dx, min_dy = 0.70 * cell_w, 0.70 * cell_h
    for density, cx, cy in candidates:
        if any(((cx - old[0]) / min_dx) ** 2 + ((cy - old[1]) / min_dy) ** 2 < 1.0
               for old in selected):
            continue
        selected.append((cx, cy, density))
        if len(selected) >= max_candidates:
            break
    return selected


def centered_crop_with_padding(image, cx, cy, crop_w, crop_h):
    image_h, image_w = image.shape[:2]
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x2, y2 = x1 + crop_w, y1 + crop_h
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(image_w, x2), min(image_h, y2)
    crop = image[sy1:sy2, sx1:sx2]
    top, bottom = sy1 - y1, y2 - sy2
    left, right = sx1 - x1, x2 - sx2
    if top or bottom or left or right:
        fill = tuple(int(v) for v in np.median(image.reshape(-1, 3), axis=0))
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=fill)
    return crop, x1, y1


def draw(image, boxes, color, thickness=2):
    output = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return output


def add_header(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(image, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def frame_paths(package_root, file_name):
    stem = Path(file_name).stem
    video = stem.split("_frame_")[0]
    frame_number = int(stem.rsplit("_", 1)[-1])
    image = package_root / "frames" / video / f"frame_{frame_number:08d}.jpg"
    label = package_root / "labels" / video / f"frame_{frame_number:08d}.txt"
    return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cell-w", type=int, default=192)
    parser.add_argument("--cell-h", type=int, default=135)
    parser.add_argument("--neighbor-threshold", type=int, default=2)
    parser.add_argument("--recheck-w", type=int, default=384)
    parser.add_argument("--recheck-h", type=int, default=270)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--recheck-confidence", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vitpose-root", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--pose-weights", type=Path, required=True)
    parser.add_argument("--direct-confidence", type=float, default=0.25)
    parser.add_argument("--pose-min-confidence", type=float, default=0.30)
    parser.add_argument("--pose-mean-confidence", type=float, default=0.38)
    parser.add_argument("--pose-min-separation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations_by_image = {}
    for annotation in coco["annotations"]:
        x, y, w, h = annotation["bbox"]
        annotations_by_image.setdefault(annotation["image_id"], []).append([x, y, x + w, y + h])

    images = coco["images"]
    if len(images) > args.samples:
        indices = np.linspace(0, len(images) - 1, args.samples, dtype=int)
        images = [images[i] for i in indices]

    model = YOLO(str(args.weights))
    pose_verifier = PoseVerifier(
        args.vitpose_root, args.pose_config, args.pose_weights, args.device
    )
    summary = []
    for image_info in images:
        image_path, label_path = frame_paths(args.package_root, image_info["file_name"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        baseline = read_yolo_labels(label_path, width, height, args.confidence)
        gt = annotations_by_image.get(image_info["id"], [])
        centers = dynamic_low_density_centers(
            baseline, width, height, args.cell_w, args.cell_h, args.neighbor_threshold,
            max_candidates=args.max_candidates,
        )

        crops, crop_meta = [], []
        for cx, cy, density in centers:
            crop, origin_x, origin_y = centered_crop_with_padding(
                image, cx, cy, args.recheck_w, args.recheck_h
            )
            crops.append(crop)
            crop_meta.append((cx, cy, density, origin_x, origin_y))

        additions = []
        if crops:
            results = model.predict(crops, imgsz=640, conf=args.recheck_confidence, iou=0.45,
                                    device=args.device, half=True, batch=min(16, len(crops)), verbose=False)
            for result, (cx, cy, _, origin_x, origin_y) in zip(results, crop_meta):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    # 复检切片边缘的截断预测不可信，直接丢弃。
                    if x1 <= 3 or y1 <= 3 or x2 >= args.recheck_w - 3 or y2 >= args.recheck_h - 3:
                        continue
                    full = [x1 + origin_x, y1 + origin_y, x2 + origin_x, y2 + origin_y, float(score)]
                    box_cx = (full[0] + full[2]) / 2
                    box_cy = (full[1] + full[3]) / 2
                    # 仅接收中心落在较小目标密度区中的框。
                    if not (cx - args.cell_w / 2 <= box_cx < cx + args.cell_w / 2 and
                            cy - args.cell_h / 2 <= box_cy < cy + args.cell_h / 2):
                        continue
                    full[0] = max(0.0, full[0])
                    full[1] = max(0.0, full[1])
                    full[2] = min(float(width), full[2])
                    full[3] = min(float(height), full[3])
                    additions.append(full)

        # 低检测置信度框可由两个高置信关键点共同确认；关键点还必须位于框内。
        additions = nms(additions, 0.45)
        additions = [box for box in additions if not any(iou(box[:4], old[:4]) >= 0.45 for old in baseline)]
        pose_predictions = pose_verifier(image, additions)
        verified_additions = []
        verified_keypoints = []
        accepted_direct = 0
        accepted_by_pose = 0
        for box, keypoints in zip(additions, pose_predictions):
            x1, y1, x2, y2 = box[:4]
            both_inside = all(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for point in keypoints)
            pose_min = float(keypoints[:, 2].min())
            pose_mean = float(keypoints[:, 2].mean())
            diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), 1e-6)
            separation_ratio = float(np.linalg.norm(keypoints[0, :2] - keypoints[1, :2]) / diagonal)
            direct = box[4] >= args.direct_confidence
            pose_confirmed = (
                both_inside
                and pose_min >= args.pose_min_confidence
                and pose_mean >= args.pose_mean_confidence
                and separation_ratio >= args.pose_min_separation_ratio
            )
            if direct or pose_confirmed:
                verified_additions.append(box)
                verified_keypoints.append(keypoints.tolist())
                accepted_direct += int(direct)
                accepted_by_pose += int(not direct and pose_confirmed)
        additions = verified_additions
        merged = baseline + additions
        before = match_counts(baseline, gt)
        after = match_counts(merged, gt)

        gt_panel = add_header(draw(image, gt, (40, 210, 40), 2), f"GT: {len(gt)}")
        before_panel = add_header(draw(image, baseline, (220, 0, 220), 1),
                                  f"Before: {len(baseline)} TP/FP/FN={before['tp']}/{before['fp']}/{before['fn']}")
        candidate_panel = draw(image, baseline, (220, 0, 220), 1)
        for cx, cy, density in centers:
            tx1, ty1 = int(cx - args.cell_w / 2), int(cy - args.cell_h / 2)
            tx2, ty2 = int(cx + args.cell_w / 2), int(cy + args.cell_h / 2)
            rx1, ry1 = int(cx - args.recheck_w / 2), int(cy - args.recheck_h / 2)
            rx2, ry2 = int(cx + args.recheck_w / 2), int(cy + args.recheck_h / 2)
            cv2.rectangle(candidate_panel, (rx1, ry1), (rx2, ry2), (0, 150, 255), 2)
            cv2.rectangle(candidate_panel, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)
            cv2.circle(candidate_panel, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(candidate_panel, f"n={density:.0f}", (tx1 + 3, ty1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        candidate_panel = add_header(candidate_panel, f"Dynamic recheck regions: {len(centers)}")
        after_panel = draw(image, baseline, (220, 0, 220), 1)
        after_panel = draw(after_panel, additions, (255, 255, 0), 3)
        for keypoints in verified_keypoints:
            head, tail = keypoints
            cv2.circle(after_panel, (round(head[0]), round(head[1])), 4, (0, 0, 255), -1)
            cv2.circle(after_panel, (round(tail[0]), round(tail[1])), 4, (255, 180, 0), -1)
        after_panel = add_header(
            after_panel,
            f"After: {len(merged)} (+{len(additions)} det={accepted_direct} pose={accepted_by_pose}) "
            f"TP/FP/FN={after['tp']}/{after['fp']}/{after['fn']}",
        )
        canvas = cv2.vconcat([cv2.hconcat([gt_panel, before_panel]),
                             cv2.hconcat([candidate_panel, after_panel])])
        output_path = args.output / f"{Path(image_info['file_name']).stem}_comparison.jpg"
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        summary.append({
            "image": str(image_path), "ground_truth": len(gt), "baseline": len(baseline),
            "dynamic_regions": len(centers), "added": len(additions),
            "before": before, "after": after, "visualization": str(output_path),
            "dynamic_centers": centers, "added_boxes": additions,
            "added_keypoints": verified_keypoints,
            "accepted_direct": accepted_direct, "accepted_by_pose": accepted_by_pose,
        })
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

    (args.output / "summary.json").write_text(
        json.dumps({"parameters": vars(args), "samples": summary}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
