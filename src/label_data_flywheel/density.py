"""密度残差选择单个局部视野复检；像素运动只提供候选证据。"""

import numpy as np
from .geometry import overlap, center


def count_preserving_map(points, size, sigma=1.5):
    width, height = size
    result = np.zeros((height, width), np.float32)
    radius = int(np.ceil(3 * sigma))
    for x, y in points:
        cx, cy = int(round(x)), int(round(y))
        x1 = max(cx - radius, 0)
        x2 = min(cx + radius + 1, width)
        y1 = max(cy - radius, 0)
        y2 = min(cy + radius + 1, height)
        if x1 >= x2 or y1 >= y2:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        result[y1:y2, x1:x2] += patch / max(patch.sum(), 1e-12)
    return result


def select_recheck_region(density, detections, image_size, window_fraction=0.3):
    import cv2

    density = np.asarray(density, float)
    h, w = density.shape
    iw, ih = image_size
    points = [center(d) * [w / iw, h / ih] for d in detections]
    observed = count_preserving_map(points, (w, h))
    residual = np.maximum(density - observed, 0)
    ww = max(1, int(w * window_fraction))
    hh = max(1, int(h * window_fraction))
    sums = cv2.boxFilter(
        residual, -1, (ww, hh), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    y, x = np.unravel_index(sums.argmax(), sums.shape)
    x1 = int(np.clip(x - ww // 2, 0, w - ww))
    y1 = int(np.clip(y - hh // 2, 0, h - hh))
    roi = [
        int(x1 * iw / w),
        int(y1 * ih / h),
        int((x1 + ww) * iw / w),
        int((y1 + hh) * ih / h),
    ]
    return {
        "roi_xyxy": roi,
        "estimated_missing_mass": float(residual[y1 : y1 + hh, x1 : x1 + ww].sum()),
        "window_fraction": window_fraction,
    }


def recheck(image, detections, density, detector, threshold=0.2):
    region = select_recheck_region(
        density, detections, [image.shape[1], image.shape[0]]
    )
    x1, y1, x2, y2 = region["roi_xyxy"]
    candidates = detector(image[y1:y2, x1:x2])
    kept = list(detections)
    for d in candidates:
        d = dict(d)
        d["bbox_xyxy"] = (np.asarray(d["bbox_xyxy"]) + [x1, y1, x1, y1]).tolist()
        om, area, _ = overlap([d["bbox_xyxy"]], [x["bbox_xyxy"] for x in kept])
        if not np.any((om >= threshold) & (area >= 16)):
            d["origin"] = "density_recheck"
            kept.append(d)
    return kept, region


def motion_evidence(previous, current, threshold=10.0):
    import cv2

    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    residual = cv2.absdiff(a, b)
    return {
        "flow": flow,
        "candidate_mask": (residual > threshold) & (np.linalg.norm(flow, axis=2) > 0.2),
        "meaning": "运动/亮度变化候选，不能独立证明蜜蜂存在",
    }
