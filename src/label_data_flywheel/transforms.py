"""图像/框/姿态同步变换；派生样本继承源group和split。"""

import copy
import numpy as np


def affine_labels(frame, matrix, size):
    f = copy.deepcopy(frame)
    m = np.asarray(matrix, float)
    w, h = size

    def points(p):
        return np.c_[p, np.ones(len(p))] @ m.T

    f["image_size"] = list(size)
    f["derived_from"] = frame["sample_id"]
    for d in f["detections"]:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        p = points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        b = np.r_[p.min(0), p.max(0)]
        b[[0, 2]] = b[[0, 2]].clip(0, w)
        b[[1, 3]] = b[[1, 3]].clip(0, h)
        d["bbox_xyxy"] = b.tolist()
        for k, v in d.get("keypoints", {}).items():
            q = points([v[:2]])[0]
            d["keypoints"][k] = [
                *q.tolist(),
                *(v[2:] if 0 <= q[0] < w and 0 <= q[1] < h else [0.0]),
            ]
    f["detections"] = [
        d
        for d in f["detections"]
        if d["bbox_xyxy"][2] > d["bbox_xyxy"][0]
        and d["bbox_xyxy"][3] > d["bbox_xyxy"][1]
    ]
    for event in f.get("events", []):
        if "polygon" in event:
            event["polygon"] = points(event["polygon"]).tolist()
    return f


def transform_clip(images, frames, matrix, size):
    import cv2

    return [
        cv2.warpAffine(im, np.asarray(matrix, float), tuple(size)) for im in images
    ], [affine_labels(f, matrix, size) for f in frames]


def target_crop(frame, target_index, object_fraction=0.15, min_visibility=0.7):
    w, h = frame["image_size"]
    b = np.asarray(frame["detections"][target_index]["bbox_xyxy"])
    c = (b[:2] + b[2:]) / 2
    side = min(
        max(np.sqrt(np.prod(b[2:] - b[:2]) / object_fraction), max(b[2:] - b[:2])), w, h
    )
    origin = np.minimum(np.maximum(c - side / 2, 0), [w - side, h - side])
    m = np.array([[1, 0, -origin[0]], [0, 1, -origin[1]]])
    transformed = affine_labels(frame, m, [int(side), int(side)])
    original = {
        d["entity_id"]: np.prod(np.asarray(d["bbox_xyxy"])[2:] - d["bbox_xyxy"][:2])
        for d in frame["detections"]
    }
    transformed["detections"] = [
        d
        for d in transformed["detections"]
        if np.prod(np.asarray(d["bbox_xyxy"])[2:] - d["bbox_xyxy"][:2])
        / original[d["entity_id"]]
        >= min_visibility
    ]
    return m, (int(side), int(side)), transformed


def direction_rotation(angle, empirical, target, rng):
    empirical = np.asarray(empirical, float)
    p = np.asarray(target, float) / np.maximum(empirical, 1e-6)
    p /= p.sum()
    theta = (rng.choice(len(p), p=p) + rng.random()) * 2 * np.pi / len(p)
    return float(theta - angle)


def photometric(image, domain, rng, gain_range=(0.85, 1.15), noise=0.01):
    x = image.astype(float) / 255
    if domain == "IR_in":
        if x.ndim == 3:
            x = x.mean(2, keepdims=True).repeat(3, 2)
        x = x * rng.uniform(*gain_range) + rng.normal(0, noise, x.shape)
    else:
        x = x * rng.uniform(*gain_range, size=(1, 1, 3))
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def estimate_stabilization(reference, image, max_shift=40, min_inliers=0.7):
    import cv2

    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x
    a = gray(reference)
    b = gray(image)
    pts = cv2.goodFeaturesToTrack(a, 500, 0.02, 12)
    identity = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    if pts is None:
        return identity, {"accepted": False, "reason": "no_background_features"}
    nxt, valid, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    mask = valid.ravel().astype(bool)
    if mask.sum() < 8:
        return identity, {"accepted": False, "reason": "few_matches"}
    m, inliers = cv2.estimateAffinePartial2D(nxt[mask], pts[mask], method=cv2.RANSAC)
    ok = (
        m is not None
        and inliers.mean() >= min_inliers
        and np.linalg.norm(m[:, 2]) <= max_shift
        and 0.95 < np.linalg.norm(m[0, :2]) < 1.05
    )
    return (m if ok else identity), {
        "accepted": bool(ok),
        "inlier_ratio": float(inliers.mean()) if inliers is not None else 0.0,
    }


def tail_augment(images, frames, background, track_id, steps=3, seed=3407):
    """B-TCA可执行派生：用真实尾部速度延长局部软前景，不冒充真实标注。"""
    import cv2
    from .geometry import center

    anchors = [
        (im, f, d)
        for im, f in zip(images, frames)
        for d in f["detections"]
        if d.get("track_id") == track_id
    ]
    if len(anchors) < 2:
        raise ValueError("尾部增强需要至少两个真实观测")
    (im0, f0, d0), (im1, f1, d1) = anchors[-2:]
    delta = (center(d1) - center(d0)) / max(f1["frame"] - f0["frame"], 1)
    m, evidence = estimate_stabilization(background, im1)
    stable = cv2.warpAffine(im1, m, (im1.shape[1], im1.shape[0]))
    lab = affine_labels(f1, m, f1["image_size"])
    d = next(x for x in lab["detections"] if x.get("track_id") == track_id)
    box = np.rint(d["bbox_xyxy"]).astype(int)
    mask = np.zeros(im1.shape[:2], np.float32)
    residual = np.mean(np.abs(stable.astype(float) - background.astype(float)), 2)
    mask[box[1] : box[3], box[0] : box[2]] = np.clip(
        residual[box[1] : box[3], box[0] : box[2]] / 20.0, 0, 1
    )
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    outputs = []
    for step in range(1, steps + 1):
        move = np.array([[1.0, 0, delta[0] * step], [0, 1.0, delta[1] * step]])
        alpha = cv2.warpAffine(mask, move, (im1.shape[1], im1.shape[0]))[..., None]
        foreground = cv2.warpAffine(stable, move, (im1.shape[1], im1.shape[0]))
        image = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        isolated = copy.deepcopy(lab)
        isolated["detections"] = [d]
        derived = affine_labels(isolated, move, f1["image_size"])
        derived.update(
            frame=f1["frame"] + step,
            sample_id=f"{f1['sample_id']}/btca/{step}",
            synthetic=True,
            stabilization=evidence,
        )
        outputs.append((image, derived))
    return outputs
