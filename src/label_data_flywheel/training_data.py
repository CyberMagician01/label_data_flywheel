"""将确认标签、软伪正和忽略区转换为训练器实际读取的 schema v2。"""

from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .io import write_json, sha256
from .semantics import is_bee


def export_coco(frames, destination, image_root):
    images, annotations = [], []
    identities = {}
    for image_id, f in enumerate(frames, 1):
        w, h = f["image_size"]
        scope = (f["domain"], f["video"], f.get("segment", f["group"]))
        valid = f.get("status", "valid") not in ("suspect", "invalid")
        images.append(
            {
                "id": image_id,
                "file_name": f.get("image_path")
                or str(Path(image_root) / f["video"] / f"frame_{f['frame']:08d}.jpg"),
                "width": w,
                "height": h,
                "domain": "IR" if f["domain"] == "IR_in" else "RGB",
                "environment": "indoor" if f["domain"] == "IR_in" else "outdoor",
                "sensor_id": f.get("sensor_id", f["domain"]),
                "sequence_id": f["video"],
                "video_id": f["video"],
                "frame_id": f["frame"],
                "group": f["group"],
                "section_id": str(scope[2]),
                "track_scope": "/".join(map(str, scope)),
                "split": f["split"],
                "track_supervised": False,
                "sampling_weight": float(f.get("sampling_weight", 1.0))
                if valid
                else 0.0,
                "status": f.get("status", "valid"),
                "annotation_complete": bool(f.get("annotation_complete", False)),
                "is_unlabeled": not bool(f.get("annotation_complete")) and not any(
                    d.get("label_status") in ("human", "human_confirmed")
                    for d in f["detections"] if is_bee(d)
                ),
                "verified_background_boxes": [
                    [r["bbox_xyxy"][0], r["bbox_xyxy"][1],
                     r["bbox_xyxy"][2] - r["bbox_xyxy"][0],
                     r["bbox_xyxy"][3] - r["bbox_xyxy"][1]]
                    for r in f.get("verified_background_regions", [])
                ],
                "source_sample_id": f["sample_id"],
            }
        )
        candidates = [*f["detections"], *f.get("ignore_regions", [])]
        for d in candidates:
            # 双域共享实体模型学习蜂体；影子仍在原始与双类别检测交付中保留。
            if not is_bee(d):
                continue
            status = d.get("label_status", "unconfirmed")
            if not valid or status not in (
                "human",
                "human_confirmed",
                "soft_positive",
                "ignore",
            ):
                continue
            if status in ("soft_positive", "ignore") and f["split"] != "train":
                continue
            mask = d.get("supervision_mask", {})
            x1, y1, x2, y2 = d["bbox_xyxy"]
            points = d.get("keypoints", {}) if mask.get("pose", True) else {}
            keypoints = []
            if points:
                for name in ("head", "abdomen_tip"):
                    p = points.get(name)
                    keypoints.extend(
                        [float(p[0]), float(p[1]), 2 if len(p) < 3 or p[2] > 0 else 0]
                        if p
                        else [0, 0, 0]
                    )
            pose_state = (2 if any(keypoints[2::3]) else 1) if points else 0
            source_id = d.get("track_id")
            has_track = (
                source_id is not None
                and mask.get("tracking", True)
                and status != "ignore"
            )
            if has_track:
                key = (*scope, str(source_id))
                identities.setdefault(key, len(identities))
                track_id = identities[key]
                images[-1]["track_supervised"] = True
            else:
                track_id = -1
            q = float(np.clip(d.get("quality", 1.0), 0, 1))
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": sum(v > 0 for v in keypoints[2::3]),
                    "pose_state": pose_state,
                    "pose_mask": bool(pose_state),
                    "track_id": track_id,
                    "source_track_id": source_id,
                    "track_mask": has_track,
                    "supervision_mask": [
                        True,
                        bool(pose_state),
                        has_track,
                        bool(d.get("confirmed_events") or d.get("behavior_labels")),
                    ],
                    "inter_group_quality": float(np.clip(d.get("q_inter", q), 0, 1)),
                    "intra_group_quality": float(np.clip(d.get("q_intra", q), 0, 1)),
                    "hierarchy_quality": q,
                    "pose_quality": float(d.get("pose_quality", q)),
                    "track_quality": float(d.get("track_quality", q)),
                    "annotator_id": f["group"],
                    "is_pseudo": status == "soft_positive",
                    "pseudo_score": float(d.get("expert_fusion", {}).get("probability") or q),
                    "ignore_region": status == "ignore",
                    "background_weight": 0.0 if status == "ignore" else 1.0,
                    "source_entity_id": d["entity_id"],
                    "label_status": status,
                }
            )
    result = {
        "schema_version": 2,
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 0,
                "name": "bee",
                "keypoints": ["head", "abdomen_tip"],
                "skeleton": [[1, 2]],
            }
        ],
    }
    add_track_geometry(result)
    write_json(destination, result)
    return result


def add_track_geometry(coco):
    """对齐E训练器的相邻观测位移/尺度/头到尾方向监督。"""
    ims = {x["id"]: x for x in coco["images"]}
    previous = {}
    ordered = sorted(
        coco["annotations"],
        key=lambda a: (
            ims[a["image_id"]]["track_scope"],
            a["track_id"],
            ims[a["image_id"]]["frame_id"],
        ),
    )
    for a in ordered:
        a.update(track_geometry_mask=False, track_axis_mask=False)
        if not a["track_mask"]:
            continue
        im = ims[a["image_id"]]
        x, y, w, h = a["bbox"]
        points = a["keypoints"]
        key = (im["track_scope"], a["track_id"])
        old = previous.get(key)
        previous[key] = a
        if old is None or im["frame_id"] <= ims[old["image_id"]]["frame_id"]:
            continue
        px, py, pw, ph = old["bbox"]
        scale = max(float(np.hypot(pw, ph)), 1.0)
        axis = len(points) == 6 and points[2] > 0 and points[5] > 0
        dx, dy = (points[3] - points[0], points[4] - points[1]) if axis else (0, 0)
        length = float(np.hypot(dx, dy))
        axis = axis and length > 1e-6
        a.update(
            track_geometry=[
                (x + w / 2 - px - pw / 2) / scale,
                (y + h / 2 - py - ph / 2) / scale,
                float(np.log(w / pw)),
                float(np.log(h / ph)),
                dx / length if axis else 0.0,
                dy / length if axis else 0.0,
            ],
            track_geometry_mask=True,
            track_axis_mask=axis,
        )


def export_yolo(coco, destination):
    """标准YOLO姿态格式；忽略区和软权重另存sidecar，禁止静默丢失。"""
    root = Path(destination)
    by_image = defaultdict(list)
    manifest = []
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    for im in coco["images"]:
        w, h = im["width"], im["height"]
        split = im["split"]
        label = (
            root
            / "labels"
            / split
            / f"{im['domain']}_{im['group']}_{im['video_id']}_frame_{im['frame_id']:08d}.txt"
        )
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for a in by_image[im["id"]]:
            if a["ignore_region"]:
                continue
            x, y, bw, bh = a["bbox"]
            values = [0, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
            k = a["keypoints"] or [0] * 6
            for i in (0, 3):
                values.extend([k[i] / w, k[i + 1] / h, k[i + 2]])
            lines.append(" ".join(map(str, values)))
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        manifest.append(
            {
                **im,
                "image_path": im["file_name"],
                "label_path": str(label),
                "annotations": by_image[im["id"]],
                "requires_supervision_sidecar": True,
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest),
        encoding="utf-8",
    )
    return manifest


def export_round(frames, destination, image_root):
    """不可变的一轮数据快照；空/漏标图保留is_unlabeled语义。"""
    root = Path(destination)
    if root.exists():
        raise FileExistsError("每轮数据使用新目录，避免修改正在训练的输入")
    root.mkdir(parents=True)
    coco = export_coco(frames, root / "all.coco.json", image_root)
    export_yolo(coco, root / "yolo")
    paths = {}
    for split in sorted({im["split"] for im in coco["images"]}):
        ims = [im for im in coco["images"] if im["split"] == split]
        ids = {im["id"] for im in ims}
        data = {
            **coco,
            "images": ims,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
        }
        p = root / f"{split}.coco.json"
        write_json(p, data)
        paths[split] = {
            "path": str(p.resolve()),
            "sha256": sha256(p),
            "images": len(ims),
            "instances": len(data["annotations"]),
        }
    manifest = {
        "schema_version": 2,
        "splits": paths,
        "hard_labels": "human_or_human_confirmed",
        "pseudo_labels": "train_only",
        "ignore_regions": "zero_background_loss",
        "requires_training_input_gate": True,
    }
    write_json(root / "snapshot.json", manifest)
    return manifest
