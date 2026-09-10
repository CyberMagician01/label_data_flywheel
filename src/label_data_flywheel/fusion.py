"""同帧框与姿态对齐；SAM只负责身份传播，姿态保留原始来源引用。"""

import copy
import numpy as np
from .geometry import overlap
from .semantics import is_bee


def attach_pose(frame, pose_frame, min_iou=0.5):
    from scipy.optimize import linear_sum_assignment

    if (frame["domain"], frame["video"], frame["frame"]) != (
        pose_frame["domain"],
        pose_frame["video"],
        pose_frame["frame"],
    ):
        raise ValueError("姿态融合必须使用同域同视频同源帧")
    out = copy.deepcopy(frame)
    ds = [d for d in out["detections"] if is_bee(d)]
    source = [d for d in pose_frame["detections"] if is_bee(d) and d.get("keypoints")]
    iou = overlap([d["bbox_xyxy"] for d in ds], [d["bbox_xyxy"] for d in source])[2]
    matched = 0
    if iou.size:
        costs = np.where(iou >= min_iou, 1 - iou, 1e6)
        a, b = linear_sum_assignment(costs)
        for i, j in zip(a, b):
            if iou[i, j] < min_iou:
                continue
            ds[i]["keypoints"] = copy.deepcopy(source[j]["keypoints"])
            ds[i]["supervision_mask"]["pose"] = True
            ds[i]["pose_provenance"] = {
                "source_entity_id": source[j]["entity_id"],
                "same_frame_iou": float(iou[i, j]),
                "source": "existing_pose_inference",
            }
            matched += 1
    out["pose_fusion"] = {
        "matched": matched,
        "unmatched": len(ds) - matched,
        "min_iou": min_iou,
    }
    return out
