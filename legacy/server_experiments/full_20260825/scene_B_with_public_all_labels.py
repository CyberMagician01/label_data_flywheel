_base_ = ["./scene_B_with_public.py"]

own_root = "/data/bee26/vitpose_bee_full_20260825/data/bee_keypoints_all_annotated"

# 仅作为标签数量/噪声消融；验证仍固定为纯人工标注。
data = dict(
    train=dict(
        ann_file=own_root + "/annotations/scene_B_train_all.json",
        img_prefix=own_root + "/",
    )
)
