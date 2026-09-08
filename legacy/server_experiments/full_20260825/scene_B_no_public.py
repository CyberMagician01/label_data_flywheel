_base_ = ["./_own_common.py"]

own_root = "/data/bee26/vitpose_bee_full_20260825/data/bee_keypoints_all_annotated"

total_epochs = 8
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001,
    step=[5, 7]
)
evaluation = dict(interval=8, metric="mAP")

data = dict(
    train=dict(
        ann_file=own_root + "/annotations/scene_B_train_manual.json",
        img_prefix=own_root + "/",
    ),
    val=dict(
        ann_file=own_root + "/annotations/scene_B_val_manual.json",
        img_prefix=own_root + "/",
    ),
    test=dict(
        ann_file=own_root + "/annotations/scene_B_val_manual.json",
        img_prefix=own_root + "/",
    ),
)
