_base_ = ["./_own_common.py"]

own_root = "/data/bee26/vitpose_bee_full_20260825/data/bee_keypoints_all_annotated"

total_epochs = 30
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=200, warmup_ratio=0.001,
    step=[18, 26]
)
evaluation = dict(interval=5, metric="mAP")

data = dict(
    train=dict(
        ann_file=own_root + "/annotations/scene_A_train_manual.json",
        img_prefix=own_root + "/",
    ),
    val=dict(
        ann_file=own_root + "/annotations/scene_A_val_manual.json",
        img_prefix=own_root + "/",
    ),
    test=dict(
        ann_file=own_root + "/annotations/scene_A_val_manual.json",
        img_prefix=own_root + "/",
    ),
)
