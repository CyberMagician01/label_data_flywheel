_base_ = ["../vitpose_plus_small_beepose_head_tail.py"]

experiment_root = "/data/bee26/vitpose_bee_full_20260825"
beepose_root = experiment_root + "/data/public_beepose"
mendeley_root = experiment_root + "/data/public_mendeley_bee_pose"

load_from = experiment_root + "/checkpoints/vitpose+_small.pth"
data_cfg = {{_base_.data_cfg}}
train_pipeline = {{_base_.train_pipeline}}
val_pipeline = {{_base_.val_pipeline}}
optimizer = dict(type="AdamW", lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
fp16 = dict(loss_scale="dynamic")
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[40, 55],
)
total_epochs = 60
evaluation = dict(interval=5, metric="mAP", save_best="AP")
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

beepose_train = dict(
    type="TopDownCocoDataset",
    ann_file=beepose_root + "/annotations/bee_pose_head_tail_train.json",
    img_prefix=beepose_root + "/images/train/",
    data_cfg=data_cfg,
    pipeline=train_pipeline,
    dataset_info={{_base_.dataset_info}},
)
mendeley_train = dict(
    type="TopDownCocoDataset",
    ann_file=mendeley_root + "/annotations/mendeley_bee_pose_train.json",
    img_prefix=mendeley_root + "/raw/",
    data_cfg=data_cfg,
    pipeline=train_pipeline,
    dataset_info={{_base_.dataset_info}},
)
mendeley_val = dict(
    type="TopDownCocoDataset",
    ann_file=mendeley_root + "/annotations/mendeley_bee_pose_val.json",
    img_prefix=mendeley_root + "/raw/",
    data_cfg=data_cfg,
    pipeline=val_pipeline,
    dataset_info={{_base_.dataset_info}},
    test_mode=True,
)
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=64, workers_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=64, workers_per_gpu=4),
    train=[beepose_train, mendeley_train],
    val=mendeley_val,
    test=mendeley_val,
)
