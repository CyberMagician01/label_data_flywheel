_base_ = ["../vitpose_plus_small_beepose_head_tail.py"]

data_root = "H:/揭榜挂帅/复现UDMT/data/bee_keypoints_13_20260827"

# 严格对照：仅通用 ViTPose++ Small 初始化，不使用蜜蜂公开数据权重。
load_from = "H:/揭榜挂帅/复现UDMT/third_party/ViTPose/checkpoints/vitpose+_small.pth"

optimizer = dict(type="AdamW", lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
fp16 = dict(loss_scale="dynamic")

total_epochs = 8
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5, 7],
)
evaluation = dict(interval=2, metric="mAP")
checkpoint_config = dict(interval=2, max_keep_ckpts=4)
log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook")])

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    persistent_workers=True,
    val_dataloader=dict(samples_per_gpu=128, workers_per_gpu=4, persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=128, workers_per_gpu=4, persistent_workers=True),
    train=dict(
        ann_file=data_root + "/annotations/scene_B_train_manual.json",
        img_prefix=data_root + "/",
    ),
    val=dict(
        ann_file=data_root + "/annotations/scene_B_val_manual.json",
        img_prefix=data_root + "/",
    ),
    test=dict(
        ann_file=data_root + "/annotations/scene_B_val_manual.json",
        img_prefix=data_root + "/",
    ),
)
