_base_ = ["../vitpose_plus_small_beepose_head_tail.py"]

experiment_root = "/data/bee26/vitpose_bee_full_20260825"
own_root = experiment_root + "/data/bee_keypoints_all_annotated"

# 主对照统一从同一通用 ViTPose++ Small 权重出发；子配置只改变初始化权重。
load_from = experiment_root + "/checkpoints/vitpose+_small.pth"

optimizer = dict(type="AdamW", lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
fp16 = dict(loss_scale="dynamic")
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook")])

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=64, workers_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=64, workers_per_gpu=4),
)
