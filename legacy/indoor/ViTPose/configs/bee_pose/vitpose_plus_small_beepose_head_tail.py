_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/bee_pose_head_tail.py',
]

data_root = r'H:/揭榜挂帅/复现UDMT/third_party/BeePose/data/raw/bee/dataset_raw/'
ann_root = r'H:/揭榜挂帅/复现UDMT/third_party/BeePose/prepared/annotations/'

channel_cfg = dict(
    num_output_channels=2,
    dataset_joints=2,
    dataset_channel=[[0, 1]],
    inference_channel=[0, 1],
)
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=2,
    num_joints=2,
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=None,
    dataset_idx=0,
)

model = dict(
    type='TopDownMoE',
    backbone=dict(
        type='ViTMoE',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        num_expert=6,
        part_features=96,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=2,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    associate_keypoint_head=[],
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        modulate_kernel=11,
        use_udp=True,
    ),
)
load_from = r'H:/揭榜挂帅/复现UDMT/third_party/ViTPose/checkpoints/vitpose+_small.pth'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.35),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type='GaussianHeatmap'),
    dict(type='Collect', keys=['img', 'target', 'target_weight'], meta_keys=[
        'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
        'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
    ]),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect', keys=['img'], meta_keys=[
        'image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs',
        'dataset_idx'
    ]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=ann_root + 'bee_pose_head_tail_train.json',
        img_prefix=data_root + 'train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=ann_root + 'bee_pose_head_tail_val.json',
        img_prefix=data_root + 'validation/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
        test_mode=True,
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=ann_root + 'bee_pose_head_tail_val.json',
        img_prefix=data_root + 'validation/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
        test_mode=True,
    ),
)

optimizer = dict(type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=200, warmup_ratio=0.001, step=[60, 90])
total_epochs = 100
evaluation = dict(interval=5, metric='mAP', save_best='AP')
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
