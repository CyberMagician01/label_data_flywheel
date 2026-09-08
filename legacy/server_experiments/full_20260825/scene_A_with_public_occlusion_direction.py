_base_ = ["./scene_A_with_public.py"]

model = dict(
    keypoint_head=dict(
        loss_keypoint=dict(
            type="JointsMSEDirectionLoss",
            use_target_weight=True,
            direction_weight=0.01,
            softmax_temperature=50.0,
        )
    )
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    dict(type="TopDownRandomTranslation", trans_factor=0.12, trans_prob=0.5),
    dict(type="TopDownGetRandomScaleRotation", rot_factor=60, scale_factor=0.4),
    dict(type="TopDownAffine", use_udp=True),
    dict(type="PhotometricDistortion", brightness_delta=24,
         contrast_range=(0.7, 1.3), saturation_range=(0.7, 1.3), hue_delta=10),
    dict(type="BeeRandomOcclusion", probability=0.5,
         min_fraction=0.08, max_fraction=0.25,
         cover_keypoint_probability=0.7),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
    dict(type="TopDownGenerateTarget", sigma=2, encoding="UDP",
         target_type="GaussianHeatmap"),
    dict(type="Collect", keys=["img", "target", "target_weight"], meta_keys=[
        "image_file", "joints_3d", "joints_3d_visible", "center", "scale",
        "rotation", "bbox_score", "flip_pairs", "dataset_idx"
    ]),
]

data = dict(train=dict(pipeline=train_pipeline))
