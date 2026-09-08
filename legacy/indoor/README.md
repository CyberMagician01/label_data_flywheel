# 室内蜜蜂标注：YOLO、ViTPose与身份关联

本仓库保留原检测、关键点、动态密度复检、几何修正及旧版关联脚本，并增加2026-09-08选定的外观关联版本。完整数据和权重保存在私有ModelScope，GitHub只保存代码与既有benchmark标注。

## 当前选定版本

`pipelines/appearance20/` 是双4090上完成全量处理的源代码快照，保留原数值和阈值。外观特征来自 `kasiabozek/bee_tracking` 的64维Inception灰度表征；关联增加衰减动量、低权重姿态与位移门控。内部身份保留150源帧，边缘75帧；观测少于30帧的短轨迹过滤；同ID端点间隔不超过90帧时线性插值。

插值与观测框交集占较小框面积达到20%、且交集至少16像素时，隐藏插值框。插值之间同样去重，原观测优先。隐藏只改变可见性，不删除身份历史。此版本没有对全部观测坐标额外做平滑。

数据位置：`poloso/yolo_indoor/versions/v4_indoor_appearance_iomin20_20260908/`。四个视频36,010帧，14,475,363条可见框记录；累计ID分别为6306、915、8525、1646。累计ID不代表真实蜜蜂总数，复杂遮挡仍会造成碎片化。人工01/03区段评估曾参与调参，不能当作独立测试成绩。

## 运行

在Linux准备原有 `bee_tracking` 的 `repo/`、`data/`，以及TensorFlow 2.15运行环境。第三方源码固定为 `ba9ce391a59cfd10ad30f2a07bf699d9f0050db1`。输入必须是1920×1080、30FPS、从0连续编号的B-5-1至B-5-4逐帧图像与原检测JSON。

```bash
python run_appearance20.py --frames /path/frames --detections /path/final/frames \
  --assets /path/bee_tracking_assets --work-dir /path/new_run \
  --python /path/tensorflow_env/bin/python --gpus 0,1
```

`--stage prepare|embed|track|export` 可分阶段恢复。输入或代码变化必须使用新的工作目录，避免误复用缓存。输出位于工作目录下 `full_20pct_20260907/final/`；JSON帧号从0开始，MOT帧号从1开始。原 `scripts/` 和 `training_package/` 入口全部保留。

## 来源

外观关联依赖 [bee_tracking](https://github.com/kasiabozek/bee_tracking)，遵循其GPL-3.0许可；YOLO与ViTPose按各自上游许可使用。代码快照不包含模型权重、凭据或原始视频。
