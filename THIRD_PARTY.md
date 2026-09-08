# 来源与许可范围

本仓库是可分别调用的代码集合。新增 `src/label_data_flywheel`、`tools`、测试和说明采用 GPL-3.0；`legacy` 中保留的第三方或已有项目代码继续遵循其原有许可与版权声明，不因进入本仓库而被重新授权。权重、完整数据和凭据不随代码分发。

| 组成 | 来源 / 固定版本 | 说明 |
|---|---|---|
| 室内标注与 appearance20 | [CyberMagician01/yolo_indoor](https://github.com/CyberMagician01/yolo_indoor)，`9dc17b431e1cf7f539d78ccfe50c4ee29b7a7185` | 保留原检测、密度、关键点、追踪及评测脚本 |
| 蜜蜂外观嵌入依赖 | [kasiabozek/bee_tracking](https://github.com/kasiabozek/bee_tracking)，`ba9ce391a59cfd10ad30f2a07bf699d9f0050db1` | GPL-3.0；完整许可保留于 `legacy/indoor/pipelines/appearance20/LICENSE.bee_tracking`；模型另行配置 |
| 室外 SAM 标注 | [miraclrmaker/sam_track_for_yellowone](https://github.com/miraclrmaker/sam_track_for_yellowone)，`6d507c0b12709618e0760aaec020b49686dd7247` | 原 README 声明 Apache-2.0；原文件保留 |
| SAM2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | 外部环境依赖，未复制其模型权重 |
| Y 路线 | 3090 `/data/bee26/beeposetrack_y_20260827` | 保留用户已有自定义模型、损失、训练、跟踪、量化代码；Ultralytics 为独立依赖 |
| E 路线 | 3090 初始代码、4090 `/root/autodl-tmp/beeposetrack_e_route_v5_unified_20260901/source/ecdetseg` 一致版本 | 保留 EdgeCrafter、Facebook、lyuwenyu 等原文件版权头。上游链接与模型说明见其 README |
| ViTPose 与密度实验 | 3090 `/data/bee26/vitpose_bee_full_20260825/tools` 及 `configs/bee_pose/full_20260825` | 自定义脚本与配置；ViTPose/MMCV/MMPose 由原环境提供 |
| CountAnything / CountGD++ | 3090 `count/jiebang_bee_training/workspace/tools` | 保留实验包装脚本，依赖与权重在独立环境 |
| TrackEval | [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval) | 通过包导入官方指标；原项目 MIT |
| T-Rex2 / DDS | [IDEA-Research/T-Rex](https://github.com/IDEA-Research/T-Rex/tree/trex2)、[DDS SDK](https://github.com/deepdataspace/dds-cloudapi-sdk) | 按官方请求协议独立实现客户端；未复制 T-Rex 权重或持久化 API 凭据 |

集成修改记录：

1. 室内新增可迁移启动器与所选参数快照，原全量算法文件保持一致。
2. 室外原始算法不改；统一导入时按同帧同 ID 合并 SAM 重放结果，并保留审计记录。
3. E 采用 4090 上已修正的成套模型代码，避免与 3090 旧版接口混用；补入无效/存疑帧过滤。
4. E 的旧测试夹具迁移至当前 COCO 初始化、schema-v2 划分和 `2×16` 微批次/累积配置，保留哈希篡改、划分泄漏和非优势模型拒绝测试。
5. 新增飞轮控制、质量与概率校准、统计采样、行为量化、复核回流和跨格式数据桥接。

逐文件哈希见 [evidence/source_manifest.json](evidence/source_manifest.json)。
# TrackEval 评测依赖

评测固定使用 `JonathonLuiten/TrackEval` 的 `12c8791b303e0a0b50f753af204249e622d0281a` 提交（MIT）。`metrics.py` 仅对 HOTA/Identity 模块提供旧 `np.float`/`np.int` 别名兼容，指标公式不变，不修改全局 NumPy。PyPI 同名包不作为本项目评测基准。
