# label_data_flywheel

**室内红外与室外 RGB 蜜蜂标注的数据飞轮。** 保留经过筛选的检测、姿态、密度与追踪方案，整合质量校准、统计采样、行为证据、人工复核和训练数据回流。

**范围只限数据飞轮模块。** 赛事文件中的标注格式、划分、复核、来源和抽帧复现要求在此落实；四 EXE、ONNX、Windows 离线打包及完整参赛材料由其他模块负责，不作为飞轮完成标准。见 [范围与数据规范](docs/数据飞轮范围与数据规范.md)。

原始标注、算法预测、人工确认结果分别保存。每轮生成新快照；没有相同评测协议下的改进证据，就不替换已有最优版本。

```mermaid
flowchart LR
  A[室内 YOLO + 关键点 + 动态密度] --> C[统一实体与源帧协议]
  B[室外 SAM2.1 + 检测吸附 + 姿态] --> C
  C --> D[Q² / 多专家校准 / 跨层校验]
  C --> E[轨迹与 BEEG 行为证据]
  D --> F[预算约束人工复核]
  E --> F
  F --> G[分布校准与不可变训练快照]
  G --> H[E 路线六阶段训练 / 保留 Y 路线]
  H --> I[固定协议评测与版本选择]
  I --> C
```

## 入口

| 需要做什么 | 入口 |
|---|---|
| 运行标注整理与复核候选生成 | `bee-flywheel round --config ... --output ...` |
| 数量、热图、近邻网络与纯视频变化复核 | `bee-flywheel colony --config configs/colony.example.json --output 新目录` |
| 导出、校验标准标注包 | `bee-flywheel export-annotations` / `validate-annotations` |
| 转换为 YOLO，保留原标注与 ID | `bee-flywheel export-yolo` / `validate-yolo` |
| 整理检测、姿态和 MOT 三个交付目录 | `bee-flywheel export-delivery --config configs/delivery.example.json --output 05_数据标注成果` |
| 当前最优室内全量 ID 方案 | [legacy/indoor/run_appearance20.py](legacy/indoor/run_appearance20.py) |
| 室外原版 SAM2.1 与几何吸附 | [legacy/outdoor](legacy/outdoor) |
| 关键点、YOLO、密度推理 | [tools/infer_models.py](tools/infer_models.py) |
| 双域六阶段联合训练 | [E 路线](legacy/route_e/ecdetseg/configs/bee_e/e_route_continuous_1280.yml) |
| 快照绑定到 E 训练器 | [tools/prepare_e_round.py](tools/prepare_e_round.py) |
| 旧 Y 路线及全部自定义模型 | [legacy/route_y](legacy/route_y) |
| ViTPose / CountAnything / CountGD++ 历史实验 | [legacy/server_experiments](legacy/server_experiments) |
| 文档功能对应实现 | [docs/功能与验证.md](docs/功能与验证.md) |
| 学术方法正文（不插实验图，保留方法流程图） | [Hive-Q²K Dual 方法](docs/Hive_Q2K_Dual_方法.md) |
| 同一正文的室内实验配图版（检测、补框与跟踪） | [室内实验配图版](docs/Hive_Q2K_Dual_方法_室内实验配图.md) |
| 群体行为与蜂学一手文献、实测及使用 | [docs/群体行为与蜂学研究.md](docs/群体行为与蜂学研究.md) |

## 安装和使用

完整仓库在 Linux 服务器上运行。大模型保持原有独立 Python 环境；无需把训练数据、权重或视频下载到工作电脑。

```bash
git clone https://github.com/CyberMagician01/label_data_flywheel.git
cd label_data_flywheel
python -m pip install -e '.[vision,test]'
bee-flywheel --help
```

配置 [configs/round.example.json](configs/round.example.json) 中的输入路径和真实 split，然后运行：

```bash
bee-flywheel round --config configs/round.example.json --output /data/bee26/flywheel_rounds/round_001
```

默认输出统一标注、质量证据、ErrorCube、行为观测与复核队列。采样计划和训练快照由相应配置启用。`review_decisions` 可接入上一轮的人工决定；`candidate_evidence` 和 `expert_calibration` 可接入完成校准的专家结果。`pose_sources` 把已有同帧关键点与 SAM 轨迹对齐。

### YOLO 标注副本

```bash
python -m pip install -e '.[annotations]'
bee-flywheel export-yolo --config configs/yolo.example.json --output /data/bee26/yolo_new_version
bee-flywheel validate-yolo --input /data/bee26/yolo_new_version
```

每张源图对应一个 TXT。`detect/labels` 每行是 `class cx cy w h`；`pose/labels` 再接头、腹尾两个点的 `x y v`，共 11 列。坐标按源图宽高归一化；`v` 是 0/1/2 可见性编码，原始关键点置信度另存。检测与姿态标签独立，不把 ID 追加进标准 YOLO 行。

`frame_ids.jsonl.gz` 每行对应一个源帧，其中 `label_files` 指向 TXT，`rows` 中的 `line` 从 1 开始，与 TXT 行号一一对应，保留 `track_id`、来源 ID、置信度和插值来源。ID 沿用原版，并以视频、标注组为作用域。`audit_hidden` 保存隐藏框的 YOLO 副本，不放入训练 labels。原始文件只读，已存在的输出目录拒绝覆盖。人工与机器标签分别在 `confirmed`、`candidates` 中；室外已有 `bee_shadow` 类独立保留。

输出只包含标注。使用图像时，将原图放到对应的 `detect/images` 或 `pose/images` 下，与 `labels` 保持相同相对目录和文件名。`label_schema.yaml` 描述类别和关键点；已有划分写入 `splits`。未划分全集保留 `unassigned`，不会自动变成训练集；仅有检测框而没有任何关键点的帧，不列入姿态训练清单。

### 交付目录

`export-delivery` 将已转换的全量机器标注整理为 `annotations/<视频>/`（YOLO 检测）、`annotations_pose/<视频>/`（YOLO 两点姿态）和 `annotations_tracking/<视频>/tracks.txt`（MOT 十列）。MOT 的帧号和左上角坐标从 1 起算；源图文件名和 ID 数值保留，metadata 记录 TXT 与 MOT 的行号对应关系。隐藏框存入独立 audit 目录；无 ID 的影子保留检测和姿态，不生成虚构轨迹。人工标注副本独立保留。

完整包同时提供 `splits`、`frame_manifest.jsonl`、抽帧脚本和可选的 DOCX 说明。全量自动标注未指定划分时，train/val 为空、源帧全部保留在 unassigned。配套图像始终存放在标注包外。已完成的八段全量格式转换、15 项相关测试及逐行检查见 [转换证据](evidence/yolo_conversion_verified.json)；三目录及 MOT 实际数量见 [交付检查](evidence/delivery_verified.json)。

YOLO 独立副本发布到私有 ModelScope 的 `versions/v5_yolo_format_20260908/`。其中 `annotations/` 保留室内、室外、原室外姿态及人工标注的独立归档；`delivery/05_数据标注成果.tar.gz` 是按上述三个平行目录整理的全量八视频交付包。原 v2/v4 标注与已冻结 benchmark 保持原样，GitHub 另存 [benchmark YOLO 副本](legacy/indoor/benchmark_yolo)。

```bash
# 先验证具体模型命令；去掉 --dry-run 即实际执行。
bee-flywheel backend density --config configs/backends.3090.json --dry-run -- \
  --checkpoint /data/bee26/vitpose_bee_full_20260825/work_dirs/density_scene_B/best_mae.pth \
  --image /data/bee26/example.jpg --output /data/bee26/density.json

# 跟踪指标使用官方 TrackEval；此命令只比较人工标注源帧。
python -m pip install -e '.[evaluation]'
bee-flywheel evaluate --gt /data/bee26/gt.jsonl --input /data/bee26/pred.jsonl \
  --video B-5-3 --domain IR_in --group 03 --tracking --output /data/bee26/metrics.json
```

服务器配置文件是既有环境的路径模板；运行前应使用自己的 checkpoint 和数据路径。原训练配置含冻结数据哈希，不能通过修改哈希来绕过数据划分检查；新一轮应重新生成并验证快照契约。

## 保留的室内默认方案

- 9 月 6 日检测、动态密度、关键点与几何修正结果作为观测输入。
- 蜜蜂外观嵌入、动量、位移与姿态约束完成整段关联；ID 不按分块重置。
- 内部记忆 150 帧、边缘 75 帧；不足 30 次观测的短轨迹过滤。
- 同 ID 两端间隔不超过 90 个源帧时线性补框。
- 交集达到较小框面积的 20%，且交集至少 16 像素时，只隐藏冲突插值框；保留观测框和隐藏记录。

这是实际选定配置，不把早期讨论过的每个参数都混入最终版本。30 FPS 下 90 帧对应 3 秒。

旧发布保留在私有 [ModelScope 数据集 poloso/yolo_indoor](https://modelscope.cn/datasets/poloso/yolo_indoor)，当前室内选择指向 `versions/v4_indoor_appearance_iomin20_20260908/`。GitHub 保存代码、已有 benchmark 和验证摘要；完整预测、视频帧、权重仍留在原服务器与私有数据集。

## 方法与输入范围

方法保留已有代码实现的双域联合训练、Q²/Q-MoE、目标分布采样、B-TCA、主动复核与知识更新。各模块的实验状态在验证记录中分别说明，不以实验是否完成裁剪方法章节。

默认群体复核使用视频中的数量、密度、运动和近邻证据。温度、天气、施药、称重等外部条件未提供，因此不纳入当前方法的环境推断。行为入口已经排除插值和已拒绝实例对实际观测量的贡献；双域 32 帧及 A-5-1 的 5,400 帧验证见 [视频分析验证](evidence/video_only_revision.json)。

## 实际验证

验证结果集中在 [evidence](evidence)。已执行：真实 RGB/IR 关键点与密度推理、Y 路线 checkpoint 推理、3090 上 E 网络双域前向/反向及优化器更新、4090 上室外 A-5-1 连续 16 帧原版 SAM 推理、32 帧双域飞轮处理，以及真实 348 个标注框的 E schema 数据加载。

核心模块 34 项测试、E 路线 153 项测试通过。具体证据见 [verification_summary.json](evidence/verification_summary.json)；保留版本见 [model_registry.json](configs/model_registry.json)。数据标准化另用 34 个真实源帧验证，见 [annotation_standardization_verified.json](evidence/annotation_standardization_verified.json)。

这些证明实现能运行。新增整个飞轮尚未完成完整训练和独立留出集的收益实验，因此不宣称整体 mAP、IDF1 或行为识别率已提高。历史最优版本保持原样；行为输出默认是候选，人工确认后才能进入监督学习。

原始方案保存在 [docs/source_proposal.md](docs/source_proposal.md)。来源、版本与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
