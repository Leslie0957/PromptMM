# PromptMM 实验工作区

## 从这里开始

新窗口按顺序阅读：**本页 → [AGENTS.md](AGENTS.md) → [当前实验短日志](TRAINING_LOG.md)**。
随后只按任务查[实验族导航](docs/experiments/README.md)中的对应审计，不必读取全部历史。

当前阶段：Sports文本方向机制对照及缓存部署效率审计已完成，正在整理论文材料。
Baby既有三种子正式结果及审计例外继续保留。没有待自动执行的实验；旧文档的“下一步”不是当前授权。
实际实验运行继续由用户手动执行。v2在Sports发布版身份读取处失败，三次Baby训练有效保留；随后的[仅评价恢复审计](docs/research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)确认12项新预检与12项一次性学生Test全部通过。第一创新点18格固定主比较已整理为[论文写作入口](docs/paper/README.md)和[主表与表注](docs/paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md)；[本机快照审计](docs/research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md)记录原始文件副本及其同盘限制。不要重跑v1/v2或重训Baby。

**写作状态：第一创新点固定配置主比较可以写正式结果，整篇论文证据尚未闭合。** 第二创新点尚未确定并完成实验；Baby与Sports现均有主比较Test，但原机制对照仍多为Validation。[论文完成条件与缺口表](docs/paper/THESIS_COMPLETION_GAPS_2026-09-26.md)已更新；[门控候选决策单](docs/research/INNOVATION2_CANDIDATE_DECISION_2026-09-26.md)找回RGCS旧方案但未选定或实现。用户优先收尾第一创新点，第二点及门控诊断设计暂缓。先读[第一创新点正式矩阵](docs/paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)、[恢复审计](docs/research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)和[正式收尾协议](docs/research/INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md)；技术评价缺格已补，创新性、基线充分性及外部学位要求仍待核对。

## 核心结论与稿件

- [论文正文表述（已审阅）](docs/paper/SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md)：分别陈述文本方向收益、暖启动边界、阶段性效率。
- [论文写作入口](docs/paper/README.md)：第一创新点[18格主表与表注](docs/paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md)、机制文字、缺口与[本机快照审计](docs/research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md)。
- Sports随机初值严格配对支持当前设置下的文本方向收益；暖启动未证明额外蒸馏优于BPR微调。
- 学生离线表征生成成本更低；同维缓存教师对照下未证明一致在线加速或向量表压缩。
- Sports主比较现有Test；Full/image、real/sham及暖启动机制仍按Validation证据单独陈述。发布版PromptMM在Sports Test较优、Baby Test均值略低于Full，两方向均保留。

实验结束后的固定更新规则见[RUN_CLOSEOUT.md](docs/experiments/RUN_CLOSEOUT.md)：声明时列明记录目标，结束后按触发条件更新，避免依赖聊天记忆。

## 按需查阅

| 要找什么 | 入口 |
|---|---|
| 当前状态、约束、最近变更 | [短日志](TRAINING_LOG.md) |
| 每类实验的结果、失败、例外与机制边界 | [实验族导航](docs/experiments/README.md) |
| 所有源文件与文档 | [文件清单](docs/experiments/FILES.md) |
| 现存运行/诊断JSON与manifest | [运行记录清单](docs/experiments/RUN_RECORDS.md) |
| 旧实验命令、参数、结论、身份哈希 | [历史章节索引](archive/training/ENTRY_INDEX_2026-09-26.md) |
| 本机大文件、缓存、原始日志路径 | [资产目录](archive/catalog/README.md) |
| 历史快照及完整性记录 | [归档入口](archive/README.md) |

## 目录分工

| 目录 | 用途 |
|---|---|
| `codes/` | 当前训练入口 `main_mmlight.py`，默认参数 `utility/parser.py`；`run_patent.py`是独立旧线 |
| `tools/`、`tests/` | 手动运行器、维护工具和检查 |
| `docs/experiments/`、`docs/paper/` | 集中导航、论文段落 |
| `docs/research/`、`docs/research_notes/` | 保持原路径的专题审计与历史讨论 |
| `archive/training/`、`archive/catalog/` | 不可改的历史日志、可刷新的文件导航 |
| `exp/`、`logs/`、`Model/`、`data/` | 原始实验资产，保留路径，不进入普通Git提交 |
| `environment/`、`backups/`、其他旧代码目录 | 环境记录、既有备份、历史来源；通过文件清单查找 |

历史日志已逐字节归档，失败和低指标记录均保留。路径清单不是实验资产备份。
本次整理不改变训练入口、参数、数据或评估协议，也不搬动被脚本和manifest引用的产物。

## 上游来源

本仓库基于PromptMM研究代码开展本地实验。上游论文说明、作者信息及原始使用说明保存在
[原始README完整快照](archive/upstream/README_ORIGINAL_2026-09-26.md)。
其中命令反映原始代码版本；当前工作区以AGENTS和当前日志中的协议为准。
