# PromptMM 实验工作区

先完整阅读 [AGENTS.md](AGENTS.md)，再读本页与 [TRAINING_LOG.md 当前状态及最新记录](TRAINING_LOG.md)。历史 pending 和旧交接不是执行队列。

当前[初始化 × KD 立项单](docs/research/INITIALIZATION_KD_INTERACTION_PROPOSAL_2026-09-27.md)下的 Sports 四臂已由用户分别完成 [seed2022](docs/research/INITIALIZATION_KD_INTERACTION_AUDIT.md)和 [seed2023](docs/research/INITIALIZATION_KD_INTERACTION_SEED2023_AUDIT.md) 两次 Validation-only 运行；见[独立八格矩阵](docs/research/INITIALIZATION_KD_INTERACTION_RESULTS.md)。两次随机初值组 KD 增量都约 +0.0365，教师初值组都很小且正负号相反；不能据此证明语义冗余、统计显著或第二算法。用户要求的第三种子 [seed2024只读预检](docs/research/INITIALIZATION_KD_SEED2024_PREFLIGHT_2026-09-28.md)和[一次手动运行声明](docs/research/INITIALIZATION_KD_SEED2024_COHORT_LAUNCH_2026-09-28.md)已准备，下一步仅用户手动执行一次，之后审计并可交另一 AI 独立分析；共同教师缓存固定，全部 Test0。历史 [资源smoke](docs/research/INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md)、[评价等价](docs/research/INITIALIZATION_KD_EVAL_PARITY_V2_AUDIT.md)及已消耗 seed2023 声明保留作证据。跨交互 P1a 已筛查停止，不推进 P1b/P2；旧 RGCS/门控及历史“下一步”不是执行队列。

## 当前入口

| 用途 | 入口 |
|---|---|
| 正式训练代码 / 默认参数 | `codes/main_mmlight.py` / `codes/utility/parser.py`（CLI 优先） |
| 当前计划与实施交接 | [初始化 × KD准备审计](docs/research/INITIALIZATION_KD_INTERACTION_PREPARATION.md) / [立项单](docs/research/INITIALIZATION_KD_INTERACTION_PROPOSAL_2026-09-27.md)；[历史跨交互路线](docs/research/CROSS_INTERACTION_ROUTE_2026-09-27.md) |
| P0 精度复核与 P1a 筛查已完成 | [P0结果审计](docs/research/CROSS_INTERACTION_P0_AUDIT.md)：36项角色效用数值可分辨；[P1a结果审计](docs/research/CROSS_INTERACTION_P1_AUDIT.md)：共同缓存、3/10/10项、372对完成，冷/暖×图文四组的预定 G1/G2 均筛查停止；尚无联合策略或正式训练结果 |
| 实验族、已完成证据 | [实验导航](docs/experiments/README.md) |
| 文件分类与保留规则 | [工作区地图](docs/experiments/WORKSPACE_MAP.md) |
| 声明与结果记录路由 | [RUN_CLOSEOUT](docs/experiments/RUN_CLOSEOUT.md) |
| 第一创新点固定18格主比较、局限与缺口 | [论文入口](docs/paper/README.md) |
| 旧命令、身份和结果 | [历史章节索引](archive/training/ENTRY_INDEX_2026-09-26.md) |
| 运行 JSON / 本机资产位置 | [记录导航](docs/experiments/RUN_RECORDS.md) / [资产目录](archive/catalog/README.md) |

既有 Baby/Sports 正式结果、失败和低指标均保留。第一创新点主比较可写限定结果，不代表整篇论文证据闭合；暖启动未证明额外 KD 收益，缓存部署未证明一致在线加速。不要重跑旧收尾批次或重训补资产。

上游论文及原始说明见 [原始 README 快照](archive/upstream/README_ORIGINAL_2026-09-26.md)。其中命令仅描述历史版本。
