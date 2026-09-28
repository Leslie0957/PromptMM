# PromptMM 实验工作区

2026-09-29当前阶段：已完成[第一项正文整合](docs/paper/INNOVATION1_INTEGRATED_CHAPTER_2026-09-29.md)，包含方法公式、固定Test、受控Validation和独立核验的局部成本。本轮待跑实验为零；下一步审阅具体章节并确认第二项贡献约束。以下旧交接按历史保留。

2026-09-29独立复核：M0通过、M1 v1失败保留/v2通过、M2通过，整体限定接受；M1流程总时间含资产核验，M2未完整固定torch随机源，9.07–9.31倍仅短窗口更新比较。见[独立审计](docs/research/INNOVATION1_COST_INDEPENDENT_AUDIT_2026-09-29.md)。当前下一步仅整合第一项正文与限定贡献，不追加本计划实验；下方旧交接保留。

当前阶段：[M0–M2成本交接包](docs/research/INNOVATION1_COST_FINAL_HANDOFF_2026-09-29.md)已完成：M0通过，M1 v1失败且隔离v2通过30/30，[M2短更新审计](docs/research/INNOVATION1_UPDATE_COST_AUDIT.md)通过27/27，无新Test访问。下一步交另一个AI独立审计；完整训练成本和无损降本仍未证实。

先完整阅读 [AGENTS.md](AGENTS.md)，再读本页与 [TRAINING_LOG.md 当前状态及最新记录](TRAINING_LOG.md)。历史 pending 和旧交接不是执行队列。

当前第一项以[目标—证据—缺口总表](docs/paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md)统一管理：保推荐质量、降低明确环节成本是研究目标，尚未实现无损降本；18格主比较、文本/残差与三seed初始化实验均已完成。成本资产清点、M0预检、M1部署和M2短更新测量已完成，待独立复核。历史pending不是执行队列。

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
