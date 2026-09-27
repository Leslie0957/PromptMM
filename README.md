# PromptMM 实验工作区

先完整阅读 [AGENTS.md](AGENTS.md)，再读本页与 [TRAINING_LOG.md 当前状态及最新记录](TRAINING_LOG.md)。历史 pending 和旧交接不是执行队列。

当前立项结论见[初始化转移 × 持续监督计划](docs/research/INITIALIZATION_KD_INTERACTION_PROPOSAL_2026-09-27.md)：受控四臂的[隔离核心与合成准备](docs/research/INITIALIZATION_KD_INTERACTION_PREPARATION.md)已完成，Train/Val 专用适配器与父进程杀停通过合成检查；真实资产预检及启动验收的合成检查已通过，用户手动[GPU资源smoke审计](docs/research/INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md)通过（130.6秒、156MiB CUDA allocator）；完整Val约122秒，使原四臂24小时候选预算缺乏支持。精确Top-K提速合成检查通过，但[手动核验v1因Windows监控文件替换失败](docs/research/INITIALIZATION_KD_EVAL_PARITY_AUDIT.md)，尚无等价/计时结果；I/O修复及[v2手动核验声明](docs/research/INITIALIZATION_KD_EVAL_PARITY_V2_LAUNCH_2026-09-27.md)已准备，27项CPU合成检查通过；下一步用户执行带`--attempt v2`的新目录核验一次，旧v1不可重跑；正式四臂仍不可启动，旧smoke命令不可重跑。此方向定位为第一项工作的机制补充，未认定第二算法。跨交互P1a已筛查停止，不推进P1b/P2；实际实验由用户另行授权并手动执行。旧RGCS/门控及历史“下一步”不是执行队列。

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
