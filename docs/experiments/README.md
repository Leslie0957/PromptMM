# 实验导航：按问题查证据

更新：2026-09-26。新窗口只需先读根目录[总览](../../README.md)、[AGENTS](../../AGENTS.md)和[短日志](../../TRAINING_LOG.md)，然后按本表选相关资料。日期较早文档中的“当前”“下一步”均只描述当时状态，不能自动执行旧命令。

2026-09-27 研究优先级更新：[跨交互效用路线](../research/CROSS_INTERACTION_ROUTE_2026-09-27.md)已确定；[P0审计](../research/CROSS_INTERACTION_P0_AUDIT.md)及精度复核完成，36项角色效用数值可分辨。[P1a结果审计](../research/CROSS_INTERACTION_P1_AUDIT.md)核实用户手动一次执行共同缓存 U1 筛查；四组预定 G1/G2 均停止，尚无策略收益证据。

运行声明与结束审计先读[固定更新规范](RUN_CLOSEOUT.md)。第一点复用结论见[协议核查](../research/INNOVATION1_PROTOCOL_REUSE_AUDIT_2026-09-26.md)，执行边界见[正式收尾协议](../research/INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md)；v1[失败审计](../research/INNOVATION1_FORMAL_CLOSEOUT_COHORT_V1_AUDIT.md)与[v2失败/有效训练审计](../research/INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md)保留原证据，旧命令不可再执行。[仅评价恢复审计](../research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)是最新正式评价入口。

## 已完成的实验与诊断族

新规划入口：[初始化 × KD立项单](../research/INITIALIZATION_KD_INTERACTION_PROPOSAL_2026-09-27.md)及[四臂准备审计](../research/INITIALIZATION_KD_INTERACTION_PREPARATION.md)。隔离核心、Train/Val 专用适配器与父进程杀停已通过合成验证，真实资产预检和启动验收合成检查已通过，[用户手动资源smoke审计](../research/INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md)通过，130.6秒/156MiB；单次Val约122秒，正式四臂24小时预算未闭合，[评价等价v2审计](../research/INITIALIZATION_KD_EVAL_PARITY_V2_AUDIT.md)已通过：35598用户排名/指标精确一致，reference291.94秒、fast10.12秒，本次观测28.84倍；训练0/Test0。下一步准备快评价器接入隔离四臂并核算训练/保存总预算，再判断手动声明条件；正式四臂仍不可启动，parity及旧smoke不可重跑。这不是P1b/P2延续，也未认证为第二算法。

| 族 | 当前可用证据 / 状态 | 首选入口 |
|---|---|---|
| 跨交互 P0精度复核 / P1a筛查（完成） | P0暖状态7对、36/36角色均值数值可分辨；P1a共同缓存、3/10/10项、372对完成，冷/暖×图文四组 G1/G2 均 `screen_stop`；非联合策略或排名证据 | [P0审计](../research/CROSS_INTERACTION_P0_AUDIT.md) / [P1a结果审计](../research/CROSS_INTERACTION_P1_AUDIT.md) / [原声明](../research/CROSS_INTERACTION_P1A_DECLARATION.md) |
| Baby数据/冻结教师/BPR基础线 | 既有正式证据；不是整个项目唯一已完成阶段 | [Baby总册](../research/INNOVATION1_THESIS_EVIDENCE_2026-09-15.md) |
| Baby BPR / Full / 原Image-0.3 | 三种子正式组；Image2023审计例外、2024低值均保留 | [三种子审计](../research/BABY_IMAGE_ONLY_THREE_SEED_SUMMARY_2026-09-15.md) |
| Baby固定120与图像系数匹配 | 七次Validation诊断，不能替换既有Test | [总册§4–5](../research/INNOVATION1_THESIS_EVIDENCE_2026-09-15.md) |
| Sports数据与教师 | 转换身份、固定epoch37教师、Validation-only | [数据](../research/SPORTS_RAW_DATA_AUDIT_2026-09-16.md) / [教师](../research/SPORTS_TEACHER_RESULT_2026-09-16.md) |
| Sports随机初值120/300 | BPR/image/Full各三seed；两个预算不是独立重复样本 | [120](../research/SPORTS_THREE_SEED_VALIDATION120_2026-09-17.md) / [300](../research/SPORTS_THREE_SEED_VALIDATION300_2026-09-18.md) |
| 第一创新点固定主比较/PromptMM发布版适配 | v2 Baby三次训练有效；恢复批次12项新预检、12项一次性Test完成。18格本机快照已整理，同盘副本不能代替异机备份。Sports Full高于BPR、低于发布版；Baby发布版均值略低于既有Full | [论文入口](../paper/README.md) / [主表](../paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md) / [快照审计](../research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md) / [正式恢复审计](../research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md) / [v2来源](../research/INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md) / [v1失败](../research/INNOVATION1_FORMAL_CLOSEOUT_COHORT_V1_AUDIT.md) |
| 评价器工程一致性 | legacy/fast原有parity证据，不是模型效率优势 | [协议](../PROMPTMM_VALIDATION_FAST_PARITY.md)，详查历史章节/运行记录 |
| Sports初始化控制/重复性 | 历史首个暖启动、随机release、失败BPR尝试及共享张量前置诊断 | [初始化审计](../research/SPORTS_INITIALIZATION_AUDIT_2026-09-23.md) / [旧总览§6](../research/EXPERIMENT_OVERVIEW_ZH_2026-09-24.md) |
| Sports共享暖启动 Full/BPR | 相同初值；单seed未证增量蒸馏收益 | [配对审计](../research/SPORTS_SHAREDINIT_PAIR_AUDIT_2026-09-23.md) |
| 暖启动alpha3/等权匹配 | 负结果有效；没有alpha或方向最优性结论 | [alpha3](../research/SPORTS_ALPHA3_AUDIT_2026-09-24.md) / [等权](../research/SPORTS_EQUAL_MATCHED_AUDIT_2026-09-24.md) |
| 零更新梯度/F/目标方向 | 真实采样边际加权、固定批次、局部梯度，不代表AdamW全程 | [梯度](../research/SPORTS_GRADIENT_AUDIT_2026-09-23.md) / [加权](../research/SPORTS_WEIGHTED_DIRECTION_AUDIT_2026-09-24.md) / [匹配](../research/SPORTS_MATCHED_TARGET_DIRECTION_AUDIT_2026-09-24.md) |
| 所选表征几何 | 历史所选状态的描述性归因，非中介机制证明 | [几何审计](../research/SPORTS_COLDINIT_SELECTED_GEOMETRY_2026-09-24.md) |
| Sports严格Full/image | 相同初值与完整triplet tape，三对为正 | [严格配对审计](../research/SPORTS_PAIRED_COLDINIT_THREE_SEED_AUDIT_2026-09-25.md) |
| Sports真实/sham残差 | 几何及初始尺度匹配，三对为正；当前机制阶段完成 | [sham审计](../research/SPORTS_SHAM_RESIDUAL_AUDIT_2026-09-26.md) / [机制综合](../research/SPORTS_INTEGRATED_MECHANISM_CONCLUSION_2026-09-26.md) |
| 缓存部署效率 | v1失败保留，v2完成36条件；在线相近、离线生成差异 | [效率审计](../research/SPORTS_CACHED_DEPLOYMENT_AUDIT_2026-09-26.md) |
| 早期Photo/Amazon/专利及旧探索 | 历史材料，不并入当前合格主结果 | [全部历史索引](../../archive/training/ENTRY_INDEX_2026-09-26.md) / [历史目录](../../archive/README.md) |

## 稿件与最小阅读范围

- 研究动机与是否继续：[第一创新点立项决策单](../paper/INNOVATION1_RATIONALE_REVIEW_2026-09-26.md)。文献/源码差异已定位，独立创新强度未认证；效率基准不含 PromptMM 发布版学生，不能扩展解释。

- 第一创新点收尾：先读[论文写作入口](../paper/README.md)；主结果、局限和本机快照均从那里定位，原始结果见[恢复审计](../research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)。v1/v2失败审计和原声明仍保留。第二点门控设计当前暂缓。
- 查后续研究：先读[跨交互效用路线](../research/CROSS_INTERACTION_ROUTE_2026-09-27.md)；[旧候选决策](../research/INNOVATION2_CANDIDATE_DECISION_2026-09-26.md)与RGCS保留为历史参考，未选定新算法。
- 查整篇论文完成度：先读[完成条件与缺口表](../paper/THESIS_COMPLETION_GAPS_2026-09-26.md)，区分已有证据和未关闭条件。
- 写论文：先读[正文表述（已审阅）](../paper/SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md)，只按段落链接展开对应审计。
- 查某次运行：从[运行记录清单](RUN_RECORDS.md)定位profile/status/manifest，再读对应历史声明；失败记录与通过记录都保留。
- 查老命令/参数/哈希：搜索[历史章节索引](../../archive/training/ENTRY_INDEX_2026-09-26.md)的标题，按文件与行号读取一段。旧pending不是尚待运行清单。
- 找任何已跟踪脚本/文档：用[文件导航](FILES.md)。找本机未跟踪的大文件：用[原始文件清单](../../archive/catalog/README.md)。
- 此次索引收录现存记录，不把一个批次报告与其子manifest重复计为多个独立实验，也不把没有结果的计划算已完成。

## 文件布局与维护

`codes/`保持算法及入口；`tools/`保持既有手动启动器；`docs/research/`保持带日期的证据报告；`docs/paper/`放稿件；`docs/experiments/`集中导航；`archive/training/`放不可改的完整历史；`archive/catalog/`放按需检索的路径/报告元数据。

`exp/、Model/、logs/、data/`的原路径已被脚本、manifest和哈希记录引用，保留原位，通过统一索引归档管理。没有为了视觉整齐而搬动、重命名或删除这些资产；生成的清单不等于资产备份。

之后每个阶段继续在短日志追加pending/outcome，并更新本表对应行及相关审计。需要刷新文件/报告清单时，可运行元数据工具 `python -B tools/build_experiment_navigation.py`；它不导入训练代码、不读取模型权重/数据矩阵、不启动实验。历史完整快照不可用此工具重写。
