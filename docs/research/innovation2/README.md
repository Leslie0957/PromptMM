# 第二项研究：探索工作页

核心目标：[论文总体路线](../../../THESIS_ROADMAP.md)，保持不变。用户希望探索针对简化学生具体不足的算法改进；尚无已选算法或有效性结论。

2026-09-29：按[教师排序缺口诊断计划](RANKING_GAP_DIAGNOSTIC_PLAN_2026-09-29.md)完成一次 Sports Validation 诊断；[结果与执行方自审](RANKING_GAP_DIAGNOSTIC_RESULTS.md)、[独立审计交接](RANKING_GAP_DIAGNOSTIC_HANDOFF.md)。第一项暂停扩展。

| 项目 | 当前判断 |
|---|---|
| 候选问题 | 正物品方向监督没有直接传递教师用户级候选排序；尚未证明是瓶颈 |
| 支持证据 | 三seed随机Full未达到同结构教师初始化+BPR的固定预算质量 |
| 最强替代解释 | 简单复制教师表及优化轨迹已解释差距，新模块未必必要 |
| 当前诊断 | 七个既有评分状态，分解双向留出正例命中，重点检查T0之外的机会 |
| 继续标准 | 固定描述性门槛，仅决定是否值得后续因果对照评审，不认证算法 |
| 执行状态 | 七状态/六比较硬验收通过；T0三seed G均超过预定门槛但N均为负，独立审计已完成（数值通过，证据有限定）；无训练/Test |

当前已形成[最小排序监督对照方案](MINIMAL_RANKING_SUPERVISION_PROPOSAL_2026-09-29.md)；原seed2022[实现与资源准备报告](MINIMAL_RANKING_SUPERVISION_PREPARATION_REPORT_2026-09-29.md)之后，已固定[三seed串行手动启动准备](MINIMAL_RANKING_THREE_SEED_MANUAL_LAUNCH_2026-09-29.md)。正式训练未启动，不认证算法创新。

唯一下一步：用户从干净提交手动启动已声明的 seed2022/2023/2024 串行 B/R/A/M cohort 一次；结束后停止并交独立审计。长尾信息利用不足为条件备选，不是自动执行队列。P1a停止，旧门控不恢复。
