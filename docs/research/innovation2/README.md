# 第二项研究：探索工作页

核心目标：[论文总体路线](../../../THESIS_ROADMAP.md)，保持不变。用户希望探索针对简化学生具体不足的算法改进；尚无已选算法或有效性结论。

以下为历史排序路线（已停止该配方），最新判断见页尾。

2026-09-29：按[教师排序缺口诊断计划](RANKING_GAP_DIAGNOSTIC_PLAN_2026-09-29.md)完成一次 Sports Validation 诊断；[结果与执行方自审](RANKING_GAP_DIAGNOSTIC_RESULTS.md)、[独立审计交接](RANKING_GAP_DIAGNOSTIC_HANDOFF.md)。第一项暂停扩展。

| 项目 | 当前判断 |
|---|---|
| 候选问题 | 正物品方向监督没有直接传递教师用户级候选排序；尚未证明是瓶颈 |
| 支持证据 | 三seed随机Full未达到同结构教师初始化+BPR的固定预算质量 |
| 最强替代解释 | 简单复制教师表及优化轨迹已解释差距，新模块未必必要 |
| 当前诊断 | 七个既有评分状态，分解双向留出正例命中，重点检查T0之外的机会 |
| 继续标准 | 固定描述性门槛，仅决定是否值得后续因果对照评审，不认证算法 |
| 执行状态 | 七状态/六比较硬验收通过；T0三seed G均超过预定门槛但N均为负，独立审计已完成（数值通过，证据有限定）；无训练/Test |

当前已形成[最小排序监督对照方案](MINIMAL_RANKING_SUPERVISION_PROPOSAL_2026-09-29.md)；原seed2022[实现与资源准备报告](MINIMAL_RANKING_SUPERVISION_PREPARATION_REPORT_2026-09-29.md)之后，已固定[三seed串行手动启动准备](MINIMAL_RANKING_THREE_SEED_MANUAL_LAUNCH_2026-09-29.md)。三seed正式训练已完成，[独立审计](MINIMAL_RANKING_THREE_SEED_AUDIT_2026-09-29.md)通过执行验收但均screen_stop；没有算法成功结论。

最新：[独立候选评审](INDEPENDENT_CANDIDATE_REVIEW_2026-09-30.md)。新物品接入优先；TopK边界分配保留备选；预算受限持续更新暂缓。三者均未达到直接开跑条件，本次是当前AI重新判断，不是另一AI或原始产物外部认证。

当前唯一下一步：为新物品接入形成具体方法立项提案，明确ALDI/SiBraR以外的机制差异、强简单替代、内容可见性、必须重建资产及预算上限；不实现、不重建、不运行。若无独立差异则暂停该候选。[原研究交接](../RESEARCH_HANDOFF_FOR_INNOVATION2_REVIEW_2026-09-30.md)保留全部历史证据与原评审任务，没有运行队列。

[个性化分配立项评审](PERSONALIZED_ALLOCATION_DESIGN_REVIEW_2026-09-30.md)：问题值得继续，简单用户加分已有近邻；暂不认定第二创新点。

最新：[标量校准结果](SCALAR_CALIBRATION_RESULTS.md)。三seed执行通过，科学screen_stop；U仅增加2/1/0个低组命中，当前配方停止扩展。

[第二项路线收束评审](ROUTE_CLOSURE_REVIEW_2026-09-30.md)是历史建议；新物品接入须解决教师暴露与资产重建，尚未授权投入，和压缩候选一样暂缓。
