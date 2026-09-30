# 第二项研究：探索工作页

核心主题与创新判断标准：[论文总体路线v1.2](../../../THESIS_ROADMAP.md)。总体主题不变；允许有明确差异与价值的跨领域迁移、组合和适配，不要求基础组件原创，不默认采用顶会级新机制门槛。历史评审的事实/归约/负结果保留，较高原创性门槛不自动作为后续候选的否决依据。

## 当前状态（2026-09-30）

用户已授权[第一项最终收尾](../../paper/INNOVATION1_FINAL_CLOSEOUT_2026-09-30.md)，现有方法/结果/证据至初稿准备阶段冻结；FDRec只是初稿阶段可考虑的直接基线补充。**后续研究重心先全部转到第二创新点探索，不再默认回到第一项补实验或泛泛评审。** 第二项尚无成立的方法；不强制沿用第一项学生结构，也不把第一项消融换名作为第二项。

当前进入第二项候选重新筛选，见[已更新的完整研究交接](../RESEARCH_HANDOFF_FOR_INNOVATION2_REVIEW_2026-09-30.md)。用户确认：允许跨领域结合，可辨认的新意优先、候选指标其次，允许记录后的战术调整和换方向。[时间充裕后的路线重评](RESEARCH_DIRECTION_WITH_EXPANDED_BUDGET_2026-09-30.md)及新物品两轮诊断作为已有证据，未固定下一场景或算法；评分几何岭回归保留简单基线，旧停止结果不撤销。

当前：[Sports未见物品诊断v1](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROTOCOL_V1.md)的一次授权seed2022串行cohort已completed/exit0，6教师/12神经fit及R/N/校准/14报告全部完成，保存证据自审567项通过。**科学undetermined：多模态教师暖probe优势未建立，K_MM低于K_CF的冷Recall且有冷暖质量代价；没有成立的第二创新点。** [结果](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_RESULTS.md) · [审计](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_AUDIT.md) · [HANDOFF](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_HANDOFF.md)。冻结profile、原准备记录与正式声明保留来源；不改历史资格。

[质量约束前沿v1](QUALITY_CONSTRAINED_FRONTIER_PROTOCOL_V1.md)已由用户手动一次运行完成：[结果](QUALITY_CONSTRAINED_FRONTIER_RESULTS.md) · [审计](QUALITY_CONSTRAINED_FRONTIER_AUDIT.md) · [HANDOFF](QUALITY_CONSTRAINED_FRONTIER_HANDOFF.md)。30.376分钟，380状态全计算，24预算格（13选中/11不可行），19逐用户状态，2536项保存证据核验通过。**主ε=.001下最佳KD−D冷Recall仅+.000122941；科学simple_control_sufficient_in_development，当前没有成立第二创新点。** K_MM不优于K_CF，N各预算不可行；次ε不替代主判据，开发select重复使用/描述性bootstrap不作独立确认。原v1 undetermined保持，不否定所有新物品问题。

本轮按[新运行声明](QUALITY_CONSTRAINED_FRONTIER_RUN_DECLARATION_V1.json)批准的originalTrain partition-only例外恢复允许角色，完整来源容器接触已披露，probe/lock标签不构造或导出；旧Val/Test/旧probe评价/sealed读与hash/锁定确认0。运行已结束，旧命令不可重复启动/覆盖。冻结profile/协议/launch review中的未运行文字保留为历史来源，当前状态由本页/日志/新交接负责。用户既定目标的后续运行直接交命令、手动启动，不重复授权；不自动扩跑。

唯一下一步：将已更新的完整研究交接交给新AI，按跨领域可结合、可辨认的新意优先/候选指标其次的标准重新筛选第二项问题；允许换方向，仅分析，不实现或启动实验。 当前待跑实验为零；第一项继续冻结。

## 历史进度与依据（不构成当前队列）

以下保留先前排序路线及候选记录，当前判断与唯一下一步以页首为准。

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

前次：[独立候选评审](INDEPENDENT_CANDIDATE_REVIEW_2026-09-30.md)。新物品接入优先；TopK边界分配保留备选；预算受限持续更新暂缓。三者均未达到直接开跑条件，本次是当前AI重新判断，不是另一AI或原始产物外部认证。

最新提案：[用户评分几何下的闭式内容映射](NEW_ITEM_METHOD_PROPOSAL_2026-09-30.md)，可归约为已有评分对齐/各向异性岭回归，算法立项未通过；该映射候选暂停，无真实实验。[原研究交接](../RESEARCH_HANDOFF_FOR_INNOVATION2_REVIEW_2026-09-30.md)及上方候选评审保留为历史依据。

最新：[时间充裕后的路线重评](RESEARCH_DIRECTION_WITH_EXPANDED_BUDGET_2026-09-30.md)。用户接受有意义的月级投入；重新开放新物品问题诊断，评分几何岭回归仍不作为新算法。先辨识多模态教师额外能力是否可迁移，区分直接学习、协同蒸馏及简单校准的解释。

该段“准备协议/未运行”是原准备时状态；实际已完成cohort及当前唯一下一步见页首，旧建议不构成执行队列。

[个性化分配立项评审](PERSONALIZED_ALLOCATION_DESIGN_REVIEW_2026-09-30.md)：问题值得继续，简单用户加分已有近邻；暂不认定第二创新点。

最新：[标量校准结果](SCALAR_CALIBRATION_RESULTS.md)。三seed执行通过，科学screen_stop；U仅增加2/1/0个低组命中，当前配方停止扩展。

[第二项路线收束评审](ROUTE_CLOSURE_REVIEW_2026-09-30.md)是历史建议；新物品接入须解决教师暴露与资产重建，尚未授权投入，和压缩候选一样暂缓。
