# PromptMM 研究工作区

核心目标：[论文总体目标与路线](THESIS_ROADMAP.md)。研究决策遵循此页，实验状态不自动改变总体目标。

更新2026-09-30。**第一项已最终收尾，至论文初稿阶段冻结；后续研究重心先全部转到第二创新点探索。** 第二项尚无成立的方法，历史标量配方screen_stop。

AI启动：完整读[AGENTS.md](AGENTS.md)，再读本页和[短日志](TRAINING_LOG.md)。历史命令、pending及旧“下一步”不构成执行授权。

用户最新工作流（2026-09-30）：**目标及具体运行范围已确定后，agent完成准备、声明、核查与提交，直接给可执行命令，不再单独请求授权；用户手动运行，agent负责结果审计。** 已写入[AGENTS现行规则](AGENTS.md#agreed-goals-and-manual-run-handoff)，替代旧交接的重复授权依据。目标/范围确实不清时只澄清待定事项；固定数据/Test边界、单次执行及证据保留仍适用。

创新判断统一遵循[论文路线中的用户确认标准](THESIS_ROADMAP.md)：允许有明确差异与价值的组合、跨领域迁移和适配，不要求基础部件原创，不默认采用顶会级新机制门槛。第一项不再默认补实验或重复泛泛评审；FDRec仅为初稿阶段可考虑的直接基线补充，届时按最终质量与成本主张决定，当前不是待跑项。

| 想了解什么 | 入口 |
|---|---|
| 交给其他AI评审第二项方向 | [第一项成果与第二项全部探索交接](docs/research/RESEARCH_HANDOFF_FOR_INNOVATION2_REVIEW_2026-09-30.md) |
| 最新候选评审与优先级 | [第二项独立候选评审](docs/research/innovation2/INDEPENDENT_CANDIDATE_REVIEW_2026-09-30.md) |
| 新物品具体提案与判定 | [评分几何映射提案：算法立项未通过](docs/research/innovation2/NEW_ITEM_METHOD_PROPOSAL_2026-09-30.md) |
| 时间充裕后的研究路线 | [教师可迁移性与混合目录瓶颈诊断](docs/research/innovation2/RESEARCH_DIRECTION_WITH_EXPANDED_BUDGET_2026-09-30.md) |
| 当前诊断协议与实现来源 | [Sports未见物品诊断v1](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROTOCOL_V1.md) · [实现与CPU合成预检](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_RUNTIME_PREPARATION_2026-09-30.md) |
| 质量约束前沿的结果与下一步 | [开发结果](docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_RESULTS.md) · [保存产物审计](docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_AUDIT.md) · [交接](docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_HANDOFF.md)；已完成，无待跑项 |
| 做过哪些实验、粗略结论 | [实验总览](docs/experiments/README.md) |
| 第一项收尾与冻结边界 | [最终收尾](docs/paper/INNOVATION1_FINAL_CLOSEOUT_2026-09-30.md) |
| 第一项已知与未知 | [第一项阶段总结](docs/paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md) |
| 第一项完整方法与结果 | [整合章节](docs/paper/INNOVATION1_INTEGRATED_CHAPTER_2026-09-29.md) |
| 第二项接下来研究什么 | [探索工作页](docs/research/innovation2/README.md) |
| 查具体运行、旧命令或文件 | [深层导航](docs/README.md) |

当前：Sports未见物品原诊断v1 completed/undetermined保留；[质量约束前沿v1](docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_RESULTS.md)已由用户手动完成，30.376分钟、380状态，保存证据2536项检查通过。主ε.001最佳KD仅比直接学习D高.000122941，科学simple_control_sufficient_in_development；开发select复用，不认证第二创新点或泛化。原Train分区恢复按本次明确许可执行，旧Val/Test/旧probe评价/封存确认0。唯一下一步：基于两轮诊断做第二项路线重评，筛选相对“直接学习＋相同校准”有明确新增价值的问题，并设计一个最小可证伪对照；仅评审与设计，不启动实验。 当前无待跑项，第一项继续冻结。

训练入口为`codes/main_mmlight.py`，参数默认源`codes/utility/parser.py`，CLI优先。`codes/run_patent.py`是独立历史路径。评审材料移至`archive/reviews/check/`，旧代码集中在archive下；原始数据/模型/运行产物与`zhuanli/`保持原位。搬迁详见[路径映射](docs/experiments/ROOT_RELOCATION_2026-09-29.json)。

[个性化分配立项评审](docs/research/innovation2/PERSONALIZED_ALLOCATION_DESIGN_REVIEW_2026-09-30.md)：问题值得继续，简单用户加分已有近邻；暂不认定第二创新点。

最新：[标量校准结果](docs/research/innovation2/SCALAR_CALIBRATION_RESULTS.md)。三seed执行通过，科学screen_stop；U仅增加2/1/0个低组命中，当前配方停止扩展。

[第二项路线收束评审](docs/research/innovation2/ROUTE_CLOSURE_REVIEW_2026-09-30.md)保留历史建议；新物品接入和压缩均未立项，当前边界见研究交接。
