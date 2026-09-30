# PromptMM 研究工作区

核心目标：[论文总体目标与路线](THESIS_ROADMAP.md)。研究决策遵循此页，实验状态不自动改变总体目标。

更新2026-09-30。**第一项已最终收尾，至论文初稿阶段冻结；后续研究重心先全部转到第二创新点探索。** 第二项尚无成立的方法，历史标量配方screen_stop。

AI启动：完整读[AGENTS.md](AGENTS.md)，再读本页和[短日志](TRAINING_LOG.md)。历史命令、pending及旧“下一步”不构成执行授权。

创新判断统一遵循[论文路线中的用户确认标准](THESIS_ROADMAP.md)：允许有明确差异与价值的组合、跨领域迁移和适配，不要求基础部件原创，不默认采用顶会级新机制门槛。第一项不再默认补实验或重复泛泛评审；FDRec仅为初稿阶段可考虑的直接基线补充，届时按最终质量与成本主张决定，当前不是待跑项。

| 想了解什么 | 入口 |
|---|---|
| 交给其他AI评审第二项方向 | [第一项成果与第二项全部探索交接](docs/research/RESEARCH_HANDOFF_FOR_INNOVATION2_REVIEW_2026-09-30.md) |
| 最新候选评审与优先级 | [第二项独立候选评审](docs/research/innovation2/INDEPENDENT_CANDIDATE_REVIEW_2026-09-30.md) |
| 新物品具体提案与判定 | [评分几何映射提案：算法立项未通过](docs/research/innovation2/NEW_ITEM_METHOD_PROPOSAL_2026-09-30.md) |
| 时间充裕后的研究路线 | [教师可迁移性与混合目录瓶颈诊断](docs/research/innovation2/RESEARCH_DIRECTION_WITH_EXPANDED_BUDGET_2026-09-30.md) |
| 当前诊断协议与实现来源 | [Sports未见物品诊断v1](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROTOCOL_V1.md) · [实现与CPU合成预检](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_RUNTIME_PREPARATION_2026-09-30.md) |
| 做过哪些实验、粗略结论 | [实验总览](docs/experiments/README.md) |
| 第一项收尾与冻结边界 | [最终收尾](docs/paper/INNOVATION1_FINAL_CLOSEOUT_2026-09-30.md) |
| 第一项已知与未知 | [第一项阶段总结](docs/paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md) |
| 第一项完整方法与结果 | [整合章节](docs/paper/INNOVATION1_INTEGRATED_CHAPTER_2026-09-29.md) |
| 第二项接下来研究什么 | [探索工作页](docs/research/innovation2/README.md) |
| 查具体运行、旧命令或文件 | [深层导航](docs/README.md) |

当前：Sports未见物品诊断v1已completed/exit0，保存证据自审567项通过；科学undetermined（多模态教师暖probe优势未建立），第二算法尚未成立。见[结果](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_RESULTS.md)、[审计](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_AUDIT.md)、[HANDOFF](docs/research/innovation2/UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_HANDOFF.md)。[最新路线重评](docs/research/innovation2/POST_DIAGNOSTIC_ROUTE_REVIEW_2026-09-30.md)保留新物品接入问题、暂缓扩大MM教师/KD；唯一下一步是准备固定内容模型的质量约束冷暖竞争前沿诊断协议，仅准备，不实现评分或启动实验。旧Val/Test及锁定确认0，无自动重试/换seed/扩参；第一项继续冻结。

训练入口为`codes/main_mmlight.py`，参数默认源`codes/utility/parser.py`，CLI优先。`codes/run_patent.py`是独立历史路径。评审材料移至`archive/reviews/check/`，旧代码集中在archive下；原始数据/模型/运行产物与`zhuanli/`保持原位。搬迁详见[路径映射](docs/experiments/ROOT_RELOCATION_2026-09-29.json)。

[个性化分配立项评审](docs/research/innovation2/PERSONALIZED_ALLOCATION_DESIGN_REVIEW_2026-09-30.md)：问题值得继续，简单用户加分已有近邻；暂不认定第二创新点。

最新：[标量校准结果](docs/research/innovation2/SCALAR_CALIBRATION_RESULTS.md)。三seed执行通过，科学screen_stop；U仅增加2/1/0个低组命中，当前配方停止扩展。

[第二项路线收束评审](docs/research/innovation2/ROUTE_CLOSURE_REVIEW_2026-09-30.md)保留历史建议；新物品接入和压缩均未立项，当前边界见研究交接。
