# PromptMM 研究工作区

核心目标：[论文总体目标与路线](THESIS_ROADMAP.md)。研究决策遵循此页，实验状态不自动改变总体目标。

更新2026-09-29。**第一项阶段性收尾，暂停扩展；第二项最小排序监督三seed已完成独立审计，执行通过、科学筛查均screen_stop；暂停该配方。**

AI启动：完整读[AGENTS.md](AGENTS.md)，再读本页和[短日志](TRAINING_LOG.md)。历史命令、pending及旧“下一步”不构成执行授权。

| 想了解什么 | 入口 |
|---|---|
| 做过哪些实验、粗略结论 | [实验总览](docs/experiments/README.md) |
| 第一项已知与未知 | [第一项阶段总结](docs/paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md) |
| 第一项完整方法与结果 | [整合章节](docs/paper/INNOVATION1_INTEGRATED_CHAPTER_2026-09-29.md) |
| 第二项接下来研究什么 | [探索工作页](docs/research/innovation2/README.md) |
| 查具体运行、旧命令或文件 | [深层导航](docs/README.md) |

当前唯一下一步：根据[残差机制复盘](docs/research/innovation2/NEIGHBOR_RESIDUAL_MECHANISM_2026-09-30.md)设计自身锚点控制方案；R接近自身变换加共同偏移，但尚无因果证明，不自动训练。

训练入口为`codes/main_mmlight.py`，参数默认源`codes/utility/parser.py`，CLI优先。`codes/run_patent.py`是独立历史路径。评审材料移至`archive/reviews/check/`，旧代码集中在archive下；原始数据/模型/运行产物与`zhuanli/`保持原位。搬迁详见[路径映射](docs/experiments/ROOT_RELOCATION_2026-09-29.json)。
