# 文档入口

先读[仓库总览](../README.md)和[当前短日志](../TRAINING_LOG.md)，再按任务展开。

- 最新立项与实施交接：[初始化转移 × 持续监督](research/INITIALIZATION_KD_INTERACTION_PROPOSAL_2026-09-27.md)及[四臂准备审计](research/INITIALIZATION_KD_INTERACTION_PREPARATION.md)。隔离核心、Train/Val 专用适配器与父进程杀停已通过合成验证，真实资产预检与启动验收合成检查已通过，[用户手动资源smoke已审计通过](research/INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md)，但Val耗时使24小时四臂预算缺乏支持；精确提速已通过合成检查，现有[独立手动评价核验声明](research/INITIALIZATION_KD_EVAL_PARITY_LAUNCH_2026-09-27.md)，正式四臂无启动命令；定位机制补充，第二算法未成立。P1a停止状态保留，无新执行授权。下列旧路线仅供追溯。

- 当前研究路线：[跨交互蒸馏效用综合决策](research/CROSS_INTERACTION_ROUTE_2026-09-27.md)。三份AI建议已评审；P0/P1a 已按门槛完成诊断，尚无新算法或正式训练授权。旧候选不再代表当前优先级。

- [P0精度审计](research/CROSS_INTERACTION_P0_AUDIT.md)7对通过，36项角色效用数值可分辨；旧 P0 粗精度结论保留在审计中。
- 最新结果：[P1a审计](research/CROSS_INTERACTION_P1_AUDIT.md)核实共同缓存的372对单步诊断完成；冷/暖×图文四组的预定 G1/G2 均筛查停止。尚无联合策略、正式训练或排名证据；[原手动声明](research/CROSS_INTERACTION_P1A_DECLARATION.md)仅供追溯，不可重跑。

- [正式收尾协议](research/INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md)：固定配置、Baby预算和新评价入口验收；[资产复用核查](research/INNOVATION1_PROTOCOL_REUSE_AUDIT_2026-09-26.md)；[实验结束更新规范](experiments/RUN_CLOSEOUT.md)。
- [v1失败审计](research/INNOVATION1_FORMAL_CLOSEOUT_COHORT_V1_AUDIT.md)：原串行命令空跑三次Baby入口，在首次预检停止；零新Test。旧[启动页](research/INNOVATION1_FORMAL_CLOSEOUT_LAUNCH_2026-09-26.md)仅供追溯，不能再执行。
- [v2失败及有效产物审计](research/INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md)：Baby三次训练有效、v2零Test；[仅评价恢复结果](research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)完成12项新预检及12项一次性Test。
- [论文写作入口](paper/README.md)：第一创新点18格固定主比较[主表与表注](paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md)、[正式主张矩阵](paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)及[本机快照审计](research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md)。
- [第二创新点候选决策](research/INNOVATION2_CANDIDATE_DECISION_2026-09-26.md)：找回RGCS、复核前置条件及门控假设。
- [论文完成条件与缺口表](paper/THESIS_COMPLETION_GAPS_2026-09-26.md)：两项创新点、数据集、正式评价及待确认要求。
- [实验族导航](experiments/README.md)：每类实验的审计入口、失败记录及结论范围。
- [论文正文表述（已审阅）](paper/SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md)：已区分正文、讨论和附录内容。
- [全部源文件/文档](experiments/FILES.md)：按目录查找，不需要逐篇读取。
- [运行记录导航](experiments/RUN_RECORDS.md)：现存exp JSON；不是独立实验计数。
- [历史章节索引](../archive/training/ENTRY_INDEX_2026-09-26.md)：老参数、命令和声明按需查询。
- [Baby正式结果总册](research/INNOVATION1_THESIS_EVIDENCE_2026-09-15.md)：保留正式Test证据与例外。
- [第二创新点历史候选](research/SECOND_INNOVATION_ROUTE.md)：尚未实现/确认，不是当前运行计划。

`research/`保留专题审计原路径；`research_notes/`保留历史讨论。
日期较早文档中的“当前”或“下一步”仅适用于当时，不覆盖根目录当前状态。
[CLEANUP_CANDIDATES.md](CLEANUP_CANDIDATES.md)是历史清理建议，未据此删除资产。
