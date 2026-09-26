# 论文写作入口

研究动机和贡献先看[第一创新点立项决策单](INNOVATION1_RATIONALE_REVIEW_2026-09-26.md)：保留简化方向监督与适用边界；不主张二次蒸馏或线上降本。独立创新强度尚待确认。

第一创新点的**可直接引用结果**先看[18 格固定配置主表与表注](INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md)。论文撰写时从此表复制数值和脚注，不从终端截图、旧计划或文件名挑结果。原始格子与资产指纹见[本机结果快照审计](../research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md)；如需逐文件查找，使用[冻结清单](../research/INNOVATION1_RESULT_FREEZE_MANIFEST_2026-09-26.json)中的仓库相对路径。

| 要写的部分 | 首选文件 | 使用边界 |
|---|---|---|
| 第一创新点正式 Test 主结果 | [18 格主表与表注](INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md) | Baby/Sports 各三方法、三学生种子；固定配置且按各数据集内比较 |
| 文本方向、暖启动、效率的正文与局限 | [已审阅正文草稿](SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md) | 机制主要是 Validation，效率是所测阶段；与 Test 主表分开 |
| 哪些主张能写、哪些不能写 | [第一创新点最终主张矩阵](INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md) | 不称 SOTA、充分调优、单因素因果或统计显著 |
| 整篇论文还缺什么 | [论文完成条件与缺口](THESIS_COMPLETION_GAPS_2026-09-26.md) | 第二创新点及外部覆盖要求尚未完成 |
| 核查原始来源与恢复文件 | [结果快照审计](../research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md) / [原正式评价审计](../research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md) | 历史 Baby 六格与恢复批次十二格分别追溯；原始文件保持原位 |

本机副本位于 `backups/innovation1_fixed_results_2026-09-26/`，内有 `manifest.json` 和按原路径排列的 `assets/`。它与仓库同一物理磁盘，**不能防磁盘故障**；以后商量 GitHub 或异机备份时以快照审计与清单为入口。GitHub 代码仓库也不会自动包含 Git 忽略的大资产。
