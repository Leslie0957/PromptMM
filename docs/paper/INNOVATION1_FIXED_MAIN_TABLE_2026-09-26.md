# 第一创新点：固定配置主结果表（论文候选）

日期：2026-09-26。此页的“主表＋表注”可直接移入论文结果节；后面的来源说明只供作者核查。它汇总 Baby 与 Sports 各三种方法、各三学生随机种子的 **18 个最终 Test 格子**。表内所有数值先在原精度下计算，再四舍五入至六位小数。

## 可直接使用的主表

**表 X. 两个数据集上的固定配置最终 Test 结果。** 每个种子格依次为 Recall@20 / NDCG@20；末两列为三个种子的均值 ± 样本标准差。

| 数据集 | 方法 | seed 2022：R / N | seed 2023：R / N | seed 2024：R / N | Recall@20 均值 ± SD | NDCG@20 均值 ± SD |
|---|---|---:|---:|---:|---:|---:|
| Baby | BPR | 0.044280 / 0.019997 | 0.036807 / 0.017025 | 0.042308 / 0.019323 | 0.041131 ± 0.003873 | 0.018782 ± 0.001558 |
| Baby | Full | 0.066592 / 0.030812 | 0.065311 / 0.030520 | 0.066829 / 0.031615 | 0.066244 ± 0.000817 | 0.030983 ± 0.000567 |
| Baby | PromptMM 发布版共享教师适配 | 0.065213 / 0.028834 | 0.064986 / 0.028764 | 0.064917 / 0.028767 | 0.065039 ± 0.000155 | 0.028788 ± 0.000040 |
| Sports | BPR | 0.044254 / 0.022405 | 0.045681 / 0.023117 | 0.042925 / 0.022273 | 0.044287 ± 0.001378 | 0.022598 ± 0.000454 |
| Sports | Full | 0.082653 / 0.039672 | 0.082832 / 0.039488 | 0.083239 / 0.039600 | 0.082908 ± 0.000300 | 0.039587 ± 0.000092 |
| Sports | PromptMM 发布版共享教师适配 | 0.089861 / 0.043011 | 0.089999 / 0.042965 | 0.089945 / 0.042897 | 0.089935 ± 0.000069 | 0.042958 ± 0.000057 |

**表注（随表保留）：** R 表示 Recall@20，N 表示 NDCG@20；SD 为三个学生种子的样本标准差（分母为 \(n-1\)），不是独立数据集或教师的波动。每个数据集内使用固定的一份 Train/Validation/Test 划分和一位冻结教师，按 Validation Recall@20 选择学生检查点，再对同一检查点进行一次最终 Test；候选集仅排除训练交互。Baby 的 BPR/Full 是既有正式结果，发布版适配是后续固定配置训练与独立评价；Sports 三组也来自不同既有训练批次，三臂没有共享同一初始化与完整训练批次。Baby 使用最多 1000 epoch、patience 7，Sports 使用固定 300 epoch，故只比较各数据集内部，不合算跨数据集均值，也不将表中差值解释为充分调优或单一损失项的因果效应。发布版适配保留教师初始化、图结构与目标等实现差异；Sports 发布版学习率 6e-5 曾按 Validation 在两个候选值中选择，Baby 直接迁移该配置，未据 Baby Test 调参。历史 Baby BPR/Full 与新增 12 格的评价实现不同，已核对候选排除与指标语义并通过既有一致性检查，但不声称逐位相同的跨实现 Test 复评。表中没有显著性检验结论。

**正文可用结果句：** 在这组固定配置下，Full 的 Test Recall@20 在 Baby 和 Sports 的三个学生种子中均高于 BPR。与发布版共享教师适配相比，Full 在 Baby 的三种子均值略高（0.066244 对 0.065039），在 Sports 则较低（0.082908 对 0.089935）。这一排序呈现了所测配置的收益与边界；由于初始化、结构、配置选择和训练来源存在差异，不能由主表单独归因于文本蒸馏。文本方向的受控机制证据另来自 Full/image 与 real/sham 的 Validation 对照。

## 作者核查用来源（不粘贴进论文）

- BPR/Full 定义和逐种子 Baby 原正式 Test、选中轮数见 [Baby 总册 §3 与 §10 白名单](../research/INNOVATION1_THESIS_EVIDENCE_2026-09-15.md)。这 6 个原始 manifest 的路径和 SHA256 固定在[共享资产账本](../research/INNOVATION1_REUSE_ASSETS_2026-09-26.json)的 `baby_existing_formal`；不以 Image-0.3 的例外结果替代。
- Baby 发布版适配三格来自 v2 训练的所选 `best.pt` 和[仅评价恢复审计](../research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)的 B3 最终报告；Sports 九格来自同一审计的 S1/S2/S3 最终报告。恢复批次 `exp/formal_closeout_cohort/innovation1_eval_recovery_v1/batch.json` SHA256 为 `fd4d1bf7eb57678db25e3ba32a818d17ab512c5eaebc6c48ac57d20547f79e0f`；该审计列出 12 份最终报告及 12 份预检报告的逐文件 SHA256、Test 时间顺序和选中/被测检查点一致性。
- 共同 Train/Validation/Test 与教师身份、九个 Sports 来源见[共享资产账本](../research/INNOVATION1_REUSE_ASSETS_2026-09-26.json)，账本 SHA256 为 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`。协议兼容范围见[复用核查](../research/INNOVATION1_PROTOCOL_REUSE_AUDIT_2026-09-26.md)及[正式矩阵](INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)。机制证据和部署边界见[正文草稿](SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md)。
- 18 格是 Test **结果格**，并非 18 次新训练；原 Baby 6 格没有因本表再次访问 Test。此页是论文可用表稿，不代表标签、standalone bundle 或忽略资产的物理冻结已经完成；执行清单见[结果冻结清单](../research/INNOVATION1_RESULT_FREEZE_CHECKLIST_2026-09-26.md)。

后续本机文件保存与逐项校验见[结果快照审计](../research/INNOVATION1_RESULT_FREEZE_AUDIT_2026-09-26.md)。上一条保留表稿形成时的状态；本机快照不等于异机备份。
