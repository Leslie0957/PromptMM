# 做过哪些实验：粗粒度总览

更新2026-09-29。先看本表，需要具体seed/命令再点审计。第一项暂停扩展；第二项尚未立项。这里按实验族计，不把重复运行或计时窗口当作新增独立实验。

| 实验族 | 状态与主要发现 | 深入记录 |
|---|---|---|
| 两数据集主比较 | Baby/Sports各三方法×三seed，18格Test；Full高于BPR，对发布版Baby略高/Sports较低 | [正式表及来源](../paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md) |
| 文本项对照 | Baby固定图像系数诊断；Sports三seed严格配对，添加文本有Val收益 | [Sports审计](../research/SPORTS_PAIRED_COLDINIT_THREE_SEED_AUDIT_2026-09-25.md) · [Baby总册](../research/INNOVATION1_THESIS_EVIDENCE_2026-09-15.md) |
| 真实/随机残差 | Sports三seed真实残差较好，未唯一归因语言语义 | [审计](../research/SPORTS_SHAM_RESIDUAL_AUDIT_2026-09-26.md) |
| 初始化×持续KD | 三seed十二臂；随机组增益大、暖组小且符号不稳；仅Val | [独立审计](../research/INITIALIZATION_KD_THREE_SEED_INDEPENDENT_AUDIT_2026-09-28.md) |
| 早期梯度/几何/蒸馏强度 | 局部诊断与负结果保留，不证明持续冲突或最优权重 | [机制综合](../research/SPORTS_INTEGRATED_MECHANISM_CONCLUSION_2026-09-26.md) |
| 原缓存部署 | 暖学生对缓存教师无一致在线优势；未含发布版臂 | [旧效率审计](../research/SPORTS_CACHED_DEPLOYMENT_AUDIT_2026-09-26.md) |
| M0–M2成本补证 | 预检、正式检查点部署、短更新完成；M1 v1失败/v2通过；局部成本收益，无完整训练结论 | [独立审计](../research/INNOVATION1_COST_INDEPENDENT_AUDIT_2026-09-29.md) · [成本表](../research/INNOVATION1_COST_RESULTS.md) |
| 跨交互P0/P1a | 精度检查完成，P1a四组screen_stop；无后续算法或排名结果 | [P0](../research/CROSS_INTERACTION_P0_AUDIT.md) · [P1a](../research/CROSS_INTERACTION_P1_AUDIT.md) |
| 评价恢复/资产核验 | 原失败保留，独立恢复完成12格新Test；不是新增12次训练 | [恢复审计](../research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md) |
| 更早数据/教师/预实验 | 历史背景，不自动并入当前合格主表 | [详细族导航](HISTORICAL_FAMILIES.md) |

## 需要再深入时

1. 审计中找具体run及原始产物路径。
2. 找不到时查[运行JSON目录](RUN_RECORDS.md)或[历史日志章节](../../archive/training/ENTRY_INDEX_2026-09-26.md)。
3. 按文件查找用[源码/文档清单](FILES.md)与[资产目录](../../archive/catalog/README.md)。

当前决策：[第一项阶段总结](../paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md) · [第二项探索](../research/innovation2/README.md)。历史pending不是待跑清单。
