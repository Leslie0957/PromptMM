# 最小排序监督三seed交接

2026-09-29。独立审计及结果收尾完成。三个子目录执行通过，三次主Recall筛查均screen_stop；不是运行失败。源码启动commit `427459f60416a44ef00c859e835f0313546ccb36`，结果提交由最终回复给出。

[审计](MINIMAL_RANKING_THREE_SEED_AUDIT_2026-09-29.md) · [12格结果](MINIMAL_RANKING_SUPERVISION_RESULTS.md) · [逐文件指纹与289项核验](MINIMAL_RANKING_THREE_SEED_VERIFICATION_2026-09-29.json)。原始目录`exp/innovation2/minimal_ranking_three_seed_v1/seed2022|seed2023|seed2024/`保持原样，忽略产物不由Git备份。

已执行命令（已消耗，不可重跑）：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_minimal_ranking_three_seed.py' --formal`。

三个seed同一教师/固定候选，B/R/A各300轮64200步、AdamW6e-5；M为.5/.5。三B回归通过，51份验收指纹一致，18个best/final身份/有限性/AdamW通过，2712次记录Validation，Test0，无本次重评。总5.07小时，各seed/整体资源在声明上限内。

重要限制：原ndcg20是非标准命中数归一化指标，不能作标准NDCG或跨旧初始化表比较；Recall主结论不受影响。源记录和Test禁读机制支持Test0，非OS证明。固定KL、初值约束、混合的末轮Recall均低于B，不证明所有相关方法无效或长尾方向成立。

路由已完成：audit/results/handoff/verification/helper、TRAINING_LOG、当前导航与实验族；原始产物、第一项消费者、总体目标及旧声明保留。唯一下一步是用户审阅本交接后决定是否另行评审备选；不自动训练、重评、加seed、调参或修NDCG后重跑。
