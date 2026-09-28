# Sports 初始化 × KD seed2024 完成审计

2026-09-28。此前仅有启动声明，本次由三seed原始证据独立审计补记结果。详细检查及局限见[联合审计](INITIALIZATION_KD_THREE_SEED_INDEPENDENT_AUDIT_2026-09-28.md)，[全部文件指纹](INITIALIZATION_KD_THREE_SEED_VERIFICATION_2026-09-28.json)。原始目录 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2024_v1/` 原位保留。

supervisor completed/exit0，acceptance passed，无failure.json；13验收SHA、八个best/final完整模型+AdamW CPU检查通过。source `5450185776df81e77bfcc8b92ab54b0f134fd37c`，config digest `ae86caee1c1f8451cb6254579cc3292b036eaf422fceb8703e31be9f5b34b6ac`；提交配置重建与manifest一致。历史已消耗命令（不可重跑）：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_cohort_seed2024.py'`。

资产和固定配置见[声明](INITIALIZATION_KD_SEED2024_COHORT_LAUNCH_2026-09-28.md)，本次比对原manifest和提交锚。四臂共享seed2024随机表/tape、共同教师cache，各300×214×1024、64200步；AdamW全新lr6e-5/wd.01，KD仅正物品图文alpha.3/text.3，BPR与归约不变。Validation1204次；earliest-best只从1–300选，主终点300轮。Test拒绝源码+report支持Test0，无独立OS访问轨迹；没有被测checkpoint。

| seed | 臂 | epoch0 R20 | epoch300 R20 | epoch300 N20 | 最早best epoch | best R20 | steps |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | R0 | 0.0011643912579358392 | 0.044483316466293048 | 0.021284822771486286 | 299 | 0.044694002459888194 | 64200 |
| 2024 | R1 | 0.0011643912579358392 | 0.082496853220581409 | 0.037287320999579245 | 293 | 0.082542735948075457 | 64200 |
| 2024 | T0 | 0.094184495190847733 | 0.094615509685152296 | 0.043502226689665739 | 263 | 0.094985648189455651 | 64200 |
| 2024 | T1 | 0.094184495190847733 | 0.094445400253286571 | 0.043414984308554401 | 268 | 0.094649955172994057 | 64200 |

末轮ΔR=+0.038013536754288361、ΔT=-0.00017010943186572536、I=+0.038183646186154087。描述性筛查通过；暖小负效应是有效结果，不等于显著、等价或新算法证据。

parent9770.656秒、worker9760.532秒，CUDA allocator峰值161480704B、采样RSS1587085312B、末次output332162703B/free410849853440B，资源门通过，墙钟null。目标有限由运行时检查支持，无逐步loss轨迹；状态有限不证明收敛。

日志补completed/current、矩阵补四格、入口与六导航更新，旧记录/产物/check/不动；旧18格Test及论文缺口不变。不是稳定冻结。唯一下步：第一项初始化依赖与适用边界写作/贡献定位复核，不运行新seed或Test。
