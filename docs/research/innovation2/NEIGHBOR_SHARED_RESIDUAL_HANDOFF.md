# 共享残差三seed交接

2026-09-30：12臂执行通过，独立产物核验783项通过；三seed与cohort均screen_stop。无重跑、无Test，原始产物保留 `exp/innovation2/neighbor_shared_residual_three_seed_v1/`。启动提交b8be549；结果提交见本次Git交接。

| seed | N−B总体Recall | N−B低组命中 | N−F低组命中 | N−R低组命中 | 判定 |
|---|---:|---:|---:|---:|---|
| 2022 | +0.000095979 | +25 | +6 | -22 | screen_stop |
| 2023 | +0.000266869 | +25 | +5 | -22 | screen_stop |
| 2024 | +0.000096247 | +24 | +4 | -24 | screen_stop |

N相对B有稳定的小幅总体/低组收益，但低组三seed都输给同结构随机邻居R。当前无法证明模态邻域的独立价值。R仍包含目标−q0，不等于纯噪声。详见[结果](NEIGHBOR_SHARED_RESIDUAL_RESULTS.md)、[审计与限制](NEIGHBOR_SHARED_RESIDUAL_AUDIT.md)。导出仅局部探针、未重读数据复算排名，限制已保留。

唯一下一步（另行执行，不是训练授权）：

> 对已有N/F/R的固定特征及检查点做有界只读机制复盘，重点检查−q0自身锚点、随机邻居均值、共享更新和低组曝光变化。先界定能够区分的解释，再决定是否修订方案；不重跑训练、不读取Test、不自动启动C/S或新增seed，不因低效果删除或覆盖任何产物。
