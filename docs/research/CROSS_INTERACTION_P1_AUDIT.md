# P1a Sports 双状态共同缓存 U1 筛查结果审计（2026-09-27）

**结论：运行完成，预注册的 G1/G2 筛查在冷/暖 × 图文四个配置均为 `screen_stop`。** 这是有效的低效用筛查结果，不是运行失败；按[一次运行声明](CROSS_INTERACTION_P1A_DECLARATION.md)不进入 P1b、门控或正式训练。本次仅测单物品、单步、训练内部 BPR 损失的加性选择代理，不能推断联合策略、排名或最终推荐指标。

## 身份、执行和硬验收

| 项目 | 审计结果 |
|---|---|
| 用户手动命令 | 从仓库根目录执行 `& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_cross_interaction_p1a.py` 一次；agent 未启动或重试 |
| 启动源 | `codex/experiment/baby-teacher-baseline`，提交 `113e4c8b1af136f0ebb46b7a70a86eba101b3881`；`launch.json`、`report.json`、`supervisor.json` 一致；运行后源码/资产未变化 |
| 协议与计划 | `sports_train_only_p1a_single_item_u1_v1`；封存计划 SHA256 `3dc983a5eff66df500493d75aa4773ef893fa58a3af2964f874cc085ba6e4dd8`，Sports `train_mat` SHA256 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8` |
| 状态与目标 | 冷 Full epoch293、完整 AdamW 两表 step62916；暖 Full epoch282、step60562；两个学生状态都使用声明中的**暖教师语义缓存**。冷状态不是其历史缓存重放；完整 checkpoint、manifest、教师/缓存 hash 见声明及 [P0锚](CROSS_INTERACTION_P0_ANCHOR.json) |
| 配方 | seed31027，两个固定 1024 BPR context；23 物品低/中/高层 3/10/10；图像 Full `0.3/1.3`、文本 Full `0.09/1.3`，half 实际重做 AdamW 配对单步；原 batch×64 归约；372 对/744 独立分支单步（368 非零、4 零对照） |
| 运行与资源 | `supervisor.json` 为 `completed`/exit 0；`report.json` 为 `completed`；272.078 秒，CUDA reserved 峰值 127,926,272 bytes（122 MiB），未触及 1800 秒/2 GiB/6 GiB RSS/512 MiB 输出硬限；原始产物约 45.1 MB |
| 精度与局部性 | 372/372 对均有原始 JSON 和行快照，两个文件各自的 SHA256 与 report 索引相符；368 个非零焦点行均发生变化；4 个零对照的效用与焦点差为 0；所有对的 `precision.consistent`、`rows_unchanged` 为真，21,288 条成对探测三元组均通过 float64/Decimal 参考一致性检查。`analysis.json` SHA256 与 report 一致 |
| 受控访问 | Validation 读取 0、Test 读取 0、教师前向 0、排名评价 0；没有新学生 checkpoint，选模/所选与被测 checkpoint 一致性不适用。历史冷/暖 checkpoint 仅作为这次单步起点 |

运行时非零对 23 项 × 2 模态 × 2 状态 × 2 context × Full/half；每状态/模态有 46 个 Full、46 个 half，对照按每状态×context 放在图像支路的零系数臂中。旧 P0 已曝光正边没有作为新 C 正边；C 的 item467 正角色无支持，保留 N/A，不能以负角色填补。角色效用、抽样质量、方差、共享用户剔除敏感性以及逐三元组精度值均保留在原始 `report.json`、`pairs/` 和 `rows/`，没有压缩或覆盖。

## 冻结筛查的结果

主对照是中/高层各 10 项选 5 项的 **C 池单项 Full U1 加性和**，减去“同数量随机选择的精确期望”与“全层实测 half 系数效用”中的较强者；单位为按原采样概率加权的**单步 BPR loss 改善**。G1 用 B 池理想化挑选、C 池评价；G2 预定主信号为 A 池 `positive_dot`、C 池评价。物品成簇 bootstrap 2000 次，事前下限为 `1×10⁻¹⁰`。下表每格合并中、高层选集的预注册总增益及 95% 物品簇区间；低层 3 项仅描述。

| 状态/模态 | G1：B 选/C 评增益 [95% 区间] | G2：A positive_dot 选/C 评增益 [95% 区间] | 判定 |
|---|---:|---:|---|
| 冷 / 图像 | `-1.527e-12` [`-9.064e-12`, `3.796e-12`] | `4.012e-12` [`-2.173e-12`, `9.213e-12`] | G1/G2 均 `screen_stop` |
| 冷 / 文本 | `-9.549e-13` [`-2.006e-12`, `6.773e-13`] | `1.088e-12` [`-5.577e-13`, `2.208e-12`] | G1/G2 均 `screen_stop` |
| 暖 / 图像 | `-1.733e-12` [`-5.524e-12`, `6.474e-13`] | `-2.319e-12` [`-6.257e-12`, `4.361e-13`] | G1/G2 均 `screen_stop` |
| 暖 / 文本 | `3.091e-13` [`-3.067e-13`, `6.170e-13`] | `5.998e-13` [`-5.426e-14`, `8.217e-13`] | G1/G2 均 `screen_stop` |

四组的 **区间上界**均远低于预定 `1e-10`，因此按声明停止该配置；区间跨 0，不声称效用严格为零或新 KD 普遍无益。B 同池选评的乐观增益四组依次为 `1.008e-11`、`2.355e-12`、`3.534e-12`、`7.511e-13`，均不是过门依据。其他 A 信号、按相同探测条数的正/正负聚合对照、单次分层随机选集及共享用户敏感性已写在 `analysis.json`/逐对报告，不能从中事后另选一个主终点。冷/暖共用暖缓存的条件解释、仅一个种子、低层短缺和稀疏正边限制仍在；这里没有 U5、实际联合更新、排名或泛化证据。

## 原始来源与记录路由

原始目录保持 `exp/cross_interaction/p1a_commoncache_seed2022_u1_v1/`（Git 忽略，不由本次提交备份）。核心文件 SHA256：`launch.json` `35ce31baaeaa58a5c5122cc9fc4677a5b7d104c403c609ec980b26082f0e50f3`；`report.json` `e34c478c75fcb6cf4207c3c83ff05dfb7a100519534d787b8379bc64868b23d4`；`analysis.json` `3c5b36525204f29c53fcafaee6db767bbed873424beeec8c873142302aded1fa`；`supervisor.json` `a1a4561f8e73eec59add9b9bce64b833aed12b6eb20b558211054c6e3bc9e7b2`。逐对 372 个 JSON 和 372 个行快照的 hash 均由 `report.json` 索引，审计已逐一核对。

| 记录目标 | 结果 |
|---|---|
| `TRAINING_LOG.md`、本审计、实验族/根/文档入口 | 追加 completed 结果并将最新状态指向本审计 |
| 路线第 5–6 节消费者 | 仅追加本次门槛结论，不改运行前规划文本 |
| 第一创新点 18 格矩阵、正文/缺口表 | 不适用：无新正式训练、Validation/Test 或排名结果，原主张未改变 |
| 六个生成导航 | 在人工文档编辑后刷新一次；原始目录、旧声明/审计、check/ 保持原位且不提交 |

**唯一建议下一步：**停止这条已筛查的共同缓存单物品 U1 选择配置，先基于本审计重新界定值得验证的研究问题；任何新配置或训练须另行声明，当前不启动 P1b/P2/P3。
