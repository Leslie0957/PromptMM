# P1a Sports 双状态共同缓存 U1 筛查：手动运行声明（pending）

**状态：只完成代码与合成验证；真实命令未执行。** 本声明只覆盖一次 Sports seed2022 冷/暖已存在 Full 状态的训练内**单物品 U1**筛查。用户明确选择两个学生状态共用[P0已固定暖缓存](CROSS_INTERACTION_P0_ANCHOR.json)。冷状态的历史教师语义缓存没有保存逐位身份，因而本运行是新的共同目标条件诊断，**不是冷训练轨迹复现**。不训练门控，不做 U5、P1b/P2、正式训练、Validation/Test 读取或排名。

## 一次运行的固定身份与命令

- 分支 `codex/experiment/baby-teacher-baseline`。启动源必须是含本声明、封存计划、实现、测试和准备结果的**干净提交**；父提交 `3c9ab2cea281a7db96ddec306d08ad31e28d8bf3`。只有用户原文 `check/` 可以保持未跟踪且不参与源码身份。启动 HEAD 写入 `launch.json`；本声明所在提交无法在自身提前写入自己的 hash，下次结果审计核对。
- 共用锚 `SPORTS_P0_WARM2022_EXISTING_STATE_V1`：warm Full seed2022 epoch282、完整 AdamW step60562、checkpoint SHA `bd651d78ab8853ba987c407a23cd485581ccf319a1b6c9202c5413091fc6879c`；train_mat SHA `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`；六张共享初始/教师张量缓存 SHA `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`；教师 epoch37 checkpoint SHA `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。完整路径见锚 JSON。
- 冷 Full seed2022 epoch293 checkpoint `Model/sports/td_distill/td_distill_full__val_test_once_v1__2026-09-24 18_22_52.151219_sports_light_init_pid5012.pth` SHA `d2648d9f1f6c593b8720b27ad226e81152373537f7cb4d3fa18de07acf20b4a2`；manifest 同时间戳路径 SHA `d31cf835fd68b575b7312e1328f8e439ad0d459302e65c9587b524ce760d6ddf`。只读 mmap 元数据核查了 epoch/profile 和完整 AdamW 两表 step62916；运行时重验 hash、形状/有限值、超参及 step。
- [封存计划](CROSS_INTERACTION_P1A_PLAN.json) SHA256 **`3dc983a5eff66df500493d75aa4773ef893fa58a3af2964f874cc085ba6e4dd8`**。由 train_mat 和旧 P0 plan SHA `11130dd5cc71ee2da45a485bb04c3946a6d5e9f310940019cc219fe1d1557393` 再生并逐字节比较；不能在看新效用后重新抽样。seed31027，2×1024 固定 update context，23 项分层 **3/10/10**；低频层短缺7项、item467 的 C 正角色 N/A，不替换。A/B/C 正边独立，旧 P0 实际用过的 264 条正边从新 C 剔除；新 C 探测池 47,746 条边。
- 更新配方：每项、每 context、每模态从同一完整状态与 AdamW 参数/动量/step/RNG/批次分别做 A=BPR、B=BPR+焦点 item KD 一次 step。图像真实 Full `0.3/1.3`，文本 `0.09/1.3`；half 是**实际重新配对**，系数为前者一半，绝不把 Full 效用除二。原 batch×64 归约，焦点在 batch 中所有出现均保留；其它项不重归一化。两个完整学生状态、两模态、两 context、23 项、Full/half → **368 非零对**；另每状态×context 一零对照 → **4 对**；合计 **372 对/744 个独立单步**，无持续训练或 checkpoint 输出。
- A/B/C：每物品/角色/池至多 8 次按原全训练 user-uniform 边际条件抽取的基础边（三组相互独立）；正角色每基础边独立抽 3 个协议负候选，负角色固定 `n=focus`，不能假造三次不同负候选，只取最多8次合法 `(u,p)`。有放回重复边不当成独立用户；角色质量按真实采样概率，正/负角色及合成贡献分别报告。计划合计 2,646 个三元组；非零对 21,168 次成对探测三元组，4 零对另120次，A/B 两更新分支共至多 **42,576 次三元组读数**。这不是新交互泛化；既有 checkpoint 历史上见过训练边。
- 只读效用精度：原 float32 更新及旧 float32 探测保留，更新后截取同一 A/B 原 float32 行，CPU64 稳定损失差并以独立 Decimal80/120 校验；沿用[P0冻结工程容限](CROSS_INTERACTION_P0_PRECISION_DECLARATION.md)。零干预差必须为0；非焦点行/用户行差必须为0；不满足即硬失败，保留部分产物。

从仓库根目录，**仅由用户手动执行一次**：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_cross_interaction_p1a.py
```

运行前不需要重复 P0 命令。启动器拒绝非干净来源（唯一豁免未跟踪 `check/`）和已存在输出目录；一次启动后无重试、恢复、加样或下一阶段许可。

## 统计筛查及解释边界（运行前冻结）

主效用是按原采样器角色质量合成的 `loss_A−loss_B`，正/负角色均值另报。对每个状态×模态，先将同物品的两个 context 平均，**只在中、高频层各10项**做主选择比较；低层仅3项且1项 C 正角色 N/A，完整保留作描述，不补齐、不拿缺失项过门。B 池 Full 效用各层取前5项，C 池一次性评价；A 的 `positive_dot` 是预定 G2 主信号，其它 local dot/cos、同样三元组预算的正负聚合、hard margin、频次、表示差异及常数规则为比较。正-only用2k条A三元组，正负聚合用k+k，k由两角色可用条数预定；不让混合信号因额外读数据获益。A 特征只读原状态和 A 探测，不看 B/C 效用。

每项的 C 单项 U1 是选择代理。每层把“所选5项 Full 的 C 单项效用之和”与同数量随机选择的**精确期望** `5/10×全层 Full 效用之和`以及所有10项**实际测得**half 系数效用之和比较；另报告一次固定种子分层随机选集。主增益为所选之和减去两对照中较强者。B 同池选评的乐观结果另报，**不参与过门**。这只是单项效果相加的筛查分数，不是联合执行选择后的真实更新效果、全局 oracle 上界或排名收益。两 context 同一物品为同簇，分层按物品有放回重采样 **2,000 次**、固定 seed3102701 得95%区间；共享用户剔除敏感性及正负角色方向需在审计中单报。一个种子和20个中高频物品不支持总体显著性结论。

预先设定的**研究投入筛查下限**为加性代理每步 BPR loss `1×10⁻¹⁰`，高于已审计 P0 三项质量加权效用的典型量级；用于判断是否值得投入后续有限步/联合诊断，不是推荐质量或部署收益阈值。若 C 增益簇区间下界高于该值，记该状态/模态的 G1 筛查通过；上界低于则停止该配置；跨过则本预算内不确定，不扩样。G2 主比较固定为 A `positive_dot` 按同规则过门；正负 `both_dot` 对等预算下相对正-only 的差另报，不事后挑 C 上最好的规则。低/负/不确定效用是有效结果，不是执行失败。G0 数值、G1/G2 筛查、G3 有限步/联合策略及排名严格分开；本次最多产生 G0/G1/G2 的有限证据，不自动进入 P1b。

## 硬预算、停止和记录目标

- 运行器单次、独占 `exp/cross_interaction/p1a_commoncache_seed2022_u1_v1/`；外层硬超时 **1800秒**，内层 1740秒检查；CUDA allocator 限制 `min(2GiB, 设备内存25%)`，进程 RSS 上限6GiB，torch CPU线程2，原始输出上限512MiB。P0的4.86秒/7对不能线性保证372对的完成时间。超时/超限/输入hash变化/计划无法再生/非有限值/精度或零对照不符/held-out 读取/计划身份异常，均停止并保留部分文件；不能自动重试。硬限制只约束本运行进程及其 allocator，外部系统负载不在此承诺内。
- 输出 `launch.json`、`worker.log`、`report.json`、`supervisor.json`、`analysis.json`、逐对 `pairs/pair_NNN.json` 与 `rows/rows_NNN.pt`。每对保存完毕才增量更新报告；失败或部分完成不冒充372对完成。最后复核全部输入 hash 与提交身份。无学生新 checkpoint、无正式选模/Test 指标；Validation/Test 分割读取0、排名0、教师前向0，selected-versus-tested 不适用。
- **记录更新目标：**本次身份 `p1a_commoncache_seed2022_u1_v1`、冷/暖×图文×两context。结果新建 `docs/research/CROSS_INTERACTION_P1_AUDIT.md`，追加 `TRAINING_LOG.md` 的本 pending 对应完成/失败记录；更新 `docs/experiments/README.md` 跨交互行与根/文档入口（仅状态变化时），刷新六项导航。当前活跃矩阵仍只有旧第一创新点18格，本训练内诊断**无对应格子**；[路线第5–6节](CROSS_INTERACTION_ROUTE_2026-09-27.md)仅在实测条件变化时追加说明。论文主表/缺口仅在正式主张改变时更新，本次默认不适用。原始产物保持上述忽略目录，不提交；这是常规诊断，不创建标签或bundle。

**唯一后续动作：**用户只运行上面的新命令一次，返回终端结束状态；随后核查现有产物并提交结果。任何失败、时间到限或部分产物均先审计，不重试、不自动开启 P1b/P2。
