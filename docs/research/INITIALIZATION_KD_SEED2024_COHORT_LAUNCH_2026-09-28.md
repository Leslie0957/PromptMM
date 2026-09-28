# Sports 初始化 × KD seed2024：四臂一次用户手动运行声明

状态：准备和合成验证完成，**真实 seed2024 四臂尚未执行**。目的仅为检验前两次观察到的条件性模式能否在第三套随机初值和训练回放中出现。先前 [seed2022审计](INITIALIZATION_KD_INTERACTION_AUDIT.md)、[seed2023审计](INITIALIZATION_KD_INTERACTION_SEED2023_AUDIT.md)均只用 Validation；固定教师缓存不是独立教师重复。用户保留手动执行 GPU 训练，agent 不代跑。

## 唯一手动命令和来源

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_cohort_seed2024.py'
```

分支 `codex/experiment/baby-teacher-baseline`；准备基线为 seed2023 结果提交 `4c2d2872267ee134563a88c633e1b3e916ed2e9e`，必须从包含本声明的**干净、已提交 HEAD** 启动，manifest 捕获实际 SHA。只允许一次用户手动启动，无 CLI 覆盖参数。新独占目录 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2024_v1/` 在准备时不存在；已消耗的 seed2022/2023 命令与输出保留。四臂 R0→R1→T0→T1 **串行**，任一失败即停，保留部分产物，不重试/恢复/跳臂/进入下一种子。内部 worker 受父进程 token/PID、源码/配置 digest、独占 claim 绑定。

[固定 seed2024 delta](INITIALIZATION_KD_SEED2024_COHORT_CONFIG_2026-09-28.json) SHA 绑定既有 seed2022 四臂配置和 [seed2024只读预检](INITIALIZATION_KD_SEED2024_PREFLIGHT_2026-09-28.json)。与已审计种子相比，只改变 `seed=2024`、随机初值、训练 tape、run_id、输出目录和命令；共同教师缓存、Train/Val、教师来源、模型、损失、优化器、评价、数值环境及阈值均不变。随机初值 SHA `e2c6885debdec09a8ff8401fe4f4f96ea5f46dcc1fd90f24dcc4ecdbd2188e37`，tape SHA `131cc14ebe6853e79216183f06923f3689d47d26c69eac2521d7f93b1b314bc7`，共同缓存 SHA `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`。运行时再次核对输入字节、形状、环境和来源，不复制或补建资产。`PYTHONHASHSEED=2022` 保持共同环境值，Torch 种子取 spec.seed=2024。

## 配置、终点与停止

| 臂 | 初始两表 | 目标 | alpha | 配对 |
|---|---|---|---:|---|
| R0 | seed2024 随机 ID | BPR | 0 | 与 R1 同初值/同 tape |
| R1 | 同 R0 | BPR+Full 图文物品 KD | 0.3 | 与 R0 配对 |
| T0 | 固定共同教师最终 ID | BPR | 0 | 与 T1 同初值/同 tape |
| T1 | 同 T0 | BPR+Full 图文物品 KD | 0.3 | 与 T0 配对 |

每臂 300 epoch×214 batch×1024 三元组、64200 AdamW 更新，共256800更新；同一不可写 seed2024 tape 供全部四臂回放。BPR=`mean(-logsigmoid(dot(u,p)-dot(u,n)))`，无显式 embedding L2；Full 仅正物品图文方向目标，`0.3*(image_mse_mean+0.3*text_mse_mean)/1.3`，无 user KD。每臂从空状态新建 AdamW：lr6e-5、wd0.01、betas(.9,.999)、eps1e-8。教师只提供固定初值和缓存目标，不 forward。

epoch0 全用户 Validation 仅描述；epoch1–300 每轮完整 Validation，总1204次。Train 候选排除、35598用户、batch256、K10/20/40/50、float32确定性精确 TopK 路径不变。主终点固定 epoch300 Recall20；次终点各臂最早最大 Recall20、同轮 NDCG20/曲线。保存每臂 best/final 全模型+AdamW。Test 文件打开拒绝在加载分割前安装，Test 文件读取/排名/最终 Test 均0，无被测 checkpoint。

事前定义 `delta_R=R1−R0`、`delta_T=T1−T0`、`I=delta_R−delta_T`；沿用 `I>=0.002` 及 `T0−T1<=0.001` 的**描述性**候选筛查，不要求暖 KD 为负，也不以低指标判失败。必须分别报告四臂值、两种增量、I、最佳轮及完整轨迹。第三种子可强化随机性稳健性描述，但仍不能区分初值质量/随机初值/回放/用户表信息/优化轨迹各自作用；三 seed 与同一教师缓存不能自动支持显著、等价、长期终点或第二算法。

资源门：**无墙钟硬停止**；CUDA allocator2GiB、采样 worker RSS8GiB、输出1GiB、可用盘至少4GiB，attempt1。前两次父进程分别5821.125秒和8279.407秒，仅供参考，不能保证本次耗时。运行时每30秒父进程打印资源心跳，逐轮进度写入当前臂的 `curve.json`，无墙钟限制意味着卡住需用户自行中断；任一身份/环境/数值/finite/资源/产物门失败即停，不临时扩限或调参。RSS采样和CUDA allocator不等于OS/GPU总占用。

预期 `launch_manifest.json`、`worker_claim.json`、`worker.log`、`telemetry.json`、`report.json`、`acceptance.json`、`supervisor.json` 和 R0/R1/T0/T1 的 `curve.json`、`best.pt`、`final.pt`。exit0单独不足以验收：需四臂各300连续epoch/64200step、1204次Val、有限目标/指标/权重/moments、最早best/末轮完整模型+AdamW状态一致、报告I可重算、SHA匹配、Test0与资源合格；失败产物保留，不自行重试。

## 准备验证与运行后的记录路由

只读预检16.704秒检查历史来源 SHA、seed2024 初值/回放与 seed2022/2023 不同、共同缓存及 Train/Val 身份、全部65,740,800个三元组归属和新目录空闲；未加载模型、GPU、排名、训练或 Test。合成 CPU 验证覆盖精确配置 delta、错误身份拒绝、四臂完整模拟 worker/1204次模拟 Validation、保存/验收、旧 seed 行为与 Test 拒绝。正式运行前再核对干净源码与环境。

记录更新目标：

- run/cohort `sports_init_kd_interaction_seed2024_v1`，R0/R1/T0/T1×seed2024；实际源提交由 launch manifest 固定。
- outcome 审计：新建 `docs/research/INITIALIZATION_KD_INTERACTION_SEED2024_AUDIT.md`，包含每臂身份、曲线/检查点、Val、资源、Test chronology、SHA、失败或低结果；不借旧运行填格。
- TRAINING_LOG：保留本 pending，运行后另记 completed/failed/partial 与当前交接。
- 实验族导航：`docs/experiments/README.md` 初始化×KD 行和根 `README.md` 按状态更新。
- 活跃矩阵：`docs/research/INITIALIZATION_KD_INTERACTION_RESULTS.md` 的 seed2024 四个 Validation 格；旧18格主 Test 矩阵不变。
- 条件性论文/缺口/立项消费者：仅在结果实质改变可支持主张时更新；第三种子本身不是触发条件。
- 六项生成导航：结果人工记录后刷新。历史数据、checkpoint、manifest、输出、`check/` 原位保留；这不是稳定论文冻结，不自动 tag/bundle/push/merge。

唯一下一步：用户手动执行本页命令一次，发回成功或失败供审计。之后可请另一位 AI 复核三次审计和原始曲线；其意见要与预定终点及机制限制逐项对照，不能成为自动执行队列。不得自行启动 seed2025、Test、P1b/P2 或门控。
