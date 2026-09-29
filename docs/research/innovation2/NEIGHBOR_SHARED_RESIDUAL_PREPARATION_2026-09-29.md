# 第二项共享残差：冻结三seed协议与手动启动准备

状态：实现准备完成；正式训练未启动。此文与 `NEIGHBOR_SHARED_RESIDUAL_PROFILE_V1.json`、启动器同一提交构成运行前冻结材料。设计 v1 的“seed2022 首筛后再决定”已被用户本次明确改为固定三seed队列，科学筛查不控制执行。没有 C/S、其他种子、参数试探或 Test。

## 冻结身份和四臂

Sports；教师初始化固定 fused `users/items`，模态 `image_items/text_items` 仅确定已保存的 image/text k20 图。Train、Validation、教师缓存、两图及每 seed 的原始训练三元组 tape 的 SHA256 见 profile；图取此前 modal_neighbor_v1 实邻居和各模态精确频次匹配随机邻居 rep0，不重建或择优。Train 频次按 `(degree,item_id)` 升序三等分；仅两个模态都有效的低组 6114 个物品有非零特征，其余为零。两个固定邻居均值与 q0 差分按有效低组 RMS 分别归一，再各除以 sqrt(2)。若 RMS<1e-12、图和低组不一致，硬停止。共享锚点的 ignored 文件原位保留，Git 提交不保护它们。

每个 seed 的 B/N/F/R 从相同 p0/q0 和空 AdamW 状态起跑，使用该 seed **同一完整** 300×214×1024 tape。B：纯 ID BPR。N：可训练零初始化 W(64×128)，`q+Wx_real`。F：可训练零初始化两个标量，作用于同一 x_real 的图文 64 维块。R：N 的相同 W 结构，仅 x 改为固定 rep0 图并独立按相同公式归一。四臂对正例和采样未观测负例均用完整评分；不加 KD、伪正例或持续教师损失。所有参数 AdamW lr6e-5、weight_decay=.01、betas(.9,.999)、eps1e-8，固定 300 轮、每轮 214 步。N/R 各新增8192参数，F新增2，B新增0；这是设计规定的机制对照，不是参数量完全匹配。训练后折叠为 64 维 ID 表。

## 评价、筛查和失败

只用 Validation，Train 物品排除，全部剩余候选精确 Top20，分数并列按 ID 升序。每轮算总体 Recall@20；第0/50/100/200/300轮存 Top20 和低/中/高三组原始 Recall 贡献、正例 micro、绝对命中/分母及曝光。分母固定 6347/6389/25163，Validation 用户 35598。主终点严格第300轮；best checkpoint/曲线仅补充。历史 NDCG 实现有问题，此配方不以 NDCG 判定。Test0，审计钩子拒绝 Test 文件读取。

B 各 seed epoch0 必须在 .094184495190847733 ±1e-6，epoch300 依序 .095151699965148426/.094343178530468197/.094615509685152296 ±1e-6；所有臂 epoch0 与同一初始评分一致。非有限 loss、梯度、参数、身份、导出、分母、资源、步骤/检查点/文件完整性均硬失败，立刻停整个队列，保留原始产物，不重试/恢复。效果低或科学筛查 `screen_stop` 仍继续预定臂和 seed。

逐seed描述性首筛：N−B 低组 micro≥.001、总体 Recall≥−.0002，且 N−F 与 N−R 低组 micro 各≥.0005；四条件同时满足标 `candidate_signal`。三seed仅当三次均为 candidate_signal 标 `all_three_candidate_signal`，否则 `screen_stop`。同时输出三种臂相对 B 的逐seed配对差及三seed均值、最小、最大与原值。不解释为显著、等价、正式非劣或独立数据集验证，不根据前一 seed 改配方。后续 C/S 需要另行授权。

## 资源与记录

单臂墙钟 8h，队列墙钟 96h；CUDA allocator 2GiB、进程 RSS 6GiB、输出 8GiB、启动与运行剩余磁盘至少 12GiB。资源为进程内协作检查、非 OS 硬隔离。按已有 9 训练臂队列约 5.07h 的实测经验，对 12 臂加同类逐轮评价的预计约 8–12h；这是估算，非保证，单臂/队列上限独立约束。磁盘建议预留至少 12GiB，预计原始输出约 1–3GiB，8GiB 输出上限；备份忽略产物需用户自行物理复制。失败时保留整个 `exp/innovation2/neighbor_shared_residual_three_seed_v1/`，包括 manifest/exit/failure、已完成与部分臂目录、曲线/检查点/Top20/资源/哈希。不得用旧命令覆盖。

手动一次性命令（仅在本提交已形成且工作树除原有 `archive/reviews/` 外干净时执行）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_neighbor_shared_residual.py' --formal
```

启动器拒绝既有输出，先验所有资产哈希和干净提交，逐 seed/臂自动验收；全完成才生成 `summary.json`。`manifest.json`、逐臂 `status.json`/`acceptance.json`、`resource.json`、`completeness.json`、`exit.json` 和运行目录的 `HANDOFF.md` 是机器交接材料，不是独立审计。用户运行后将 HANDOFF 与原始目录交给另一 AI 统一审计。

记录更新目标：原始目录如上；正式 outcome 追加 `TRAINING_LOG.md`，审计 `NEIGHBOR_SHARED_RESIDUAL_AUDIT.md`，结果矩阵 `NEIGHBOR_SHARED_RESIDUAL_RESULTS.md` 的 12 格，交接 `NEIGHBOR_SHARED_RESIDUAL_HANDOFF.md`；更新第二项入口及 `docs/experiments/README.md` 对应族，必要时更新根状态。论文/缺口仅在独立审计支持新主张时触发；第一项表、旧审计、原资产、THESIS_ROADMAP 不适用。完成审计人工文件后刷新六个生成导航。不能以机器摘要替代上述 outcome 审计。

## 准备证据

四项合成测试通过：零初始化评分/冻结特征/折叠评分、正负两侧梯度与共享作用、三组分母/贡献、冻结配置与不完整摘要拒绝。已声明且仅一次非正式资源 smoke：`exp/innovation2/neighbor_shared_residual_smoke_v1/`，seed2022 四臂各20个完整 batch，80步，Validation0/Test0，600秒/2GiB CUDA/6GiB RSS/256MiB 输出/4GiB余量；exit0，耗时5.125秒，采样RSS峰值1,521,123,328B、CUDA allocator峰值176,160,768B，四臂每臂20步。图缩放 real 2.4219/2.3704、rep0 3.1191/3.1033。smoke 不产生效果判定，也未调效果参数。后续代码只补强了资源采样、每臂验收/哈希与交接，没有第二次 smoke。

补充准备验收：按日志中的独立 preflight amendment，`exp/innovation2/neighbor_shared_residual_epoch0_preflight_v1/report.json` 一次无优化的完整 Validation 评分通过；Recall@20=.09418449519085098，相对锚点误差约3.25e-15，三组正例分母6347/6389/25163、命中10/32/3520，总计35598个用户；Test0、训练0步、Validation1次，5.672秒，结束时RSS955,342,848B，CUDA allocator62,914,560B。它不能验证 epoch300 B 回归；正式队列把每seed B末轮回归设为硬停止。此 preflight 与资源 smoke 是不同的隔离产物，资源 smoke 本身 Validation0。
