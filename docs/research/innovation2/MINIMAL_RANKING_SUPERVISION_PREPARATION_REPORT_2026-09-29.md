# 最小排序监督：实现与资源准备报告

2026-09-29。状态：**实现准备完成；正式训练未声明、未启动。** 依据[方案 v1.0](MINIMAL_RANKING_SUPERVISION_PROPOSAL_2026-09-29.md)。本报告记录执行方准备检查，不是正式 cohort 结果或独立审计。

## 固定实现与启动边界

- 新的独立目标模块 `codes/minimal_ranking_supervision.py`、正式/非正式 runner `tools/run_minimal_ranking_supervision.py`、一次性候选准备器 `tools/prepare_minimal_ranking_candidates.py`；不改第一项训练入口、旧 checkpoint、旧结果及原始数据。
- [固定 profile](MINIMAL_RANKING_SUPERVISION_PROFILE_V1.json) SHA256 `3aa8cb62977b1079f8d05eefe3bd12cc70b9ae34f6cc26726bd28cd1a13335a8`。Sports seed2022，原初始化实验 manifest/source/config 为共享锚；B→R→A 各 300×214 步，教师融合表共同初值，原完整 triplet tape，空 AdamW moments、lr 6e-5、wd .01。R= BPR+权重1的固定64候选 KL；A=BPR+权重1的归一化坐标初值约束；M 为 B 末轮与冻结教师固定 0.5/0.5 分数混合。候选 NumPy 独立 RNG seed 20260930；CPU/CUDA/全局 NumPy seed 2022，TF32 关闭，确定性算法及 CUBLAS 配置固定。
- 正式评价只用 Validation，B/R/A 的 epoch0 与 1–300 共 903 次，加 M 末轮一次为 904 次；epoch300 Recall@20 为主终点。B epoch0 与原教师、B epoch300 与旧 T0 的误差必须 ≤1e-6；若 B 回归失败即停止在 B。保存曲线、best/final checkpoint、末轮四状态 top20、候选、资源样本与源/资产指纹。Test 路径拒读钩子在资产加载前安装，报告真实被拒尝试次数。
- 正式输出固定 `exp/innovation2/minimal_ranking_seed2022_v1/`，目前不存在。profile 状态 `preparation_only_not_launchable`，runner 的正式入口会在加载资产/创建输出前拒绝。后续只有在用户明确授权**这一** cohort、追加完整正式 pending 声明、将状态改为 `launchable_after_explicit_authorization` 并提交干净源码后，才执行以下精确命令一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_minimal_ranking_supervision.py' --formal
```

正式上限：8 小时、CUDA allocator 2 GiB、RSS 6 GiB、新输出 2 GiB、盘余量至少 4 GiB；协作式 1 秒采样，不是 OS 硬隔离。不得从本报告将当前 profile 直接视作已授权启动。固定候选资产位于被 Git 忽略的 `exp/innovation2/minimal_ranking_candidate_prepare_seed2022_v1/candidates.npz`，正式前必须确认文件仍在且 SHA 一致；源码提交不能备份它。

## 合成验证和真实资源预检

六项合成测试通过：KL 在相同教师/学生 logits 处为零且梯度近零、教师目标 detach；anchor 逐坐标平方与缩放；固定 0.5 混合与 128 维拼接内积等价；候选排除/去重/同分与 NumPy RNG 隔离；BPR 步和非有限值异常；Test 路径拒读与尝试计数。三个新 Python 文件通过编译。profile 的固定字段、锚和状态通过静态校验。

仅执行**一次**独立非正式 smoke：`--resource-smoke`，输出 `exp/innovation2/minimal_ranking_resource_smoke_seed2022_v1/`。B/R/A 各 10 步、候选最多 256 个 Train 用户，Validation 评价 0、Test 被拒尝试 0；`acceptance.json` passed、`exit.json` completed/0，stderr 为空。耗时 7.625 秒，CUDA allocator 峰值 127,926,272 B，采样 RSS 峰值 1,371,414,528 B，最低可用盘 401,991,491,584 B，采样间隔约 1 秒，无 OS 硬隔离。为守住 256 用户上限，每步从原 1024 三元组中仅保留 6–11 条；这项 smoke **不能代表**正式 1024 批大小、全量候选访问、904 次 Validation 或 8 小时总时长。没有用 smoke loss 调权、选候选或作效果判断。

另按方案要求单独进行**一次**全量候选准备，输出 `exp/innovation2/minimal_ranking_candidate_prepare_seed2022_v1/`；仅读取 SHA 固定的 Train 与共享教师融合表，未读取 Validation、Test 或 ranking-gap NPZ，训练步数 0。35,598 个 Train 用户各有 32 个教师 top 与 32 个均匀无放回非 Train/非 top 物品，形状 35,598×64；文件 SHA256 `9ef0f59bb457a51539856f8def1f156658f0af10808470c3a1c94f074dc6e571`、逻辑数组 SHA256 `f9acea50a73e7c8ba0bb3bbb7fc02cc7f033d3fdc2f5aa152b17686263ada90e`，均固定到 profile。`acceptance.json` passed，耗时 22.422 秒，采样 RSS 峰值 642,064,384 B、最低盘余量 401,966,374,912 B，输出候选约 13.53 MB。正式 runner 只加载并校验该文件，不再生成另一套候选。

原始入口：smoke 的 `manifest.json`、`report.json`、`resource_samples.json`、`acceptance.json`、`exit.json` 和 stdout/stderr；候选准备的同名 manifest/acceptance/exit/资源样本及 `candidates.npz`。关键 SHA：smoke report `5145bf2546765d8f2ea660b56c27abbf5f49ec3b14e0ef6c35d5a71a69570762`，smoke acceptance `d5c23859fd361ccb03cd8137fec6010ba757087b577188489c514ae0d5516073`；候选准备 manifest `a4e20bb1cdd66b999e9d8b98d4d584201de01fc02e0e3039e798788de7a9b1e3`。两次准备均在未提交的新实现工作树执行，原始 manifest 所记 HEAD `e4e81da` 仅为基线提交，不代表准备源码已提交；本任务末尾的准备提交与最终 profile SHA 才是后续审查依据。

## 判断、路由与下一步

合成逻辑、真实资产读入、固定候选生成与受限三臂更新路径已通过。正式端到端数值回归和完整资源上限**尚未验证**，只能由明确授权的唯一正式 cohort 给出；旧 seed2022 四臂运行耗时约 5,821 秒可作粗背景，不是新 runner 的 8 小时保证。特别是 R 的完整批 KL、904 次评价及 best/final 写盘不在 smoke 中。若正式门失败，保留产物、记录 technical_fail/partial，不自动重试、加 seed、改预算或访问 Test。

本阶段只更新 active log、第二项入口与生成目录。正式 B/R/A/M 结果矩阵 `MINIMAL_RANKING_SUPERVISION_RESULTS.md` 和 `MINIMAL_RANKING_SUPERVISION_HANDOFF.md` 尚不存在；实验族结果行、根阶段、论文正文/缺口仅在正式结果触发时更新。THESIS_ROADMAP、旧初始化审计和第一项矩阵不变。原始准备资产被忽略，须保存以供之后验证；无 tag/bundle/merge。

唯一下一步：由用户决定是否明确授权 seed2022 B/R/A/M 这一正式 cohort；授权后先完成干净启动声明提交，再按上面的精确命令启动一次。
