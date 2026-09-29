# Ranking-gap v1 执行交接，供独立审计

2026-09-29。执行方已完成用户委托的一次 Sports Validation 诊断并自行核对，**尚未进行独立审计**。结果矩阵与解释在 [RESULTS](RANKING_GAP_DIAGNOSTIC_RESULTS.md)，原始产物目录 `exp/innovation2/ranking_gap_v1/`。该目录被 Git 忽略，不受源码提交保护；请保留原样审计。此文档所在的结果提交由交接方最终报告精确 SHA，启动源码提交为 `4012c5a`，分支 `codex/experiment/baby-teacher-baseline`。

## 启动身份与边界

- 唯一启动命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_ranking_gap_diagnostic.py'`，从干净跟踪源码 `4012c5a` 启动；原有未跟踪 `archive/reviews/` 未触碰。无第二次启动、恢复、追加 seed、参数更新或 Test 访问。
- 固定 profile：`docs/research/innovation2/RANKING_GAP_PROFILE_V1.json`，SHA256 `f2a3b633224b567045b209fa3fabcd4fc6c01605b127ab336a2f4b75e714a509`；完整七状态、六 checkpoint SHA、共享融合表、Train/Val 锚和门槛均在其中。`manifest.json` 记录实际 source commit、Python 3.10.20、torch 2.11.0+cu128、NumPy 2.2.6、SciPy 1.15.3、psutil 7.2.2、RTX 5060 及每个 checkpoint 原 source/config。
- 实际状态：`exit.json` completed/0；`acceptance.json` passed，七状态/六比较完整；`report.json` screen=`opportunity_present`。七份逐用户 npz 各含 user ID、top20 ID、20 位命中布尔值和 Validation 分母；`report.json` 含完整精度、三等分贡献和 2,000 次簇 bootstrap 区间。
- 数值门：教师/六学生 Recall@20 对原值误差均小于 1e-6；全部 G-L 与 Recall 差一致（1e-10 内），物品组贡献和一致。仅 2023 R1 有一行 top20 边界同分；固定按较小 item ID。有效分数、top20 合法性、Train 过滤、数量和无更新检查均通过。程序 Test 读取计数 0，且 Test 文件拒读钩子安装于 Train/Val 加载前；没有独立 OS 文件访问轨迹。
- 资源：58.187 秒，峰值 CUDA allocator 75,497,472 B，采样 RSS 1,004,249,088 B，输出约 9.1 MB，最低盘余量约 402.2 GB；1 秒采样，无 OS 硬隔离。原始 stdout/stderr、每次采样和退出状态均保留。

## 原始文件 SHA256

下表为运行结束后只读哈希，路径均相对 `exp/innovation2/ranking_gap_v1/`。空 stderr 的 SHA 是标准空文件摘要。

| 文件 | SHA256 |
|---|---|
| `manifest.json` | `0eb92ea5e5013bf3e667f1802eee147614314541be5cd25e92ff4449067d9015` |
| `report.json` | `28f76be91d9a4c2af7addd68809819773b6d3580aee1512c670a48076bb4f6be` |
| `acceptance.json` | `2be95c42b06724a5e9f75a7fabeac43b6e3d9802f47ddebc183fce322b88252a` |
| `resource.json` | `5ea89620c501d401fa60978d9ead800b0cb281f3a59fa6629708d4a2a296f132` |
| `exit.json` | `04dde27e66640904a81c1c0e215af5b99ddb048ea1ef50e5b7d95880bc4fc494` |
| `stdout.log` | `22f650ed37d4bfc8c3b4826fe81043026a6f6e2284767bd2db71d5e9870503bb` |
| `stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `teacher_per_user.npz` | `c47902fe60dd65918ce31c663073520084d2517b6ed5c22475638483435ceddc` |
| `2022_R1_per_user.npz` | `3e3b8ff86ab50b658915906699ba16a81a9a2b52a3b108d3d88f998a2e1addde` |
| `2022_T0_per_user.npz` | `fff7bfbf0807f6a0a6a09f9806c7b475c0c3f61aa4a94b6666e46a67f71b30f0` |
| `2023_R1_per_user.npz` | `0085e6e296c29581d0933866d7f7fc5311263d7b6a8d8b2d1cd1d4ff6828d9ea` |
| `2023_T0_per_user.npz` | `08c70cf831ffd4e02b507ba870120aa64248c28775198f22b46be8c4e7c0f327` |
| `2024_R1_per_user.npz` | `163657a043f43571f6cbb9d0e20803ff2114df9e48f0fa02d0f3a9e178ff9145` |
| `2024_T0_per_user.npz` | `7840d75942224cd893c79059ac76af62c72db4acb905c43f763af18de6691c6b` |

## 待审计点与记录路由

请独立核对七份逐用户文件的用户顺序、Train 过滤、命中布尔值/分母及 G/L 恒等式；复核原 manifest/checkpoint/共享表 SHA 与原评价协议的对应关系、边界同分和资源/Test 防护证据。只读审计不应重新运行排名，也不应把执行方程序计数称为 OS 证明。描述性机会不等于可实现收益；三个 T0 的 N 全为负，教师整体 Recall 不高于 T0。

本次更新 `TRAINING_LOG.md`、RESULTS、本 HANDOFF、第二项入口、实验族总览和根状态，刷新六个生成导航。无活跃第一项矩阵格子适用；第一项正文/缺口/旧初始化审计及 THESIS_ROADMAP 未变，因本次不产生新训练结论。无 Test 指标、无 selected-versus-tested checkpoint：本诊断只有既有 epoch300 final 与 Validation；原 Test 主表不可填。无标签、bundle、合并或原产物覆盖。

唯一下一步：由用户将此 HANDOFF 交回独立审计；执行方在此停止。
