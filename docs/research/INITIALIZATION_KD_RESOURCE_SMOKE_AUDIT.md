# 初始化 × KD 资源 smoke 审计

2026-09-27：**completed，资源与功能检查通过；正式四臂仍不可启动。** 用户手动执行了一次[声明](INITIALIZATION_KD_RESOURCE_SMOKE_LAUNCH_2026-09-27.md)中的命令，agent仅审计保存产物，没有重跑或开启后续训练。

## 执行身份与协议

历史执行命令（已完成，不可重跑）：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_resource_smoke.py'`。

启动与结束源检查绑定 `42336f5355dd76f4f17f91009deb8a4477ab2e44`，分支 `codex/experiment/baby-teacher-baseline`；本次审计开始时 tracked tree 干净，仅原有未跟踪check/。manifest配置与当前冻结smoke配置逐字段一致，config digest `785fcb537a9d564ab9494eebe6be2a43523cb8484bc67d6dbe706ef9bb5a0168`。parent53556、worker39304的claim与manifest对应。worker通过源码中的输入哈希、环境和干净源门后才产生完成报告；本审计不重复读取旧资产或数据分割。

共同资产沿用[CPU预检锚](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)，base-config SHA `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d`。seed2022、T1、共同缓存教师ID两表、原Full图文目标、回放第1轮前8批×1024、ID64；AdamW全新状态，lr6e-5/decay.01/betas(.9,.999)/eps1e-8，Full系数`.3*(image+.3text)/1.3`，用户KD0，无额外显式L2。没有训练R0/R1/T0、没有四臂交互或正式结果。

环境与声明一致：Python3.10.20、torch2.11.0+cu128、numpy2.2.6、scipy1.15.3、psutil7.2.2；设备报告NVIDIA GeForce RTX 5060，cuda0/float32、TF32off、deterministic algorithms、4个torch CPU线程。epoch0明确跳过排名；8步后一次完整Val，35598用户，候选排除Train、K10/20/40/50。Test打开/排名和teacher forward为0，依据冻结执行路径、Test-open审计钩子和完成报告；没有另行操作系统级文件访问追踪，`test_file_reads`不是独立OS计数器。审计自身只读新产物，无Test/Train/Val分割读取。

## 验收与资源

| 项目 | 保存结果 | 声明限制 / 结论 |
|---|---:|---|
| 父进程总耗时 | 130.594秒 | 900秒内，exit0、supervisor completed、artifact acceptance passed |
| worker训练/评价段 | 123.266秒 | 完成8步+1次Val |
| 完整Val耗时 | 121.984秒 | 占主要耗时；不是8批训练耗时 |
| CUDA allocator峰值 | 163,577,856 bytes = 156MiB | 2GiB内；不代表驱动/桌面总显存 |
| report采样RSS峰值 | 2,328,125,440 bytes ≈ 2.168GiB | 4GiB内，见下方测量限制 |
| 最后telemetry RSS | 2,353,360,896 bytes ≈ 2.192GiB | checkpoint验收后略高于report数值，仍在4GiB内 |
| 最终文件总量 | 82,893,712 bytes ≈ 79.053MiB | 256MiB内 |
| 最后可用磁盘 | 411,721,310,208 bytes | 高于4GiB门槛 |

RSS限制由父进程约1秒轮询实施，但未保存完整采样序列；report的峰值在最后checkpoint验收/强制采样之前写出，因此不能把2.168GiB称为全程真实峰值。最后telemetry是2.192GiB，报告与此不同是记录时点差异；现有父进程成功记录支持没有触发采样超限，不能排除采样间瞬时峰值。最终文件总量为审计时10个文件之和；最后telemetry写入前的output_bytes稍小。上述限制不改变本次通过结论，也不保证长程4臂峰值/耗时。

agent在CPU上重新运行保存产物验证函数：完整文件集与acceptance SHA一致、report/curve/checkpoint源与config一致、两表shape/dtype/有限性、AdamW两组moments/参数和step=8通过。best/final都为epoch1，最早最优index0/固定末轮index0；两个checkpoint的模型及optimizer张量逐位相同，序列化文件哈希不同不表示状态不同。保存Val Recall@20=0.09418449519084773、NDCG@20=0.04333248926110577，均有限；只作功能检查，不与历史结果比较、不用于阈值选择。没有student Test，因此selected-versus-tested等式不适用。

## 保存指纹

原始目录保持原位：`exp/initialization_kd_interaction/sports_init_kd_resource_smoke_seed2022_v1/`。本次只提交审计文本及导航，不提交模型/日志。

| 文件 | SHA256 |
|---|---|
| launch_manifest.json | e5e16af0e6a4f0836d32dee62cc827323f4906d73b82cec6cb0a44dffee24aac |
| report.json | 87a99e8d984d43b13cacb4f25e0ffb79716c06e2bcca2a9cb58ad2c0cd8a5f34 |
| acceptance.json | 76eb4fc2a1037d1253a56c882418886c908f24c45000f62672ff96f0979065a9 |
| supervisor.json | 3c81f75246ea8324ef1cfd8232cdcb5f9cbec0f3e87ba1276d30a16ca5e404a3 |
| telemetry.json | f901d4a7f93d72c5fb33d73a94c7b93c10b33094c598a37a0610dbe1ae7acd75 |
| worker_claim.json | bf962e39c021ba4652acf9bfc3b0265aba2a9d162abca644bd61eace903729b7 |
| worker.log | 6d89a8ee19ffd5bf7a3fdf277ad87595f5f423b0d22e3ad6950dcd2d33a09f47 |
| T1/curve.json | dd5de9ab7a02db967c24777d8737948f43322ed4f18017de15f0a6383eb62b64 |
| T1/best.pt | 2f818e3e53e445f65e7f6314ac6d3bb7eb370ce424c20a9f432eace6559aa447 |
| T1/final.pt | ec18349231ded3a4b1dcd400bff31b80a2e23b1c819519ad6c6fd0eb64ddbcc7 |

## 实验意义和下一步

真实GPU上的短程更新、一次完整Val和保存/验收链条通过，当前资产与实现可以完成所声明的smoke。**这并未验证初始化×KD假说，更没有证明KD增益或四臂完成。** 8步状态不得续作正式初值。

时间预算有实质缺口：按本次单次Val速度，原计划4×301=1204次Val约为 `121.984×1204/3600=40.80小时`，尚未包含256,800批训练及保存。此为单次测量的条件性粗估，不是统计下界或保证时长；但现有证据不能支持候选24小时总预算。不得为赶预算静默减少Val频率、用户、候选集或训练轮次，也不得按新曲线反选实际意义门槛。

**唯一下一步：保持研究/评价协议不变，准备评价器提速与等价性验证方案，先做合成正确性检查并重新核算预算；真实性能复核须另行声明。** 暂不提供正式四臂命令；不重试本smoke、不扩大预算或进入P1b/P2/门控。

## 记录路由

本审计、TRAINING_LOG追加结果与短当前交接、根/docs/实验族入口已更新；生成导航刷新。旧声明/历史审计不重写；四臂新矩阵和旧18格不适用，论文正文/缺口不变（资源检查无科学主张变化）。check/与所有历史资产、当前原始产物原位保留；无tag、bundle、推送或合并。Git提交保护本次文本源，不备份忽略产物。
