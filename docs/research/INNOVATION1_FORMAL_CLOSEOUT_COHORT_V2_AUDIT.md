# Innovation1 v2：批次失败，三次 Baby 训练有效保留

## 状态、来源与命令

用户手动执行一次，启动源码 `da2371ebf6de2a5a0daf06444c7f1fd32837ce37`，分支 `codex/experiment/baby-teacher-baseline`；源代码清洁并在各已完成步骤保持不变。时间 2026-09-26 14:55:41–15:24:11（Asia/Shanghai）。原声明曾标为待授权；用户实际手动执行并返回进度及最终终端输出，构成本次该确切命令的执行确认；保留旧声明原文。没有 agent 正式执行。

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_formal_closeout.py' --conditions-confirmed --cohort-id innovation1_fixed_v2
```

参数及比较边界沿用[原协议](INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md)和当前日志 v2 声明：Baby release 共享教师、seeds2022/2023/2024、lr6e-5、cap1000/patience7、每轮 Validation Recall@20 严格提高选模、相等保留最早；Sports 复用固定九项来源。资产锚 `INNOVATION1_REUSE_ASSETS_2026-09-26.json` SHA256 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c` 未修改。Python3.10.20/Torch2.11.0+cu128/CUDA12.8/RTX5060。

## 已完成 Baby 训练（不是最终 Test）

| seed | 实际轮数 | 最佳轮数（1起算） | Val Recall@20 | Val NDCG@20 | optimizer steps |
|---|---|---|---|---|---|
| 2022 | 9 | 2 | 0.065426293943 | 0.028433000406 | 1044 |
| 2023 | 9 | 2 | 0.065529148147 | 0.028422812248 | 1044 |
| 2024 | 10 | 3 | 0.065297726187 | 0.028351932710 | 1160 |

三项均 `validation_completed`，停止原因 `patience7`，每轮116批；初始 Validation 不参与选模。已核对训练源码哈希、最佳文件 SHA256、恢复回环、学生更新、教师/prompt不变、完整步骤数和验证次数；JSON全部数值有限（含逐轮损失/指标）。三项 Test 次数0且未加载 Test。较晚轮数指标回落不使训练失败，也不构成重训理由。完整曲线和损失仍在原报告。

## 预检及 Test 顺序

| slot | 预检状态 | Recall@20 重放绝对误差 |
|---|---|---|
| b3_2022 | passed | 0.0 |
| b3_2023 | passed | 0.0 |
| b3_2024 | passed | 0.0 |
| s1_2022 | passed | 0.0 |
| s1_2023 | passed | 0.0 |
| s1_2024 | passed | 0.0 |
| s2_2022 | passed | 0.0 |
| s2_2023 | passed | 0.0 |
| s2_2024 | passed | 0.0 |
| s3_2022 | failed | N/A |

`s3_2023`、`s3_2024`未启动。第13步 `preflight_s3_2022` 在读取身份元数据时报 `KeyError('data_identity')`，未进入该项真实Train/Validation读取与排名。其余九项预检均通过、报告和所选checkpoint哈希复核相符。总批次为 **failed / 部分完成**；任何新的最终 Test 均未启动，12项学生Test尝试/评价为0，教师Test为0，所有预检 Test访问/加载标记为false；最终评价目录不存在。所选与被测checkpoint相等和Test指标不适用，不能把预检Val写入正式Test表。

## 根因、修复与恢复边界

固定来源清单中，Sports BPR/Full使用嵌套 `data_identity`，Sports release使用 `input_sha256` 和teacher转换身份；评价入口误把三法当成相同JSON格式。此前测试没有覆盖真实发布版来源的身份解析路径。这是实现缺陷，与训练质量无关。

修复新增元数据适配：要求所有现存Sports BPR/Full身份锚一致，将release的train/val/features/teacher及conversion身份逐项匹配后才取得完整划分身份；也核对实际发布版报告的输入身份。缺失、冲突或不匹配均拒绝，不修改原清单，不读取Test文件来补身份。14项合成/无Test测试通过，新增真实九项Sports清单/三项release报告的纯元数据检查及冲突拒绝测试。未重放真实Validation、未训练、未访问Test。

**唯一建议下一步：另行准备评价恢复批次，固定复用本次三个Baby最佳检查点，修复源码下在新路径做12项无Test预检，再按既定顺序做12次最终学生Test；不重训Baby。** 旧九项预检保留作证据，但旧提交的通过报告不能冒充修复源码的预检。此处未实施新启动入口、未声明或授权恢复执行；不得重跑原v2命令，后续需要明确授权。

## 记录更新与保留

本审计、TRAINING_LOG、矩阵B3缺口（已训练、缺Test）、实验族及root/docs导航更新；论文缺口表更新训练工作量，正文结论不变（尚无最终Test比较）。生成目录刷新。v1/v2总账、训练报告/检查点及预检原位保留；未删除、覆盖、改写、重试。普通Git提交不包含这些忽略产物，不宣称已做物理备份或结果冻结。

## 原始产物 SHA256

- `exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed2022_lr6e5_v1/report.json`：`53ea4868f6a6109f31a86a590e9d0adb506a4c515085d7067b20868771a73384`
- `exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed2022_lr6e5_v1/best.pt`：`de1cbc51d5150d4b6d42a9a747d9fe14afd58998379eb409a3d030c2934bf3ac`
- `exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed2023_lr6e5_v1/report.json`：`d5edac3a1719c874872735f55ffea87e6eb46831cf054308b77962a350e7189d`
- `exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed2023_lr6e5_v1/best.pt`：`2526c852b2bd3459e11b71a3542dc6407342c8b8c913f4e81a74bb3593323c9f`
- `exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed2024_lr6e5_v1/report.json`：`e4140d87c587cfa47801de7c54e6d7d8bd1f9e9f94f5ec858602d9abbebb266e`
- `exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed2024_lr6e5_v1/best.pt`：`27f8546df27f2a9de4781398595782de7d55a94edefde6571cfb9ecb83b6548d`
- `exp/formal_closeout_cohort/innovation1_fixed_v2/batch.json`：`e7febbd5ba054996a53c1dcce96eac5b6f1ab97238bfdee5a61faf5dc167b82f`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/b3_2022/report.json`：`ca44fd95e69137af74aa82a71fe0943600d89e88de2b236b4c7fa3bd1eddd58a`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/b3_2023/report.json`：`85889a9172d966027b11094a50c58413b38aca1256d5bcd96bb447a8b73c70e7`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/b3_2024/report.json`：`50d831170075294b4ff0d4be477a8760551a72abbddc70004ef5e7eddaa830cf`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s1_2022/report.json`：`63596e2d7c6a83085eecee48b6eff589c88d9353478c47efd61b9da49dbab792`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s1_2023/report.json`：`3d201c145a61fe3fe6c351bea505ca3b0861dc4ef49dfc070ee46723a0b3f42d`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s1_2024/report.json`：`7bc844f97beed3c9b6a5cce7489c775aa1f7a14a9da50d966a3faaaa07d49a8a`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s2_2022/report.json`：`642a6f460765ff552140d6c4b8acccf627bdaf4b7ad170dc548c85eef2e23987`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s2_2023/report.json`：`36de54fd6c359b316fd19d9561e616ad5ab591df431ba030fa2673807b8a08df`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s2_2024/report.json`：`0862a4a124f62208a9e13026234e35efa9d687ff2494bea5bcd9c4e3a7ab5634`
- `exp/formal_closeout_preflight/innovation1_fixed_v2/s3_2022/report.json`：`5589100c9de6593adcb494a199e18116f9d3343b4af74776c38d18962084df40`
