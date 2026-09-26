# 第一创新点：仅评价恢复批次（待另行授权）

本页是 `innovation1_eval_recovery_v1` 的启动声明入口，不是已完成结果。v2 批次在 Sports 发布版身份解析处停止；三次 Baby 训练及最佳检查点有效，九项旧预检通过，学生与教师 Test 均为零。来源与原始指纹见 [v2 审计](INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md)。本恢复批次不训练、不重新选源、不调整参数，也不写入 v1/v2 产物。

## 固定来源与执行边界

- 共享数据、教师、划分及九项 Sports 选择锚：`INNOVATION1_REUSE_ASSETS_2026-09-26.json`，SHA256 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`；Baby 教师 `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`，Sports 教师 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。
- B3 seeds 2022/2023/2024 读取 `exp/promptmm_release_baby/innovation1_fixed_v2/` 下原 `report.json` 和 `best.pt`。这三份训练的来源提交为 `da2371ebf6de2a5a0daf06444c7f1fd32837ce37`；原报告与检查点 SHA256 由脚本逐一对照 v2 审计。B1/B2 历史 Baby Test 不重评。
- S1/S2/S3 各三 seed 使用共享锚固定的 Sports 所选检查点。B3 训练配置沿用 v2 的 lr6e-5、cap1000、patience7 等完整固定配置；本批次只评价，不产生新的训练配置选择。比较边界见 [正式矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)。
- 执行环境为 `D:\miniconda\envs\run_5060\python.exe`，CUDA 设备 0。用户须从本声明所在的**干净、已提交 HEAD** 手动启动一次。启动前脚本校验源指纹、v2 零 Test、恢复目录未被占用；运行中检验 HEAD 不变。

## 待授权的一次性命令

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_eval_recovery.py' --conditions-confirmed
```

固定顺序：B3、S1、S2、S3 各 seeds 2022/2023/2024 的 **12 次全新无 Test Validation 预检**，随后相同顺序的 **12 次最终学生 Test**。预检必须逐项达到 `passed`、同一启动提交、相同账本哈希、Validation Recall@20 重放绝对误差不超过 `1e-10`；12 项全部通过之前，评价器禁止首次 Test。每项最终评价在加载 Test 前持久标记访问开始，最多一次学生 Test，教师 Test 为零；选中与被测检查点 SHA256 必须相同，目标和指标有限。首次失败即停，不自动重试、续跑、跳项或启动下一阶段。低指标只要硬条件通过仍属有效结果。

独立输出：`exp/formal_closeout_cohort/innovation1_eval_recovery_v1/batch.json` 与步骤日志，`exp/formal_closeout_preflight/innovation1_eval_recovery_v1/<slot>/report.json`，`exp/formal_closeout_eval/innovation1_eval_recovery_v1/<slot>/report.json`。原 v1/v2 报告和检查点保留原位。脚本 `--describe` 仅列出计划，不作真实评价。

## 结束后固定路由

无论完成或失败，先核对总账及逐项原报告，再按 [RUN_CLOSEOUT](../experiments/RUN_CLOSEOUT.md) 更新：`TRAINING_LOG.md` 的独立结果条目；新建 `docs/research/INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md` 记录全部 12 项的来源、Validation/Test、访问顺序、选择与被测哈希及限制；更新 `docs/experiments/README.md` 发布版实验族一行及正式矩阵 B3/S1/S2/S3 各三格。论文缺口表和正文仅在覆盖或结论确实变化时更新；未触发的消费者写明原因。人工编辑后刷新六项生成导航。忽略的大资产仍原位保存，稳定里程碑是否额外备份另议。

本页准备阶段未运行真实 Validation 或 Test。唯一下一步是用户明确授权并手动执行上面这**一条**命令，然后返回总账供审计。
