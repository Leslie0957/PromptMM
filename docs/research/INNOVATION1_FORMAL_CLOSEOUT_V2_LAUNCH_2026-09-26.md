# 第一创新点固定配置 v2 恢复批次：待授权启动页

> **本命令已执行并失败，禁止重跑。** Baby三次训练和九项预检有效保留，零Test；当前状态以[v2失败审计](INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md)为准。以下为原准备声明的历史内容。


本页对应[当前日志中的 v2 待执行声明](../../TRAINING_LOG.md)，承接[v1 失败审计](INNOVATION1_FORMAL_CLOSEOUT_COHORT_V1_AUDIT.md)。截至本页提交时，v2 **仅准备完成，未执行**；用户需要单独明确授权实际运行。v1 旧命令不可重试。

## 唯一拟执行命令

在干净且已提交的当前任务分支上，由用户在 PowerShell 手动执行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_formal_closeout.py' --conditions-confirmed --cohort-id innovation1_fixed_v2
```

这不是 v1 续跑：它从头执行原固定配置下未实际发生的 3 次 Baby 发布版训练、12 次只读 Train/Validation 预检和 12 次一次性学生 Test，严格按此顺序串行。所有预检通过后才进入 Test；任一步失败立即停止，不自动重试、跳过或继续下一阶段。B1/B2 既有 Baby Test 只复用，不重复访问。Sports S1/S2/S3 按[固定来源清单](INNOVATION1_REUSE_ASSETS_2026-09-26.json)复用所选检查点，不重训。

## 独立产物和验收

- 总账及逐步日志：`exp/formal_closeout_cohort/innovation1_fixed_v2/`。
- 三次 Baby 训练：`exp/promptmm_release_baby/innovation1_fixed_v2/baby_promptmm_release_cap1000_patience7_seed{2022,2023,2024}_lr6e5_v1/`。每项必须有 `validation_completed` 报告与匹配 SHA256 的 `best.pt`，Test 为零。
- 12 个无 Test 预检：`exp/formal_closeout_preflight/innovation1_fixed_v2/<slot>/report.json`。每项必须 `passed`，按既定 `abs(Validation Recall@20 delta)<=1e-10` 重放所选检查点，Test 访问和评价均为零。
- 12 个最终评价：`exp/formal_closeout_eval/innovation1_fixed_v2/<slot>/report.json`。每项最多一次学生 Test 尝试、零教师 Test；报告须为 `completed`、指标有限、所选与被测检查点 SHA256 相同。Test 第一次访问前持久记录标记；访问失败也不得重试。

`<slot>` 固定为 B3/S1/S2/S3 各 2022/2023/2024，文件名使用小写 `b3_2022` 等。v2 的四类目录在准备时均不存在；旧 v1 总账和失败预检及其哈希保留原位。只看进程退出码不能判定训练或论文结果完成。跑完或报错后把 v2 `batch.json` 路径与终端末尾信息交给 agent，按[固定更新规范](../experiments/RUN_CLOSEOUT.md)审计。正式验收及比较局限以[原协议](INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md)和当前日志声明为准。

无需运行数据的命令清单可用 `--describe --cohort-id innovation1_fixed_v2` 查看；它不是正式执行，也不访问 Test。
