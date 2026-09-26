# 第一创新点固定配置串行收尾：用户启动页

本页是[当前日志的正式批次声明](../../TRAINING_LOG.md)的便捷入口；完整身份、预算、Test 权限、资产锚和记录目标以该 pending 声明为准。[收尾协议](INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md)规定解释边界，[来源清单](INNOVATION1_REUSE_ASSETS_2026-09-26.json)固定既有检查点。实现源码提交 `67fc82e`；实际启动以包含声明的干净 HEAD 为准。用户执行，本任务没有替用户运行训练或 Test。

## 一条启动命令（PowerShell，任意当前目录）

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_formal_closeout.py' --conditions-confirmed
```

脚本先确认 Git 干净、批次和每格输出目录未被使用；然后严格串行：Baby 发布版 2022/2023/2024 三次训练 → B3/S1/S2/S3 各三种子的 12 次只读 Train/Validation 预检 → 同一 12 格各一次最终学生 Test。所有 12 项预检通过后才会开始第一项 Test。任一步失败立即停止，不跳过、不重试、不覆盖；指标低但通过硬验收仍保留。命令只能启动一次，不能在失败后原样重跑。

每一步同时显示在前台并写入 `exp/formal_closeout_cohort/innovation1_fixed_v1/<step>.log`。总账为 `exp/formal_closeout_cohort/innovation1_fixed_v1/batch.json`；Baby 训练报告和最佳文件在 `exp/promptmm_release_baby/`，无 Test 预检在 `exp/formal_closeout_preflight/`，最终评价在 `exp/formal_closeout_eval/`。这些 `exp/` 产物不在普通 Git 提交中。结束或报错后把总账路径和终端末尾信息交给 agent 审计，再按[固定更新规范](../experiments/RUN_CLOSEOUT.md)更新日志、族导航与[正式矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)；不要根据脚本退出码自行将格子标为论文正式完成。

无需计算的只读命令清单可用 `--describe` 查看。它不是一次预检，也不会启动训练或评价。真实来源的 Validation 重放安排在上方主命令内部，尚未被这次合成工程测试替代；如果与旧记录不一致，脚本会在 Test 前停下并保留报告。
