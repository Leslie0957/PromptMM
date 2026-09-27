# P0 精度复核：实现验收与用户手动运行声明

日期：2026-09-27。状态：**实现与合成验证完成；真实运行 pending，未执行。** 本声明落实 [精度复核方案](CROSS_INTERACTION_P0_PRECISION_REVIEW_2026-09-27.md)，仅用于测量，不进入P1。

## 唯一启动命令

在 `D:\Download\PromptMM` 的 PowerShell 只执行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_cross_interaction_precision.py
```

不要再执行旧 `run_cross_interaction_p0.py`。新程序打印开始和完成提示；逐对进度写入新目录的worker.log。

启动源：分支 `codex/experiment/baby-teacher-baseline`，包含本声明、实现、合成测试与准备outcome的同一提交，其父为方案提交 `9012a6d`；最终完整hash在交接及launch.json中记录。源码须已提交且tracked-clean，只有用户未跟踪 `check/` 被允许且不读取/提交。尚未推送不妨碍本机的已提交源门槛。

## 输入、比较与不变项

- run ID：`p0_precision_warm2022_v1`。共享锚仍为 [SPORTS_P0_WARM2022_EXISTING_STATE_V1](CROSS_INTERACTION_P0_ANCHOR.json)，不改任何资产身份。checkpoint/cache/manifest/train文件哈希运行前后核验，不重建教师缓存。
- Warm Full seed2022 selected epoch282，ID64，两张float32表；AdamW两表step60562，lr6e-5、weight_decay.01、betas(.9,.999)、eps1e-8，无投影/图传播/用户KD。image/text实际系数.3/1.3与.09/1.3；原1024×64归约及multiplicity保持。
- 直接复制原P0 `plan.json` 字节，不生成划分或新负例，不加载train矩阵。该plan SHA256=`11130dd5cc71ee2da45a485bb04c3946a6d5e9f310940019cc219fe1d1557393`；原report SHA256=`9eee8804963aa2db9691d53472d94350b8dafeb95d56000e68e80219d9fac02a`；原supervisor SHA256=`d35150d6308a6bfd4362ff67a18285ad0fbd0f867eef4d9d7ffd57968ed77e47`。三文件前后必须匹配，原目录不可写入。
- 同一batch1024，物品1494/1744/1646各图文一对，加1494-image零系数对，共7对/14个独立float32虚拟step；每对重新从同一原状态出发，不是连续14步。无U5、放大KD、增样、换物品或另一个seed。
- 原BPR/KD/backward/AdamW表达式未改。`paired_u1`仅增加默认关闭的更新后observer，原所有更新与float32读数完成后才调用。三个精度读数均来自**本次同一A/B状态**；旧运行没存A/B，故不能声称已逐位证明与历史A/B完整状态相等，而是检查旧float32读数完整复现。
- 环境保持指定run_5060、torch2.11.0+cu128、GPU0、deterministic、TF32关闭、seed2022；仍为新声明的诊断RNG，不是恢复历史训练RNG。
- Validation/Test读取与排名0、教师前向0。只使用已曝光训练内A/B/C，不把它们当新P1审计集；没有新增Validation/Test选择或测试指标。

## 测量实现与预先冻结的容限

新模块 `codes/cross_interaction_precision.py` 在CPU上读取少量原float32参数行：

1. 原batched GPU float32效用留在原结果结构中，同时报告其相对参考误差。另存CPU逐条float32读数，它与GPU批量读数有明确区分，不相互替代。
2. 参数在点积前转float64；同时计算普通loss差和局部参数差产生的margin delta。小delta（绝对值≤.5）用 `-log1p(sigmoid(-mA)*expm1(-delta))`，大delta用稳定softplus差。
3. 使用相同原float32行的精确数值输入，独立Decimal80位、120位标量点积/exp/ln做参考；不再次更新模型，不以torch同函数做参考。
4. 每三元组保存margin、delta、各读数、误差、参考字符串和容限；角色均值、方差、角色概率及加权池贡献分别保留。捕获参数行保存为每对单独文件，并记录SHA及张量哈希，可在后续审计中脱离GPU更新核对。

固定工程一致性规则（合成测试后、真实精度结果前冻结）：

|检查|容限|
|---|---|
|stable64相对Decimal120|`1e-15 + 1e-10*abs(reference)`|
|direct64相对Decimal120|`5e-13*max(1,abs(mA64),abs(mB64))`|
|Decimal80相对Decimal120|`1e-30 + 1e-25*abs(reference)`|
|零干预|参数差与精度角色均值严格0|
|旧float32输出|每对旧结果字段完全相等，包括差分、均值、方差、参数差范数等|

角色均值容限取逐样本stable容限平均；池贡献容限按固定角色概率加权和。只有超过相应容限才标记numeric resolved；**这是合成验证的工程规则，不是严格数学误差界、统计置信区间或实际效应意义阈值**。独立参考一致也不证明总体选择有效。原P0的保守门槛及原resolved=false不改。

## 准备阶段验证

20/20 CPU合成测试通过（原P0 10项＋精度10项）：非零AdamW矩下observer前后原结果相同、源模型/矩不变；正负角色、margin[-1000,1000]、极小可表示更新、舍入消减、大delta安全分支；零/空角色；参考不一致检测；非局部变化拒绝；捕获行往返与重复样本容限聚合；describe无加载、held-out拒绝、超时保留/再次启动拒绝。合成消减样例中原float32读数0而stable64与独立参考一致且非0，证明测试能识别这类精度损失；不外推真实效用。

额外核对：源语法、原P0三个输出哈希、共享锚路径、旧launcher/正式训练源码不变、原日志追加保留和本地链接。准备过程中没有加载真实模型、读split矩阵、计算真实梯度、使用GPU或创建真实新输出目录。

## 资源上限、输出与停止

- 7对上限；单worker，torch CPU线程2，GPU0；父进程600秒硬超时（包含导入/加载/Decimal计算与保存），worker570秒主动检查。超时杀掉子进程，保存supervisor失败，不重试。
- PyTorch CUDA allocator上限为2GiB与设备显存25%较小者，逐对检查峰值；不含驱动/上下文或整机内存。高精度仅处理CPU少量行，不将完整训练状态转double。首次实际耗时/峰值尚未测，不能承诺与原5.55秒相同。
- 独占新输出 `exp/cross_interaction/p0_precision_warm2022_v1/`：`launch.json`、原字节`plan.json`副本、`report.json`、`supervisor.json`、`worker.log`、`rows_00.pt`至`rows_06.pt`（小型被探测参数行，不是完整checkpoint）。目录已存在就拒绝，不能删除它来重跑。
- 哈希/参数/有限值/非局部/零对照/旧读数/高精度参考一致性任一失败即停止。参考不一致时已保存行和报告；中断/超时可能只有部分记录，以supervisor失败为准，保留而不自动修容限、换状态或重试。
- 成功验收：两报告completed、exit0、7对齐全、旧读数匹配、参考一致、输入及源码不变、资源和零split访问满足。效用未超出容限仍可completed/unresolved，不因信号小回滚或重训。

## 记录更新目标

- outcome追加 `docs/research/CROSS_INTERACTION_P0_AUDIT.md`，保留原P0与方案历史；`TRAINING_LOG.md`本pending后追加completed/failed/partial。
- `docs/experiments/README.md`更新P0精度复核状态；root/docs入口按结果更新。新输出后刷新六项生成导航。
- 正式矩阵无新增格子；原18格、论文正文及缺口表不变，除非后续经审计科学主张实际改变。路线仅按真实数值/资源边界补充，不将数值通过当P1通过。
- 原始资产全部原位；无标签、merge、bundle或备份里程碑。原始新输出被Git忽略，不会随源码推送自动备份。

**本次准备后唯一下一步：用户执行本页新命令一次，随后审计产物；无自动重试或下一阶段。**
