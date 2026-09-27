# P0 用户手动运行声明：p0_warm2022_v1

状态：pending，准备完成，**尚未运行**。这是一轮非正式测量/资源检查，非正式训练，不是 P1 信号筛查。依据 [资产审计](CROSS_INTERACTION_P0_AUDIT.md) 和 [固定锚](CROSS_INTERACTION_P0_ANCHOR.json)。

## 唯一命令与来源

在 `D:\Download\PromptMM` 的 PowerShell 执行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_cross_interaction_p0.py
```

分支 `codex/experiment/baby-teacher-baseline`；启动源必须是包含本声明、锚、实现、测试和准备结果的同一完整本地提交，或后续经独立声明验证的等价来源。该准备提交以整理提交 `6e8faa0` 为父提交，最终 hash 见任务交接；launch.json 自动记录实际完整 HEAD。启动前 tracked source 必须干净；唯一允许的未跟踪文件是用户原始 check/，不被读取为配置、不提交。任何其他改动阻止启动。

## 固定对象、比较与 delta

- 命名锚 `SPORTS_P0_WARM2022_EXISTING_STATE_V1`：完整路径与 SHA 在 JSON。复用 Sports 既有 warm Full selected epoch282、固定教师缓存和原 train_mat；以暖启动 [原配对审计](SPORTS_SHAREDINIT_PAIR_AUDIT_2026-09-23.md) 为来源锚，不重新选模型。
- 与原配方相比仅诊断数据与执行方式改变：四组 train 内正边；一个固定 update batch；每对独立恢复完整状态；base=BPR 对比 base+物品模态 KD。ID64、无投影、原 AdamW 与图文系数保持。无 teacher forward，无新 checkpoint 写入，无原正式训练配方改动。
- seed2022，确定性 PyTorch，禁TF32；环境 torch2.11.0+cu128，指定 run_5060 Python，GPU0。原 RNG 未保存，当前 RNG 是新声明，非历史轨迹复现；无调度器状态需要补造。
- 焦点物品固定1494/1744/1646，各 image/text 一对，共6对，加1494-image 系数0一对。总7对，每对两次虚拟更新，14个独立分支step；不存在连训7步。每一对都从同一原状态重新恢复。只U1，无U5、Full删除背景、oracle筛选或门控。
- A/B/C 每角色最多8条，独立负采样种子，正边分离；只作测量可用性记录。低频/缺失按审计约定 N/A；不为提升测量结果更换物品、扩样或调系数。
- Validation/Test split 读取及排名均0，教师前向0；历史checkpoint选择指标不再评价。选中与被测checkpoint的正式Test相等性不适用。

## 资源上限、输出和停止

- 单进程 worker、CPU torch线程2、GPU0；父进程对子进程施加600秒超时（包含Python导入、身份核查、读取、更新与保存），到限杀掉 worker，不重试；worker570秒主动停止为收尾留余量。
- 最多7对（低于路线12对上限），batch1024，最多3×3×2×8条角色探测抽样；CUDA allocator 上限取2GiB和显存25%中较小者，逐对检查峰值。该限制针对PyTorch allocator，不包含CUDA驱动/上下文；不是整机内存或总显存占用保证。OOM或显存风险即失败停止；不缩batch重试。
- 独占新目录 `exp/cross_interaction/p0_warm2022_v1/`，已存在即拒绝（包括上次失败）。输出 `launch.json`、`plan.json`（划分覆盖/焦点/完整三元组/角色概率）、`report.json`（零对照、逐角色U1/方差/数值分辨率、参数差、时间/显存/身份）、`worker.log`、`supervisor.json`。不保存训练模型，不覆盖旧产物。
- 身份/参数/形状/step不符、非有限值、非局部差、零干预不一致、source变化、时间/显存上限或任何异常均停止，保留已有产物。supervisor失败优先于不完整的worker报告；退出不代表全部测量已完成。

## 验收与解释

硬验收：两报告完成、7对齐全；固定资产哈希/launch HEAD可核对；step60562及模型/矩均有限；原资产不变；held-out与teacher forward为0；零干预参数差与U1严格0；正边隔离、完整 batch multiplicity、角色概率与coverage吻合；在预算内完成。

效用正负或接近0都不是硬失败。差异不超过逐角色 `8*float32_eps*max(1,abs(loss_A),abs(loss_B))` 的保守数值尺度，标记 unresolved；这不是已测 GPU 噪声或P1科学意义阈值。有限样本方差与重复边必须一起解释。P0完成也不自动通过G1/G2，不宣称新算法、性能收益或负迁移。

停止后唯一后续动作：审计这一次已有产物，追加结果记录；需要新执行必须另行声明授权。无自动重试/恢复/下一seed/下一阶段。

## 记录更新目标

- run：p0_warm2022_v1，仅上述warm seed2022。
- outcome：`docs/research/CROSS_INTERACTION_P0_AUDIT.md` 追加实际执行节，保留准备审计。
- `TRAINING_LOG.md`：保留本次pending，追加completed/failed/partial及新交接。
- `docs/experiments/README.md`：P0准备行更新为实际测量状态。
- 活跃结果矩阵：无；不得填入旧18格或改变第一创新点资格。
- 条件消费者：跨交互路线第5–6节仅在资源/测量边界确实改变时补充；论文/缺口表无新主张则不变。
- 新输出后刷新六项生成导航；原资产保留，不属于正式冻结/标签/备份里程碑。
