# 自身锚点干预：启动准备

2026-09-30。按[固定方案](OWN_ANCHOR_CONTROL_PLAN_2026-09-30.md)完成实现与合成验证，**未运行真实数据评价、训练或Test**。这是准备交接，不是结果报告或启动授权。

## 固定范围与命令

仅Sports既有seed2022/2023/2024的R第300轮检查点；每seed原R和共同中心替换C，共6个评分状态。零参数更新，Train/Validation only，Test0。不重新训练，不选择最佳轮，不加入新seed或插值。

用户授权或手动执行时，从仓库根目录运行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_own_anchor_control.py' --formal
```

分支`codex/experiment/baby-teacher-baseline`，启动源码为当时干净的committed HEAD（只有原有未跟踪archive/reviews/例外），manifest记录实际提交及profile SHA。未来AI代运行须先追加该一次执行的pending并提交；用户手动启动本命令则表示执行这一固定范围，结果审计须补录实际执行时间及来源，不能把准备误记为已运行。

输出固定为`exp/innovation2/own_anchor_intervention_v1/`，目录存在即拒绝启动，不重试/续跑/覆盖。失败保留全部文件，恢复需新授权。原始12臂目录保持只读。

## 身份、实现和协议

[profile](OWN_ANCHOR_CONTROL_PROFILE_V1.json)固定原manifest/completeness的SHA；teacher/Train/Validation/两图引用已审计共享anchor，12份R final/status/Top20及B status的SHA取自原completeness。准备只读取JSON元数据并核对路径存在，没有加载checkpoint/图张量或交互矩阵；正式启动再校验实际资产SHA。

工具`tools/run_own_anchor_control.py`复用原`rank`和exact_topk、group_metrics，保持CUDA FP32、TF32关闭、确定性算法、4CPU线程、batch256。R特征必须逐字节回归原manifest SHA，R评分必须逐位回归原Top20及Recall误差≤1e-6。C中心采用原FP32教师表计算的邻居中心在固定目标集合上的FP32均值，保持原RMS；不重新缩放。

C由原W参数化计算，并与float64代数A/b表达交叉验证（rtol/atol=1e-5）；非目标行与用户/参数不变。调用torch.inference_mode，不创建优化器、不调用反向传播。模型权重在评分前后逐位比较。只选final，第300轮和原seed/arm/source字段必须一致。

逐seed保存R/C分组与总体指标、Top20、逐用户gain/loss/overlap/正例数、逐物品gain/loss；验证净命中与分组差、逐用户Recall差恒等。80%低组收益保留和Recall差≥−0.0002仅作描述筛查。指标低不判执行失败，不恢复原N路线，不宣称等价或算法成立。

## 验证与资源限制

5项合成测试通过：代数展开/掩码/尺度不变、共同中心退化一致性、配对命中与Recall恒等、并列排序/排除/非有限分数拒绝、数据路径白名单拒绝。初次测试因预期数组未转存储FP32失败，修正预期dtype后通过；未改实验计算或容差。语法、CLI help、全部绑定路径存在及输出目录不存在检查通过。

没有真实数据smoke，因此**未证明完整六状态能在1800秒内完成，也未证明实际R Top20必能逐位回归**。这些是运行硬验收，失败即保留并停止，不自动放宽。

外部父进程每秒采样worker RSS/墙钟/输出/剩余磁盘，超限terminate；worker每个评分批次核查时间、RSS和CUDA allocator，后者另设2GiB进程内分配限制。上限1800秒、RSS4GiB、输出512MiB、剩余盘2GiB。采样有间隔、不是OS作业对象硬隔离，瞬时峰值仍可能漏测；RSS上限针对worker，父进程开销未计。CUDA allocator不等于整卡显存。

data目录open白名单只允许Sports train_mat/val_mat；计数区分允许读取和拒绝尝试。合成拒绝测试传入虚构路径，不实际打开Test。Python审计hook不是OS级文件隔离，Test0结论需结合实际代码路径、计数与manifest审计，不能只凭常量字段。

## 运行后记录路由

- 原始报告：输出目录manifest/intervention/report/acceptance/supervisor/worker_resources/completeness、console和六份Top20、三份配对明细。
- 按RUN_CLOSEOUT核对外部退出、身份、资源、六状态、原R回归、数值、Test0；写本目录`OWN_ANCHOR_CONTROL_AUDIT.md`、`OWN_ANCHOR_CONTROL_RESULTS.md`（三seed×R/C）、`OWN_ANCHOR_CONTROL_HANDOFF.md`。
- TRAINING_LOG追加outcome，实验总览增加自身锚点干预行，根/第二项入口更新，刷新生成目录。第一项矩阵/论文、旧12臂矩阵、总体路线、历史审计均不改。
- 本轮无新结果矩阵；普通准备提交不打tag、不创建里程碑备份。Git不保存原始忽略资产。

**唯一下一步：用户运行上述固定命令一次，完成后提供输出目录进行独立审计。** 若委托AI执行，只授权这一次六状态诊断及收尾，不授权训练或后续阶段。
