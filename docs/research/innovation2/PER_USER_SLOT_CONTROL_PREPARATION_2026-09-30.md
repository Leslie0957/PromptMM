# 逐用户槽位对照启动准备

2026-09-30。按[固定方案](PER_USER_SLOT_CONTROL_PLAN_2026-09-30.md)实现并通过合成验证，尚未评价真实数据、训练或访问Test。

用户决定执行时，在仓库根目录运行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_per_user_slot_control.py' --formal
```

固定seed2022/2023/2024，三个新H列表、B/Q/R九个保存参照；共12个指标状态，不是12次模型运行。H按R每用户S位置模式填入B两组候选前缀。CPU，无模型加载/GPU/训练/全候选打分/β搜索。完成本次后统一研究判断，不默认继续诊断或恢复旧算法。

## 来源与执行边界

[profile](PER_USER_SLOT_CONTROL_PROFILE_V1.json)固定上一cohort的manifest/report/completeness、三candidate和三lists SHA，及Train/Validation/两图身份。曝光诊断源码为6dbdc400cb029340d2c40bdc65e8f4a442941108，原B/R训练源码b8be549613498e5cbb98412ffa766c62279ad7a6；本次工具启动源码为干净committed HEAD，在manifest记录。

分支codex/experiment/baby-teacher-baseline，仅原未跟踪archive/reviews/可保留。输出`exp/innovation2/per_user_slot_control_v1/`，存在即拒绝，不覆盖、重试、续跑。工具默认不执行，需明确--formal。AI代执行前需追加这一次pending声明并提交；用户手动启动仅表示运行固定诊断，不扩展到后续实验。

原始资产只读；准备阶段只读JSON元数据及检查路径存在，没有载入真实NPZ、交互矩阵或checkpoint。正式运行核验SHA，先构造三个H并写construction_seal，然后开放Train/Val。data目录在封存前全部拒绝，封存后只允许train_mat/val_mat；Test始终拒绝。Python hook不是OS级隔离，审计须结合源码和读取记录。

## 验证与验收

6项合成测试通过：零/满/混合槽位、换物品但保留位置模式、padding/错序/并列合法性、封存前Train/Val及始终Test拒绝、S/nonS/低组非S指标、决策边界。语法/CLI、全部绑定路径存在、输出不存在检查通过。没有真实数据smoke；合成通过不保证完整运行资源或原指标回归。

固定预算600秒、worker RSS2GiB、输出128MiB、剩余盘2GiB；父进程每秒采样，worker阶段检查，超限保留产物并停止。父进程RSS不计，瞬时峰值可能漏测。CUDA_VISIBLE_DEVICES为空且没有torch/model调用，GPU0是执行路径约束，不是显卡全局活动监测。

硬验收包括输入身份、缓存顺序、H前缀/逐用户位置及数量、无Train已见项、原B/Q/R Recall≤1e-6回归与命中/曝光精确回归、分组分母、配对差与分解、封存先于标签/封存不变、资源和Test0。科学结果按预定3/4命中及0.0002 Recall的三类规则，不能称显著、等价或语义贡献。

## 产物及收尾

输出manifest、三个H与k/targets、construction_seal、12状态report、R−H/H−Q/H−B配对明细、acceptance、worker_resources、supervisor、console和completeness。另报S、非S和5个低组非S物品的命中/曝光，避免归因混淆。

运行后按RUN_CLOSEOUT创建本目录PER_USER_SLOT_CONTROL_RESULTS.md/AUDIT.md/HANDOFF.md，追加TRAINING_LOG，更新实验族、当前入口与生成导航；旧矩阵/审计、第一项、THESIS_ROADMAP和原始产物保持。没有结果前不创建空结果文件。普通提交不建立标签或宣称原始资产已备份。

**唯一下一步：运行上方命令一次，将输出目录交回审计及统一研究判断。** 失败后不要重复启动。
