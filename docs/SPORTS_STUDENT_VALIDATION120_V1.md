# Sports 学生配对验证配置（2026-09-16）

三个配置已经准备；本次仅做参数、源码及数据/教师元数据检查，没有训练、模型前向、PCA拟合或Test指标评估。

## 比较设计

seed2022，ID-only、无投影、随机初始化64维，BPR推荐目标；AdamW学习率6e-5，weight decay0.01，batch1024。120轮（0–119），patience120，窗口内不早停。Validation Recall@20选模，恢复最佳权重后跳过Test。三组使用同一份Sports数据、官方划分、预处理和Epoch37教师，不用教师权重初始化学生。

|组别|alpha|图像rate|文本rate|实际目标|
|---|---:|---:|---:|---|
|BPR|0|0|0|BPR|
|完整方法|0.3|1|0.3|BPR + (0.3/1.3)L_image + (0.09/1.3)L_text|
|匹配图像权重去文本|0.3/1.3|1|0|BPR + (0.3/1.3)L_image|

用户蒸馏权重均0。去文本指去掉学生文本蒸馏项；图像目标仍来自同一个多模态教师，不能解释为纯视觉教师。BPR路径也加载同一教师以维持当前代码流程，但不使用语义损失或教师初始化；计算代价不等于独立纯BPR实现。

## 教师复用边界

教师原件及manifest均不改写：paper_ready_eligible=false，唯一blocker为final test evaluation is disabled。新入口只接受以下SHA256的教师，且只用于这三个Sports无覆盖配置、仅Validation学生训练：

57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea

保留原有协议、Validation选模、K、数据身份、推理参数、候选排除、hard-token provenance、资格标记和blocker精确一致检查，以及模型/提示权重严格加载。仅对该受限路径使用既有require_paper_ready_checkpoint=false分支。正式Test、教师重训、其他权重、配置覆盖均不能通过此入口。运行manifest新增teacher_reuse_policy和授权指纹。此改动不使教师或学生变为正式Test结果。

数据锚点：SPORTS_CONVERTED_20260916（转换manifest SHA256 3772a17c8b70fa4739653534dca8d542e67e0649f1f44ccba4194fec17bc1104）；教师及PCA身份见TRAINING_LOG.md的Sports教师completed记录和SPORTS_TEACHER_RESULT_2026-09-16.md。

## 手动命令

从包含本声明的干净提交运行。一次仅启动一个命令，失败或异常时保留结果并停止，不自动重试或启动下一组。单一下一步是BPR；其余命令一并备好，正式执行顺序按记录推进。不要改参数、种子、Test开关或并行启动。

```powershell
Set-Location D:\Download\PromptMM
git status --short --branch
git rev-parse HEAD
```

### sports_student_bpr_seed2022_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2022_val120_v1 --gpu_id 0
```

### sports_student_full_seed2022_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2022_val120_v1 --gpu_id 0
```

### sports_student_image_matched_seed2022_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2022_val120_v1 --gpu_id 0
```

## 产物与验收

每次独立run名；exp/runs/sports/run_manifest__<run>.json、logs/<run>、Model/sports/下对应td_distill模型完整/推理权重及exp/converge/sports/曲线，具体路径以manifest为准。验收120轮、有限目标和指标、零Test/零teacher训练、无配置覆盖、相同教师/数据/cache及随机初始化设置，比较最佳Validation和完整曲线。单种子仅验证结果不等于多种子Test泛化证据。

59项相关测试通过；三组实际教师SHA256及数据/推理元数据检查通过。未做GPU训练测试，运行时风险仍需按日志审计。源版本以本准备提交为锚点；现有runtime manifest不记录Git HEAD/dirty状态，不声称自动保存了启动版本，手动保留上面Git命令输出。

seed2023历史审计例外及原manifest false保留，第二创新点未确定。
