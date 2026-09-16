# Sports seed2023/2024 三组配对手动运行

本次完成六组配置准备，未启动训练、Test或PCA拟合。每组120轮、patience120、仅Validation Recall@20选模，seed是学生随机种子。固定教师为Sports Epoch37，PCA/hard_token_seed保持2022。全部沿用seed2022的randomID64、无投影、AdamW lr6e-5/decay0.01、batch1024和官方划分。

BPR alpha0；完整方法alpha0.3/image1/text0.3；匹配图像去文本alpha0.3/1.3/image1/text0。用户语义项0、无教师初始化。即完整方法与去文本实际图像权重相等。

教师SHA256 57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea；原资格false不改写。受限复用白名单随registry扩展到9个已声明配置，其余数据身份、推理参数、blocker、Test禁用及指纹检查不变。没有通用跳过校验开关。

59项测试通过，六组真实parser及实际教师指纹/冻结元数据检查通过；三个旧seed2022配置与1093348源码对象逐项相等。新旧每个arm解析参数差异仅seed和配置name/source。此准备未访问Test数据；沿用已审计数据身份，正式运行仍会结构预检。

## 手动执行

先进入根目录；保留git输出，工作区应干净，提交应为包含本声明的准备提交。一次运行一条，不并行；每条一次。异常时停止并保留输出，不重跑。成功后可按下列已准备顺序继续，完成后统一审计各次结果；这不是自动训练脚本，不授权额外seed/调参/Test。

```powershell
Set-Location D:\Download\PromptMM
git status --short --branch
git rev-parse HEAD
```

### sports_student_bpr_seed2023_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2023_val120_v1 --gpu_id 0
```

### sports_student_full_seed2023_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2023_val120_v1 --gpu_id 0
```

### sports_student_image_matched_seed2023_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2023_val120_v1 --gpu_id 0
```

### sports_student_bpr_seed2024_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2024_val120_v1 --gpu_id 0
```

### sports_student_full_seed2024_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2024_val120_v1 --gpu_id 0
```

### sports_student_image_matched_seed2024_val120_v1

```powershell
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2024_val120_v1 --gpu_id 0
```

## 完成后审计

逐次核对120轮、有限损失/曲线、best checkpoint恢复、Test次数0、同一数据/教师/cache、无配置覆盖；汇总三种子均值、样本标准差和配对差值，以及峰值是否贴近窗口边界。当前120轮不能保证充分收敛，若后续诊断增加预算必须保持配对。只变化学生种子，不代表教师种子稳健性。

运行manifest没有Git HEAD/dirty快照这一历史限制仍存在，保留上面的输出；忽略的大文件不受Git备份保护。历史Baby seed2023审计例外及原manifest false保留；本次Sports seed2023不是旧异常运行的重跑。第二创新点未确定。
