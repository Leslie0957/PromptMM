# Validation排名加速一致性检查

新增NumPy partition阈值筛选、候选原顺序处理边界并列、稳定排序前50名。GPU评分仍用batch256；保留候选set顺序及现有逐用户指标/累计顺序，以严格一致优先。未引入全量用户×物品缓存，未改指标定义。旧训练默认仍用旧评估器；新版待本次真实检查审计后才考虑接入。

手动运行一次：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B codes/check_promptmm_validation_parity.py
```

固定使用seed2022/lr6e-5/300最佳checkpoint（epoch294），SHA e32f9907d68874f09f4327a6ccfba8b4c86e1b7c73c7d0a06713f990b6e94e27。仅加载train_mat、val_mat及该权重；不加载教师/特征/Test，不训练、不更新或覆盖模型、不重新选模。

先在完全相同的分数数组上对所有用户比较旧/新前50名ID，要求逐项完全相同，指标严格相等；再分别执行完整旧/新评估各一次，要求指标与前一步严格相等。共三个Validation遍历（首次同时对比两个排名器），零Test。报告旧/新排名时间和完整函数时间；单次计时有顺序、缓存、系统负载影响，不是论文正式效率实验。历史保存指标差异单独报告，不静默忽略新版与旧版的差异。

报告：exp/validation_checks/sports_promptmm_validation_parity_seed2022_lr6e5_v1/report.json。排他创建，失败保留；不删除结果绕过保护，不自动重试。源码需已提交且干净；运行期间保持不变。预计数分钟至十余分钟，实际提速与耗时须以报告为准。

无论通过或失败，都把结果发回。通过后再审计是否可以用于后续训练；未通过则定位差异，不改变已有实验结果。原eligibilityfalse、Babyseed2023审计例外/原manifestfalse不变，第二创新点未定。
