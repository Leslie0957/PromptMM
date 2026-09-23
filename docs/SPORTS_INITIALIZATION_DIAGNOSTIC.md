# Sports 初始化重复性诊断（零参数更新）

手动运行一次：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/diagnose_sports_initialization.py
```

沿用真实TD入口的BPR教师初始化profile、seed2022和教师/数据身份检查，在教师初始向量生成后提前返回；不会构造TD优化器或进入学生训练，不会调用Validation/Test排名或验证进程池。历史加载器仍有Test结构读取，缓存及preflight准备照常。不是失败BPR训练的重试，不解除原运行目录保护，不更改系统页面文件。

脚本串行启动两个独立进程。每个进程保存首次完整6项教师输出为initial_tensors.pt，并追加2次同条件前向，记录每次SHA、最大/平均绝对差、不同元素数量；最终比较两个进程的首次输出。记录教师/prompt状态及权重未变、环境/数据/源码身份。GPU模式不改变确定性/精度开关，先观察原路径。哈希不一致是诊断结果，有限且完整的诊断可以正常completed；它不等于BPR初始化验收通过。

输出：exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/report.json及process0/process1子目录；另有标记initialization_diagnostic_*的时间戳manifest。已有failed BPR和已完成full产物均保留。源码须已提交且干净，输出目录排他创建，不自动重试。只运行上述顶层命令，不单独调用内部worker。

预计几十秒至数分钟，取决于加载和preflight；不是300轮。看到最终Diagnostic report并返回提示符后反馈；异常也保留并反馈，不删除目录重跑。

报告回答：原路径在一次进程内能否重复、跨进程能否重复、是否匹配历史指纹。由于历史初始张量没有保存，不能仅凭本次不同张量间的小差值就认定它们与历史张量也接近。不会自动设容差、采用新初始化训练或恢复BPR。下一步先审计报告，再决定如何冻结一份共享初始化或进一步定位原因。
