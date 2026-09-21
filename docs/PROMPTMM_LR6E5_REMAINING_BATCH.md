# PromptMM lr6e-5：seed2023、2024串行复核

两组各300轮，仅Validation、不早停；固定lr6e-5，其余配置与已完成seed2022相同。从固定epoch37教师重新初始化，单组64200更新/301次Validation（含不参选的初始验证），按Recall@20严格改善保存最佳权重，同分保留较早轮次。Test不读取、不评估。不会重跑seed2022。

一次运行：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B codes/run_promptmm_lr6e5_remaining.py
```

依次执行2023、2024，预计共26–30小时。保持供电、不睡眠，运行期间不修改源码或切换commit。每组退出成功且预算、种子、实际学习率、有限指标、选模、checkpoint哈希、零Test等检查通过才进入下一组。失败保留结果并停止，不自动重试/续训/跳过；勿删除目录绕过保护。

输出：
- `exp/promptmm_release/sports_promptmm_release_validation300_seed2023_lr6e5_v1/`
- `exp/promptmm_release/sports_promptmm_release_validation300_seed2024_lr6e5_v1/`
- 批次汇总：`exp/promptmm_release/sports_promptmm_release_validation300_lr6e5_remaining_v1/batch.json`

每组有report.json和best.pt，独立目录，不覆盖旧2e-5结果或已完成2022。批次固定参数，无额外CLI覆盖项；普通旧300批次仍为2e-5，请使用上面的新命令。

完成后一起审计，与已有2022 lr6e-5组成三个种子，并与2e-5及现有完整方法配对比较。相同教师不等于独立教师稳健性，相同seed不保证DGL完全相同随机轨迹。6e-5在seed2022上已经反超当前完整方法，先验证跨种子情况，不预先宣称最终方法优劣或论文价值。

原teacher/student eligibility=false、Baby历史seed2023审计例外及原manifest false均保留，第二创新点未定。后续实验需根据联合审计另行准备，本批次不自动追加。
