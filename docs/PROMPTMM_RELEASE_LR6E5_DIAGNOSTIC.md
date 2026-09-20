# PromptMM seed2022 学习率单因素诊断

仅将已审计seed2022/300轮的学习率从2e-5改为6e-5。300轮、64200更新、初始加每轮共301次Validation，不早停，Recall@20严格改善选模、同分取较早轮次；不读Test、不评估Test。其余参数、共享epoch37教师、数据、目标函数和初始化方式保持不变。

手动运行一次（不是三种子串行命令）：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2022 --epochs 300 --student_lr 6e-5 --gpu_id 0
```

预计约12.5–15小时，以实际负载为准。源码须已提交且保持干净；运行期间不要改文件/切换commit。新目录：`exp/promptmm_release/sports_promptmm_release_validation300_seed2022_lr6e5_v1/`，产物report.json和best.pt。独立目录不覆盖旧实验，已有目录时拒绝启动；失败保留、不自动重试/续训，也不自动启动其他种子。

对照：原seed2022、lr2e-5最佳第299轮，Recall@20=0.07697886138664704，NDCG@20=0.034808348170557626；对应report SHA256 a0872a978fd0087190b1bb431897ca7d6dd73e840a12712d533c7933df657330。此次从同一固定教师重新初始化，不从旧best.pt继续。

学习率明确写入配置、优化器、报告及checkpoint。此独立入口的--student_lr覆盖其固定默认2e-5，不修改通用parser或旧三种子命令。CLI只允许seed2022/300轮使用6e-5。

此实验检验学习率选择对固定预算内的验证表现和曲线速度的影响，不预先假定一定改善；即使变差，硬性协议通过仍保留为有效结果。AdamW的weight_decay数值不变，但有效每步衰减随学习率改变，因此不能解释为完全排除了正则化影响。DGL随机性也使相同seed不能保证完全相同采样轨迹。

完成后只审计本次与既有seed2022的初始/最佳/最终、30/120/300轮窗口、末段变化及稳定性，再决定是否值得其他种子复核；不自动扩展调参、加轮数或开启Test。单种子学习率对照不能替代完整公平调参或证明创新性。

保留所有eligibility=false、Baby seed2023审计例外及原manifest false；第二创新点仍未确定。
