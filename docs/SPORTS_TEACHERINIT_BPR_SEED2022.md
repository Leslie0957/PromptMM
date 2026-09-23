# Sports 教师初始化 BPR-only，seed2022，300轮

手动运行一次：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_teacherinit_bpr.py
```

与已完成的教师初始化完整方法相比，仅td_distill_alpha从0.3改为0。保留image/text rates1/0.3、user0（保证相同模型构造路径），总系数0会完全跳过方向损失及其梯度。仍保留教师语义缓存准备，避免另改实验流程；不是效率测试。

seed2022，300轮，lr6e-5，batch1024，64维ID表，AdamW weight_decay0.01；仅Validation，patience300，窗口内不早停。实际初始用户/物品向量的形状、dtype、SHA256须与上一组完全相同，否则首个优化器步骤之前报错退出。初始Validation只记录，不参与选模，结束后还要求与参照初始指标完全一致。

对照：未训练Recall0.094184495191/NDCG0.043320782869；完整方法最佳Recall0.095438076853/NDCG0.043780603134。BPR可能更好、接近或更差，所有完成且合法的结果均保留，用于判断方向蒸馏相对BPR微调的增量。

脚本要求干净已提交源码；已有同profile或运行目录时拒绝启动，无重试、恢复或下一组。参照manifest指纹也必须一致。运行期间不要改源码。上次同预算TD运行约3小时8分钟，仅供参考。

记录：exp/initialization_checks/sports_bpr_teacherinit_seed2022_val300_v1/run.json；新manifest位于exp/runs/sports，profile为sports_student_bpr_teacherinit_seed2022_val300_v1；曲线/独立checkpoint路径在manifest。运行完验证300轮有限曲线、最佳checkpoint、方向损失全0、总损失与BPR相等、初始指纹和初始指标一致。

不执行教师/学生Test排名；历史TD加载器仍读取Test结构，不宣称零Test文件读取。原结果不覆盖，eligibilityfalse保留；Babyseed2023审计例外接受、原manifestfalse不改。下一步仅审计这一组与完整方法/未训练起点的差值，不自动增加种子或访问Test。
