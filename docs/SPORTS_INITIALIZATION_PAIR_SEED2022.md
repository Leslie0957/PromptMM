# Sports seed-2022 初始化对照：手动串行两组

准备阶段；不重跑原始两组，不执行Test排名。运行一次：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_initialization_pair.py
```

顺序：
1. 当前完整方法 + 教师初始化：sports_student_full_teacherinit_seed2022_val300_v1。
2. PromptMM-release-Sports-sharedTeacher-v1 的随机初始化控制变体：sports_promptmm_release_validation300_seed2022_lr6e5_randominit_v1。

各300轮，seed2022，lr6e-5，训练batch1024，维度64，AdamW weight_decay0.01；窗口内不早停。各方法内部仅改变初始化来源，保留原生损失、采样、图结构和更新方式。TD教师向量独立复制；release用原构造器Xavier向量建立原有共享参数，保留教师初始前向，避免额外随机数消耗。初始向量shape/dtype/SHA记录，初始Validation仅描述不参与选模；此后每轮Validation Recall@20严格改善选模。

TD沿用旧评估器，release用已核验的快速评估器。TD预计耗时仍参考历史运行；release粗估约5小时，负载波动可能较大，不承诺两组合计时间。TD沿用历史加载器，会读取Test结构用于协议检查，但不执行Test推荐评估；release不读取Test。不得将两者统称为零Test文件访问。

参考：已有TD随机初始化seed2022最佳Recall0.0823282598、NDCG0.0372074078；已有release教师初始化seed2022最佳Recall0.087634556329、NDCG0.040125451936。共享SPORTS_CONVERTED_20260916和固定epoch37教师。保持各自的训练协议，不宣称仅初始化就消除了所有跨方法混杂。现有其他种子无需重跑。

串行脚本要求干净已提交源码；任何已存在的本批次/目标尝试都会拒绝启动。第一组退出异常、曲线不满300、指标非有限或检查不通过，就停止，绝不自动重试。请勿运行期间修改仓库，不要删除失败目录绕过保护。

输出：
- exp/initialization_checks/sports_initialization_pair_seed2022_val300_v1/batch.json
- TD manifest在exp/runs/sports/run_manifest__*.json，按新profile识别；模型/曲线路径记录在manifest。
- release在exp/promptmm_release/sports_promptmm_release_validation300_seed2022_lr6e5_randominit_v1/。

跑完或异常后发回结果，下一步审计两组并与已有初始化参照比较；不自动补其他种子。Babyseed2023既有审计例外接受且原manifest=false；本批次eligibility=false，第二创新点未定。
