# Sports 共享初始化与语义目标：完整方法 / BPR-only

这是新配对，不是重试旧失败目录。手动执行一次：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sharedinit_pair.py
```

顺序：full共享初始化300轮，然后BPR共享初始化300轮；第一组完整性检查通过才继续。seed2022、lr6e-5、batch1024、64维、AdamW weight_decay0.01，Validation Recall@20严格改善选模，patience300，不在窗口内早停；初始Validation不参与选模。两新profile默认参数仅alpha不同：full0.3，BPR0；保留image/text rates1/0.3和user0，以维持同构造路径。

共享文件固定为诊断process0第一次输出：exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt，SHA256 e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20。按进程顺序选择，没有根据指标挑选。每次加载检查SHA、键、shape、float32和有限值；两组独立复制同一用户/物品表。full的图文语义目标也固定来自该文件；BPR禁用方向损失。保留历史教师前向准备调用后替换输出，不改变种子或采样逻辑。

这是对新保存张量的严格配对，不声称恢复了9月22日原始初始向量。旧full/BPR失败、PromptMM和诊断产物保留。正式新证据来自两组之间的比较；不能把新BPR单独与旧full称为逐位同起点对照。

预期耗时参考旧TD约3小时8分/组，两组合计约6–7小时，系统负载可能延长。沿用TD原验证器；此前Windows验证子进程页面文件不足风险尚未通过本任务修改，运行时避免叠加其他大内存任务，但本次不改系统设置。遇到异常保留输出并反馈，不删除目录重跑。

脚本要求干净提交，两个profile及批次目录无旧尝试；每组源码/共享文件前后核验，检查300轮有限曲线、严格选模、checkpoint和初始指纹。BPR检查蒸馏全0及总损失等于BPR；两组完成后核验初始指标完全相等。原哈希保护未删除，仅新profile改用共享文件身份。

输出：exp/initialization_checks/sports_sharedteacherinit_pair_seed2022_val300_v1/batch.json；各profile时间戳manifest在exp/runs/sports，checkpoint/曲线路径在manifest。两profile分别为sports_student_full_sharedteacherinit_seed2022_val300_v1和sports_student_bpr_sharedteacherinit_seed2022_val300_v1。

不执行教师/学生Test排名，历史TD加载器Test结构读取保留；eligibility=false，Babyseed2023审计例外及原manifest=false不变。跑完下一步只审计蒸馏相对BPR微调增益，不自动加种子、重试或启动其他实验。
