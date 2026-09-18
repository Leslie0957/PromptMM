# PromptMM 发布版：Sports seed-2022 的 30 轮 Validation 诊断

身份：`PromptMM-release-Sports-sharedTeacher-v1`，官方源码锚 `70da1002a35d6f2c7712c16cd0b2ca24c8813008`；共享本地 Sports epoch37 教师。这是发布版适配的有限预算诊断，不是已调优的最终对照，也不是原论文端到端复现。

## 手动运行一次

在仓库根目录、当前已提交且干净的源码上运行：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2022 --epochs 30 --gpu_id 0
```

只运行这一组。30轮，每轮214个batch，共6420次更新；batch1024，不早停。先做一次训练前Validation参照，再每轮验证一次，共31次学生Validation，教师和学生Test均为0。CLI固定预算和参数，不接受其他seed、轮数、调参、Test或resume参数。

首次Validation结束显示初始Recall@20；之后每轮显示训练/验证耗时、Recall@20和最佳轮次。现有3步资源检查不能可靠预测整次耗时，以本次实际每轮耗时为准。运行中保持源码/工作区不变，结束时会再次核验启动commit及源码。

## 固定配置与用途

沿用资源检查的seed2022、维度64、图层1、lr0.00002、AdamW weight_decay0.01、embedding decay0.00001、pair/list系数1000000、feature系数0.1、SCE幂2、10个列表负样本、foreach=False。教师drop_rate0.2、prompt_dropout0；教师和prompt不更新，保留发布版的共享embedding存储、图列顺序、目标函数和CPU DGL候选适配。

不在观察结果前改系数。本轮回答：该未调参配置在多轮中是否保持有限、Validation相对初始化如何变化、曲线方向如何、验证阶段需要多少显存。即使结果下降，只要硬性执行条件通过，也保留为有效诊断结果；不能据此直接否定或证明任一方法。

## 选模、指标和产物

- 独立加载train/features/val_mat及固定教师，不导入会读取Test的Data/batch_test，不读取或散列test_mat/test.json。
- 验证用户为val_mat非空行，全物品候选只排除训练交互；Ks=[10,20,40,50]。选择指标Recall@20严格变好才保存，同分保留较早轮次。
- 训练前Validation单独记录，不参加最佳轮次选择；候选轮次为1至30。后续审计须与初始化对比，避免将教师初始化的效果误认为训练收益。
- 排名候选顺序、heapq tie行为及Recall/Precision/Hit/NDCG公式与当前TD评估的纯函数合成对照通过。保留已有NDCG定义（其理想DCG来自top50命中向量排序），不把它混称为按全部相关物品数归一化的其他NDCG定义。评分用户分块256以限制显存；浮点并列边界的跨GPU/分块逐位等价不作保证。
- 产物目录：`exp/promptmm_release/sports_promptmm_release_validation30_seed2022_v1/`，`report.json`保存完整曲线、配置/源码/数据身份、耗时及显存，`best.pt`保存最佳学生权重和选模元数据。
- JSON和checkpoint均临时文件写完再原子替换；跨两个文件不是联合事务，中断状态须审计。成功结束会将最佳权重恢复到显式共享存储的学生并校验摘要及元数据，不进行额外排名。
- best.pt用于选模和后续推理检查，不包含恢复训练所需的优化器/RNG状态，不支持断点续训。图仍须由同一train_mat和发布版语义重建。
- 目录排他创建；失败保留报告、已保存的最佳权重及可能的临时文件，不自动重试。不要删除目录绕过保护；将失败输出交回审计。

所有checkpoint/report保持paper_ready_eligible=false，教师原manifest不变。历史Baby seed2023按审计例外接受、原manifest仍为false；第二创新点未定。原版未参与目标的多余诊断计算仍省略，本次耗时不能作为正式效率加速比。

本次准备只做合成CPU检查和源码/元数据检查，没有启动真实数据训练或Validation/Test排名。运行结束后回复“跑完了”，下一步仅审计该运行及决定合理的后续调参/同预算对照安排。
