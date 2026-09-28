# Sports 初始化 × KD seed2023：四臂一次用户手动运行声明

状态：独立seed2023入口与合成验证准备完成；**真实四臂尚未执行**。只为检验[seed2022现象](INITIALIZATION_KD_INTERACTION_AUDIT.md)是否依赖特定随机初值与三元组回放。不能把固定教师缓存当作第二个独立教师，不改变先前“第一项机制补充、第二算法未成立”的定位。

## 唯一手动命令、来源与身份

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_cohort_seed2023.py'
```

准备基线 `a16bcc1d9c7c6ccd2dd8e7007ff0e5fac59faaf7`，分支 `codex/experiment/baby-teacher-baseline`；用户必须从包含本声明的干净提交HEAD启动，manifest记录实际提交。正式入口无可覆盖CLI参数，内部worker经父进程token/PID、source/config、独占claim绑定。新目录 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2023_v1/` 准备时不存在；seed2022目录及全部旧资产原位保存。只允许一次用户手动启动，按R0→R1→T0→T1串行；任一失败停下并保留部分产物，不跳过、不重试、不恢复或进入下一seed。

固定配置：[seed2023受限delta](INITIALIZATION_KD_SEED2023_COHORT_CONFIG_2026-09-28.json) SHA绑定[seed2022四臂配置](INITIALIZATION_KD_COHORT_CONFIG_2026-09-27.json)和[seed2023只读预检](INITIALIZATION_KD_SEED2023_PREFLIGHT_2026-09-28.json)。解析后与seed2022相比，仅`seed=2023`、随机初值、回放、run_id、输出目录和命令不同；其他配置逐字段相同，包括PYTHONHASHSEED=2022，模型torch RNG种子使用spec.seed=2023。保留固定共同教师缓存、Train/Val、教师checkpoint来源、优化器/损失/评价、环境/数值、候选阈值和资源保护。共同原始配置SHA `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d`。运行前逐资产SHA/结构/有限性检查，身份错即停，不补建或复制资产。

seed2023初值路径 `exp/paired_coldinit/sports_three_seed_v1/seed2023/initial.pt`，SHA `bb5429aa8a1edfe53b36fed977146460d1101ae79539d2badf856d5514ef9d85`；回放 `exp/paired_coldinit/sports_three_seed_v1/seed2023/triplets.npy`，SHA `3763128ba003228d60c09924b9370b4cab0b8c16679366742835fc4559d89085`。两者都不同于seed2022，历史full/image_matched两项同用它们；预检对全部65,740,800三元组的Train正负归属错误均为0。共同缓存SHA `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`；Train/Val SHA沿用已固定资产锚，不读取Test。

## 四臂协议与事前判断

| 臂 | 初值 | 训练目标 | alpha | 配对关系 |
|---|---|---|---:|---|
| R0 | seed2023随机ID两表 | BPR | 0 | 与R1同初值/同回放 |
| R1 | 同R0 | BPR+Full | 0.3 | 与R0配对 |
| T0 | 共同教师最终ID两表 | BPR | 0 | 与T1同初值/同回放 |
| T1 | 同T0 | BPR+Full | 0.3 | 与T0配对 |

每臂300epoch×214batch×1024三元组，64200更新、全cohort256800更新；同一个seed2023不可写回放用于全部四臂，用户/正/负例与批次顺序相同。BPR mean(-logsigmoid(dot(u,p)-dot(u,n)))、无显式embedding L2；Full只对正物品用固定image/text方向监督，`0.3*(image_mse_mean+0.3*text_mse_mean)/1.3`，无user KD。每臂初始化之后新建空状态AdamW，lr6e-5、wd.01、betas(.9,.999)、eps1e-8。教师只提供已固定ID初值/模态目标，不forward。

epoch0完整Validation仅描述，不参与选best；epoch1–300每轮完整Validation，四臂合计1204次。Train候选排除、全35598用户、batch256、K10/20/40/50、float32相同点积和指标求和；精确`exact_topk_v1`此前对固定状态[真实parity](INITIALIZATION_KD_EVAL_PARITY_V2_AUDIT.md)通过，合成边界/同分检查通过。主终点第300轮Recall20，次终点最早最大Recall20及同轮NDCG20/完整曲线；每臂保存best/final完整模型+AdamW状态。Test文件打开拒绝在加载分割前安装，Test读取/排名/最终测试均0。

事前计算末轮delta_R=R1−R0、delta_T=T1−T0、I=delta_R−delta_T；沿用I>=.002及T0−T1<=.001的描述性候选筛查（均为绝对Recall单位），**不要求暖KD为负**。同时报告四臂值、两个delta、最佳轮及轨迹，不只看I。若同向，削弱seed2022特定随机轨迹解释；不能分离随机初值与回放各自作用，也不能证明语义重叠、统计显著、等价、正式非劣或第二算法。低值或相反方向均是有效完成结果。

## 资源、产物、验收与停止

与seed2022用户修订相同：**无墙钟硬停止**，持续记录elapsed；CUDA allocator2GiB、workerRSS8GiB、输出1GiB、可用盘至少4GiB，attempt1。seed2022实际父进程5821.125秒（约1小时37分），只作时间参考，不保证本次时长；如卡住，须用户自行中断，仍要保留部分产物与审计，不能自行重试。RSS采样/allocator分数不是OS全显存上限。任一身份、源码、环境、数值、有限性、资源或完整产物门失败即停；不临时扩限或调参。

输出launch_manifest、worker_claim、worker.log、telemetry、report、acceptance、supervisor，以及四臂curve.json/best.pt/final.pt；失败尽可能有failure.json，全部保留。exit0不能单独验收，必须四臂各300连续epoch/64200step、共1204Val、finite指标/权重/moments、最早best/epoch300检查点和optimizer状态一致，来源config/hash对应、报告interaction可重算、Test0、资源达标。worker检查checkpoint内容，parent独立复核文件集SHA；无Test对应checkpoint。

## 准备验证与结果记录路由

合成CPU检查覆盖：seed2023解析配置仅上述六类差异、锁定预检/历史配置SHA、拒绝错seed/路径/额外字段、同一新spec在完整3用户/60物品/300轮×四臂的模拟GPU worker路径中运行、1204快Val及保存/验收；原seed2022、Test拒绝、资源和Windows I/O合成检查继续通过。本准备不读取真实模型/分割、不运行GPU/真实训练或评价，用户执行时runtime再复核输入字节与环境。

记录更新目标：

- 本次run/cohort身份：`sports_init_kd_interaction_seed2023_v1`，R0/R1/T0/T1×seed2023。
- Outcome审计：新建`docs/research/INITIALIZATION_KD_INTERACTION_SEED2023_AUDIT.md`，完整保存身份/参数、各臂曲线与checkpoint、Validation终点、Test0、停止和资源、所有指纹；无效/部分也如实记录。
- TRAINING_LOG：保留本pending，运行后另追加completed/failed/partial与当前交接。
- 实验族导航：`docs/experiments/README.md`初始化×KD行及root短入口按状态更新。
- 活跃矩阵：`docs/research/INITIALIZATION_KD_INTERACTION_RESULTS.md`增加seed2023四臂Val状态/格子；旧18格主矩阵/Test格不变。
- 条件性消费者：立项单、论文主张/缺口，仅在实际机制覆盖/主张改变时更新；不得凭第二seed自动宣称显著或新算法。
- 六项生成导航：结果审计后刷新。旧seed2022/历史运行、cache、数据、manifest和check/原位保留；非稳定冻结，不自动tag/bundle/push/merge。

唯一下一步：用户执行本页命令一次，返回成功或失败供审计。不得自动执行seed2024、重跑任何旧命令、访问Test或进入P1b/P2/门控。
