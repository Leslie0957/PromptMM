# Sports 初始化 × KD 四臂 seed2022 完成审计

2026-09-28。用户手动运行一次，从干净提交 `484a4bef36030e0d76a5d77e4da14d84aea46589` 启动；supervisor `completed`、exit0、artifact acceptance passed。审计只读既有 JSON/曲线/检查点，没有重训、重排、加载真实分割、GPU或Test访问。原始输出原位于 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2022_v1/`。

## 冻结身份、命令与协议

[启动声明及用户取消墙钟限制修订](INITIALIZATION_KD_COHORT_LAUNCH_2026-09-27.md)是本次依据。实际历史命令（已消耗，不可重跑）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_cohort.py'
```

manifest/config/report/acceptance source均匹配上述提交，config digest与resolved config一致；共同base SHA `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d`，共同资产由原[预检锚](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)及本次运行时hash门定义。本审计不重新哈希六大输入。user35598/item18357/ID64、seed2022、同一只读300×214×1024回放、固定随机和共同教师缓存初始化、全新AdamW lr6e-5/wd.01、BPR/Full原归约不变。环境run_5060/RTX5060/cuda0/float32/TF32off/deterministic/CPU线程4由干净源码环境门及manifest固定。

R0/R1同一随机初值、T0/T1同一教师最终ID表；R1/T1用同一图文物品目标，KD alpha0.3、文本率0.3，另两臂只BPR。实际初始化张量没有单独保存到本次目录；成对身份依据运行时加载共同锚、`run_arm`逐位复制检查、固定source与两组epoch0指标相同，而非事后从最终检查点倒推。没有使用smoke-final表作正式初值。未打开Test文件、未做Test排名/选模或教师forward的证据来自冻结代码及report `test_file_reads=0`；没有独立OS文件访问轨迹。不存在selected-versus-tested checkpoint，Test成绩均空。

## 完成性与Validation结果

四臂顺序R0→R1→T0→T1，各300epoch、64200 AdamW更新，共256800；各有epoch0描述性Val及epoch1–300曲线，共1204完整Val。最早最大Recall20选择，主终点末轮epoch300。每臂best/final模型+优化器检查点保存，CPU只读的既有验收函数逐一反序列化检查身份、形状、有限权重/moment、AdamW参数和步数，并核对曲线/选择及报告与SHA；审计另行核对所有连续epoch、有限[0,1]指标、交互重算。运行时每步拒绝非有限训练目标，但没有逐步loss轨迹；不能据此评估收敛细节。

| 臂 | seed | epoch0 Recall20 | epoch300 Recall20 | epoch300 NDCG20 | 最早最佳epoch | 最佳Recall20 | AdamW步数 | 状态 | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| R0 | 2022 | 0.0012196378073674552 | 0.045566338918469319 | 0.021737701988833878 | 299 | 0.045566338918469319 | 64200 | 完成 | 未读取 |
| R1 | 2022 | 0.0012196378073674552 | 0.082049530069735588 | 0.037171971739143245 | 294 | 0.082328259836077014 | 64200 | 完成 | 未读取 |
| T0 | 2022 | 0.094184495190847733 | 0.095151699965148426 | 0.043719393957035953 | 286 | 0.095477873095973498 | 64200 | 完成 | 未读取 |
| T1 | 2022 | 0.094184495190847733 | 0.09506040270125718 | 0.043686891922287814 | 283 | 0.095438076852738862 | 64200 | 完成 | 未读取 |

末轮 delta_R=R1−R0=`0.03648319115126627`；delta_T=T1−T0=`-0.00009129726389124626`；I=delta_R−delta_T=`0.036574488415157515`，与report逐字段相同。I超过预定描述性候选门槛`+0.002`；T0−T1=`0.00009129726389124626`，小于预定代价容限`0.001`。两者仅支持“这一seed及配方下，KD对随机初值收益大，对教师初值的末轮增益未显示且略低”；暖状态最多是候选有限增量边界，**单seed不证明统计显著、等价或正式非劣**。按最佳epoch，T0=`0.0954778730959735`、T1=`0.09543807685273886`，同样不显示增量收益；R1最佳0.08232825983607701。最早最佳epoch均由实际曲线确认，未把best充作末轮主指标。

两组epoch0 Recall20分别0.0012196378073674552与0.09418449519084773，起点质量不同；四臂差分描述完整初始化方案×指定KD配方交互，不能唯一归因为模态语义冗余或“教师起点使KD无用”。R1末轮0.08205仍低于T0/T1约0.095，不能称随机+KD全面赶上教师起点。Test完全未访问，这些均仅Validation结果，不能填论文主Test格子或宣称第二算法成立。

## 资源与停止条件

父进程5821.125秒（1小时37分1.125秒），worker5812.125秒；各臂总秒数R0 1403.500、R1 1499.937、T0 1423.641、T1 1485.015。报告CUDA allocator峰值161480704字节（154MiB）；采样RSS峰值2275549184字节（约2.119GiB）。最后遥测可用盘411364511744字节>4GiB，输出332162190字节<1GiB，均未触发上限。RSS是采样值，CUDAallocator不含驱动/桌面显存；遥测worker wall5812.953秒不是父进程完整elapsed。按用户修订没有墙钟硬限；本次自然完成，不能把原24小时历史限额当成本次门槛。

此前基于短smoke的14.817小时粗估显著高于实际1.617小时，说明短程成本外推失真；本次真实时间只描述这台机器/这次状态，不保证其他seed耗时。本次已完成全部固定工作，未因低指标触发失败、重试、回滚或追加seed。

## 原始产物与路由

以下指纹按本次审计时原位文件计算；目录内无failure.json。旧smoke/parity及check/不改、不提交原始检查点。当前seed2022独立结果矩阵见[四臂矩阵](INITIALIZATION_KD_INTERACTION_RESULTS.md)，旧18格不动。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `acceptance.json` | 1348 | `5ee7fc677e63eed1e42338a0a9ac7ec0108135925c6951c5d42c88b3972e10c1` |
| `launch_manifest.json` | 7495 | `d1eb28f82a34c0a8e45b06729336a08260d9668215a3994e184d903f3af3664f` |
| `R0/best.pt` | 41441421 | `de5fd969498ae59f57084f1df4eb38ac566d5f578d35e044e82a87645b2ac037` |
| `R0/curve.json` | 140180 | `572a10e8036813527d01326c19708197341369eddbd591aa64875cdf4dc800ae` |
| `R0/final.pt` | 41441755 | `a18e0b19baaae05cdda8d669af746fc775a0be18696fdf5d1e4a7e9742397723` |
| `R1/best.pt` | 41441421 | `e2ac4cea1c56369f4c7a1a7861e832599c8ff5ce876801d524812fd9b0a5ade6` |
| `R1/curve.json` | 139268 | `d3adb184b4b30bb797f9096b9e1a7f94cef5f0e029c3945cb58ad9f731ffc831` |
| `R1/final.pt` | 41441755 | `b8720f29855eece4b1da584367ce34f402e209ba32257081d9790fead3550d62` |
| `report.json` | 63167 | `122b6d40929bff2052f92954eaa9d3b270f1564a5af1886baae404c5c0aa6cfe` |
| `supervisor.json` | 118 | `013ad0cdea49ab6969c3bf6c9af47ec51e05602e88dc8465819b644f045907c6` |
| `T0/best.pt` | 41441421 | `bc1b5e0707d0ee371cd35873cafc4757f34a1f00ddc21d1f73bb3461a9db9f12` |
| `T0/curve.json` | 138833 | `3da01c3dda4db808e423d9221e93fa43298949d6e7066eb0d3c7cad6c5a2e6e8` |
| `T0/final.pt` | 41441755 | `d432ee6a211caf77c1999f6357d4d55689bafc7daa95b3b6157a8503cef364b0` |
| `T1/best.pt` | 41441421 | `a4774934f838b4298f358312ea60bc465d1622587fa6645f7f2246d7771b5c11` |
| `T1/curve.json` | 138786 | `966b9474ae23e5fecb6ebe41307889b201a9879cd29a8a845d49ec3603a4f8bb` |
| `T1/final.pt` | 41441755 | `2aeba3de096f58a24790348b909c57e0e171225a2e006c2c1bf62471306d1dcc` |
| `telemetry.json` | 189 | `544dd59af6c3bf16bae51aec6bbd78273523b4b27aae65d47d12245adff5cd19` |
| `worker.log` | 184 | `5dfc47265e4ccee5a17596a297d65101af5da89314006a63390078096fb9b5a0` |
| `worker_claim.json` | 35 | `d4dab03d65a7d796430184838dfe4f1932bc17e5fe75b7f63b18579e7ed3af0e` |

记录目标：本审计、TRAINING_LOG独立completed/current、docs/experiments族入口和root入口、上述新独立矩阵、六项生成目录均更新；docs/README无状态细节不改。立项单的规划状态由本完成结果取代，但不重写其历史文本；论文正文/缺口与旧18格主矩阵不因单seed Val变化而改写，当前第二创新点/整篇覆盖主张不变。无tag/bundle/push/merge；忽略产物不受Git提交保护。

唯一下一步：针对本次四臂的初始化质量差异与暖KD微小负差，做**只读机制与预算复核**，确定是否值得单独声明第二seed或归入第一项机制补充；本次不提供另一个run命令，不自动训练/访问Test/进入P1b/P2或门控。
