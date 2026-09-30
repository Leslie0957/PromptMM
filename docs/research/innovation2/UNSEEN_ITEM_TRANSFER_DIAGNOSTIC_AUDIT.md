# Sports未见物品诊断v1：执行与保存证据自审

2026-09-30。**completed，exit0；预定科学判定undetermined。** 原因为`MM warm teacher advantage not established`。所有预定fit/校准/报告完成，有效低结果原样保留；未重试、未恢复、未启动下阶段。此为执行方自审与另写保存证据核查，不冒充另一AI/外部独立审核。数值见[14行矩阵](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_RESULTS.md)，[review交接](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_HANDOFF.md)。

## 1. 授权、来源与实际执行

用户对唯一具体串行诊断回复“授权”。[正式声明](UNSEEN_ITEM_TRANSFER_RUN_DECLARATION_V1.json)与日志pending所在clean launch commit为`07eaeee8db0fc62b8fc998f7ebed1c30dd024e78`，分支`codex/experiment/baby-teacher-baseline`，exec session65504仅launch一次。实现基线`e353dc1f48d229b2405c42493bc94dcf3254c4e1`；共享协议anchor`e565a949b7417a2c4268552c59a8918954ee53e2`，profile SHA`8fa483db8ee8fd1a80086b6f7bcbb090803adc4d57ccddf8c85e6cf96ad9ed8e`。profile中的preparation/launch=false是原设计快照，后续真人授权和已提交正式声明是本次执行依据；不改旧profile伪造资格。

实际PowerShell命令（历史执行记录，不是新的待跑命令）：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_unseen_transfer_diagnostic.py --execute --declaration docs/research/innovation2/UNSEEN_ITEM_TRANSFER_RUN_DECLARATION_V1.json
```

过程按assets→6教师→16学生配置（12神经+3ridge+1N）→18校准选择→2教师/12学生probe报告串行执行。全部神经fit300轮，每10轮选择；实际参数/搜索网格与冻结profile一致，无追加配置。原始产物独占目录`exp/innovation2/sports_unseen_transfer_v1/`保留，未覆盖任何第一项/基线资产。

环境：Python3.10.20，PyTorch2.11.0+cu128，NumPy2.2.6，SciPy1.15.3，sklearn1.7.2，psutil7.2.2；run_5060既有环境，一台RTX5060串行，BLAS/Torch线程4，deterministic_algorithms启用，CUBLAS_WORKSPACE_CONFIG=:4096:8。未安装/更换包；未证明跨硬件逐位一致。PyTorch sparse invariant提示出现，但未中断或触发硬失败。

唯一三份实际来源指纹（启动时核验，本次审计没有再次打开原始输入）：

| 来源 | bytes | SHA256 |
|---|---|---|
| `data/sports/train_mat` | 1890060 | `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8` |
| `data/sports/image_feat.npy` | 601522304 | `222f924a0694b6c7e2bc26ea4bc2ef4ec89f7045aa5105ef463af8455c7a7695` |
| `data/sports/text_feat.npy` | 28196480 | `27d6087f0d8644b8245c60052425530c6446dffe61024b8126db0009ad94de84` |

## 2. 实际数据角色及公平范围

原Train正物品按冻结SHA划分。固定暖历史用户13011，warm_fit84695对，warm_select12671对，warm_probe12650对；warm_active13930个，751个零fit暖物品被排除且不重采样。cold_select/probe各917个候选，分别5351/5488条正例、4067/4114个正例用户、881/886个有正例物品。lock1837个ID/11584条固定用户正例仅在partition builder封存，不参与训练、选择或报告。用户不按冷标签筛选。

warm_active SHA`b8849c576aa4a4e40d665b843064d443768383cc36b283fd5f15c0e49f3bcdca`，用户SHA`c84719c11fbd0d0790153864b6b267e0fe9746c4c8c1c91d96669e9e78dc40d4`；各角色pair SHA在split_manifest中。PCA每模态64维、full solver/无whiten/行L2，拼接128维，只用warm_active拟合；图仅warm_fit，双向二值边、对称度归一化、无self-loop。新配对2层LightGCN教师维度64，ID初始化及triplet tape共享；T_MM另有8192个内容投影参数，不能称教师参数完全相同。外部预提取encoder对冷内容的历史暴露未知，不宣称端到端encoder严格未见。

T_CF冻结P/Q作为R/V/D/K_CF/K_MM公共暖评分和用户坐标。R为带不惩罚截距的ridge（λ.001/.1/10）；V为归一目标能量向量回归；D为固定P_CF的BPR；K_CF/K_MM为同结构MLP的BPR+标准化教师margin MSE，β.1/1。MLP128→128ReLU→64。两个KD臂各4个配置，D/V各2个，R3个/N1个；全体搜索预算并不相等。N为暖历史内容均值的完整内容评分参考，暖评分也不同，不能作为同P/Q因果臂。

warm Q能量0.089748530，σ_CF2.437380852、σ_MM2.668835046，均未触发退化门槛；scale tape100000，排除无负例用户0。BPR正负triplets和教师目标均仅warm_fit；cold_select标签用于选模/校准，不进入梯度。旧全目录教师、PCA或缓存未复用。

## 3. 选择与报告来源

教师lr[.0003,.001,.003]，warm_select Recall最大、再早epoch/小lr。表中probe只在全部学生/校准选择冻结后计算。

| 教师 | lr | 选中epoch | warm_select R20 | warm_probe R20 | 检查点 | SHA256 |
|---|---|---|---|---|---|---|
| T_CF | 0.0003 | 220 | 0.077578723 | 0.084031621 | `teachers/T_CF_lr0.0003/selected.pt` | `504716b6e4989a3e704fcd8ca39728c66720e701886e2ebf43561853bcc19b00` |
| T_MM | 0.0003 | 120 | 0.079314971 | 0.083399209 | `teachers/T_MM_lr0.0003/selected.pt` | `167cbdb1b32e8a0f1b4529188528b1167a1d83399e0270bf2163bf9e14e6aa3b` |

学生lr[.0003,.001]，混合select总体R20优先，其次冷R20、早epoch、小lr、小β/λ；原始选择固定后，仅对该检查点搜索冷系数.5/1/2，暖分数不变。下表β/λ分别按臂解释，N无训练参数。没有按probe改选epoch或系数。

| 臂 | lr | β或λ | epoch | mixed_select R20 | cold_select R20 | 校准系数 | 检查点/来源 | SHA256 |
|---|---|---|---|---|---|---|---|---|
| R | — | 0.001 | 0 | 0.064761308 | 0.000000000 | 1 | `students/R_lambda0.001/selected.npz` | `81f97ceef4668e57965b25d175a36df3a507069f65fa7cd66944243cbf285e11` |
| V | 0.001 | 0.0 | 240 | 0.064924438 | 0.000942546 | 1 | `students/V_lr0.001_beta0.0/selected.pt` | `ca132d4cc47af4585055f693679cb5aa25f6f7ddc7942cd0b0c0b0c0a04c91a3` |
| D | 0.001 | 0.0 | 10 | 0.065563908 | 0.007827227 | 1 | `students/D_lr0.001_beta0.0/selected.pt` | `5f8999de0145244dc9c5d9e15ffa63be5a1422f7743fb882b3d64f47c3d8b89d` |
| K_CF | 0.0003 | 0.1 | 20 | 0.064023961 | 0.020592574 | 0.5 | `students/K_CF_lr0.0003_beta0.1/selected.pt` | `97695d3984ccda2dc9d1a01ff56e015ac823b51f58e555653b0abcc22cb474f7` |
| K_MM | 0.0003 | 0.1 | 20 | 0.064428523 | 0.019486108 | 0.5 | `students/K_MM_lr0.0003_beta0.1/selected.pt` | `3e846289ec593f0e40419b048d99a44eb33356ee1fd7a7000a8bb908c6d7bd6a` |
| N | — | — | 0 | 0.027017685 | 0.021182690 | 2 | `content-reference; no learned checkpoint` | `2b786220a1f52877d6ee4ec51e54bf76ea664ebc589bae8d1a6e56e0791cf736` |

两KD臂均选择lr.0003/β.1/epoch20，D为epoch10，V为epoch240；每个配置依旧训练完整300轮并保留best/final/curve。末尾5次选择范围检查如下。这只是预定有限平台检查，不证明全局最优或不存在早期峰值/后期退化。

| 选择配置 | 末尾范围 | 验收 |
|---|---|---|
| T_CF | 0.001262726 | ≤0.002，通过 |
| T_MM | 0.000631363 | ≤0.002，通过 |
| V | 0.000117454 | ≤0.002，通过 |
| D | 0.000849581 | ≤0.002，通过 |
| K_CF | 0.000630334 | ≤0.002，通过 |
| K_MM | 0.000711247 | ≤0.002，通过 |

选模文件封存身份`0668df48fb97db898b5916e5765c69734085777929d63a852b22c5250f6d9740:1cb24b2cab2f8c1982b2f585ec34d7d1f9b5ffef846c37852ab547aad2779a84`。审计逐项核对6/16配置网格、30次选择/300轮优化、best epoch/global tie规则、校准仅依附raw选择、所选checkpoint SHA及封存事件。报告函数从所选state的固定副本生成相同暖表/内容函数，之后没有优化或改选；这支持“选中=报告来源”。审计未加载checkpoint再打分，因此不声称独立重构报告分数或逐参数证明被测模型。

## 4. 访问时序、排名与数值核查

1. 源码白名单/manifest只记录originalTrain、image、text三份输入。originalTrain中lock接触仅builder按固定规则划分封存；之后没有重新加载原Train。
2. teacher/student/calibration均在selection phase，probe角色拒绝访问。一个`selection_sealed`事件绑定上述两个文件SHA；在它之前无warm_probe/cold_probe事件。
3. seal后只report：2次教师暖probe+12次学生混合probe，warm_probe label事件14次/cold_probe12次；没有后续选择集事件。原Val0、原Test0、sealed lock下游重开0。这是应用guard/event与源码证据，不是OS级审计计数。本诊断不使用第一项val_test_once_v1，不能把internal probe填为正式Test。
4. 每个评分状态float32全候选精确Top20，边界tie按原ID升序、warm_fit屏蔽；组Recall用各组正例用户分母，总体用正例并集用户分母。cold-only从同一评分批次取冷子目录，仅辅助。
5. 另写无Torch/无输入读取/无评分调用的`exp/innovation2/sports_unseen_transfer_v1_audit/verify.py`，**567项通过**。核查全部非lock产物SHA、选模与事件、数组有限性/用户SHA/合法唯一Top20、曝光与分组hit/denominator/Recall/DCG代数、汇总NDCG、全部30个配对区间、资源与冻结decision重建。bootstrap2000次、seed20260930、用户配对百分位95%、linear quantile；不以它证明跨seed稳定性，未作多重比较校正。

保存的Top20/命中数组没有完整评分、正例标签和fit屏蔽集；审计没有为了重构它们重开原Train或重新排名。因此Top20正确排序、实际标签和屏蔽的外部独立核验仅有源代码/前期合成测试支持，保存数组验证只能确认内部一致性。不得将567项夸大为独立重算真实ground truth。

## 5. 预定判断与诊断意义

样本门槛和全部末尾范围均满足，但T_MM暖probe0.083399209低于T_CF0.084031621，差-0.000632411（1055对1063暖命中），未达+0.001教师额外能力前提。程序和另写核查均得到**undetermined**，不能转记screen_stop或宣布整个未见物品任务失败。没有算法创新成立结论；[用户确认的创新标准](../../../THESIS_ROADMAP.md)不变，此次不足来自贡献效果/教师对照，未因基础loss已存在而否决。

以下为K_MM减各对照；全部同variant配对，warm/overall列是差值，不是与教师暖目录直接比较。

| variant | 对照 | 冷R20差 | 冷差95%区间 | 暖R20差 | 总体R20差 |
|---|---|---|---|---|---|
| raw | R | 0.026920272 | [0.022322152, 0.031782227] | -0.005849802 | -0.000088812 |
| raw | V | 0.026555664 | [0.021917031, 0.031397363] | -0.005849802 | -0.000154115 |
| raw | D | 0.017622752 | [0.013855129, 0.021552423] | -0.004584980 | -0.000568137 |
| raw | K_CF | -0.004172743 | [-0.006685505, -0.001782531] | 0.000790514 | 0.000045712 |
| raw | N | -0.002815589 | [-0.009460582, 0.003686599] | 0.048142292 | 0.039000153 |
| calibrated | R | 0.000000000 | [0.000000000, 0.000000000] | 0.000000000 | 0.000000000 |
| calibrated | V | -0.000364609 | [-0.000972290, 0.000000000] | 0.000000000 | -0.000065303 |
| calibrated | D | -0.009297521 | [-0.012011830, -0.006643980] | 0.001264822 | -0.000479325 |
| calibrated | K_CF | 0.000000000 | [0.000000000, 0.000000000] | 0.000000000 | 0.000000000 |
| calibrated | N | -0.149811619 | [-0.159722340, -0.140216183] | 0.082134387 | 0.040058126 |

raw K_MM比D冷Recall高0.017622752，但暖Recall低0.004584980，暖差区间[-0.005770751,-0.003399209]，超过允许代价0.001；比同预算K_CF冷Recall低0.004172743且区间全负，额外多模态收益未建立。calibrated K_MM/K_CF均选择0.5，cold probe命中0；该选择在select总体指标上事先发生，不是看到probe后退回暖目录。即使不应用教师前提，raw/calibrated均不能通过所有对照及质量门槛，但这不能改写冻结decision。

冷内部排序有信号（K_MM0.145254116、D0.159619513），进入混合目录后K_MM为0.026920272、D0.009297521；这种差异和KD缩放/暖质量代价提示冷暖竞争需要进一步解释，**不证明分数尺度是唯一因果瓶颈**。N系数2冷Recall0.149811619但暖Recall0.001897233、总体0.028576777，直接展示“更多冷曝光”并不保证总体价值。

本次只有一个固定划分/seed、一个warm-only教师结构、有限LR/β/λ/校准范围；calibrated搜索条件于raw最佳，未遍历所有模型×系数组合。预提取encoder暴露未知；用户暖历史资格限制；N参考坐标不同；搜索预算不完全相同。故不能排除更强教师、其他预声明搜索或不同场景的机会，也不能凭本次冷内部排序宣布新方法有效。后续任何更改/执行需要独立声明与授权。

## 6. 资源、产物与指纹

实际5496.547s（91.61min）；assets44.125s、教师2121.406s、学生含目标/选择3249.500s、report81.438s，全部在48h/2h/24h/18h/4h预定上限内。5s采样RSS峰2.488GiB、CUDA allocated0.152GiB、最低空闲370.032GiB；最终输出254568974 bytes，低于8GiB。采样/合作检查不是OS硬隔离，CUDA allocated不等于整张显卡总占用。

917个冷向量单次生成计时R/V/D/K_CF/N记录0.000s、K_MM0.015s；平台时钟粒度与单次小批量测量使这些数值无法支撑速度优劣或零成本主张。全目录报告计时是诊断成本，不是正式部署benchmark；未追加成本实验。

| 产物（均在run根） | bytes | SHA256 |
|---|---|---|
| `report.json` | 2692434 | `41cccd5a23d3368ee80ccdee38539b699dea8788cdd98a76994bf8de459781b6` |
| `batch.json` | 11929 | `9d4861d0089a230db2b76613443b6b5e3c1f1173dfda0f4e9db25356af026764` |
| `per_user_probe.npz` | 7264693 | `e3ba290fe1e79f02d72f943d6d8ab0bbc99867baf43bdd40ff530b6ffc4ad628` |
| `teacher_selection.json` | 1698200 | `0668df48fb97db898b5916e5765c69734085777929d63a852b22c5250f6d9740` |
| `student_selection.json` | 14236589 | `1cb24b2cab2f8c1982b2f585ec34d7d1f9b5ffef846c37852ab547aad2779a84` |
| `source_manifest.json` | 1472 | `73aa1a7697209627fcd4de12a1f25fae08ad4381b216b0da97db5f94edda4170` |
| `split_manifest.json` | 1410 | `06c1509d572f23c9b8969fbf9cbbcfcb5640c4e5afa4bf12513477e208092efd` |
| `transform.npz` | 2233807 | `349047a3e398cfa2d63b7aa3d4d22c571d589c8655bdf7c359b1bc44b6a231c2` |
| `transform_manifest.json` | 858 | `9dc0c3465bd3dfe80a3967386e9ea2767a04c48aa32da6351aa0b16f21d01e4f` |
| `graph_manifest.json` | 277 | `6091121964597588054e6419d3905dae0bf7f06c43f3641febee0355c567b5d4` |
| `target_manifest.json` | 351 | `db4a2eed22296f66ce38d25461eda3b5fd467143e11265fd071f93231cf3ae27` |
| `access_audit.json` | 182465 | `064d6a273ee315f1fe1d954c429e009b7e462210ccc6c07eb105c20a405f920f` |

sealed_lock SHA`bffb8a4120ae679969f964dab17a8b14e591661b048bc7a6fe6aa814f38e78d8`仅引用写入时缓存，**没有为审计重新打开或重哈希该payload**。全部fit best/final/curves及冷向量身份由batch/report索引。审计脚本SHA`a436e7478efea24e5c672c2c10d8247f937801b6902e73f6da2dd5500fd21f4a`，verification JSON SHA`dced5832a0b4a0a39cb5566b8ca8b13b1de8d6a8d1111e751d1f1b66d881d39e`；它们在独立忽略audit目录，不修改run的原始batch/selection/report。

## 7. 记录路由与收尾范围

| 目标 | 处理 |
|---|---|
| 本审计/14行seed2022矩阵/HANDOFF | 新增，指标与限制如实记录 |
| TRAINING_LOG | pending原文保留；追加completed及导航guard结果，更新短当前状态 |
| root README/innovation2 README/experiments README | 完成状态、结果/审计入口及唯一讨论步骤 |
| docs/README | 核对，稳定第二项入口仍有效，无需重复改写 |
| 第一项正文/矩阵/gap、THESIS_ROADMAP、旧审计/声明、冻结profile与runtime | 无变化；第一项冻结，未形成新论文主张 |
| 6个生成导航文件 | 已在sealed读取拒绝guard下生成；3个封存条目仅path/bytes，最终证据说明后再刷新元数据 |
| 原始输入/输出/旧基线、评审untracked文件 | 保留原位，不提交大产物、不覆盖/删除 |

运行中为必须的closeout识别并事前记录导航guard amendment：旧catalog遍历exp JSON会打开sealed_lock；**在任何post-launch catalog刷新之前**将该文件名改为metadata-only。仅修改builder/两个针对性测试/log，不改已加载runtime/profile；两测试通过，三份运行源SHA与launch一致。138条本地链接、14矩阵行、Git-canonical旧日志历史保留、runtime/profile/charter不变和diff空白检查通过。最终同一结果commit保存此修复与收尾；hash见Git/最终交接，不为写入自身hash循环追加commit。

无merge/main/tag/bundle/push、无重跑/恢复/换seed、无备份里程碑承诺。Git保护跟踪源/文档，不保护忽略的raw资产；本次仅确认它们留存于本机原路径。工作树保留既有untracked archive/reviews/。

唯一下一步（可复制）：**“依据Sports未见物品诊断v1，重评第二项路线：解释教师优势缺失和冷暖质量冲突，讨论是否继续未见物品方向；只评审，不运行实验。”**
