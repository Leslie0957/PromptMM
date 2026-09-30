# 质量约束前沿：实现与CPU合成预检

2026-09-30。起点提交`c3adf8eb7e6cb04ca7df3abb641bccfe261dc039`，分支`codex/experiment/baby-teacher-baseline`。本次只完成实现与非正式合成验证，没有真实Sports诊断结果。第一项、论文路线v1.1、原诊断结果及固定profile保持原样。

## 当前结论与执行边界

独立评分器已实现，小型CPU预检通过。真实运行未授权，正式声明仍未创建，`exp/innovation2/sports_quality_frontier_v1`仍不存在，当前待跑正式实验为零。

[冻结协议](QUALITY_CONSTRAINED_FRONTIER_PROTOCOL_V1.md)及[profile](QUALITY_CONSTRAINED_FRONTIER_PROFILE_V1.json)的“未实现”字段是协议冻结时的历史快照，本页与活跃TRAINING_LOG记录后续实现状态；不改动冻结搜索范围、门槛或原结果。profile规范化LF SHA256：`a665ed386d58c47067fc33d04304f06752d814b07b0eb8077c524367dfd986cd`。

本次原Train、图像/文本数值、旧模型、原Val/Test、旧probe与sealed_lock均没有读取或哈希。元数据预检仅stat了11个复用资产与3个原始输入。应用访问审计不是OS独立追踪，不能把它表述成进程隔离。

## 实现

- [独立数学与角色模块](../../../tools/quality_frontier_core.py)：按原SHA分区恢复fit/warm_select/cold_select；暖留出rank1直接丢弃，不构造probe/lock标签数组，不导出其统计，不写新seal。身份核验包括固定用户/暖物品与三个允许角色的计数、SHA；越界访问拒绝。
- [CLI评分器](../../../tools/run_quality_constrained_frontier.py)：默认只检查文件元数据；真实模式必须通过已提交声明、明确人工授权、干净源码、锚点/命令/预算/边界/路由一致性及独占输出检查。真实模式未来会先核验允许资产，再恢复角色、读取保存的CF/PCA/固定学生状态；不拟合、不重新选checkpoint、不访问原MM教师表。
- 保存的PCA沿用当前原库先投影、后减均值投影的计算顺序，避免破坏N查询指纹；两模态分别归一化后拼接。N查询按暖历史均值归一化，N物品表采用原内容参考的完整行余弦归一化。未来真实运行必须核验固定N查询SHA。
- CPU线程4；暖基础评分仅2次（五个核心臂共享CF暖缓存，N单独缓存），冷基础评分6次。暖分数的总体标准差来自全部固定用户×暖候选、fit屏蔽之前的float32分数，在float64下累计。缓存只保存分组TopK。
- 在全部63组正仿射候选下，检查所有相邻不同冷基础分数的严格次序，覆盖TopK边界；数值舍入合并不同分数时硬失败。通过后以float64合并暖/冷Top20，按原物品ID破同分。没有保存全用户×物品分数矩阵。
- 全部378候选与2暖参考逐状态排名一次；D/raw提前计算后复用。参考使用同一混合标签分母，不进入搜索。双质量约束、固定选择tie、有限三维非支配前沿、四预算嵌套检查、不可行显式状态及去重逐用户记录均实现。
- 保存冷单组Recall、冷槽位比例、冷曝光HHI、冷向量范数、可行候选比例、阶段时间，以及固定选择后的最多36项配对描述性区间。组内共用重采样索引；真实bootstrap2000次。低样本或sigma缺失写development_undetermined；无可行候选不制造零冷收益赢家。
- 真实资源保护沿用CPU2h/RSS6GiB/output2GiB/freeDisk12GiB及阶段上限，5秒采样配合循环时间检查。它不是OS强制隔离。发生失败保留batch/access与当时产物，拒绝覆盖、自动重试或推进下一阶段。

## 验证及保留产物

命令均从仓库根执行；以下属于非正式CPU验证，不使用真实数据。

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B -m unittest discover -s tests -p test_quality_frontier.py -v
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_quality_constrained_frontier.py --synthetic --output exp/innovation2/synthetic_quality_frontier_cpu_final_v1_20260930
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_quality_constrained_frontier.py
& 'D:\miniconda\envs\run_5060\python.exe' -B exp/innovation2/verify_quality_frontier_preparation_20260930.py
```

**不要重跑已有fixture命名空间。** 原始stdout另存同目录前缀文件；所有合成产物均为Git忽略资产，源码提交不备份它们。

| 验证 | 实际结果 |
|---|---|
| 12项单元检查 | exit0；纯分区参考一致、越界拒绝、全63点全量排名对照（含fit掩码/同分/边界）、舍入坍塌拒绝、分母/NDCG、质量选择tie、Pareto等价点、PCA运算顺序、共同bootstrap、声明拒绝、决策顺序、零sigma、覆盖/资源拒绝 |
| 最终CPU fixture | completed，380排名状态，0跳过；36配对区间，13个去重逐用户状态；0.641秒运行主体，RSS采样峰值502,669,312字节，产物427,813字节 |
| fixture delta | 16用户/160物品；内容8维、隐藏16、输出6、K5、batch8、bootstrap40；随机固定状态，不训练/PCA拟合；每次总120秒/output512MiB/freeDisk2GiB保护。真实规格保持不变 |
| 科学标记 | development_undetermined（样本不足）；scope=synthetic_no_Sports_conclusion，innovation_certified=false；合成数值不填入论文或Sports结果矩阵 |
| 产物审计 | 343检查通过：所有产物SHA/大小、最终源码绑定、24选择格、嵌套预算、保存数组聚合与共同bootstrap独立重算、缓存/访问与输出身份；不重新排名 |
| 元数据预检 | 14资产存在/大小符合；payload load/hash0，正式声明与真实输出均不存在 |
| 故障注入 | 独立synthetic_quality_frontier_failure_v1_20260930退出1；角色恢复后按预期失败，batch/restoration/model/fixture/access/资源记录保留；未重试或删除 |

首次`synthetic_quality_frontier_cpu_v1_20260930`已成功并保留。随后修正零sigma的D/raw计数并提取纯声明校验，按日志amendment使用独立final命名空间验证最终源码；没有覆盖首次结果。故障路径未变，未重复故障fixture。非正式验证发生于提交前，最终源码以内容SHA绑定，再作本地提交；没有从未提交源码发起正式运行。

最终fixture batch SHA：`53a65b961aac51df1c8c72b14cd29a353481f1bd43c2682460d5b11eedc3b1ad`。

故障fixture batch SHA：`55ae4be567687205b741554c42b77111442852ca21b647ac8f9d0e25e3b00430`。

最终评分器SHA：`824279111354b93f4b11b0354d46572ecd1475d6a06f59a446847aa805aa2509`；数学/角色模块SHA：`922a2b96f0bdd56fcce18225572f015fa5c94d76394529b2bfb5b38b832efa20`。核验记录：`exp/innovation2/quality_frontier_preparation_verification_20260930.json`。

## 局限与记录路由

小型CPU验证不能保证真实规模速度、内存、PCA/角色/N查询身份或科学效果；这些仍是未来真实执行的硬门槛。合成质量约束、Pareto前沿与bootstrap只验证运算，不提供独立推荐收益证据。真实数据只有已开发select，选择后区间不构成独立显著性；良好开发结果也不自动认证第二创新点。

| 目标 | 本次处理 |
|---|---|
| TRAINING_LOG | 事前pending、合成验证amendment、独立completed及当前状态 |
| 本页、根README、第二项README、实验族导航 | 更新实现/预检证据和唯一下一步 |
| 六生成目录 | 人工编辑完成后，sealed读取拒绝条件下刷新 |
| docs/README、正式结果矩阵、论文/第一项gap | 入口稳定或没有真实结果，无需修改；不造Sports数值格 |
| 原v1代码/profile/声明/结果/审计、路线与论文charter、冻结前沿profile/协议 | 保持不变；新状态由本页和活跃日志承载 |
| 原始资产、未跟踪archive/reviews | 不改动、不提交；本次不是标签/bundle/物理备份里程碑 |

## 唯一下一步

准备本诊断的一次真实执行声明供用户审核，绑定本次实现提交、冻结profile/复用锚点、24格结果路由、CPU上限及one-launch/no-retry/no-next-stage。必须明确未来原Train分区重读会接触整个容器中的probe/lock源字节，只有允许标签导出；不得把它写成零源接触。

未来接口（**本次不执行，正式声明和真实执行授权均缺少**）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_quality_constrained_frontier.py --execute --declaration docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_RUN_DECLARATION_V1.json
```

AGENTS的正式运行门槛要求明确授权、已提交精确源码/声明及一次启动；“下一步”本身在当前阶段只指声明准备，不把历史训练授权转移到这项新诊断。
