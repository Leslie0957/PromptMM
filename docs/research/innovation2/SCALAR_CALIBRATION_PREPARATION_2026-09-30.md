# 全局／逐用户标量校准：固定协议与启动准备

2026-09-30。承接[立项评审](PERSONALIZED_ALLOCATION_DESIGN_REVIEW_2026-09-30.md)。准备已完成；未打开真实数据/检查点/候选张量，未运行真实拟合或评价。身份是简单基线，不是已认定第二算法。既有H诊断链已结束。

## 单次范围与身份

Sports三个既有B学生状态seed2022/2023/2024，各比较B（冻结原表）、G（一个全局标量）、U（每用户一个标量）。六次标量拟合、九个Validation状态，不重训B、不启动新学生seed、不恢复R/N。采用现有300轮B的final，而非best；标量只取固定求解终态，不早停或按Validation选轮。

[profile](SCALAR_CALIBRATION_PROFILE_V1.json)冻结Train/Validation及图身份、B final/status/Top20、原全局加分生成的B两组候选缓存SHA。基础训练anchor为neighbor_shared_residual_three_seed_v1/manifest.json，来源b8be549613498e5cbb98412ffa766c62279ad7a6；缓存身份沿用已审计全局加分和槽位控制记录。图文件只用于读取S/degree，不用邻居内容或模态特征。R/H列表与结果不是运行输入。

分支codex/experiment/baby-teacher-baseline，launch_source_rule=committed_head；本次准备提交（或仅追加同一协议声明的提交）作为启动HEAD，工作区须干净，原有未跟踪archive/reviews例外。运行manifest记录精确HEAD、profile哈希及完整配置。资产由运行时预检校验；准备不重复读取大资产。环境D:\miniconda\envs\run_5060\python.exe；CPU，torch仅加载冻结表，NumPy执行求解，禁CUDA；运行记录版本。

## 唯一固定目标与求解

s'(u,i)=s_B(u,i)+a_u h_i；S为已钉定6114物品集合，h_i=1[i∈S]。G约束全部a_u相同，U不共享标量。B不更新，组内次序不改变。

每用户64对：正例在Train已观测项中均匀有放回抽样，负例在Train未观测补集中均匀有放回抽样，统一NumPy default_rng种子8721。共2,278,272对，三个B状态及G/U使用同一份保存的tape。未观测不等于真实负偏好。无过采样、无曝光匹配、无人工固定正加分。报告跨组两个方向数量及无跨组用户数。

m_ut=s_B(u,i)−s_B(u,j)，d_ut=h_i−h_j。目标固定为

L(a)=(1/n)Σ_u[(1/64)Σ_t softplus(−m_ut−a_u d_ut)+0.01 a_u²]。

训练margin使用加载的FP32表转FP64后的内积；不改变基础参数。G与U采用相同用户权重与正则归一化。这里只拟合Train；基础B已经见过Train，不声称训练外效用估计。有限采样和λ=.01是固定探索选择，不主张最优。

导数为mean_t[−d_ut sigmoid(−m_ut−a_u d_ut)]+0.02a_u，二阶导≥.02，所以各用户严格凸；G对用户求平均同样严格凸。数据梯度绝对值≤1，根在[−50,50]。固定40次导数二分，区间宽最多100/2^40；G先平均用户导数，U逐用户求解。数据项无跨组信号的用户解0。检查终态最大驻点误差<1e−8、U目标≤G目标≤B目标（容差1e−10）。这些是数值验收，不证明泛化。无学习率、AdamW状态、训练epoch或损失创新。

## 排名、封存和数据边界

沿用B全候选打分得到的每组Top20缓存，合并为总Top20，组内FP32缓存分数按FP64比较有符号加分，并列按item ID升序。这与对同一已缓存B分数加常量后的全候选Top20集合等价，不需要全用户×全物品分数矩阵。FP64训练margin与原CUDA FP32缓存分数数值路径不同，明确保留该区别；不声称跨精度完全同分。零加分必须逐位复现原B Top20。

对三个seed先全部拟合并保存offset与B/G/U列表；全部封存哈希及时间后才能打开Validation。Python访问白名单在此前仅允许Train，在此后允许Train/Val，Test始终拒绝；不是OS级访问隔离保证。旧B status仅在封存后打开作回归。禁止用Validation标签选择a_u、λ、采样或终态；不访问Test。

检查候选身份/排序/组别、用户一致、列表20项唯一且排除Train已见项、S属于低组。复算B并回归旧final Recall≤1e−6且组命中一致。仅主要使用Recall@20及三频次组绝对命中/曝光、S/nonS/low_nonS；分母6347/6389/25163。保存U−B、U−G、G−B逐用户/物品配对明细并核对净差。没有新NDCG或Test结果。

## 事前描述性筛查

唯一candidate_signal条件：三个seed分别同时满足U低组命中>B且>G，并且U总体Recall≥B且≥G。其余screen_stop，同时如实报告取舍。这是本次新协议的严格描述性条件，不沿用H的0.0002容忍线，不是显著性/非劣/等价；即使通过也不认证算法新颖性。没有测试结果后放宽条件、换λ/采样或追加seed的队列。

需要承认：G/U总曝光可能不同，所以U>G不唯一证明同等曝光下分配更好。原H控制支持那个窄问题，本次检验的是独立训练出的简单策略实际表现。若通过仍需已有个性化校准对照与独立泛化证据，不能自动开始下一轮。若停止，只否定本固定采样/正则/基线，不否定所有个性化方法。

## 成本与资源

冻结表约13.2MiB FP32；FP64副本约26.4MiB。单份margin约17.4MiB、两份int32采样ID合计约17.4MiB、delta约2.2MiB；缓存与临时数组、torch运行时和原checkpoint其他字段另外计入。逐块256用户计算margin；不缓存全评分矩阵。二分G/U各40次、三个seed，总计240次全tape导数遍历。

CPU cohort墙钟上限1800秒、worker RSS4GiB、输出256MiB、可用盘≥2GiB。父进程每秒采样，worker每阶段/求解迭代检查；资源超限终止并保留失败，不自动重试。采样不保证捕获瞬时峰。**没有真实资源smoke，无法证明完整运行会在30分钟内完成；30分钟是终止上限，不是耗时预测。**

U部署附加n_users标量与item指示，可显式加分或拼成65维内积；输出只保存标量和列表，不制造新全表checkpoint。相比64维B非零成本；R本就可折叠ID表，不宣称比R线上更省。求解、基础B生成/训练、缓存准备分别计账，不能把只拟合标量的耗时与完整模型训练等同。

## 验证和执行入口

新增tools/scalar_calibration_core.py、tools/run_scalar_calibration.py、tests/test_scalar_calibration.py。7项合成测试通过：有限差分导数、凸解/符号/零信号、极端margin、可复现采样及排除已见、合并与全排序/单调名额/负加分/并列/缺项、65维代数恒等、数据屏障及筛查边界。help和语法检查通过。测试只用构造数组，未读取真实资产。未完成真实端到端资源验收，这只能由一次授权运行验证。

用户决定执行后，从仓库根目录运行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_scalar_calibration.py' --formal
```

默认无formal拒绝执行；输出exp/innovation2/scalar_calibration_v1存在即拒绝，失败也不允许自动覆盖/重试。父子进程绑定；源码脏则拒绝。仅这一固定cohort；禁止另一个seed/arm、Test或后续阶段。用户尚未在本准备任务授权实际运行，本页提供手动入口。

## 记录更新目标

运行产物：manifest、train_pairs、每seed fit/lists、fit_seal、求解轨迹及报告、配对明细、acceptance、worker_resources/supervisor/completeness。原始资产不改。

审计路由：本目录SCALAR_CALIBRATION_RESULTS.md（3seed×B/G/U矩阵）、SCALAR_CALIBRATION_AUDIT.md、SCALAR_CALIBRATION_HANDOFF.md；追加TRAINING_LOG outcome、实验总览“标量校准基线”行、第二项/根当前状态和生成目录。第一项矩阵/正文与总体路线不变，除非另有明确主张触发；无tag/备份冻结里程碑。有效低结果记completed，执行/硬验收失败才记failed。

唯一下一步：用户手动执行上述固定cohort一次，返回后审计；本准备阶段不代为启动。
