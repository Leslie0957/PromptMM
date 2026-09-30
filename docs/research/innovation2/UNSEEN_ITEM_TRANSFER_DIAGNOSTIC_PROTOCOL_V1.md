# Sports未见物品教师可迁移性与混合目录瓶颈诊断 v1

状态更新（2026-09-30）：[独立运行实现与CPU合成预检已完成](UNSEEN_ITEM_TRANSFER_RUNTIME_PREPARATION_2026-09-30.md)，用户随后已明确授权既定单次cohort，正式声明/pending见[活跃日志](../../../TRAINING_LOG.md)。下文“当前”及JSON状态保留协议冻结时的快照；实验参数不变，真实进度与结果以活跃日志和独占输出batch为准。

2026-09-30。用户授权准备这项诊断；本次冻结协议和[机器可读配置](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROFILE_V1.json)，不执行真实数据构建、训练或评价。当前状态为 **protocol_frozen_runtime_not_implemented**，不是可立即开跑的训练包。按[用户确认创新标准](../../../THESIS_ROADMAP.md)判断具体差异与价值，不要求基础工具原创；第一项继续冻结。

## 1. 问题与本阶段产出

在暖物品专属训练资产、相同内容输入及内容模型容量下，多模态教师是否提供超出直接内容学习、协同教师迁移及简单校准的未见物品收益？加入暖目录后，这种收益是否保留？

这是单seed探索性机会诊断，不是新的算法或学位创新认证。线性/MLP回归、BPR、评分KD和校准都作为已有工具使用。如果有效，可以进入具体组合/适配方法设计；如果表现不佳，记录有效负证据后与用户讨论，不自动加模块或转跑备选。也允许“未判定”，避免将教师不够强、容量限制或优化不足误写成没有迁移信号。

参数以JSON为执行规范；本文给出数据角色、公式、含义和准备边界。任何影响划分、目标、预算、选模或评价的修改，须在结果出现前版本化声明，不能靠低指标事后改变。

## 2. 源资产与禁止复用的路径

只允许原Sports `train_mat`及预提取图文特征进入新诊断。来源锚点为 `data/sports/conversion_manifest.json` 的output身份；JSON保存三个允许载荷的bytes/SHA256。已有元数据为35598用户、18357物品、218409条原Train交互，图像4096维、文本384维；这是本仓库特征，不替换为其他论文的1024维文本。

运行准备阶段必须重新核验这三个允许文件与锚点一致；本次仅读转换元数据和文件大小，未重新哈希或加载载荷。旧 `val_mat`、`test_mat`、包含全部split的 `sports.inter`均禁止加载。元数据含历史Test统计不等于本次读取Test载荷或进行Test评价。

源码检查发现：[通用loader](../../../codes/utility/load_data.py)会同时加载旧Train/Val/Test；[硬token缓存](../../../codes/utility/hard_token_cache.py)对传入整表fit。故后续必须实现独立Train-only loader及warm-only变换器，**不能直接导入会实例化旧Data的训练入口**，不能复用原全目录教师、缓存、PCA/ICA或从旧教师删行伪装成合法资产。[main_mmlight.py](../../../codes/main_mmlight.py)仍为第一项活跃训练线，本诊断不改它。

冻结外部特征编码器的预训练暴露未知。结论限“给定预提取特征的身份留出内容接入”，不称真实时间到达任务、原始图文端到端接入或无任何预训练暴露。

## 3. 身份划分及独立的信息角色

只从原Train正交互取得物品ID集合，不用旧Val/Test或效果选择ID。对每个ID按JSON的固定键取SHA256，按摘要字典序、ID升序排序，依次分配：W=80%、C_select=5%、C_probe=5%、C_lock=剩余10%。前三段取floor，锁定集合接收余数；seed20260930。先去重正交互，负/非有限值或非法ID为硬失败；不按流行度、内容相似度或结果重新划分。

对每个用户，仅看W交互；W度数至少5者进入U。按固定warm_pair摘要排序，前两条分别作为warm_select和warm_probe，其余作为warm_fit，至少3条。以warm_fit正度数确定W_active，零训练度的W物品从所有暖候选/暖留出中排除并报告；不移动到冷集合，不重新抽样。所有训练、选择和报告只针对U，用户不能依据冷标签筛入。warm_select/probe标签被排除后的分母照实报告，不用它们反向重筛用户。

| 角色 | 允许用途 | 禁止用途 |
|---|---|---|
| warm_fit / W_active | 教师/学生梯度、图、频次、变换器fit、内容邻居与尺度统计 | 引入冷交互 |
| warm_select | 两教师暖目录选模；学生混合选择 | 梯度、负采样过滤依据 |
| C_select内容/标签 | 学生选模及冷分数校准选择 | 教师、变换器fit、学生梯度、内容邻居训练 |
| warm_probe / C_probe | 所有选择冻结后最终探索报告 | 选checkpoint、参数、校准或重新划分 |
| C_lock | 留待另行授权的独立确认阶段 | 本诊断模型、选择、候选池、内容数值处理或评价 |

split构建器必须读取原Train才能分区，因此会接触之后归属C_lock的交互；只允许做固定分区与封存，不输出其效果或将其用于过滤/优化。封存后模型/评价进程不能再读该载荷。特征整文件身份哈希不作为数值fit；数值访问按上述角色逐行进行，锁定行不传入模型。原Train过去已用于研究，新留出不抹除研究者先验；C_probe仍是探索报告，不冒充全新正式Test。

两个选择/报告目录分别是W_active∪C_select、W_active∪C_probe。C_select不参与最终probe排名，C_lock始终不参与。屏蔽warm_fit已见项；不屏蔽选择/报告正例，不为负采样读取留出标签。训练负物品只来自W_active，排除该用户warm_fit正例。

## 4. 合法教师与共同评分空间

为保持明确、可实现的第一诊断，两教师采用同一两层[LightGCN骨架](https://arxiv.org/abs/2002.02126)：用户/物品ID维度64，同一warm_fit归一化二部图，输出为输入及两层传播表示的均值。图使用二值正边、对称度归一化，不加自环。T_CF物品输入是q_ID；T_MM是 `q_ID+(W_img*x_img+W_txt*x_txt)/sqrt(2)`，两个线性投影无bias，图像/文本分别64维。用户输入相同，ID初值/三元组回放相同；模态投影是T_MM额外参数。因此比较支持“本设定多模态教师监督增量”，不声称所有教师参数容量完全相同。

这是一对新建诊断教师，**不称PromptMM复现或把旧PromptMM强度转移给它们**。既有复杂教师/强方法可在后续需要时纳入，不在本阶段默认为必须全复现。

图文各自仅在W_active上fit64维PCA（full SVD、非whiten），保存均值和components；转换后逐模态L2归一化，再拼为128维内容输入。零行保持零并报告数量。两个教师及所有内容模型使用同一输入；C_select/probe只应用冻结变换。教师MLP/ID/dropout等细节以JSON固定。

每教师预定3个lr：3e-4、1e-3、3e-3，各300轮上限，batch1024，AdamW、weight_decay1e-5，无dropout，BPR。每10轮暖选择Recall@20，按最大值选择，tie选更早epoch再更小lr；不按probe选教师、不在选模后合并留出重新训练。共6个教师fit、一个seed2022，不自动追加seed。

固定选中的T_CF用户P_CF和暖物品Q_CF为共同锚点。所有核心内容接入组使用P_CF；暖物品一直用Q_CF。T_MM坐标与P_CF不同，因此仅提供合法暖三元组的**评分差目标**，不直接将Q_MM当作P_CF空间向量。

## 5. 学生与解释控制

| 臂 | 训练/评分定义 | 配置次数 |
|---|---|---|
| R | 带不惩罚截距的ridge预测Q_CF；坐标均方误差+lambda‖W‖² | lambda=0.001/0.1/10，3次解析fit |
| V | 128→128 ReLU→64带bias小MLP预测Q_CF | lr两值，2次fit |
| D | 同MLP，冻结P_CF，以warm_fit BPR训练 | lr两值，2次fit |
| K_CF | D加T_CF标准化评分差MSE | lr两值×beta两值，4次fit |
| K_MM | D加T_MM标准化评分差MSE | 与K_CF相同4次fit |
| N | 用户warm_fit历史内容均值归一化，与候选内容算余弦 | 固定1个参考，无拟合搜索 |

V/D/K_CF/K_MM容量与初值相同，各300轮上限，每10轮选择，AdamW、weight_decay1e-5，lr=3e-4/1e-3。D/K_CF/K_MM共享三元组tape；V每轮遍历W_active内容/向量对，不声称与BPR相同训练样本或更新数。KD有两值beta，因此总搜索次数高于D/V，不能声称整臂总搜索预算相同；两类KD的搜索次数严格配对。神经fit共12个；包括N才称有内容参考，不把随机新ID当唯一对照。D共享协同锚点，不称完全从零、无任何教师的系统；N使用自己的完整评分，属于额外参考，不计作同P_CF的因果隔离臂。

所有ID表、线性/MLP权重采用Xavier normal初始化，MLP bias为零；相同形状的共有参数使用同一独立初始化seed，不让T_MM额外投影改变ID随机序列。每轮BPR遍历一次固定seed打乱的warm_fit正边，负例从该用户未见W_active均匀抽取；无合法负例用户不采样并报告。BPR为batch内mean softplus(-margin)，不额外加独立embedding正则，AdamW衰减覆盖所有训练参数。教师每个更新以当前参数重算两层图传播；学生使用冻结教师表。每配置从同一初值重新开始，不接续其他学习率；tape随epoch固定回放。

设学生评分差m_S=P_CF[u]ᵀ(f(x_i)−f(x_j))，教师差m_T=s_T(u,i)−s_T(u,j)。固定warm_fit-only 100000三元组尺度tape（不足则按固定采样有放回），分别估计T_CF/T_MM差值标准差sigma，population ddof0；低于1e-6则优化前停为undetermined，不把病态尺度作负证据。教师冻结，目标为：

`L_K = L_BPR + beta * mean((m_S/sigma_CF - m_T/sigma_T)^2)`，beta=0.1/1。

这是同一数值规则的评分KD，不声称新损失。BPR使用原学生评分差。V目标均方误差除以warm Q_CF逐元素平方均值，使尺度可解释；该均值低于1e-12也停为undetermined。R使用JSON的显式mean归约而非默认库lambda尺度，不伪造与V完全相同容量。

所有配置仅以W_active∪C_select、warm_select∪C_select标签的总体Recall@20选模；tie按冷Recall、更早epoch、更小lr、更小beta/lambda。这里总体是用户联合正例Recall，不用冷热宏Recall无条件平均冒充总体。所有组公开选择次数，不能只给K_MM额外预算。最多360次神经checkpoint选择、3次R选择及1次N选择；教师最多180次暖选择。

每臂先冻结raw最佳配置/checkpoint，再仅对这个配置尝试冷分数乘a∈{0.5,1,2}，暖分数不变，以相同混合选择规则选a（tie优先1、再0.5、再2）。最多18次校准选择，不重新选其它checkpoint/学习率。记录raw与calibrated；校准是简单尺度干预，不冒充完整复现幅值收缩论文，也不凭此认证第二算法。

## 6. 最终报告与成本

所有教师/学生及校准选择写入student_selection/teacher_selection并封存后，才打开probe标签，逐臂每variant做一次报告。教师只评价warm_probe的W_active目录，用于核实教师额外能力，不对冷物品调用行为教师oracle。学生共同混合目录中同一排名报告：总体、暖/冷Recall@20、NDCG@20、命中数、正例分母、有正例用户数、曝光数；float32全候选精确排名，tie按原itemID升序，用户batch128。

总体Recall为每用户冷热联合命中/联合正例数，宏平均有任意probe正例用户。分组Recall各自以该组正例用户平均；不能简单平均两个组冒充总体。分组NDCG用共同Top20中该组gain、除以该组理想DCG；总体使用联合gain与联合IDCG。冷目录单独排名仅为辅助尺度竞争诊断，由同一评分在报告阶段生成，不替代混合目录终点，不用来调a。

配对用户bootstrap2000次、seed20260930、95% percentile区间（2.5%/97.5%，线性分位数），报告K_MM对每个对照的差值；冷终点用共同的有冷probe正例用户，总体/暖用各自固定人群。同一终点各比较共享重采样索引；这是逐比较探索区间，没有多重比较校正。不能把一个seed的bootstrap当训练seed稳定性，也不宣布全领域显著/SOTA。probe后不选择新变体或修改阈值；比较所有已冻结对照，不挑最弱者。

保存向量norm及分数分布、曝光集中度、raw与校准变化、各fit loss和梯度范数/有限性、选择曲线和最后5个选择点范围。暖向量拟合只能称合法暖目标拟合；冷物品不存在行为教师真值，禁止拿旧完整教师作冷真值。

资源仅为前瞻保护上限，不是已测耗时：本机只读设备查询RTX5060、8151MiB；单GPU串行，CUDA allocated≤6GiB，RSS≤12GiB，free disk≥12GiB，output≤8GiB，总墙钟≤48h。阶段上限资产2h、教师合计24h、学生合计18h、报告4h，threads4，5s遥测；未使用旧六小时草案。CUDA allocator与RSS限制不等于OS硬隔离；实现须记录OS GPU使用和资源触发。任一上限触发保留partial/failed产物并停，不自动续跑、换seed或扩大预算。

记录真实教师重建、变换、目标/cache准备、学生fit/选择、最终报告、给定特征后的冷向量生成及混合服务成本。服务小计不替代完整生命周期；内容N亦计准备/评分成本。输出身份包含代码commit、环境版本、源SHA、角色ID/交互SHA、暖图/变换SHA、teacher与选中checkpoint SHA及选择时间顺序。

## 7. 预先冻结的判断规则

阈值为**本诊断的机会筛查约定**，不是理论必要值、正式非劣标准或由旧Test推导。最终按事实保存结果，科学screen_stop仍是有效completed；只有执行/硬条件失败才叫failed。

先检查有效性：集合互斥、零旧Val/Test/锁定确认访问、合法暖上游、有限loss/score/grad、正确mask/分母/选模；冷probe至少200物品、1000用户、2000交互，warm_probe至少1000用户。样本不足不重新抽样，记undetermined。teacher/学生各选中配置最后5个选择点Recall范围≤0.002作最低平台检查；明显仍增长、全部拟合病态或尺度检查失败则undetermined，不称收敛已被证明。

| 结果标签 | 冻结规则与含义 |
|---|---|
| support_transfer | T_MM暖probe R20比T_CF至少高0.001；在相同variant下，K_MM冷混合R20比R/V/D/K_CF/N各至少高0.002，配对区间下界各>0；总体及暖R20相比各对照下降都不超过0.001；样本/平台/硬验收均有效。仅支持本设定有机会，尚未证明多seed稳定或已成立新方法 |
| calibration_signal | raw没过上述机会门，校准后通过，或冷-only改善而混合不改善且预定校准明显改变差距。报告为尺度竞争线索；不得据此声称语义迁移机制成立或标准校准自动无创新，后续按整体差异/价值讨论 |
| screen_stop | 有效教师额外能力/优化检查通过，但raw和校准均未过门；相对D/K_CF/最强简单参考的冷增量95%上界都低于0.002，或明确的质量代价超出筛查容忍且区间支持。暂停本资产/容量/锚点设定，交用户讨论，不否定全部冷启动研究 |
| undetermined | 教师额外能力不足、样本/平台/尺度检查不满足，区间跨机会门、不同对照得失混合或强参考尚不足。保留不确定性，不自动加模块/扩跑；列出具体缺口再讨论 |

calibration_signal若只是描述性线索而未满足support全部条件，应明确没有通过迁移机会门。若K_MM不赢但N/D较强，优先报告“本诊断迁移必要性未显示”，检查匹配假设而非自动归因内容无信号。若效果支持，下一阶段也要明确方法整体差异与价值，近邻ALDI/SiBraR等需在实际主张范围对照；不回到基础工具必须原创的旧门槛。

## 8. 实现与运行准备边界

当前仅有协议、JSON和[stdlib协议检查器](../../../tools/validate_unseen_transfer_protocol.py)。没有专用Train-only loader、split/transform builder、warm教师/学生runner或合法新资产，也没有任何正式launch声明。**现在没有真实诊断运行命令；不得将check-only命令当开跑命令。**

可运行的准备核查（只读配置及转换元数据，合成人工ID检查）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/validate_unseen_transfer_protocol.py --self-check --verify-source-metadata
```

下一阶段实现者须按此版本：构建独立loader和角色访问审计；实现上述教师/学生/精确评价；对合成fixture核查角色互斥、只用fit上游、不同坐标评分KD、raw/校准分离、同排名分组指标、tie/mask和失败保留；实现前按AGENTS声明，验证后提交。真实资产构建、硬件smoke和正式cohort另有精确范围/命令/授权；到那时记录一个串行诊断cohort（6教师fit、12神经fit、R/N及预定校准），不可把未记录脚本或无限调参混入。源commit、环境和运行命令必须真实存在后再冻结launch，不编造现成可运行入口。

## 9. 记录路由

| 目标 | 本次准备 / 后续真实结果规则 |
|---|---|
| TRAINING_LOG | 本次pending与completed准备记录；真正launch另声明，再分别审计outcome |
| 活跃矩阵 | 后续 `UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_RESULTS.md`；seed2022，T_CF/T_MM与R/V/D/K_CF/K_MM/N raw/calibrated共14行，区分探索probe与Test；当前不造数值 |
| outcome审计 | 后续 `UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_AUDIT.md`，对应全部预定fit、选择/报告、资源、来源与零Test访问 |
| 交接 | 后续 `UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_HANDOFF.md`，一个科学标签及限制、失败恢复或下一阶段 |
| 实验族 | 当前overview仅准备状态；真正完成后添加“Sports未见物品教师可迁移性/混合目录瓶颈”行，不计作新颖方法 |
| 条件消费者 | innovation2入口和当前root/log；第一项正文/矩阵/审计、THESIS_ROADMAP及旧提案不改，未涉及其新结果 |
| 原始资产 | 未来独占 `exp/innovation2/sports_unseen_transfer_v1/`，输出目录不得预存在/覆盖；checkpoint/raw不入Git，失败保留 |
| 导航 | 人工编辑结束刷新六个metadata索引；协议准备无需tag/bundle/物理资产里程碑 |

本阶段唯一下一步：按已冻结协议准备独立运行实现及合成预检，形成真实可审查的启动包；本次不进入执行，不自动启动任何臂或确认评价。
