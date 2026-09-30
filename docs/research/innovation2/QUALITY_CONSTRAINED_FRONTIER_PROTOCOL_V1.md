# 固定内容模型的质量约束冷暖竞争前沿诊断 v1

2026-09-30，依据`1dea1c88393c67e2644cddc5bb3dde440291853f`的[路线重评](POST_DIAGNOSTIC_ROUTE_REVIEW_2026-09-30.md)，结果来源`6714bf14880b636261a03584f731bac70e776d09`。参数以[冻结JSON](QUALITY_CONSTRAINED_FRONTIER_PROFILE_V1.json)为准。**本次仅完成协议准备；评分程序未实现、正式声明未创建、未运行。** 本协议不能追认原诊断成功，不能消费旧probe或封存确认集。

## 1. 只回答一个问题

在已经选定的内容模型不变时，给各方法相同的全局仿射校准预算，并约束暖与总体质量损失，哪种表示能够取得更多冷物品收益？这用于区分模型表示与冷暖竞争的解释，不认证新算法，也不证明分数尺度是唯一瓶颈。

本诊断明确是**复用选择集的开发诊断**：warm_select/cold_select已参与原模型选择，原probe也已被研究者看过。现在冻结的新规则受到这些历史知识影响，不称独立确认；原v1的completed/undetermined和阈值原样保留。新方法需要后续独立冻结与确认，这一步不使用cold_lock。

## 2. 固定模型与资产

复用v1的固定用户13011、warm_active13930、cold_select917个候选、暖fit84695对；暖select12671对、冷select5351对/4067位正例用户/881个有正例物品。确切角色SHA与模型/变换文件路径、bytes、SHA已复制到JSON的anchor/restoration/models。只复制原manifest的身份，本次没有重新哈希或加载运行payload。

| 臂 | 固定状态 | 作用 |
|---|---|---|
| R | λ0.001，原selected.npz | 简单线性内容映射 |
| V | lr0.001，epoch240 | 非线性向量回归 |
| D | lr0.001，epoch10 | 直接暖Train排序，固定P_CF |
| K_CF | lr0.0003，β0.1，epoch20 | 协同排序蒸馏 |
| K_MM | lr0.0003，β0.1，epoch20 | 多模态教师排序蒸馏 |
| N | 原暖fit内容历史均值参考，query SHA固定 | 完整内容参考，不是同P/Q因果臂 |

核心五臂共享原T_CF缓存P/Q和暖分数，不加载T_MM表/教师图。固定128→128ReLU→64映射，不重训、不更换epoch/模型、不扫β/LR/λ，不重新拟合PCA。原transform.npz只有两模态mean/components，未来按原公式变换warm_active/cold_select，行L2、float32后拼接；N按原history与normalize公式恢复，并核查query SHA。已选模型本身不是此次新增候选，不能利用新校准指标改选原all_fits。

允许的原来源为原Train和预提取图像/文本，沿用原锚点与维度；旧Val/Test/incoming.inter、原report/per_user_probe、T_MM_tables及sealed_lock均禁止读。原teacher/student选择JSON允许仅提取已固定身份，不拿其中probe结果作选择。运行时加载小型模型/变换/表之前核验anchor SHA；恢复原Train须在反序列化前核验原Train SHA。未变的特征用已验证锚点和stat；只有锚点过时/不可信或尺寸身份改变时重哈希，不为一般文档维护反复扫描特征。

## 3. 角色如何合法恢复

v1没有保存warm_fit/select标签，旧Top20也没有完整分数，不能从现有指标计算新工作点。未来执行需要一个**新的、明确授权的originalTrain partition-only重读例外**：

1. 用原item SHA key与80/5/5/10划分，按原warm degree≥5确定用户。暖pair SHA排序第0项为warm_select，第1项仅保留位置空缺，第2项以后为fit；零fit暖物品排除，不重采样、不按cold标签筛用户。
2. 只导出warm_fit/warm_select/cold_select、允许的用户/暖/冷选择ID。cold_probe/cold_lock边不进入导出对象，不计数、不创建标签对象或新的seal文件；暖保留位置不输出为warm_probe。固定源容器反序列化不可避免地含有这些边，故必须披露“源容器partition接触”，不能声称源字节零接触。
3. 核对允许角色的原pair SHA/count/users/items、warm/user SHA和维度；原始整矩阵及临时全部pair在评分前释放。只保存允许角色的恢复manifest或数组，访问日志记录例外和导出范围。
4. 评分器仅持有允许角色；禁止复用会建立probe/lock标签并写seal的原Roles类。不打开sealed_lock，连哈希也不读；没有确认集排名或特征变换，只有warm_active/cold_select数值行可处理。

真实恢复不属于本次准备，也不能用旧v1运行授权覆盖。角色身份不匹配属硬失败，保留输出，不自动重采样/重试。应用层guard/events不是OS隔离证明；future runtime需合成越界测试和封存读拒绝验证。

## 4. 固定校准网格与精确排名

暖分数始终保持不变；只对冷物品做

`s_c'(u,i) = a * float64(s_c(u,i)) + b * sigma_w`。

sigma_w为固定用户×warm_active所有**屏蔽前**float32暖分数的全局人口标准差，用float64统计，与标签无关。核心五臂完全相同，N使用自己的暖评分标准差；这是将相同offset网格放在各臂的暖评分单位下，不能称N暖模型与核心相同。sigma<1e-8时科学undetermined并明确未执行项，不伪造归一尺度。

- a = 2^(j/2)，j固定为-4至4：9个正尺度，含0.5/1/2。
- b固定为[-1,-0.5,-0.25,0,0.25,0.5,1]：7个offset。
- 每臂9×7=63个候选，6臂共378个排名状态。a=1,b=0是raw；所有方法预算相同。无幅值收缩、个性化标量或逐物品偏移，避免本阶段并入另一方法。
- 另计算CF_W_ONLY、N_W_ONLY共2个暖参考，总计380个状态。候选身份仍为warm_active∪cold_select，参考冷分数为排除标记；总体分母保留全部暖/冷正例，暖参考冷命中为0。它们是质量参考，不是63点搜索的fallback候选。

固定用户batch128，全目录基础内积float32，仿射及跨组比较float64，按分数降序/原ID升序tie；掩蔽warm_fit，冷select正例不屏蔽。这个精度选择是新协议显式delta，不要求逐位复现v1的float32校准结果，也不替换旧结果。

实现可只算一次基础评分，保存暖Top20与冷Top20的ID/原分数后，在每点合并两组列表；对正仿射，组内顺序不变，因此数学上等价于全目录Top20。**float64舍入不能悄悄破坏此条件**：runtime必须证明本批次全部网格下冷边界与不同基础分数的严格次序保持，原分数并列仍按ID；无法认证则硬失败，不能继续用近似排名。合成测试需覆盖边界ties/负分数/offset/屏蔽/组DCG，并与朴素全候选float64参考一致。禁止物化或保存全用户×物品分数矩阵。

缓存可供380个状态和所选状态复用，不重复真实评分。基础cold-only排名对正尺度和共同offset保持不变，只报告一次/臂并核查恒等；它不能替代混合排名主指标。

## 5. 质量基准、可行集与选择

**暖质量基准为CF_W_ONLY的warm_select Recall，整体质量基准为本次D/raw的mixed-select总体Recall。** 二者对应同一固定用户/正例集合；教师暖目录的overall不能直接拿来作为混合参考。N也接受这套共同约束，其N_W_ONLY仅辅助说明自身暖基线。

ε固定为[0,0.0005,0.001,0.002]，主ε=0.001，单位为绝对Recall比例。每个臂/ε要求同时满足：

- warm Recall ≥ CF_W_ONLY warm Recall − ε；
- mixed overall Recall ≥ D/raw overall Recall − ε。

两个参考分别表达保留既有暖系统质量和保留直接接入模型的整体质量；不是声称两个参考来自同一模型。没有事后更换参考、选择“最好看”的ε或修改v1容忍量。

可行集中依次选冷Recall最大、总体Recall最大、暖Recall最大、|log2(a)|最小、|b|最小、a较小、b较小；所有比较用已计算未四舍五入值。没有可行点则明确`no_feasible_grid_candidate`，不补上无冷参考当赢家。每臂最多4个预算所选状态，共24格；重复工作点去重保存，不再次排名。ε增加时冷最优值不能下降（只对两者均有可行点比较），作为选择一致性检查。

另保存全部三维warm/overall/cold非支配候选；它只是63点有限前沿，既不是理论最优，也不是全checkpoint空间最优。相等指标可用同一参数tie规则记录代表并保存等价候选列表。

## 6. 报告与判定

每点记录整体/暖/冷R20、N20、命中、正例、用户分母；组NDCG仍来自同一混合Top20。报告冷槽位占比、冷曝光集中度、可行候选比例、cold-only、冷向量范数、恢复/评分/选择/评价成本。所有展示均标development/select，不能填入第一项Test主表。

保存2参考、6raw以及每个不同预算所选状态的逐用户数组，可用来核查汇总代数、曝光、DCG和所选工作点。开发集曾被复用，因此它们不足以认证泛化或产生“真实不劣”保证。bootstrap只对每预算K_CF−D、K_MM−D、K_MM−K_CF、每组Recall作配对，至多3×4×3=36比较；2000次、seed20260930、linear百分位95%，组用户及resample索引一致。无可行臂则相应比较缺失。**这些是选择后的描述性区间，不经选择偏差修正，不以显著性授予成功资格。** 不bootstrap挑新系数。

事先冻结以下开发分流（质量和sample门槛缺失优先undetermined）：

| 主ε下情况 | 标记与含义 |
|---|---|
| 角色正确但暖select用户<1000，冷用户<1000，冷正例<2000或有正例冷物品<200 | development_undetermined；缺信息 |
| 核心五臂全无可行点 | development_no_feasible_candidate；有限网格未找到共同质量约束内的点 |
| 核心可行点的最佳冷R20<0.002 | development_no_useful_tradeoff_in_grid；当前有界搜索无足够开发信号 |
| 有冷信号但D/K_CF/K_MM任一无可行点 | development_incomplete_control_comparison；不能伪造公平KD结论 |
| D和两KD可行，最佳KD冷R20−最佳可行R/V/D冷R20≤0.002 | simple_control_sufficient_in_development；本有限网格无大于0.002的KD点估计优势 |
| D和两KD可行，上述差>0.002 | KD_difference_remains_in_development；值得解释的开发差异，仍不认证泛化/因果/创新 |

记录K_MM−K_CF差及质量代价，但这里不检验暖教师总体优势，不再读probe。N单列参考、不能挤入核心最大值改变判定。次ε只展示；标签不追溯改变v1科学undetermined。良性低指标为completed，不删产物；异常/身份/资源/精度硬验收失败为failed，保留现场。任何结果都停止为用户讨论，不自动训练门控/重建教师/换seed。

## 7. 资源与未来执行接口

CPU串行，Torch/BLAS线程4，不要求CUDA、不训练；RSS≤6GiB、输出≤2GiB、空闲磁盘≥12GiB，5s采样和批次合作检查。总上限7200s：恢复/资产900、评分网格4500、bootstrap900、closeout900。上限不是耗时预报；真实性能尚未测量。超过上限即保留失败，无自动retry/resume。

未来独占输出`exp/innovation2/sports_quality_frontier_v1/`必须不存在；资产只读，新缓存存新目录。source/restoration/model/score-cache manifest、resolved profile、grid/frontier/budget_selection、per_user_selected、report/access/resources/batch必须齐全并绑定来源SHA。源数据/模型/旧run不修改；忽略产物留本机，不声称Git或off-device备份保护。

预定接口（**目前程序与声明文件均不存在，不可执行**）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_quality_constrained_frontier.py --execute --declaration docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_RUN_DECLARATION_V1.json
```

将来只有runtime实现及CPU合成预检完成、真实读取例外被具体声明并获用户授权、profile/源身份/资源门槛验证且exact source提交干净后，才能launch一次。这里没有授权声明、训练/真实评分、旧probe/Test/lock确认或恢复重读权限。

## 8. 固定记录路由及当前下一步

未来结果必更新QUALITY_CONSTRAINED_FRONTIER_RESULTS.md（6臂×4预算24格，缺失如实记录，6raw/2参考另表）、AUDIT.md、HANDOFF.md；同前缀且与JSON完全一致。TRAINING_LOG追加独立run outcome，保留新run pending；实验族仍Sports未见物品教师可迁移性/混合目录瓶颈。根与第二项入口按实际阶段更新，六个生成导航用sealed guard刷新。第一项正文/主表/gap、charter、v1旧结果/profile/runtime不适用且保持冻结。不创建结果矩阵占位数值或假报告。

**唯一下一步：实现独立评分器与角色隔离，并做有界CPU合成预检（含仿射合并精度、质量选择、角色越界、失败保留和声明门槛）；不读取真实数据/模型，不启动本诊断。** 准备验证通过之后，再提供具体真实执行声明给用户授权。
