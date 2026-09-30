# 质量约束前沿v1：一次真实诊断的运行声明审核

2026-09-30。**用户已明确授权本次诊断及原Train分区恢复；启动声明已更新，待用户手动单次启动，agent尚未运行。** [机器声明](QUALITY_CONSTRAINED_FRONTIER_RUN_DECLARATION_V1.json)的`human_authorized=true`、`originalTrain_partition_exception_acknowledged=true`，authorization_evidence保存用户原文与所批准声明提交5e5bdc9。用户最新要求是以后准备好可运行任务后直接给命令，不再增加授权环节；本次完成核查与提交后交命令，不由agent启动。

本页承接[独立实现与CPU合成预检](QUALITY_CONSTRAINED_FRONTIER_RUNTIME_PREPARATION_2026-09-30.md)；[冻结协议](QUALITY_CONSTRAINED_FRONTIER_PROTOCOL_V1.md)和[profile](QUALITY_CONSTRAINED_FRONTIER_PROFILE_V1.json)保持不变。旧profile/实现准备页中的不存在/未授权文字是各自历史状态；当前阶段由本页与活跃日志负责。没有第二创新点已成立的结论，第一项继续冻结至初稿。

## 本次要判断什么

在固定内容模型和相同仿射校准预算下，同时约束暖物品与混合目录整体质量，比较可取得的冷物品收益。它帮助判断简单接入模型是否足够，或KD表示是否仍留下值得研究的开发差异；不把校准本身认定为第二创新点。

选择集已经参与过原模型选择，本次也是**开发诊断**。即使表现较好，也不能称独立确认、泛化保证或创新认证；无可行点、低收益与undetermined都是可以保留的科学结果，不自动删结果、换seed或扩网格。

## 一次运行的固定范围

| 项目 | 声明 |
|---|---|
| 身份 | sports_quality_frontier_v1，一次串行cohort，复用原seed2022已选模型；新训练次数0 |
| 六固定模型 | R λ.001；V lr.001/epoch240；D lr.001/epoch10；K_CF、K_MM lr.0003/β.1/epoch20；N原暖历史内容均值。复用CF的P/Q和保存PCA，不加载MM教师表、不重训/重新选模/PCA拟合 |
| 网格 | 每臂a=2^(j/2)，j=-4..4；b=[-1,-.5,-.25,0,.25,.5,1]，单位为固定暖分数标准差；63×6+2暖参考=380状态 |
| 质量约束 | 暖R20≥CF_W_ONLY暖R20−ε；整体R20≥D/raw整体R20−ε。ε=[0,.0005,.001,.002]，主ε=.001，不事后更换主预算或基准 |
| 输出 | 六臂×四预算24格，6raw/2参考单列；不可行明确记录。最多36项选择后配对描述性区间，bootstrap2000/seed20260930 |
| 数值/目录 | 固定13011用户，暖13930/冷选择917候选；K20/batch128；基础float32，冷仿射和组间排序float64，原ID破同分；fit正例屏蔽。严格冷排序证书不通过即失败 |
| 资源 | CPU4线程、串行，最多7200秒；恢复900/评分4500/bootstrap900/收尾900秒。RSS6GiB、输出2GiB、空闲磁盘至少12GiB，5秒采样加合作检查；不是OS强制隔离，也不是预计要耗时2小时 |
| 新输出 | exp/innovation2/sports_quality_frontier_v1，必须不存在；只在此处产生新缓存/记录，原资产保持只读 |
| 停止规则 | 只启动一次，无自动retry/resume/另一seed/另一实验/后续算法训练。运行失败仍保留现场并审计；任何恢复需新授权 |

复用共享锚点：原launch`07eaeee8db0fc62b8fc998f7ebed1c30dd024e78`、结果`6714bf14880b636261a03584f731bac70e776d09`、路线`1dea1c88393c67e2644cddc5bb3dde440291853f`及原batch SHA`9d4861d0089a230db2b76613443b6b5e3c1f1173dfda0f4e9db25356af026764`。14个输入/模型身份与固定角色SHA逐项继承冻结profile，未为声明重新扫描真实payload。

## 明确请求的原Train分区重读例外

原运行没有保存允许角色的完整标签，必须重新读取原Train来恢复固定分区。**反序列化会接触完整源容器，包括其中probe/lock来源的字节；这不是“源字节零接触”。** 用户本次授权已包含这项具体重读；授权依据是新的直接回复，未沿用旧诊断许可。

恢复时只导出warm_fit/warm_select/cold_select和允许的ID；暖pair第1项直接丢弃，不保留为warm_probe目标。核对原角色SHA/counts后，在评分前释放完整矩阵和临时全部pairs。probe/lock标签不构造、不导出、不计数，不生成新的seal。

图像/文本数值只处理warm_active与cold_select行。完整预提取特征容器可mmap，但不得变换probe/lock数值行。沿用已核验来源的特征SHA并stat大小；这是继承身份，不是本次新做整文件哈希。原Train及小型模型/变换/身份文件在加载前实际核验SHA。

本次仍要求：原Val/Test读取0，旧probe评价0，锁定确认0；sealed_lock**读取与哈希均0**，原report/per_user_probe、T_MM_tables与incoming.inter也禁止访问。应用guard和访问记录不是OS独立追踪。

## 声明准备时的必要修正与验证

基于实现提交`06d99b6b4c46ef94b2c2b977cb6f06e71fdb9ef2`，核查发现runner对未变特征无条件全量哈希，与冻结协议的可信锚点/stat要求不一致。已作小范围修正：先核验允许的原batch JSON元数据SHA，再引用特征锚点并stat；大小/元数据不匹配直接失败，不自动全量扫描或更换来源。原Train/小资产SHA规则保留，评分、分区、门槛、网格和资源均未变。

13项测试通过，其中新增假文件测试覆盖特征零digest调用、缺少可信锚点拒绝、特征大小异常、Train哈希异常及元数据失效。没有重跑已有合成fixture，历史预检报告和原产物保留其原源码指纹，不能把它们静默改写成新runner的执行证据。评分core保持原SHA`922a2b96f0bdd56fcce18225572f015fa5c94d76394529b2bfb5b38b832efa20`；新runner SHA`e97a58221177fe9efe160a525ec79cd8bcf20b1864a844126f726881e164398f`。

元数据预检：14个允许资产存在/大小一致；一个允许的原batch JSON SHA与锚点一致；磁盘空闲约370GiB（声明时快照，运行时还需重查）。声明准备期间真实Train/特征/模型/Val/Test/probe/封存payload加载与哈希0，无真实角色恢复、训练、排名或新fixture。

环境：run_5060，Python3.10.20、numpy2.2.6、torch2.11.0+cu128、scipy1.15.3、sklearn1.7.2、psutil7.2.2；本命令评分设备仅CPU。真实规模的角色/N-query/数值证书与资源门槛仍未执行，不能由小型预检保证一定通过。

## 命令、源码与结果路由

声明提交干净且启动门槛通过后，用户从`D:\Download\PromptMM`执行一次以下精确命令：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_quality_constrained_frontier.py --execute --declaration docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_RUN_DECLARATION_V1.json
```

分支`codex/experiment/baby-teacher-baseline`。启动源为**包含已授权声明的干净committed HEAD**；提交后的真实运行会记录实际launch SHA。授权声明提交和启动门槛核查不等于实验已发起。本次许可覆盖准备和后续结果审计；用户最新要求手动启动，agent不代执行，不再次索要相同许可。

| 记录目标 | 运行后处理 |
|---|---|
| TRAINING_LOG | 保留pending，追加独立completed/failed运行结果；更新当前状态 |
| QUALITY_CONSTRAINED_FRONTIER_RESULTS.md | seed2022×六臂×四预算24格；raw/两暖参考另表，全部标开发select，无伪造Test值 |
| QUALITY_CONSTRAINED_FRONTIER_AUDIT.md、HANDOFF.md | 硬验收、选择/指标、来源/产物指纹、访问/资源、失败原因或实验意义及一个后续动作 |
| 根/第二项入口、实验族总览 | 按真实结果更新Sports未见物品教师可迁移性/混合目录瓶颈行 |
| 六生成导航 | 手工更新完成后刷新，sealed读取拒绝；本次声明也刷新 |
| 第一项正文/gap/Test主表、论文charter及原v1记录 | 无触发，不改动；本次不创建虚假结果表或数值占位 |

源码提交只保护已跟踪源文件。新输出是Git忽略资产，不等于有物理/off-device备份；本次不是标签、bundle或稳定结果冻结里程碑。失败/低结果保留，不自动回滚、覆盖或重跑；后续准备按用户最新工作流直接交可运行命令，由用户手动决定执行。

## 唯一下一步：用户手动启动

用户2026-09-30原文：“授权，以后不要再授权了，以后如果这次能跑起来就把跑起来的指令给我就行”。这条最新工作流指示已写入活跃日志和根入口，后续对话应遵守，不再复制旧文案制造额外授权轮次。

本次只完成已授权声明的记录、核查与提交，没有真实Train/特征/模型/Val/Test/probe/封存payload访问、恢复角色、训练或排名。后续由用户执行上方精确命令一次；运行后的既有产物可按原路由直接审计，不再请求同一次运行的许可。启动门槛包括committed source/declaration一致、允许资产stat与磁盘上限、输出未存在；真实角色/N-query/数值与资源硬验收仍须由真实运行验证，不能提前保证成功。
