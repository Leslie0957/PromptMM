# Sports 初始化 × KD：seed2022 四臂用户手动运行声明

**现行修订：用户已取消24小时硬停止，详见文末“用户授权修订”。下方原声明的墙钟限制仅作历史记录；现配置parent_wall_seconds=null。**

状态：快评价器接入和29项合成CPU检查通过，准备一次串行四臂cohort；真实训练尚未启动。用户请求准备并给出手动命令，AI只完成准备，不执行训练。本页是当前唯一启动声明；旧smoke/parity和历史pending不是执行队列。

## 唯一命令与源码

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_cohort.py'
```

分支 `codex/experiment/baby-teacher-baseline`；准备基线 `fc9062831cc3a6be6e060d44aa761992d47f311d`，启动源为包含本声明的干净提交HEAD（仅允许未跟踪check/），manifest记录实际提交。父子进程同一独立wrapper，配置无CLI覆盖；token/PID/manifest/config/source及一次claim绑定，结束时再核对源码。训练期间不要修改仓库源代码、配置或提交新源码。

新独占目录 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2022_v1/`；准备时不存在。一次启动按R0→R1→T0→T1串行完成，同一个seed2022，没有并行臂/其他seed。任何一臂失败即停止，保留已完成臂和部分产物，不跳过、不恢复、不自动重试或进入后续实验。终端成功或失败均交回审计；低指标不是运行失败，不删除结果。

## 冻结资产与配置表

共同锚：[CPU资产预检](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)，来源提交5ebf5df：六资产SHA及形状/有限性通过，全部65,740,800条回放正负例归属通过。原共同配置[base JSON](INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json)保持不可直接启动，SHA `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d`。

新增固定[cohort delta JSON](INITIALIZATION_KD_COHORT_CONFIG_2026-09-27.json)加载并校验base SHA，仅声明可启动身份/命令、原候选硬限、环境和`exact_topk_v1`。共同参数/资产/四臂/阈值均继承base，严格运行时校验，不能覆写。此准备不重新读真实张量或分割；正式运行时先安装Test拒绝，再校验输入SHA及结构。资产若变动则停止，不能重建替代。

| 项目 | R0 | R1 | T0 | T1 |
|---|---|---|---|---|
| 初值 | 固定随机两表 | 同R0逐位初值 | 共同缓存教师最终两表 | 同T0逐位初值 |
| 目标 | BPR | BPR+Full | BPR | BPR+Full |
| alpha | 0 | 0.3 | 0 | 0.3 |
| 模态目标 | 不计算KD | 固定共同image/text物品表 | 不计算KD | 与R1同一目标 |
| 优化器 | 新AdamW | 新AdamW | 新AdamW | 新AdamW |
| 回放/预算 | 同一300×214×1024 tape | 同R0 | 同R0 | 同R0 |

- Sports user35598/item18357/ID64，纯ID无投影学生；R初值SHA `2357868cefeb086dba14c3808415eac97936f67cf5e6bef8382d95ff43bcffff`；共享六表SHA `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`。教师最终ID表为初始化字段，image/text字段才是KD目标，不混用，不使用smoke-final作为初值。
- replay SHA `e8d982c6a2d1be20a934502d4a1ad6ba7e2b699b21098668343341f10db3d77b`，int32 `[300,214,3,1024]`，只读，同批用户/正/负例和顺序；每臂64200更新，共256800更新。不重采样、补batch或共享优化器状态。
- BPR=`mean(-logsigmoid(dot(u,p)-dot(u,n)))`，无显式embedding L2。Full=`0.3*(image_mse_mean+0.3*text_mse_mean)/1.3`，仅当前正物品；无user KD。复用原BPR/方向蒸馏函数，AdamW全参数lr6e-5、weight_decay.01、betas(.9,.999)、eps1e-8、amsgradFalse，初始化之后新建空moment。
- Train/Val SHA及教师checkpoint来源SHA完全沿用base资产表；教师checkpoint只hash来源，不教师forward，不再训练教师。Train218409/Val37899 nnz，35598验证用户。无Test读取、排名或最终Test；不使用旧会打开Test的Data/batch_test。
- 每臂固定300epoch，之前不早停；epoch0只描述，不参与选最优。epoch1–300每轮完整Val，合计1204遍；batch256，Train排除、全部候选、K10/20/40/50、float32点积及原指标归约。主指标第300轮Recall20，次指标最早最大Recall20及同轮NDCG20、完整曲线。保存各臂best/final模型+AdamW，低值同样保留。
- 仅新四臂入口显式选exact_topk_v1；旧smoke/默认参考仍heapq。等价锚：[parity v2审计](INITIALIZATION_KD_EVAL_PARITY_V2_AUDIT.md)，启动42d1fd8，固定状态35598用户全分数行/有序Top50/指标完全相同；保存排名独立核验。新训练状态依赖同一精确排序实现及随机/同分/边界合成验证，不声称事先测过未来全部checkpoint。
- 环境冻结：Python3.10.20、torch2.11.0+cu128、numpy2.2.6、scipy1.15.3、psutil7.2.2，run_5060绝对解释器；cuda0/float32/TF32off/deterministic，CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=2022、CPU线程4。

## 事前解释标准

固定末轮delta_R=R1−R0、delta_T=T1−T0、I=delta_R−delta_T。预先冻结候选实际意义筛查I>=0.002绝对Recall20（0.2百分点）；暖BPR相对暖Full的描述性非劣筛查T0−T1>=−0.001（0.1百分点）。理由：只将至少0.2百分点的交互作为继续投入多seed研究的候选，停止KD省去训练监督时可接受的候选代价上限取其一半；属于研究资源决策约定，无外部应用效用标定，不能称普适最小临床/业务效应。不是从本次新曲线、Test差异或P1a数值门槛选择。

单seed仅描述；满足筛查不证明显著性/等价/非劣，也不授权新增seed。不满足仍是有效结果。I正也不代表暖起点KD完全无用，要同时报告两个delta和各臂值。四臂识别整个初始化方案×指定KD配方，不唯一识别语义知识冗余；主结果与最佳轮不同需如实区分。不得因效果小而改门槛/删结果/重跑。

## 总预算、停止与不确定性

硬限沿用原候选：父进程86400秒（24小时，含加载/训练/评价/保存/验收）、CUDA allocator2GiB、workerRSS8GiB、输出1GiB、可用盘至少4GiB、一次attempt。父进程约1秒轮询并终止超限worker，CUDAallocator设置硬分数、worker补充检查。监控采样并非操作系统级RSS配额；驱动/桌面显存不包含在allocator值。永久I/O错误、身份/源码/环境不符、非有限loss/指标、产物缺失或任一预算超限立即停止。

预算依据均来自已完成资产，不新做GPU探测：

| 项目 | 计算 | 估计 |
|---|---|---:|
| 训练及附带非Val开销 | smoke T1总123.266−Val121.984=1.282秒，除8步，再乘四臂256800步 | 11.431小时 |
| 全部Val | parity fast10.123664秒×1204 | 3.386小时 |
| 基准合计 | 上两项 | 14.817小时 |
| 计划余量 | 合计额外50%，覆盖加载/保存/验收与运行波动 | 7.408小时 |
| 带余量估计 | 14.817×1.5 | 22.225小时 |

这不是可靠置信上界：8步非Val含模型初始化/保存、无法分离纯更新，可能高估稳态步成本；长程缓存、系统竞争、磁盘与温度又可能低估。BPR两臂也按T1 Full成本估，不据此保证24小时内完成。若全部基准变慢超过约1.62倍或出现异常开销，会到硬限停止并保留部分结果；不临时放宽预算。

smoke每份完整checkpoint41,441,755字节，8份最终best/final加1份同时写入临时文件约372,975,795字节（356MiB），另加曲线/报告远低于1GiB；按最多每epoch刷新best估计全程累计写入约50GB，与最终占盘不同。上表smoke非Val外推已含保存成本，50%余量额外覆盖I/O波动。前次smoke CUDA峰值156MiB、RSS约2.2GiB，低于候选上限；四臂串行不保留前臂模型/优化器，tape只读mmap。长程真实峰值仍由运行监督检查。

## 输出与硬验收

输出manifest、claim、worker.log、telemetry、report、acceptance、supervisor，以及每臂curve.json、best.pt、final.pt；失败时failure.json及所有部分产物。report给出每臂epoch0/选择/末轮/步数/逐轮Val耗时、1204次评价和I。非有限目标逐步检查，无完整逐batch损失轨迹；不能据此绘制损失曲线或证明收敛。

必须exit0且supervisor completed/acceptance passed；四臂全部300epoch/64200steps，曲线连续，所选最早最大与第300轮checkpoint/AdamW步数和指标对应，所有保存权重/moment/指标有限，input/source/config身份一致、1204Val、Test0、exact_topk_v1及报告I重新计算一致，资源均过门。worker检查checkpoint内容，parent独立复核完整文件集SHA。没有Test checkpoint对应项，因为协议禁止Test。

29项合成CPU检查通过，新增真实worker流程在3用户/60物品/2维/每epoch1batch的合成输入上串行四臂300epoch，1204次全部走快路径，保存/验收/初值保护通过，篡改interaction拒绝；CUDAAPI全部mock。原有Test拒绝、配对初值/回放、全新AdamW、Full归约、精确排名和Windows文件占用检查保留。准备未读真实模型或分割、未GPU/训练/排名。提交后只读检查源码/环境和输出目录未占用；实际asset bytes运行时检查。

## 记录更新目标与唯一下一步

- 本cohort outcome：`docs/research/INITIALIZATION_KD_INTERACTION_AUDIT.md`，逐臂seed2022完成/失败/未开始及科学解释。
- 活跃独立矩阵：结果审计时新增`docs/research/INITIALIZATION_KD_INTERACTION_RESULTS.md`，R0/R1/T0/T1×seed2022；旧18格不变，禁止把Val填Test格子。
- TRAINING_LOG：本次正式pending另行追加completed/failed/partial及当前交接。更新root/family初始化×KD行、准备记录链接；docs总览仅实际入口变化才更新。
- 立项单在机制结论变化时追加状态；论文/缺口仅实际支持范围改变时更新，不因运行完成自动声称第二创新或填补全部缺口。
- 六项生成导航在人工编辑及结果审计后刷新；所有历史资产、失败目录和check/原位保留且不顺带提交。单seed探索并非最终多seed冻结，无自动tag/bundle/推送/合并；忽略资产不受Git保护。

唯一下一步：用户执行本页命令一次。完成或失败后只审计，不自动进入另一seed/重试/P1b/P2/门控。命令只授权本声明四臂串行cohort，不包含Test或其他阶段。


## 用户授权修订：取消墙钟硬停止（2026-09-27）

用户明确要求“24小时硬停止 不需要这个”。检查时正式输出目录不存在，尚未保留/启动任何attempt。本修订以4eff39232fd7c86cc0300e6934dd85252ee560e4为修改基线，启动仍为包含本修订的干净提交HEAD，由manifest记录；原准备提交的24小时限制被本修订替代，不改变旧日志/审计及base候选配置。

唯一配置delta：cohort hard_caps.parent_wall_seconds从86400改为JSON null，明确表示无墙钟超时。父进程与worker仍记录elapsed并检查其他资源，超过24小时不因时长被杀停。固定四臂×300epoch结束时退出；若卡住，不再有墙钟自动终止，需要用户自行中断。没有隐藏的替代时限。

仍保留2GiB CUDA allocator、8GiB RSS、1GiB输出、至少4GiB空闲盘，身份/有限值/错误/完整验收/一次attempt/不重试/Test0全部不变。原15小时量级和22.23小时含余量估计仅是规划参考，不再作为时长验收条件。smoke/parity原900秒限额不改变。命令、输入、输出目录、四臂/seed/参数/阈值及结果记录目标与原声明完全相同。

新增合成检查：elapsed十亿秒且wall=null不触发时间失败，同时内存/显存/输出/磁盘超限仍拒绝；900秒有限时限仍拒绝901秒。完整30项合成CPU检查通过，无真实GPU/训练/模型/分割读取。唯一下一步仍为用户手动执行本页命令一次，之后审计。
