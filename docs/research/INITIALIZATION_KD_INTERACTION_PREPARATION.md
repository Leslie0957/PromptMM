# Sports 初始化 × 持续 KD 四臂准备审计（2026-09-27）

**状态：隔离四臂训练核心与合成 CPU 协议已准备；真实 seed2022 cohort 仍不可启动。** 按[立项单第 7 节](INITIALIZATION_KD_INTERACTION_PROPOSAL_2026-09-27.md)仅完成静态来源核对和合成验证。没有加载真实模型、真实张量或任何数据分割，没有训练/GPU、Validation/Test 排名。四臂只是第一项工作的机制补充，不是已选定的第二算法。[机器可读配置](INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json)记录固定值、来源和候选预算；`launch_command=null` 是有意的阻断状态。

## 四臂已解析配置

| 臂 | 初值 | 优化目标 | 共享输入 |
|---|---|---|---|
| R0 | 既有 seed2022 随机 ID `initial.pt` | BPR；alpha=0 | 同一只读 `triplets.npy`、同一批序 |
| R1 | 与 R0 逐位相同 | BPR + 原 Full；alpha=0.3 | 同一 tape，固定语义缓存图像/文本目标 |
| T0 | 共同缓存 `users/items` 教师最终表 | BPR；alpha=0 | 同一 tape；缓存身份仍记入共同锚 |
| T1 | 与 T0 逐位相同 | BPR + 原 Full；alpha=0.3 | 同一 tape 与 R1 相同模态目标 |

四臂统一 Sports seed2022、ID64、无投影/无图传播学生、点积 BPR `mean(-logsigmoid(u·p-u·n))`。现有 TD 路径的 BPR **没有额外显式 L2 正则项**；parser 的 `regs='[1e-5,1e-5,1e-2]'` 在该路径未进入目标，不能误加。Full 的正物品图像/文本目标分别按 `1.0/0.3` 加权、除以 `1.3`，再乘 alpha `0.3`；用户图文 rate 均为 0，`torch.nn.functional.mse_loss` 默认均值归约。BPR 两臂直接不计算 KD，不因缓存存在而反向传播。源码对应 `codes/main_mmlight.py:1078–1219` 与 `codes/td_distill_model_no_projection.py:61–82`；隔离核心直接复用现有模型/BPR/方向损失函数，不改旧训练器。每臂单独建 AdamW，lr `6e-5`、decay `0.01`、默认 betas `(0.9,0.999)`/eps `1e-8`，空 moments 和 step0；不继承历史完整 checkpoint 的优化器状态。历史 Sports 固定300 profile 和 seed2022 manifest 的上述字段已静态交叉核对；新核心**不解析旧 CLI 或使用旧 profile**，故无隐藏 CLI 覆盖。未来真实入口必须锁定本配置，拒绝额外 CLI 调参。

预算固定 **300 epoch × 214 batch × 1024**，每臂64,200批、四臂256,800批，tape 形状 `[300,214,3,1024]`，不提前 early-stop。每臂在 epoch0 只记录初始 Validation，不参加选择；训练后每 epoch 评价一次（每臂301次含初值，四臂共1,204次）。主终点是固定第300轮 Validation Recall@20；次终点是训练 epoch1–300 的最早最大 Recall@20 及其同轮 NDCG@20、完整曲线；最早最大不能代替末轮交互。Validation 排名须只排除 Train 已见物品，K `[10,20,40,50]`、`test_flag=part`。Test 文件打开与排名均为零；新进程必须在任何加载器导入/读取前安装 Test 文件审计钩子。当前老 `codes/utility/load_data.py:21–59` 在构造时读取 `test_mat`，`utility/batch_test.py` 导入即构造该加载器，故这两者**不能用于实际新入口**。失败时保存独立部分产物，停 cohort；不得跳过失败臂、重试或报告完整交互。

主交互为 `I=(R1−R0)−(T1−T0)`，差值均采用固定末轮 Recall@20。事前**候选**实际意义门槛 `delta_I=+0.002` 绝对 Recall（0.2 个百分点）：约为既有 Sports 严格配对 Full/image Validation 平均增益 `0.004178` 的一半，以免把远小于已可观察组件效应的差异当成值得继续的机制。若未来只声称暖启动 KD 不损失，候选非劣界 `epsilon=0.001` 绝对 Recall（0.1 个百分点），需预定合适的重复实验和区间使 `delta_T` 下界高于 `−epsilon` 才能判断；单 seed 只能描述，不能认证非劣、等价或显著性。这两个是立项判据候选，未受用户任务价值确认，也未形成可启动阈值；与 P1a 的损失尺度和旧 Test 差值无关。看新曲线后不能反选门槛。

## 静态资产与成本核查

只查了源、旧 manifest、路径存在与文件长度、旧审计记录及 tape 的 64 字符 digest 文本，没有打开 `.pt`/`.npy`/Train/Val/Test。现存共同缓存 `initial_tensors.pt` 41,440,437 bytes，旧审计 SHA `e5573bbd…4da20`；随机 `initial.pt` 13,814,245 bytes，manifest SHA `2357868c…cffff`；tape `triplets.npy` 788,889,728 bytes，旁置 `.sha256` 与 manifest 均记 `e8d982c6…3d77b`。旧 Full manifest 的 seed2022、epoch300、214批、batch1024、上述随机初值与 tape 路径/hash，以及 `run_final_test=false`/Validation 身份均吻合。固定 epoch37 教师 checkpoint 路径存在，357,779,639 bytes；新四臂预期只需其已固定输出缓存，不执行教师前向。共同缓存的 users/items 同时给 T0/T1，image_items/text_items 同时给 R1/T1；R0/T0 的目标锚也固定但不参与损失。旧资产只作引用，不重建、移动、覆写；`check/` 原文不动。

**尚未核实当前资产字节与旧 SHA 相等、实际张量 shape/dtype/有限性、tape 数值范围、缓存六表与教师/物品索引的实际适配。** 要闭合这一步，未来单独声明的只读预检应只获准哈希固定 cache/initial/tape/teacher/train/val 来源，并加载 cache/initial/tape 做形状、dtype、有限性/索引检查；不得加载训练后学生模型、读 Test 或排名。当前没有以旧记录冒充当前字节验收。

粗工作量：四臂训练 256,800 批、验证约1,204次；同一约789MB tape 逻辑读取四遍，另有两表 13.8MB/臂及 AdamW 两组 moments。旧暖 Full/BPR 串行配对约8小时17分，系统负载和评价器未受控，故四臂**不能可靠线性外推**；P1a 的分钟级耗时更不适用。提出候选硬限：整 cohort 24 小时、CUDA allocator 2GiB、进程 RSS 8GiB、原始输出 1GiB、起始可用盘至少4GiB、一个独占 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2022_v1/`，一次尝试不恢复。此预算有待真实独立适配器实施监督、无模型的静态估算和必要的另授权资源 smoke；**当前核心不执行这些进程/磁盘硬门，不能当运行上限保证**。若达到限制，先保存部分记录并停止，不自动扩限。

## 隔离实现与验证结果

新增 `codes/initialization_kd_interaction.py` 提供冻结四臂参数、纯内存单臂/四臂训练核心、初值/缓存/tape 形状与有限性检查、独立 AdamW、最早最大选择、Test 路径访问拒绝和独占目录原语。模块没有真实资产加载器或可执行 cohort CLI，绝不调用旧 `main_mmlight.py`/旧 runner。合成测试用微型 CPU 表和同一 tape 检查四臂、初值不变、各臂各两步新 AdamW、Full/alpha0 目标、epoch0与末轮/并列最早最大、缓存错配/越界停机、实际 Test 文件打开被拒、目录二次使用被拒；只打开临时合成文件。尚不能据此宣称真实 Validation 排名语义一致，或真实资产/源/资源门通过。

未来必须补齐且先通过合成测试的**隔离真实适配器**：Train/Val 专用读取与旧评价器同义性校验（不能导入会读Test的模块）；在加载前安装全进程 Test 拒绝钩子；只读哈希/张量身份预检；GPU 显存/墙钟/RSS/输出限制；每臂固定末轮与最早最大状态保存、完整 manifest/失败部分产物；干净提交和无历史命名空间覆写检查；对已存在输出拒绝启动。上述缺口使本准备**不能生成或提供真实启动命令**，也不形成正式运行 pending。四臂相同 tape 与成对初值在合成核心中通过，不是对真实资产逐位执行的证据。

## 后续声明与记录路由

获准的实际 seed2022 四臂若未来能启动，需独立 `pending` 声明：精确源提交、参数/身份 hash、上述每个硬限的执行点、唯一命令、一次性串行 R0→R1→T0→T1、一个失败即停/不重试、Validation-only/Test文件读取0。运行结束审计路径 `docs/research/INITIALIZATION_KD_INTERACTION_AUDIT.md`；`TRAINING_LOG.md` 追加 completed/failed；`docs/experiments/README.md` 更新实验族；新建**独立四臂×seed2022**结果矩阵，旧第一创新点18格不改。立项单仅在机制结论变化时追加状态；论文主张/缺口表仅在证据真的改变时更新；生成六项导航刷新。忽略原始产物留在上述新命名空间，普通准备不创建标签/bundle或冒充资产备份。

**唯一下一步：**单独准备并合成验证 Train/Val 专用实际适配器及限制监督，再做限定范围只读资产字节预检；这些门未通过前不提供启动命令，不运行四臂。

## 2026-09-27 下一准备阶段：隔离适配器（未获运行声明）

按上述单一步骤新增 `codes/initialization_kd_adapter.py` 与封闭的 `tools/run_initialization_kd_interaction.py`。新适配器只定义 Train/Val 两个固定名称的稀疏矩阵读取，先比对 SHA256，再反序列化；新 worker 在调用任何分割加载前安装 Test 文件打开拒绝钩子。Validation 对有 Val 正例的用户，以点积给全部物品打分，排除 Train 物品、用旧 `part` 路径相同的 `heapq.nlargest` 顺序取前 K，按旧 recall/NDCG 定义求平均。旧 `Data`/`batch_test` 均未导入。新增缓存/随机初值/tape 的只读哈希和张量形状/有限性/索引预检函数；教师 checkpoint 仅校验来源哈希，不参与学生训练。

四臂内核现可接受设备、每轮保存回调及资源检查回调。封闭 launcher 设计为一次独占命名空间、串行 R0→R1→T0→T1，按轮保留完整 Val 曲线与最早最优/第300轮模型及 AdamW 状态，部分失败保留已有文件。父进程轮询墙钟、RSS、输出字节和空闲盘；worker 每秒检查这些数值与 CUDA allocator，并设 CUDA allocator 分数限制。边界采样间隔约一秒，超过门限后终止；此处未以真实 GPU 测过实际超限反应时间。现配置仍为 `preparation_only_not_launchable`、`launch_command=null`，所以新脚本在任何资产读取前拒绝启动。`hard_caps`、`launch_branch` 和 `launch_source_rule` 只有经后续预检和正式声明后才能变成可运行字段；当前的候选数值仍标为未执行门。Val SHA256 来源于既有资产清单，但本阶段没有重新哈希当前字节。

合成 CPU 检查覆盖 Train/Val 稀疏读入、Test 实际打开拒绝、哈希错配、排除 Train 的 Val 排名、预算边界、不可启动状态、父进程墙钟超限杀停并保留部分产物，以及原有四臂配对/新 AdamW；未读取任何真实 `.pt`、`.npy`、Train/Val/Test 分割，未跑 GPU 或训练。正式运行前仍需：单独的只读真实资产字节/张量预检；确认候选资源阈值在目标机器上的实际可用性与真实 GPU 监督；冻结阈值和完整正式声明后提交干净启动 HEAD。当前**没有经核验的启动命令**，不得把脚本名称当成手动运行许可。

## 2026-09-27 CPU 只读资产预检通过（仍非运行声明）

本次用户批准上一阶段唯一下一步后，执行新建 `tools/preflight_initialization_kd_assets.py` 一次，耗时 6.234 秒；报告见 [资产预检 JSON](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)。配置 SHA256 为 `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d`。六项资产全部与固定 SHA 相同，包括仅作来源哈希、不反序列化的教师 checkpoint；没有重建、搬移或改写任何资产。

- CPU 读取共享六表与随机初值两表：float32、ID64、user35598/item18357、全部有限，键和尺寸通过。教师最终 ID 表和模态语义表是不同字段，未混用；身份依据是固定缓存和既有来源记录，不重新执行教师。
- 只读 tape 为 int32 `[300,214,3,1024]`，索引合法；遍历全部 65,740,800 条三元组，正例不在 Train 的数量0，负例在 Train 的数量0。未来四臂引用这同一个不可写 tape；本次没有采样或更新。
- Train/Val 形状均35598×18357，nnz分别218409/37899；Val用户35598；Train/Val交集0；最少可排名候选18120。只检查集合与结构，没有计算任何用户分数或排名。
- 新正式输出目录尚不存在。磁盘可用411,798,528,000字节；预检时可用RAM3,018,383,360字节，进程RSS1,411,309,568字节。GPU未初始化/分配/测量；不能把候选8GiB进程上限当成当前可用内存保证。数据/模型/日志/manifest和check/原位保留。

本阶段不改正式启动配置；`launch_command=null` 继续生效。预检通过仅关闭资产身份/结构/tape内容门，不代表完整运行器验收。最终源码审阅还发现需补强：父进程不能仅凭exit0认定cohort完成，必须验收四臂report/曲线/检查点；内部worker入口应绑定独占attempt及父进程manifest，不能作为旁路启动；声明参数与实际Protocol必须逐字段核对，锁定环境与数值设置。以上属于同一准备工作的启动可靠性缺口，尚未形成新研究结果或协议修改。

**唯一下一步：完成最终启动验收修补和合成测试，并准备一个另行声明、有界、用户手动执行的 GPU 资源 smoke，以便关闭真实资源门。** 不直接启动四臂，不恢复旧入口，不提供未经核验的四臂命令。实际意义门槛仍为有依据的候选+0.002，单seed仅描述；非劣界.001不能用于单seed非劣主张。资源及科学判据须在新结果出现前冻结于最终声明。

记录路由：本准备审计、资产预检JSON、TRAINING_LOG当前状态与append-only结果、根/docs/实验族入口更新；六项生成导航刷新。新四臂结果矩阵、旧18格、论文正文/缺口不适用（没有新实验/指标），历史记录不重写。

## 2026-09-27 最终启动保护与资源 smoke 准备完成（GPU未执行）

补强了 `tools/run_initialization_kd_interaction.py`，共用新的 `codes/initialization_kd_runtime.py`：严格核对配置与固定实现、解释器/依赖版本，拒绝脏源（仅豁免未跟踪check/）；内部worker必须与父进程的pipe token、PID、manifest、source/config和独占claim对应。末尾重新核对源，worker检查完整曲线、模型两表和AdamW参数/moments/步数，父进程独立核对完整文件集与SHA；exit0但缺产物不得报成功。修复合成测试发现的Windows路径分隔符不一致。保留失败日志/部分产物，不重试。

新增独立smoke入口与配置，完整范围/命令/上限/输出/验收见 [用户手动smoke声明](INITIALIZATION_KD_RESOURCE_SMOKE_LAUNCH_2026-09-27.md)。只做T1前8批更新和一次完整Val，900秒、2GiB CUDA allocator、4GiB workerRSS、256MiB输出、4GiB空闲盘、一次尝试。正式四臂配置仍disabled，不产生四臂结果。smoke科学验收不使用候选interaction门槛，低Recall不是失败；只核实功能和资源。

16项合成CPU检查通过，覆盖相同worker保存/验收整条路径（GPU接口全部模拟）、父进程超时、exit0缺结果、损坏checkpoint、错步数、错配置/环境与worker旁路拒绝。此准备阶段没有读取真实模型/tape/Train/Val/Test，没有运行GPU、真实排名或训练；共同资产引用5ebf5df的既有预检锚，不重复哈希。真实资源适配只待用户手动执行本次smoke。历史资产/check/原位保持，旧18格和论文主张不变；下一步仅用户运行独立smoke一次后审计，不自动进入正式四臂。

## 2026-09-27 Validation提速与精确等价核验准备（未执行真实评价）

已实现独立exact_topk：分区仅求K边界，显式按ID处理边界同分，最后只排序K项；原heapq仍是默认/参考。新旧路径共享GEMM、user batch256、全用户/候选、Train排除及全部指标公式/求和顺序，训练频率/轮数/配方没有改动。21项合成CPU检查通过，包含dense ties、signed zero、近邻浮点、mask、非有限/不足候选、完整指标精确相等、错误排名和损坏保存表拒绝。合成128x18357的rank-only三遍中位数加速约63.75倍，不含GPU/GEMM/指标，不能替代真实资源结论。

下一步已准备[一次用户手动评价核验声明](INITIALIZATION_KD_EVAL_PARITY_LAUNCH_2026-09-27.md)：只读固定smoke-final表和Train/Val，先reference再fast两次完整Val，逐用户全分数行hash/有序Top50/完整指标精确相等，保存两份排名表供审计，训练步数0/Test0；新独占目录，900秒/2GiB CUDA allocator/4GiB workerRSS/256MiB输出/4GiB空闲盘。AI本次只执行合成检查，没有真实模型或分割读取/排名/GPU/训练。旧smoke与其结果保留，四臂默认评价器和不可启动配置保持不变。四臂24小时预算仍待真实核验结果后重算，不自动扩大预算或进入正式运行。


## 2026-09-27 快评价器接入与四臂手动声明准备完成

真实[parity v2](INITIALIZATION_KD_EVAL_PARITY_V2_AUDIT.md)通过后，新增独立cohort wrapper/config，只在声明four_arm中显式选择exact_topk_v1；旧训练配方、base候选JSON、smoke和parity入口语义不变。严格检查共同参数、数值配置、原24小时/2GiB CUDA/8GiB RSS/1GiB输出/free4GiB上限及原候选效应门槛；parent另核实评价器/Test0/四臂末轮interaction一致。无需恢复或使用smoke训练状态。

29项合成CPU检查通过，含3用户/60物品/2维的四臂300epoch（每epoch1batch），1204次Val全部走快路径、每臂完整checkpoint和空起始AdamW、验收及错误interaction拒绝。未加载真实模型/分割、未运行GPU或真实训练。共同资产沿用预检锚，byte hash由运行时复核。

预算：smoke非Val1.282秒/8步外推256800步约11.431小时，加1204次fastVal约3.386小时=14.817小时；额外50%余量后22.225小时，原24小时硬限不变。短程外推不保证长程完成，约356MiB峰值checkpoint槽位、最坏约50GB累计写入已纳入保存/波动讨论。I>=.002及T0−T1>=−.001冻结为描述性资源决策筛查，不是单seed显著性/非劣证据。

当前唯一下一步由用户执行[完整四臂手动声明](INITIALIZATION_KD_COHORT_LAUNCH_2026-09-27.md)命令一次，之后只审计。独立wrapper以干净提交启动、独占新目录、无CLI参数覆盖，一臂失败即停、不重试或跨阶段。上方旧待准备/不可启动交接保留为历史；base候选入口仍disabled，只有新cohort命令可用于本声明。


## 2026-09-27 用户授权取消四臂墙钟限制

现cohort配置parent_wall_seconds=null，无墙钟硬停止；固定四臂300epoch及其他资源/身份/Test0/一次attempt守卫保留，smoke/parity时限不变。详见[启动声明文末修订](INITIALIZATION_KD_COHORT_LAUNCH_2026-09-27.md)。30项合成CPU检查通过，尚未启动真实训练，输出目录未占用；原24小时记录保留为历史。唯一下一步仍是同一用户手动cohort命令一次。
