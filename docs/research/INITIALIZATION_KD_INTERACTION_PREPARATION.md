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
