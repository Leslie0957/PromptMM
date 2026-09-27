# P0 资产与实现准备审计（2026-09-27）

状态：**准备完成；实际 P0 未运行。** 当前只有资产、训练边覆盖与合成测试证据，没有真实效用、吞吐或显存结果。下一步仅为用户手动运行 [P0 声明](CROSS_INTERACTION_P0_DECLARATION.md)。不进入 P1/P2。

## 已核查的资产

具体路径与完整 SHA256 固定在 [P0 共享锚](CROSS_INTERACTION_P0_ANCHOR.json)，不是扫描后随意选最新文件。

| 资产 | 核查结果 / 限制 |
|---|---|
| Sports 冷启动 paired Full seed2022 | 完整 checkpoint 41,444,269 字节，SHA256 `d2648d9f1f6c593b8720b27ad226e81152373537f7cb4d3fa18de07acf20b4a2`；selected epoch293，从0计。路径见 [原审计](SPORTS_PAIRED_COLDINIT_THREE_SEED_AUDIT_2026-09-25.md) 和其 batch.json 第一项。模型及 AdamW 状态存在；没有保存当次教师语义缓存逐位相等证据，故本 P0 不使用它 |
| Sports 暖启动 sharedteacherinit Full seed2022 | 41,444,781 字节，SHA256 `bd651d78ab8853ba987c407a23cd485581ccf319a1b6c9202c5413091fc6879c`；selected epoch282，从0计。manifest SHA256 `825029a48243ac8780b366f088a754d0f9f3f0738e6e69449aaaaf3cc05530fa`；[原配对审计](SPORTS_SHAREDINIT_PAIR_AUDIT_2026-09-23.md) 来源闭合 |
| 两个完整状态的模型 | 仅 user `[35598,64]`、item `[18357,64]` 两张 float32 ID 表，无图传播/投影头 |
| 两个完整状态的优化器 | 参数顺序 `[0,1]`，各自包含 step、exp_avg、exp_avg_sq，矩形状与表相同；lr=6e-5，weight_decay=.01，betas=(.9,.999)，eps=1e-8，amsgrad/maximize=false |
| 已有固定教师语义缓存 | process0/initial_tensors.pt，41,440,437 字节，SHA256 `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`。6张表含 image_items/text_items，各 `[18357,64]`。暖启动 manifest 明确引用此文件与张量哈希；代码 shared_initialization.py 直接使用该缓存，P0 无教师前向 |
| 缓存来源 | process0/report.json SHA256 `293eb18c629aef64a598167b3e20a628cf50112898a0c66b514b2cc18caa4ea9`；来自提交701a646的首次教师前向，固定epoch37教师 SHA256 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。历史生成器曾结构性读取 Test，但未做 Test 排名；本 P0 不复制那条读取路径 |
| 训练矩阵 | data/sports/train_mat，1,890,060 字节，SHA256 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`；shape `[35598,18357]`、218409条边 |
| 推理表与中间状态 | infer_only 只是导出的 ID 表，不含优化器，不可用于此干预。原保存函数在最佳选择轮次覆盖 full，未保存每轮状态；没有验证可用 early/mid/late 序列，不补训、不把初始表或 final best 伪称中期状态 |

本次对 checkpoint/cache 使用 PyTorch FakeTensorMode + weights_only 元数据检查，不物化权重 storage，不实例化真实模型、不做前向。文件哈希已比对上述已审计身份；并未声称本次重新测得实际 step 数值或权重有限性。真实加载时必须复核有限性、形状与 warm step=60562（283×214）；失败即停。没有原 RNG 快照；TD 代码无学习率调度，lr 在优化器中固定。P0 使用新声明的 seed2022 状态，不能声称恢复历史训练轨迹。

## 采样、损失和更新核查

- `Data.sample` 在 batch≤n_users 时均匀无放回抽活跃用户，每用户均匀抽一条训练正边；负候选在全部物品中均匀拒绝该用户所有训练正例。负候选不是已知负偏好。每三元组边际概率为 `1/(Nactive × degree(u) × (Nitems-degree(u)))`。
- 原 TD BPR 是逐样本 `-logsigmoid(u·p-u·n)` 的 batch mean，无额外显式 L2 项；正物品模态 KD 是归一化后 MSE，在 batch×64 上取平均。Full 实际 image/text 系数为 `.3/1.3`、`.09/1.3`，用户系数0。
- 单物品 KD 保留该物品在 batch 中所有出现，先求平方差之和，再除以原 `1024×64`；删除其他项不重新归一化。一个总损失只执行一次 AdamW step，不能先 BPR step 再 KD step。
- AdamW 对完整 dense 表执行动量和 decoupled decay，包括未出现行的已有动量；本实现保留全量分支与完整状态，没有冻结其他行。独立深拷贝优化器状态避免 PyTorch load_state_dict 的张量别名。无全局梯度裁剪。
- U1=`ProbeLoss(A)-ProbeLoss(B)`，A=BPR，B=BPR+单项KD。两分支同模型/矩/step/RNG/批次；每个干预重新从相同完整状态出发。验证 user 行及非焦点 item 行分支差严格为0后，才按正/负角色概率合成该探测池下的总体贡献。

## 训练内部探测与实测覆盖

只读取 train_mat，不导入旧 Data/parser/trainer/evaluator；运行器额外安装拒绝 val_mat/test_mat/val.json/test.json 的读取保护。每个物品的用户正边按 seed2022 打乱并轮流放入 update/A/B/C；四组正边完全分离。负候选使用各自独立种子，并排除**所有**原训练正例，不仅排除所在池。

| 覆盖项 | 只读实测 |
|---|---:|
| update / A / B / C 正边 | 61345 / 56591 / 52463 / 48010 |
| 训练零频物品 | 5 |
| 仅1–3条正边，无法覆盖四池的物品 | 2050 |
| 至少4条边，可分到四池的物品 | 16302 |

update 按其可用用户均匀抽取、用户内均匀抽 update 正边，batch1024。它是预先声明的训练内限制分布，不是原全量正式训练批次。探测分布则用原 user-uniform 采样器的边际 `w(u,p)=1/(Nactive×degree(u))`，条件于 A/B/C 各池。正角色权重为该池 p=i 的边 w；负角色为 i 不在用户训练正例时 `w/(Nitems-degree(u))`。直接按条件权重有放回抽样，无需事后重要性权重；不能用50:50混合替代角色概率。

每角色每池最多8次抽样，实际上限为可用正边数；重复可能减少 unique_edges，报告 nominal ESS 与 unique_edges，不能将其当独立用户数。小样本总体 variance=0 不证明无噪声。池内整体贡献是两角色条件均值×实际角色概率之和，**不是**全训练分布效用；P0 不估计跨物品相关性或泛化收益。

按所有训练非零频物品排序（同频按ID）等分三层，再在固定 batch 已曝光物品中按固定种子选一项。记录曝光选择偏差；层内无曝光就 N/A，不换层、不补 batch。实际清单在看真实效用之前固定：

|层|物品|训练频次|batch multiplicity|A/B/C 正角色可用边|A/B/C 实际独立正边|
|---|---:|---:|---:|---|---|
|低|1494|5|1|1/1/1|1/1/1|
|中|1744|7|1|2/2/1|1/1/1|
|高|1646|16|1|4/4/4|4/3/2|

三项负角色各池均抽8条，实际8个不同正边；本次所抽探测用户与焦点 update 用户无重叠。A/B/C 正边分离不等于所有用户完全分离；计划 JSON 保留每条三元组及共享用户计数供核查。稀疏项目正角色缺失返回 N/A，可单报负角色，不偷换热门物品。既有 checkpoint 已看过训练边，因此只是条件更新测量，非未见交互泛化。

## 实现与验证

新隔离模块 `codes/cross_interaction_p0.py`，手动受监督入口 `tools/run_cross_interaction_p0.py`，合成测试 `tests/test_cross_interaction_p0.py`；原正式 trainer/parser/profile 未改。

10项 CPU 合成测试全部通过：零干预与来源不变；实际归约/重复计数；独立正式损失表达式下 AdamW 双分支及符号一致；非零矩深拷贝隔离；缺优化器拒绝；正边分离/低频缺失；非50:50角色概率；describe无加载；仅豁免未跟踪check/；held-out拒绝及600秒超时保留/禁止再次启动。合成测试只用6用户9物品4维随机张量，不加载任何真实模型，不执行真实训练。另做语法、链接、追加日志前缀、Git差异检查。

不实现 local shortcut，因此“local vs full”验收不适用，full reference 与独立公式已经合成比对。GPU deterministic 数值行为、实际优化器step值、吞吐、显存和效用可测性由首次手动 P0 验证；没有用合成通过冒充 GPU 通过。

## 路由与剩余条件

- 已有：可识别的 warm 完整模型/矩状态、实际复用教师缓存与来源、train-only覆盖、隔离实现及合成测试。
- 待实际 P0：真实数值有限性/零干预一致、资源与测量精度。cold 本轮不运行；其当次缓存等同性、中期状态序列、P1预算与实际意义阈值尚未满足/确定。不自动重建缓存、补训或扩大样本。
- 日志追加准备与手动声明，实验导航新增 P0 准备行；六项生成导航刷新。旧18格、论文正文/缺口表无变化：无新正式结果或科学主张。原始资产全部原位，check/ 不提交。
- 原始 P0 输出预留 `exp/cross_interaction/p0_warm2022_v1/`，本次没有创建。后续依据 supervisor/report/plan/log 审计，向本页与日志追加 completed/failed/partial，失败也保留，不重跑。

## 2026-09-27 手动运行结果（completed；效用仍不可识别）

用户执行原声明命令一次，launch HEAD=`aca59c5cfba5df5768eb68a3b02c226e084cb41f`，分支 `codex/experiment/baby-teacher-baseline`。supervisor和worker均completed，exit_code=0；原入口把子进程输出重定向到worker.log，成功路径没有print，所以直接返回PowerShell且空日志是正常行为。

- 实际完成7对（6干预+1零对照），即14个从原状态独立出发的虚拟分支step；没有继续正式训练或保存新模型。命令及参数与原声明一致：warm seed2022、batch1024、ID64、lr6e-5、AdamW decay.01，image/text系数.3/1.3与.09/1.3。实际恢复的两张表 optimizer step 均60562。
- worker耗时5.5469999998秒，CUDA allocator峰值127926272字节=122MiB，低于600秒/2GiB上限。此值不含CUDA上下文或其他进程，单次P0时间不能直接外推P1成本。
- 零对照参数差范数及全部角色U1严格0；六个真实干预参数差范数范围8.9819872e-7至1.8318266e-5。运行内检查模型/矩/梯度/损失有限，user及非焦点item两分支相等；审计重新核对全部报告值有限、计划哈希/覆盖、焦点与系数、固定输入文件SHA及启动身份。源与资产不变检查通过。
- 6干预×3池×2角色共36个U1，范围[-5.7271121179e-9,7.4605068789e-9]，**0/36超过预声明保守数值尺度**。程序执行完成，测量仍unresolved；不能用这些符号做门控、宣称无效/等价/负迁移，也不能据此通过G1/G2。没有因效用小而回滚或删除结果。
- Validation读取0、Test读取0、排名0、教师前向0。无新Validation/Test选模指标，selected-versus-tested不适用；已有checkpoint选择身份未改变。此次审计也未加载真实模型、未重算梯度或排名。

C池的角色U1（正值只表示该条件探测损失下降，以下全部未通过数值分辨率门槛）：

|物品|模态|正角色U1|负角色U1|
|---|---|---:|---:|
|1494|image|2.0372681320e-10|-5.7271121179e-9|
|1494|text|2.0372681320e-10|9.6360963653e-10|
|1744|image|0|1.5077432636e-9|
|1744|text|3.2014213502e-10|6.0069282881e-10|
|1646|image|-2.2737367544e-11|7.2759576142e-11|
|1646|text|0|1.2005330063e-10|

原始文件保持在 `exp/cross_interaction/p0_warm2022_v1/`：

|文件|字节|SHA256|
|---|---:|---|
|launch.json|272|ecfe26c1837ddeffee9b5a19ae458cfcf7297998157dd3bcb50c7ab3a7341b4c|
|plan.json|72488|11130dd5cc71ee2da45a485bb04c3946a6d5e9f310940019cc219fe1d1557393|
|report.json|25475|9eee8804963aa2db9691d53472d94350b8dafeb95d56000e68e80219d9fac02a|
|supervisor.json|112|d35150d6308a6bfd4362ff67a18285ad0fbd0f867eef4d9d7ffd57968ed77e47|
|worker.log|0|e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855|

原pending声明保留，不再是可执行队列。日志、入口/实验族和路线补充本结果，生成导航刷新；旧18格、论文正文和缺口表不变：这不是新排名结果或算法证据。无需标签/bundle，不提交原始输出或check/。

**唯一建议下一步：准备测量精度复核方案，先明确现有float32差分的误差与可识别性，再决定是否值得另行声明新诊断；本次不实施、不运行该方案，也不进入P1。**
