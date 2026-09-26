# 第一创新点正式协议兼容性与资产复用核查

日期：2026-09-26；基点`ac439c2`。只读JSON、源码和指定学生文件字节；无张量反序列化、模型前向、训练、Validation/Test分割读取或排名。机器可查资产正本为[15项来源记录](INNOVATION1_REUSE_ASSETS_2026-09-26.json)：6项Baby既有正式记录、9项Sports待评价来源。

## 1. 结论与新增工作量

**按保留已有固定配置/既定预算的收尾范围，建议新增3次Baby发布版学生正式训练、12次学生最终Test（Baby3次＋Sports9次）；Sports无需因现有学生资产缺失而重训。** 这是本次证据支持的复用路线及工作量，不是执行授权，也不是“充分调优后一定只需3次训练”的保证。

| 组 | 训练增量 | 最终Test增量 | 裁定 |
|---|---:|---:|---|
| Baby原BPR三seed | 0 | 0 | 08-02白名单正式结果保留，不再测Test |
| Baby原Full三seed | 0 | 0 | 同上；原cap1000/patience7的固定配置结论可用 |
| Baby发布版适配三seed | 3 | 3 | 无合格同身份结果；先完成Baby适配和协议，训练后各一次最终Test |
| Sports BPR三seed | 0 | 3 | 三个已选checkpoint存在、记录身份一致，可走独立最终评价 |
| Sports严格配对Full三seed | 0 | 3 | 来源固定为严格Full/image组，不在旧Full、real及配对组间择优 |
| Sports发布版lr6e-5三seed | 0 | 3 | best.pt均在且指纹一致，具有独立恢复路径；不可直接把ID权重当最终表打分 |
| 辅助消融、sham、暖启动、效率 | 0 | 0 | 限定Validation/阶段结论已可用，无默认补跑 |

工程工作另计：**一套隔离最终评价入口＋一套Baby发布版适配/正式运行入口**；合成或训练/Validation范围的一致性验证属于准备工作，不能伪装成零成本，也不能用Test做调试。不提供时长估计：新入口尚未实现，旧训练耗时受评价器/系统负载影响，不能外推。

剩余不确定项是科学比较范围，不是找不到文件：若要求补充对称调参、改用统一新训练预算或增加基线，须另行定预算再增列运行量。当前可支持固定配置比较，不能称各法充分调优。为避免先看Test后再调，最终Test须等配置选择范围和主张边界确认后再执行。

## 2. 协议逐项比较

| 项目 | Baby原BPR/Full | Sports BPR/配对Full | Sports发布版 | 裁定 |
|---|---|---|---|---|
| 数据/教师 | 六份记录内部train/val/test及教师指纹一致 | 六份内部一致；同一epoch37教师 | train/val/教师记录与TD一致 | 身份记录一致；本次不重复读取大数据/教师载荷 |
| 训练预算 | cap1000、patience7 | 300、patience300 | 固定300，无早停 | 同数据集TD内部可比；数据集间预算差异需披露，不能称全局统一300 |
| 学生配置 | ID64，batch1024，lr6e-5，AdamW decay0.01 | 同左；BPR无KD，Full alpha0.3、1:0.3 | ID64，batch1024，lr6e-5，decay0.01，另有发布版损失/图结构 | 发布版不是纯ID训练同构，结构差异保留为方法差异；同LR不证明调参公平 |
| 初始化 | 随机ID | 随机ID；配对Full有保存初值和triplets | 教师表征初始化 | 不把主表差值解释为只由某个损失造成；暖启动控制作为辅助 |
| 选模 | Val Recall@20首次严格最大 | 同左 | 同左；初始验证不参与选择 | 选模口径一致；best_epoch为TD零起算、release一起算，不能直接混用 |
| 候选/指标 | 排除Train已见项 | 同左 | 适配保持同一候选/legacy指标语义 | Test时也保留train-only，不能悄悄改成train+val；NDCG保留现行top-max(Ks)定义并披露 |
| Test历史 | 六学生已各有正式Test | final_test_performed=false，旧加载器结构性读取Test | test_evaluations=0、test_split_loaded=false | 已有Test不重测；新评价独立记账，禁止选模/挑配置 |

**Baby预算处理建议**：保留已发表在本地记录中的cap1000/patience7结果；Baby发布版新适配若要进入同一固定协议表，应支持并事前约定该停止规则、评价频率与选择范围。不能把现有Sports-only/固定300入口换个数据集名就拼表。若后来选择固定预算新协议，旧Baby结果保留为历史组，新增训练量另算，不因2024低Test而定向重跑。

**配置选择的残余限制**：Sports发布版已经比较过2e-5/6e-5，BPR/Full仍是选定固定配置，现有材料没有对称超参搜索证据。lr6e-5作为待冻结发布版来源有Validation依据，不能宣称三种子是对超参选择完全独立的重复。是否扩大调参须在新Test前决定，本核查不凭空增加扫描。

## 3. 指定资产的核查结果

- 9个Sports来源对应24个报告/manifest/学生文件全部存在；其中18项有直接的已存hash用于比较，全部匹配。包括9个所选完整学生checkpoint，其记录hash均匹配；其余文件记录当前hash，不把新算hash称为历史匹配。
- BPR来自`exp/audit/sports300_audit.json`的三项bpr；Full来自`exp/paired_coldinit/sports_three_seed_v1/batch.json`三项full；release来自三份`validation300_seed20xx_lr6e5_v1/report.json`。所有具体路径、字节数、hash、参数、epoch、Test标记见资产JSON，不用聊天记忆再选。
- 所选TD epoch（零起算）：BPR 298/295/298，Full 293/291/292。release epoch（一起算）：294/299/300。存在性与字节身份已核查；参数张量有限性、所选/导出相等及release roundtrip沿用已有审计，本次没有重新加载模型验证。
- Baby六项来源固定为总册白名单：08-02的BPR pid17532/36572/33332，Full pid27620/4516/5916。元数据确认Test=true、eligible=true，教师/数据身份一致。Baby Image2023例外属于辅助表，不能混淆为这六项的例外。
- 复用前执行预检仍需再次检查即将使用的文件身份、环境与源提交；本文核查不是未来文件绝不变化的保证，Git也不备份忽略资产。

## 4. 为什么当前命令不能直接评价

| 入口 | 只读源码发现 | 后续准备要求 |
|---|---|---|
| [main_mmlight.py](../../codes/main_mmlight.py) | TD先完成训练循环，再restore-best/Test；无本任务可用的独立eval-only入口；配对300门槛显式禁止run_final_test | 新增隔离加载/评价入口，不能再次启动原训练命令或篡改其profile |
| [promptmm_release_validation.py](../../codes/promptmm_release_validation.py) | dataset仅sports，教师固定epoch37及原eligible=false、数据路径硬编码；epochs限30/120/300；只加载Train/Val | Baby需显式适配身份/数据/教师与停止规则；最终Test需独立授权和隔离报告 |
| release best.pt | 保存state_dict、所选epoch/metrics与tensor digest；resumable=false | 不可当续训检查点；可按原restore_best逻辑恢复做推理，须保留共享storage别名 |
| [ReleaseStudent](../../codes/promptmm_release.py) | forward需Train图，两份别名表相加后图传播、跨层平均 | 不可直接把其中一张权重当TD的最终embedding；评价无需教师前向，但需正确的学生图计算 |
| [batch_test.py](../../codes/utility/batch_test.py) | 全局Data在导入时加载数据；排名/指标有既定候选顺序与legacy定义 | 隔离入口避免训练入口及隐式加载；Test只在正式授权路径读取；不得静默换指标 |

已有[Validation parity](../PROMPTMM_VALIDATION_FAST_PARITY.md)证明特定Sports release状态上旧/新排名一致，是复用实现的依据之一；不等于新TD/Test/Baby入口已通过验证。后续用合成或允许的Train/Val案例验证恢复、候选、并列与指标一致性，不能为调试而多次访问Test。

## 5. 正式评价需新增的记录

独立评价要有新run_id、冻结来源清单、源码/环境、授权的split和一次性消费记录、无优化器更新证据、每个来源selected-versus-evaluated身份相等、完整指标与失败状态。原训练manifest继续保留Validation-only及eligible=false；若新协议接受复用，资格只在新评价报告中按其规则判断，不回写历史。crash/部分Test同样计入实际访问，不能无记录重试。

这说明**资产与推理结构支持复用**，不等于**当前已有可直接执行且合规的Test工具**。本次无Test访问，无新正式结果，无新paper-ready判定。

## 6. 当前唯一下一步

**准备第一创新点正式收尾协议**：以本次固定来源与工作量为基准，明确采用固定配置比较的主张边界、Baby停止/配置选择规则、隔离Test工具验收和具体文件更新目标。协议准备阶段不运行Test或训练；之后实现与运行范围须由用户明确授权，实际命令仍交用户。

来源：[正式矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)、[Baby总册](INNOVATION1_THESIS_EVIDENCE_2026-09-15.md)、[Sports300](SPORTS_THREE_SEED_VALIDATION300_2026-09-18.md)、[严格配对](SPORTS_PAIRED_COLDINIT_THREE_SEED_AUDIT_2026-09-25.md)、[release身份](PROMPTMM_BASELINE_IDENTITY_2026-09-18.md)、[lr6e-5结果](PROMPTMM_LR6E5_THREE_SEED_2026-09-22.md)。
