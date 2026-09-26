# 第一创新点正式收尾协议（执行前版本）

日期：2026-09-26；制定基点 `263499b`。本协议规定固定配置比较的解释边界、Baby 新增发布版对照预算，以及隔离评价入口的验收要求。它不是某次运行的 `pending` 声明或执行许可；入口尚未实现，没有新训练、Validation 排名或 Test 访问。实际实验由用户执行，逐阶段按 [AGENTS.md](../../AGENTS.md) 声明和授权。

> 后续状态：[Baby与隔离评价入口](INNOVATION1_FORMAL_CLOSEOUT_LAUNCH_2026-09-26.md)已实现并通过合成无Test检查；当前短日志另有用户手动执行的完整批次声明。上述制定时状态保留作历史语境；真实训练/Validation/Test仍未执行。

依据：[正式主张矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)、[协议和资产核查](INNOVATION1_PROTOCOL_REUSE_AUDIT_2026-09-26.md)、[15 项来源清单](INNOVATION1_REUSE_ASSETS_2026-09-26.json)、[Baby 总册](INNOVATION1_THESIS_EVIDENCE_2026-09-15.md)、[发布版身份](PROMPTMM_BASELINE_IDENTITY_2026-09-18.md)。来源清单固定记录路径/哈希，正式执行前仍须按声明核对。文内“决定”只固定后续准备所依据的方案；如果修改预算、配置、来源或 Test 规则，先追加协议修订和对应日志记录，再重新明确授权。

## 1. 配置选择边界与可写主张

**决定采用既有固定配置的透明比较，不称充分调优的公平竞赛。** 主表保留 Baby/Sports、BPR/随机初值 Full/发布版共享教师适配、学生 seed 2022/2023/2024；每格唯一来源由来源清单和后续声明指定。Baby BPR/Full 使用 08-02 白名单六项既有正式 Test，不重复评价；Sports BPR 使用历史 300 轮三项，Full 使用严格配对组 Full 三项，发布版使用已做过 Sports Validation 配置比较后选定的 `lr=6e-5` 三项。不得从历史 Full/real/配对重复组中逐 seed 挑高分；不得按未来 Test 调整来源。

Baby 新发布版适配的**唯一预选配置**是从 Sports 发布版 300 轮三种子组转移 `lr=6e-5`，沿用已核对的发布版学生结构、教师初始化、损失/采样/teacher 与 prompt 模式、64 维、batch 1024、AdamW decay 0.01 及其余未声明改变的发布版系数。它不是使用 Baby Test 选择的配置，也不宣称对 Baby 最优。Baby BPR/Full 的既有 `lr=6e-5`、学生 decay 0.01 等保持原始记录；同 LR 不是三法同等调参机会。Baby 适配为新数据/教师/停机/评价协议分支，保留原 Sports 运行身份和历史报告；只要损失、梯度、采样或学生机制需要修正，就另命名 corrected 对照并修订主表计划，不能沿用 release 身份。

主表按每个 seed 报 Test Recall/NDCG@20，辅以均值及样本标准差，并列出所选 Validation Recall@20、选择 epoch、预算、初始化和已知不对称。结果低也纳入；不能在看过 Test 后增加 LR、延长轮次、换来源或补种子来优化主表。对发布版 Sports 的 `2e-5/6e-5` 比较必须披露三 seed 的 Validation 已参与配置选择，故三 seed 不是独立的超参搜索确认。现有随机初值 BPR/Full 没有对称搜索，故主张限定“这些固定配置下的得失”；发布版教师初始化与 Full 随机初始化也是方法差异，不能把主表差值单独归因于某一个损失。若后续需要对称调参或新数据集，必须在任何新 Test 前定义独立预算/选择规则，并重算工作量；旧结果继续保留，不自动成为新协议的同组比较。

## 2. Baby 发布版预算与选模

**决定 Baby 新发布版学生使用每 seed 最多 1000 个完整 epoch、每 epoch 后一次 Validation、连续 7 个未严格提高的 epoch 早停。** 这与 Baby 既有 BPR/Full 的 `cap=1000, patience=7`、每 epoch Validation 和 `val_test_once_v1` 停机口径对齐；不是承诺三个算法有相同优化步数、运行时间或充分收敛。发布版 `codes/promptmm_release_validation.py` 当前仅支持 Sports 30/120/300 固定预算、`early_stopping=False`，必须实现新的显式 Baby 入口，不能把旧脚本仅改 dataset 名称或直接跑满 1000 来充数。

选择只看完成训练后的 Validation Recall@20；分数**严格大于**历史最佳才替换最佳检查点，相等保留最早者；初始 Validation 可记录但不得参与选择。首次完成的 epoch 必须产生最佳检查点；每次提高重置未提高计数，否则加 1，达到 7 即停，至多完成第 1000 个 epoch。以 1 起算记录发布版训练 epoch，并同时保存停止原因、实际 epoch/optimizer steps/Validation 次数、最佳 epoch、完整曲线、最佳文件 SHA256、有限性和恢复一致性；不要拿 TD 的 0 起算 epoch 直接比较。使用与 Baby 原组相同的 Train/Validation/Test 身份、冻结教师来源及 train-only 候选排除，Validation Recall@20 为唯一选模指标。若发布版实现无法在该预算下保持其核定计算身份或有限目标，记录失败和可用产物，另行修订/授权；不隐性修损失或自动重试。

该预算允许早停时间因方法而异，不能据此声称相同训练资源。Sports 既有 300 轮 BPR/Full 和无早停发布版仍保持原记录；跨数据集不拼成“统一 1000 轮”协议。为避免把历史 Baby Test 当调参反馈，新增 Baby 发布版不做 Baby LR 先导或赛后调参；若研究目标改为充分调优，先另定对称的 Validation-only 搜索预算，不能继续宣称本协议只需 3 次新训练。

## 3. 隔离评价入口：实现验收与 Test 闸门

### A. Test 之前完成的工程验收

1. 新增独立的 `eval-only` 入口及独立报告命名空间；不调用训练循环、不构造优化器、不执行反向传播/参数更新，不修改旧 Validation-only manifest、`best.pt` 或旧正式 Baby 报告。TD 的历史配对 profile 禁止 Test，不能通过改其开关绕过。发布版须按图传播及双 ID 表共享 storage/别名语义恢复完整学生；不可只取一个原始 ID 权重表评分。
2. 用不含 Test 的合成样例核对 TD 完整检查点与推理导出表、发布版 `restore_best` 回环及权重别名、参数有限性、用户/物品维度和推理分数。用允许的 Train/Validation 样例核对旧/新评价的用户集、train-only 排除、候选顺序/并列处理、K 列表 `[10,20,40,50]`、Recall/Precision/Hit/NDCG 定义与聚合口径；对**同一已选 checkpoint**重放 Validation Recall@20，应与原记录在预先声明的浮点容差内一致。已有 Sports release fast/legacy parity 是参考，不能替代新入口对 TD、Baby 和 Test 语义的验收。任何不一致先修实现并重新做无 Test 验收，不靠 Test 调试。
3. 启动前核对干净且已提交的代码、环境/依赖、固定来源清单、所选 checkpoint 哈希、数据划分/教师共享锚、源报告的选模指标与 epoch、实际模型结构、旧 Test 历史和目标报告路径的唯一性。Baby 既有六项 Test 禁止重测；Sports 九项来源原 `final_test_performed=false`、release `test_evaluations=0` 是既有记录，不修改为 true。未来资产漂移或身份不一致即停止并审计，不打开 Test。
4. 实现需支持预检/Validation 阶段和单独显式 `final-test-once` 阶段，后者默认禁用；Test 文件读取和排名只能发生在所有预检通过、来源与选模锁定、干净 source commit 且用户明确授权该具体评价后。预检不得结构性加载 Test 矩阵。每个 Test 格子保留独立新 run ID 与报告；声明列出该格的唯一 Test 访问、失败不重试、预期产物及更新目标。

### B. 一次正式 Test 的硬验收

- 在首次尝试打开 Test 之前持久写入 `test_access_started`、时间、来源哈希、选中 checkpoint 哈希、代码/环境身份和该格 attempt=1；通过独占路径/已用标记拒绝第二次尝试或覆盖报告。随后只评价该学生所选状态一次，不评价教师 Test，不做新选模。即使在读取或排名中崩溃，也保留 `started/partial/failed` 记录；不能因为没有最终指标就把访问次数记为 0 并重跑。
- 报告记录 Test 数据身份、候选规则、K/指标实现身份、真实访问时间顺序、所选与被评 checkpoint 的文件哈希及状态/embedding digest 相等证据、有限分数和指标、完整指标数组、唯一一次学生 Test 次数、教师 Test 次数 0、源码在评价前后未改变，以及 `completed/failed/partial` 状态。新报告独立给出该固定协议下的资格；旧诊断 `eligible=false` 保留原貌，不回写。
- 只有预检、Val 重放、一次 Test、身份相等、有限性及完整记录都通过，格子才由“缺最终评价”变为正式完成。若硬验收失败，保留失败报告和原资产，审计根因；后续重试或修订入口为新的明确授权阶段。指标低但硬验收通过仍是有效完成，不重训追分。

## 4. 执行顺序、工作量与记录路由

先实现并验收 Baby 适配和独立评价器，再逐项声明、提交干净来源，由用户手动运行。评价器工程验收不授权 Test；Baby 每个 seed 的训练与每个最终 Test、Sports 每个来源的最终 Test 都需在对应明确授权范围内按一阶段规则执行。可以准备串行命令清单供用户执行，但一次授权不自动覆盖后续 seed、arm、重试或 Test。正式声明须写全命令、共同身份锚、差量、source commit、预算/选择、来源哈希、唯一 Test 权限、产物和失败处理；本协议不提供可直接运行的正式命令。

在**此固定方案且全部工程门槛通过**时，新增工作量是 Baby 发布版 3 次学生训练及 3 次最终 Test、Sports 9 次最终 Test；Baby 原六项 Test 直接复用，共形成 18 个主表格子。实现/预检、Val 重放及后续审计另计，不把它们说成已有结果。预算或配置边界改变时，按 [矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)重新核算，而不是机械套用 3+12。

每次声明按 [RUN_CLOSEOUT.md](../experiments/RUN_CLOSEOUT.md)事先写明记录更新目标：`TRAINING_LOG.md` 的 pending/outcome、该次独立 `docs/research/` 审计、[实验族导航](../experiments/README.md)对应行、[正式矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)对应 B3/S1/S2/S3 seed 格、生成的文件与记录目录；若论文主张或整体缺口状态变化，再更新正文稿/缺口表，未触发则注明原因。旧声明、原 manifest、失败/低值、忽略的大资产不覆盖。全部正式格完成后仍需按真实结果收缩或保留结论，核对相关工作/学位要求及第二创新点，不能把本协议当整篇论文完成证明。

**当前唯一下一步：实现 Baby 发布版适配与隔离评价入口，并先完成不访问 Test 的工程验收；实现与正式运行范围另行明确授权，实际训练/评价命令仍由用户执行。**
