# 第一创新点固定配置结果冻结清单（待执行）

日期：2026-09-26。对象为[18 格论文主表](../paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md)及其可复核来源。**当前只完成表格整理；没有进行结果冻结、标签创建、standalone bundle 更新或忽略资产的物理/异地备份。** 冻结是一项后续独立授权的里程碑，不产生新训练或 Test。

## 冻结边界

冻结的是 Baby/Sports 各 BPR、Full、PromptMM 发布版共享教师适配 × seed 2022/2023/2024 的固定配置 Test 结果与解释边界；不是第一创新点新颖性认证、充分调优排名、全论文定稿或第二创新点完成。原 Baby Image-0.3 例外/低值、Sports 机制 Validation 对照、暖启动及效率结果保留在各自审计，不混入 18 格。

| 核对项 | 已有证据 / 当前状态 | 真正冻结前的动作 |
|---|---|---|
| 18 格数值、汇总与表注 | **表稿已整理**；Baby 原 6 格加恢复批次 12 格，均按原始精度计算均值与样本 SD | 以保存的 18 份原始 JSON 再次只读核对最终版本；论文中维持固定配置、同数据集比较和评价实现差异表注 |
| 训练与评价源码 | 原 Baby 正式来源列于[Baby 总册](INNOVATION1_THESIS_EVIDENCE_2026-09-15.md)；B3 训练提交 `da2371ebf6de2a5a0daf06444c7f1fd32837ce37`，12 格最终评价提交 `b3ea32fd7f9685631d33a60f52591e8dd3e0cbb8` | 记录最终表格文档提交、所有被引用的源码提交及清洁状态；不将后续文稿提交误写成训练来源 |
| 数据、教师和固定来源 | [共享资产账本](INNOVATION1_REUSE_ASSETS_2026-09-26.json) SHA256 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`；Baby 教师 `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`，Sports 教师 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea` | 核对账本当前 SHA、记录其引用的 Train/Validation/Test、特征、教师及 Sports 9 个所选源的路径/指纹；若物理资产有差异，先审计，不能沿用旧资格 |
| Baby 原 6 格 | `baby_existing_formal` 白名单含 6 个正式 manifest 的路径/SHA；原 Test、选模、资格见[总册](INNOVATION1_THESIS_EVIDENCE_2026-09-15.md) | 核对原文件仍存在且 SHA 一致，连同引用的选中检查点、曲线/运行记录和必要环境文件列入备份清单；不重跑 Test |
| B3 训练 3 格 | [v2 审计](INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md)记录 `exp/promptmm_release_baby/innovation1_fixed_v2/` 三个 `report.json` 与 `best.pt`、选模和 SHA | 核对三对原文件与审计指纹；保留失败 v1/v2 总账作为执行历史，不把失败视为缺失 B3 有效训练 |
| 新增 12 格最终评价 | [恢复审计](INNOVATION1_EVAL_RECOVERY_V1_AUDIT.md)列出 12 份预检与 12 份最终报告逐文件 SHA；批次总账 `exp/formal_closeout_cohort/innovation1_eval_recovery_v1/batch.json` SHA256 `fd4d1bf7eb57678db25e3ba32a818d17ab512c5eaebc6c48ac57d20547f79e0f` | 再核对 12 份报告及总账指纹、源/所选/被测检查点相等、12 个先预检后 Test、每格学生 Test 一次且教师零次；保存相应 `.log` 与 `exp/formal_closeout_preflight/`、`exp/formal_closeout_eval/` 家族 |
| 大资产的真实保存 | `data/`、`Model/`、`exp/`、`logs/` 等被 Git 忽略；当前 Git 提交和生成目录**不是备份** | 先确定由用户控制的物理/异地目的地和容量，再复制本表依赖的数据、教师、选中学生检查点、manifest/report、必要日志与环境身份；生成含路径、字节数、SHA256 的冻结清单并校验副本，至少抽样读回；原路径不移动/删除 |
| 论文解释与资格 | [正式矩阵](../paper/INNOVATION1_FINAL_CLAIMS_MATRIX_2026-09-26.md)的 R5（新意、强基线充分性、外部要求）仍开放；18 格仅是技术主比较 | 作者确认表注、方法身份、相关工作和论文需要的覆盖要求；若改变主张或协议，另行声明相应工作，不拿已有 Test 做新配置选择 |
| 版本里程碑 | 此清单状态为**待执行**；没有冻结标签、bundle、备份验收记录 | 另行授权后，按[AGENTS](../../AGENTS.md)在干净且已验证的提交创建有注释标签、刷新 standalone bundle，并把实体备份清单、目的地、校验结果、标签/提交写入新的完成记录；不改旧原始报告 |

## 冻结完成的判定

上述依赖文件的当前身份、独立副本校验、表格/措辞复核及里程碑提交/标签/bundle 全部完成，并在 `TRAINING_LOG.md` 另记完成结果后，才能称这 18 格的**结果资产已冻结**。若只完成表格或 Git 提交，状态仍为“表稿可用、冻结待执行”。新窗口优先读根 README、AGENTS、当前短日志，再从本清单沿链接查证，不必扫全部历史。

下一步仅为确定冻结范围、备份目的地与该里程碑的独立授权；这里不包含可执行训练或 Test 命令。
