# 第一创新点 18 格固定结果：本机快照审计

日期：2026-09-26。用户授权对现有结果作便于论文写作的归档；用户明确没有指定外部备份路径，接受先记录和校验本机副本，GitHub 事宜以后另议。此处“快照”指已经选定的固定配置结果和来源文件的本机可复核状态，不表示全论文完成或具备异机灾备。

## 从哪里开始

1. 写论文读[论文入口](../paper/README.md)和[18 格主表与表注](../paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md)。不得从本地副本中择优重选 Test 或训练配置。
2. 查来源用[逐文件冻结清单](INNOVATION1_RESULT_FREEZE_MANIFEST_2026-09-26.json)：每项是仓库相对路径、字节数、SHA256。总文件数 **168**，总字节数 **2,018,503,879**；清单 SHA256 为 `9f957609709d220a477d9778994457677ccb11b665e30bcefe60ad3a6ab378b7`。
3. 原始文件仍在原路径；已核对的实体副本在 `backups/innovation1_fixed_results_2026-09-26/`，其 `assets/` 保留仓库相对目录层次，根部 `manifest.json` 与跟踪清单逐字节相同。仓库 `backups/` 被 Git 忽略，不会随 Git bundle 或日后推送自动上传。

## 范围与验收

| 项目 | 本次状态 |
|---|---|
| 结果 | Baby 原正式 BPR/Full 6 格 + 恢复批次 Baby 发布版 3 格、Sports BPR/Full/发布版 9 格；全部为最终 Test R/N@20。原精度取 JSON，论文表六位小数和 6 组均值/样本 SD 逐项复算一致 |
| 资格 | 原 Baby 6 份 manifest 状态 completed、paper_ready_eligible=true、原 Test 已执行；恢复 12 份报告 completed、paper_ready_eligible=true，学生 Test 各一次、教师零次、所选和被测检查点 SHA256 一致 |
| 时序 | 恢复批次总账 SHA256 `fd4d1bf7eb57678db25e3ba32a818d17ab512c5eaebc6c48ac57d20547f79e0f`；12 个无 Test 预检全部先于 12 次最终 Test，批次 completed；v1/v2 失败史和有效 B3 训练来源保留 |
| 身份 | 共享账本 SHA256 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`；Baby/Sports 教师分别为 `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4` / `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`；旧 6 manifest、Sports 9 组选中资产、恢复 12 组来源/检查点均按已记录 SHA256 验证 |
| 保存 | 处理后 Baby/Sports 数据与特征、两位教师、选中学生检查点、旧 Baby 六组 manifest/曲线/预检、Sports 九组来源、新 B3 三组报告/检查点、v1/v2/恢复总账及步骤日志、12 预检/12 最终报告、环境记录；共 168 项，按清单逐项复制并读回 SHA256，168/168 通过 |
| 源码 | Baby B3 训练来自 `da2371ebf6de2a5a0daf06444c7f1fd32837ce37`；新 12 格最终评价来自 `b3ea32fd7f9685631d33a60f52591e8dd3e0cbb8`；论文表稿基础提交 `4ddbce65c705d91b5ea826be81a07dcf1300314d`。本审计与脚本另经一个干净提交，标注标签 `innovation1-fixed-results-20260926-local`；其指向的提交就是跟踪源码/文档快照 |

核验命令（元数据、JSON、文件字节操作；**不加载模型或运行评价**）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/freeze_innovation1_results.py verify --destination 'D:\Download\PromptMM\backups\innovation1_fixed_results_2026-09-26'
git cat-file -p refs/tags/innovation1-fixed-results-20260926-local
git rev-parse 'innovation1-fixed-results-20260926-local^{}'
git bundle verify backups/PromptMM_innovation1_fixed_results_20260926.bundle
```

本里程碑采用**带注释、未加密签名**的标签；上述命令读取标签及目标提交。bundle 的 `.sha256` 旁文件用于校验 bundle 本身。标签与 bundle 在本审计提交之后创建，精确提交与 bundle 指纹见本机副本根部 `FREEZE_RECEIPT.txt` 和任务交接。

旧 Baby manifest 中 `artifacts.student_run`/`artifacts.teacher_run` 是当时计划路径，现有文件并不都存在；本次保存的是 manifest 实际记录的 `td_full_checkpoint`、`td_infer_checkpoint` 和已验证的教师别名，不伪造缺失文件。复制不修改、不移动、不删除原始资产。

## 限制与后续

Baby 和 Sports 使用不同预算、教师和划分；发布版适配与 Full 的初始化、结构、配置选择路径不同。主表不证明文本损失在 Test 上的独立因果贡献、充分调优或统计显著；受控 Full/image 与 real/sham 仍按 Validation 机制证据写。第一创新点相关工作定位、第二创新点和学位覆盖要求另在[论文缺口表](../paper/THESIS_COMPLETION_GAPS_2026-09-26.md)。

C: 和 D: 经核查都属于物理磁盘 0。本次副本与原件同盘，源代码 bundle 也在本仓库 `backups/`；**没有异机、云端或离线设备备份**。用户选择暂不做更严格备份。之后若迁到 GitHub，需要单独商量隐私、仓库体积及忽略大资产的保存方案，不能把普通 `git push` 视为这 168 个文件已备份。唯一下一步是论文写作时从主表/表注引用本快照；更广的备份与第二创新点分别另议。
