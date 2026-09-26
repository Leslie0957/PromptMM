# 第一创新点仅评价恢复批次：正式结果审计

## 结论与执行身份

用户于 2026-09-26 16:13:02–16:44:50（Asia/Shanghai）从干净的 `codex/experiment/baby-teacher-baseline` 提交 `b3ea32fd7f9685631d33a60f52591e8dd3e0cbb8` 手动执行一次声明的命令：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_eval_recovery.py' --conditions-confirmed
```

`innovation1_eval_recovery_v1` 总账状态为 `completed`，24/24 步均为退出码 0 且各自报告通过：**12 项全新无 Test Validation 预检，随后 12 项一次性学生 Test**。没有训练步骤、自动重试或跳项。报告均记录该启动提交、账本 SHA256 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`，一致的 Python 3.10.20 / Torch 2.11.0+cu128 / CUDA 12.8 / RTX 5060 环境及评价源码指纹。Baby 三个来源仍为 v2 在 `da2371ebf6de2a5a0daf06444c7f1fd32837ce37` 训练的所选文件；Sports 九个来源仍为共享账本固定选择。原 v1/v2 产物未重跑或改写。完整命令、参数和 Test 策略见[恢复声明](INNOVATION1_EVAL_RECOVERY_V1_LAUNCH_2026-09-26.md)，v2 训练身份与有效性见[v2 审计](INNOVATION1_FORMAL_CLOSEOUT_COHORT_V2_AUDIT.md)。

只读复核时，共享账本、v2 总账及 v2 失败的 `s3_2022` 预检报告 SHA256 分别仍为 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`、`e7febbd5ba054996a53c1dcce96eac5b6f1ab97238bfdee5a61faf5dc167b82f`、`5589100c9de6593adcb494a199e18116f9d3343b4af74776c38d18962084df40`，均匹配原审计。

## 逐格结果与协议验收

表中 Val 为原所选检查点的 Recall@20，预检与最终评价均精确重放该值（差值 0）；Test 为同一检查点一次评估的 @20 指标。三个种子固定为 2022/2023/2024，未据 Test 重选轮数或配置。

| 格子 | 方法/数据集 | 所选 epoch | Val R@20 | Test R@20 | Test N@20 | Test 用户数 |
|---|---|---:|---:|---:|---:|---:|
| B3-2022 | Baby PromptMM release 共享教师适配 | 2 | 0.065426294 | 0.065212928 | 0.028833987 | 19445 |
| B3-2023 | 同上 | 2 | 0.065529148 | 0.064985792 | 0.028764021 | 19445 |
| B3-2024 | 同上 | 3 | 0.065297726 | 0.064917222 | 0.028767068 | 19445 |
| S1-2022 | Sports BPR | 298 | 0.045566339 | 0.044253801 | 0.022405469 | 35598 |
| S1-2023 | 同上 | 295 | 0.045193547 | 0.045681350 | 0.023116953 | 35598 |
| S1-2024 | 同上 | 298 | 0.044694002 | 0.042925465 | 0.022272670 | 35598 |
| S2-2022 | Sports Full | 293 | 0.082328260 | 0.082653464 | 0.039671728 | 35598 |
| S2-2023 | 同上 | 291 | 0.082050199 | 0.082831767 | 0.039488463 | 35598 |
| S2-2024 | 同上 | 292 | 0.082542736 | 0.083239127 | 0.039600124 | 35598 |
| S3-2022 | Sports PromptMM release 共享教师适配 | 294 | 0.087634556 | 0.089861155 | 0.043010790 | 35598 |
| S3-2023 | 同上 | 299 | 0.087612083 | 0.089998937 | 0.042965309 | 35598 |
| S3-2024 | 同上 | 300 | 0.087472362 | 0.089945429 | 0.042897203 | 35598 |

| 三种子组 | Test R@20 均值 ± 样本 SD | Test N@20 均值 ± 样本 SD |
|---|---:|---:|
| Baby B3 | 0.065038648 ± 0.000154777 | 0.028788359 ± 0.000039545 |
| Sports S1 BPR | 0.044286872 ± 0.001378240 | 0.022598364 ± 0.000453993 |
| Sports S2 Full | 0.082908119 ± 0.000300204 | 0.039586772 ± 0.000092359 |
| Sports S3 release | 0.089935173 ± 0.000069461 | 0.042957767 ± 0.000057168 |

已逐份只读核验 24 个报告及总账，非仅凭终端输出：所有步骤顺序无交叠，12 份预检先于首个 Test；同一格子的预检/最终报告来源路径、训练/Validation/Test身份哈希、所选状态与指标实现指纹一致。所有预检 `passed`、Test 访问/加载均 false、学生/教师 Test 计数均 0；所有最终报告 `completed`、Test 访问与加载 true、`student_test_attempts=student_test_evaluations=1`、教师 0，持久首次访问时间位于报告开始与结束之间。每份 `selected_checkpoint_sha256` 与 `evaluated_checkpoint_sha256` 相等，来源报告和所选检查点的文件 SHA256 与记录相等；Test 文件身份匹配账本预期。Val 重放误差均 0，所有 Val/Test 指标有限，候选排除均为 `train_only`，K 固定 `[10,20,40,50]`。此前 v2 训练已审计有限目标/损失；此次评价没有新的训练目标。全部最终报告 `paper_ready_eligible=true`；这是本次固定协议的技术资格，不等于方法新颖性或论文整体充分性。

## 结果含义与限制

- 固定配置主比较的 Sports Test 上，Full 均值 0.082908119 高于 BPR 0.044286872（差 +0.038621247），三个 seed 均如此；发布版适配对照 0.089935173 又高于 Full（Full−release = −0.007027054），三个 seed 均如此。该负向直接基线结果须随主表保留。发布版适配与 Full 的初始化、结构、配置选择机会不同，不能把差值解释为单个机制的因果效应，也不称充分调优或 SOTA。
- Baby 既有正式 Full Test R@20 均值 0.066243941、固定 BPR 为 0.041131438（见[Baby 总册](INNOVATION1_THESIS_EVIDENCE_2026-09-15.md)）；新 B3 release 均值 0.065038648，略低于既有 Full（Full−B3 约 +0.001205293），但高于既有 BPR。三法来源与初始化/优化路径不同；本批结果补齐固定配置直接对照，不证明文本项在 Test 上的独立效应。原 Image-0.3 2023 审计例外与 2024 低值仍照原记录保留。
- 两数据集的固定主比较格子现有 Test 来源，但两者训练预算不同、教师和划分各固定一份；三学生 seed 不是三个独立数据集或教师。此前严格 Full/image 与 real/sham 是 Sports Validation 机制对照，新 Test 不把这些诊断自动升级为 Test 机制证据。暖启动与缓存效率的原结论不变。论文创新性、强基线充分性、第三数据集是否必要、第二创新点和外部学位要求仍未由本次评价解决。

## 原始产物与保留

总账：`exp/formal_closeout_cohort/innovation1_eval_recovery_v1/batch.json`，SHA256 `fd4d1bf7eb57678db25e3ba32a818d17ab512c5eaebc6c48ac57d20547f79e0f`；各步 `.log` 同目录。以下每行在 `exp/formal_closeout_preflight/innovation1_eval_recovery_v1/<slot>/report.json` 与 `exp/formal_closeout_eval/innovation1_eval_recovery_v1/<slot>/report.json`，依次为预检与最终报告 SHA256。原始 JSON 与检查点保持 Git 忽略、原路径不动；提交记录这些指纹但不是大资产的物理备份。

| slot | 预检报告 SHA256 | 最终报告 SHA256 |
|---|---|---|
| b3_2022 | `5435b15dc422170a326a0dee32fc212be712657f56e51d47b9e3186a8b450d95` | `670497bfbe8092b8fb8f57e53b1292fe4b5ae1b8e70a82e4c5a866f8592347bb` |
| b3_2023 | `32d31862bb563333e48f29c102c62783a06a87d5581b6e24a24ebd67ea32817b` | `5bb57a7c196f5af3ccc5ab45ec61d8475c82524e659352a87d876367bd98523b` |
| b3_2024 | `05700dfef6d8a901c1bfd6b88e7c00f8cdd6ce4f3502462db6c42c51ab20516a` | `0759cf6a078122d733b830d56d649ec06b4cbfd366f0f8e160e9a0892c025d17` |
| s1_2022 | `3b9a79f4597d081a0ea6b56cc528132c16eae777973454337b91fa2f5487c9d3` | `d98b132ef76e10eaca216c397abb4b7cab09a34b2c10e84c9f58dcc00de0e417` |
| s1_2023 | `6e01eeb7af514b49dd3bb0a8031b8ae9b7f1f25875b0d44914df315f60d658b1` | `47bfc688f502f73d6ac5f8abc131787a5cac9256459480de37f5f00a934ed874` |
| s1_2024 | `83fcf913f285814bef77c17480bd4ead6c6913045f24a8a0312466b59180607f` | `2f6ea3c5045616cbcc5fe4bdd53ce5deb9b514375c3125e367551b8b4df36c51` |
| s2_2022 | `559b1c6c8a4198813e687be598ad01f7b1470c579b32ec39bb89d52665a2a578` | `d97dd40f2a461f1223988a2d0ff2e919e0a8d86e46e65961a844570035e6d59f` |
| s2_2023 | `06f9896428f61dc0f447624a63c96d49c203704f2a9fa37d0f383461fb0ce129` | `7ee7a875f6ae9b6ba9fa793be071d5d797ca891408a04acfe621a6b14b0a5f7a` |
| s2_2024 | `478a3fe67b6816f4b48554fe3d5e179203963389ab25512d0a9a9c2b6b6f7f07` | `3c3c8ddcafbc7aeae45f34f45cb17e7c44579a2ebe05c12a9fceaa86ba0d23c2` |
| s3_2022 | `a023dae957f9a4399ae1144d48b334ba1e10be7ba18414224d2c4601ca86c040` | `67c72111ba2fa9cf2742a74e8e9cc77f01c99f691d7b75d885ad05098f590762` |
| s3_2023 | `65aca8ecb3ab36e7e4092c9ea3542a0f75055070ff3cc398fda65255c5d483c0` | `e868e07d96092f4373056b493ae1e0f73e41899b2d634338c0d894ff0780b648` |
| s3_2024 | `6f982db35edda6a0ecb4545825500cc4d2c40d58c78d603731a8e12c94ab67e3` | `5881820c488c2073eeec10883b2ba570df4c9b5c1c086d457fc4717203c8f60d` |

记录路由：本审计和 `TRAINING_LOG.md` 必更；`docs/experiments/README.md` 发布版行、正式矩阵 B3/S1/S2/S3 全部三 seed、论文缺口表的 G2/G7/G8 与覆盖矩阵、Sports 稿件的正式比较范围及 root/docs 导航因状态变化而更新；生成目录刷新。第二创新点候选、机制原审计、v1/v2 原报告、参数/协议和历史 Baby 结果无变化；没有根据 Test 修改配置。**唯一下一步：只读完成第一创新点 18 格正式主表的跨来源资格与论文表述复核，再决定是否进入结果冻结；不自动开始第二创新点实验。**
