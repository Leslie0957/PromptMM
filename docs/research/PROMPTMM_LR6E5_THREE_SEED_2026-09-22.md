# PromptMM lr6e-5 三种子联合审计（2026-09-22）

三个种子均完成并通过核验。当前完整方法在三个种子的两项Validation指标上均低于该发布版适配配置；此前对lr2e-5配置的优势不能概括成方法整体超过PromptMM。

|配置，300轮|Recall@20均值±样本SD|同一选中轮次NDCG@20均值±样本SD|
|---|---:|---:|
|PromptMM lr2e-5|0.07693267 ± 0.00004001|0.03480523 ± 0.00002518|
|当前完整方法|0.08230706 ± 0.00024695|0.03720290 ± 0.00010589|
|PromptMM lr6e-5|0.08757300 ± 0.00008788|0.04001322 ± 0.00014882|

新发布版配置相对当前完整方法均值Recall +6.3979%，NDCG +7.5540%。分母为完整方法均值；反向差距不应复用此百分比。既有完整方法相对BPR、匹配图像权重去文本的消融证据仍成立。

|seed|最佳epoch（1起算）|Recall|NDCG|完整方法Recall|最终Recall|末20轮均值增量|
|---|---:|---:|---:|---:|---:|---:|
|2022|294|0.08763456|0.04012545|0.08232826|0.08717947|0.00047954|
|2023|299|0.08761208|0.04006979|0.08205020|0.08761208|0.00035881|
|2024|300|0.08747236|0.03984441|0.08254274|0.08747236|0.00008598|

末20轮为281–300，对比261–280；三个种子增量仍正，但2024增量较小。最佳294/299/300不等于充分收敛，也不支持立即无限追加轮数。三个种子初始Recall均0.0473432010，来自同一固定教师；很小SD不是独立教师或数据划分稳健性证明。

|seed|前30轮最佳Recall|前120轮最佳Recall|运行小时|平均训练秒/轮|平均验证秒/轮|
|---|---:|---:|---:|---:|---:|
|2022|0.06857338|0.07874371|13.00|16.50|138.89|
|2023|0.06845633|0.07868635|15.85|17.28|171.75|
|2024|0.06810987|0.07880925|15.58|17.21|169.16|

2023/2024合计约31.44小时，超过此前26–30小时估计。验证约占每轮训练+验证的91%；代码将GPU评分搬到CPU后逐用户Python/heapq排序和计算指标，属于已定位的实现瓶颈。具体时间波动的系统原因未测，不能认定前后台、电源策略或温度中某一因素。不同实现/负载下日志耗时不可用于论文效率结论。

## 核验与身份

2023/2024启动09dce8b5654e88e0b6ff510c1c28215c2b25e521，2022启动b6c801e3e8a09073d3f1c4e6b62fde45cd38917b，分支codex/experiment/baby-teacher-baseline。两源码间训练入口唯一差异为允许2023/2024使用6e-5的CLI守卫；其余记录源码哈希相同，无训练语义变更。最新两组当前源码指纹一致，2022原report固定SHA复核通过。

每组300轮/64200更新/301学生Validation，全部运行完成、原启动干净、结束源码未变；教师/prompt未变、学生更新、alias/roundtrip检查通过。逐轮有限值及严格选模标记重算，CPU载入best.pt核验配置、输入身份、launchcommit、epoch/metrics、有限权重、tensor digest和alias副本。未执行模型前向或新增排名。教师/学生Test均0，未读Test；selected-versus-tested不适用。

公共配置沿用PROMPTMM_LR6E5_REMAINING_BATCH_V1：SPORTS_CONVERTED_20260916、固定epoch37教师、batch1024、214步/轮、dim64、layer1、lr6e-5、AdamW decay0.01、embedding decay1e-5、pair/list1e6、feature0.1、SCE2、negative10、foreach=False、teacher drop0.2/prompt0；初始Validation不参选、每轮Recall@20严格改善选模、同分较早、不早停；环境run_5060/Python3.10.20/torch2.11.0+cu128/DGL2.2.1/RTX5060。三组公共配置和输入/初始指标一致，仅seed不同。

## 解释与唯一下一步

已证明的是当前固定配置、同300轮Validation比较中，发布版适配6e-5在三个种子均更好；不证明原论文端到端重现、统计显著、Test泛化或最优调参。lr6e-5在2022结果后选择，两个额外种子是训练随机性复核，仍使用同一个Validation集合；不是独立测试。初始化、图传播、损失及历史调参预算仍不同。

当前第一创新点不能以超过该基线的准确率作为已证实主张；轻量化价值需要受控训练/推理/表示更新成本与准确率取舍证据，不能自动成立。也不能由该比较单独断定课题必须重做。暂不加门控或铺开新长训练。

唯一下一步：优化并验证Validation实现，保持评分、候选排除、tie规则、指标定义、验证频率和选模完全一致；先做合成极端/并列对照，再准备固定已有checkpoint的仅Validation一致性及计时命令交用户手动运行，零Test、不重训。GPU topk并列行为不能假定等价。此步骤是基础设施，不是论文创新；之后再据可靠评估工具设计初始化/结构损失诊断和受控效率比较。本次未改代码或运行该验证。

## 运行命令与产物

批次命令：`D:/miniconda/envs/run_5060/python.exe -B codes/run_promptmm_lr6e5_remaining.py`。batch.json位于exp/promptmm_release/sports_promptmm_release_validation300_lr6e5_remaining_v1/，SHA256 `448823af7eaf5d2e734c6ea267623fb7affb94826f93767def0c6218cea5ab4c`。

### seed2022
命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2022 --epochs 300 --student_lr 6e-5 --gpu_id 0`。
目录：exp/promptmm_release/sports_promptmm_release_validation300_seed2022_lr6e5_v1/
- best.pt SHA256 `e32f9907d68874f09f4327a6ccfba8b4c86e1b7c73c7d0a06713f990b6e94e27`
- report.json SHA256 `84b6a90f2f5510ff22df24dfa2e267d634a4fbb6a260f0500f3062bf48b9ad03`

### seed2023
命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2023 --epochs 300 --student_lr 6e-5 --gpu_id 0`。
目录：exp/promptmm_release/sports_promptmm_release_validation300_seed2023_lr6e5_v1/
- best.pt SHA256 `9b85a76b655b5107211b59baba5e1325289b38a6cf0c191d25a26cad8166e9da`
- report.json SHA256 `8ec8b80d9be20c6b6f6e4b3e77e2d8ce5267870490616c1981df65fad17747b4`

### seed2024
命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2024 --epochs 300 --student_lr 6e-5 --gpu_id 0`。
目录：exp/promptmm_release/sports_promptmm_release_validation300_seed2024_lr6e5_v1/
- best.pt SHA256 `d8589d349cc388a5a1e73ed2517744640571ec59aeae1349eacdafcbd5da27a9`
- report.json SHA256 `4e51e8ea14f0b1c948f1616ca0664e90f3e30f039dc010a61c167497d2bb8721`

派生摘要exp/audit/promptmm_lr6e5_three_seed.json。原始产物保留，Git不保护忽略产物，此提交非物理备份或论文结果冻结。原eligibility=false不变，Babyseed2023按审计例外接受且原manifest仍false，第二创新点未定。
