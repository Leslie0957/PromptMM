# Sports：PromptMM 发布版适配三种子300轮审计（2026-09-20）

三组均完整通过；这是共享教师、未调参发布版适配的Validation对照，不是原论文端到端复现或最终Test结论。

## 结果与配对比较

|方法|Recall@20 均值±样本SD|同一选中轮次 NDCG@20 均值±样本SD|
|---|---:|---:|
|ID-only BPR|0.04515130 ± 0.00043770|0.02153059 ± 0.00020296|
|PromptMM-release-Sports-sharedTeacher-v1|0.07693267 ± 0.00004001|0.03480523 ± 0.00002518|
|去文本（图像系数匹配）|0.07812907 ± 0.00068555|0.03555797 ± 0.00016149|
|完整方法|0.08230706 ± 0.00024695|0.03720290 ± 0.00010589|

|seed|发布版最佳轮（1起算）|发布版Recall|发布版NDCG|完整方法−发布版Recall|相对Recall增益|
|---|---:|---:|---:|---:|---:|
|2022|299|0.07697886|0.03480835|0.00534940|6.9492%|
|2023|300|0.07690957|0.03477864|0.00514063|6.6840%|
|2024|296|0.07690957|0.03482872|0.00563317|7.3244%|

完整方法均值相对发布版适配：Recall +6.9858%，NDCG +6.8888%；两项逐种子均为正。去文本匹配组也在三个种子上高于发布版适配，故不能把完整方法的全部优势归因于新增文本监督。三个种子不是统计显著性证明。TD值引用既有审计表（显示到10位小数），本次未重新评估。

## 曲线与初始化

|seed|初始Recall|前30轮最佳|前120轮最佳|前300轮最佳|最终Recall|末20轮均值−前20轮均值|
|---|---:|---:|---:|---:|---:|---:|
|2022|0.04734320|0.06579641|0.06954956|0.07697886|0.07692268|0.00083386|
|2023|0.04734320|0.06557636|0.06951679|0.07690957|0.07690957|0.00090300|
|2024|0.04734320|0.06558572|0.06927801|0.07690957|0.07680189|0.00080582|

发布版前120轮均值0.06944812，300轮均值0.07693267，提高10.7772%。与已有TD完整方法120轮均值0.07373697相比，差距从约6.18%变为300轮约6.99%；当前预算下没有追平。末窗口比较是261–280与281–300轮，均值和最高值三个种子全部仍提高；不能宣称充分收敛，也不自动追加轮数。
三个种子初始指标完全相同是固定教师初始化的预期结果；后续采样/训练态dropout等随机性不同。很小的跨种子SD不代表独立教师或数据划分稳健性。seed2022前30轮最佳0.06579641与独立30轮0.06575427不完全一致，符合已记录DGL随机性限制。

总目标约从−1.5e8降至−3.9e8，但BPR约从0.12升至0.60，feature从约0.784至0.790；验证仍提高。它表明各目标与推荐指标不必同步，不能把负loss当失败或把总loss下降当收敛证据，也不能仅靠标量大小断言梯度主导。

## 已证明与未证明

- 支持：固定数据、共享教师、相同300轮/每轮214更新及每轮选模下，当前完整方法在三种子Validation上优于这个明确命名、固定未调参配置的发布版适配；已有文本消融结论仍成立。
- 尚未证明：超过合理调优后的PromptMM、原论文端到端性能、Test泛化、显著性、独立教师稳健性、受控效率收益或论文创新性。
- 混杂/边界：发布版lr2e-5而TD为6e-5；教师初始化与随机ID初始化不同，图结构/目标不同、调参历史不同；CPU DGL适配、共享embedding别名和发布版异常KL语义保留；原版未参与目标的诊断计算省略。相同轮数不等于相同算力或调参预算。
- 现有NDCG保留top50相关向量构造理想DCG的既有定义，不与其他定义直接混比。

## 唯一建议下一步

准备seed2022、300轮、仅Validation的学习率单因素对照：将发布版lr从2e-5改为6e-5，其余全部保持本批次一致，从固定教师重新初始化，独立目录、窗口内不早停。对照本次seed2022，检验最直接的优化速度差异。先声明并准备手动命令，本次不实施或启动。该实验本身也不能替代完整公平调参，但比不加区分地继续堆轮数或立即加门控更直接回答当前不确定性。

## 执行与身份核验

启动commit42fe4aff67480a596a7017214fd00309b68ff5c9，分支codex/experiment/baby-teacher-baseline；三组启动干净且运行结束源码未变。批次核验与CPU checkpoint独立核验通过：源码指纹、配置/数据一致、全部有限值、共享别名副本一致、tensor digest、严格Recall选模和best.pt哈希正确。每组64200更新、301次学生Validation；总计192600更新、903次Validation。教师/学生Test均0，Test划分未读取；本次没有任何模型前向或新增排名。selected-versus-tested不适用。

共同参数/身份引用TRAINING_LOG.md的PROMPTMM_RELEASE_SPORTS_VALIDATION300_BATCH_V1：seed2022/2023/2024、batch1024、dim64、layer1、lr2e-5、AdamW decay0.01、embedding decay1e-5、pair/list1e6、feature0.1、SCE2、negative10、foreach=False、teacher drop0.2/prompt0，固定epoch37教师与SPORTS_CONVERTED_20260916。

批次约39.83小时；单组12.60/12.61/14.61小时。训练平均约16.5–17.0秒/轮，验证约134–158秒/轮。训练allocator峰值约1.6484GiB。系统负载及评估实现未控制，不据此声称相对TD效率优势。

总启动命令：`D:/miniconda/envs/run_5060/python.exe -B codes/run_promptmm_validation300.py`。每组实际命令与指纹如下。

### seed2022

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2022 --epochs 300 --gpu_id 0`
目录：`exp/promptmm_release/sports_promptmm_release_validation300_seed2022_v1/`
- best.pt SHA256 `41d921d313e01a061b02c3c00573a620add3706e6a873f576ea1de8013a41c01`
- report.json SHA256 `a0872a978fd0087190b1bb431897ca7d6dd73e840a12712d533c7933df657330`

### seed2023

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2023 --epochs 300 --gpu_id 0`
目录：`exp/promptmm_release/sports_promptmm_release_validation300_seed2023_v1/`
- best.pt SHA256 `279c5d21a1ef312286280f3997981f0e79f26d657b9c40ddf82cc7189fb7e5ea`
- report.json SHA256 `d5d3b5ff5a712e9b5fc98efd67ac73fe91cf9277036b7e7cb0e699502f3dfc31`

### seed2024

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2024 --epochs 300 --gpu_id 0`
目录：`exp/promptmm_release/sports_promptmm_release_validation300_seed2024_v1/`
- best.pt SHA256 `d991fe012e42311234032dd6f18a0eab0aeca6ed52db8d5532fa2c77dda6b2a5`
- report.json SHA256 `5c873d5c49cb9092a3cc1130a7a3df80d7f7f22cb6df1e9708da3b6ba7f38190`

批次batch.json SHA256 `e851278736b2d3e6909b2ed046ae4be6a969d94d3562462b2be9007867b4897c`。派生审计摘要在exp/audit/promptmm300_joint_audit.json。原始产物均保留且被Git忽略，本提交不构成物理备份或最终论文结果冻结。

所有原eligibility=false保留；Baby历史seed2023按审计例外接受、原manifest仍为false，未改写。第二创新点未确定。
