# Sports 三种子固定300轮 Validation 审计（2026-09-18）

九组执行审计通过；这是固定教师、变化学生种子的有限预算验证诊断，不是最终Test结果冻结或充分收敛声明。

## 协议与来源

启动源码：`4060dc0495672905361c6f3c1d66fe95e9b271c7`，分支 `codex/experiment/baby-teacher-baseline`。串行sidecar记录launch_dirty=false、九组completed；启动器每组前检查HEAD和工作区。九份manifest源码指纹与审计时文件一致。

共享锚：SPORTS_CONVERTED_20260916；conversion SHA256 `3772a17c8b70fa4739653534dca8d542e67e0649f1f44ccba4194fec17bc1104`；教师epoch37 SHA256 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`，PCA seed2022。九组数据、教师与推理配置身份一致；沿用已核验共享锚，不重复读取整个数据集或执行评估。

环境沿用run_5060 Python3.10.20 / torch2.11.0+cu128；学生随机ID64、无教师初始化，lr6e-5、weight_decay0.01、batch1024；epoch=patience=300；Validation Recall@20选模；用户蒸馏项为0。full的alpha0.3、image/text rates1/0.3；image_matched的alpha0.3/1.3、rates1/0；BPR alpha0。

九组日志各连续300轮（0–299），曲线目标和指标有限，checkpoint模型/优化器张量有限；最佳轮次与manifest/曲线/checkpoint一致；推理导出两张ID表与完整checkpoint逐元素一致。恢复最佳后跳过Test：教师Test0、学生Test0；无selected-versus-tested比较。读取已存在的preflight结构统计不属于新增Test排名评估。

## 三种子汇总（样本标准差ddof=1）

|方法|Recall@20 均值±SD|同轮NDCG@20 均值±SD|
|---|---:|---:|
|bpr|0.04515130 ± 0.00043770|0.02153059 ± 0.00020296|
|image_matched|0.07812907 ± 0.00068555|0.03555797 ± 0.00016149|
|full|0.08230706 ± 0.00024695|0.03720290 ± 0.00010589|

## 单次结果

|seed|方法|最佳epoch|Recall@20|同轮NDCG@20|
|---|---|---:|---:|---:|
|2022|bpr|298|0.0455663389|0.0217154295|
|2022|full|293|0.0823282598|0.0372074078|
|2022|image_matched|299|0.0786500158|0.0357300061|
|2023|bpr|295|0.0451935473|0.0215629346|
|2023|full|291|0.0820501989|0.0370948217|
|2023|image_matched|287|0.0783847734|0.0355342423|
|2024|bpr|298|0.0446940025|0.0213133927|
|2024|full|292|0.0825427359|0.0373064581|
|2024|image_matched|285|0.0773524120|0.0354096525|

## 配对差值

|seed|full−image Recall|相对增益|
|---|---:|---:|
|2022|0.0036782440|4.6767%|
|2023|0.0036654255|4.6762%|
|2024|0.0051903239|6.7100%|

均值之比：full相对image_matched Recall +5.3476%、NDCG +4.6260%；相对BPR Recall +82.2917%。三个种子的两项配对指标均为正。不以三个种子方向一致声称统计显著性。

## 120轮前缀与预算影响

九组前120轮Recall@20逐点完全一致。其他指标/目标不统一声称逐位相等：同名曲线列表的最大绝对差不超过2.80915e-5（full seed2024），未影响Recall@20前缀；不同预算的选模来自更多候选轮次。

|方法|120轮平均Recall|300轮平均Recall|相对提高|
|---|---:|---:|---:|
|bpr|0.04037980|0.04515130|11.8165%|
|image_matched|0.06997523|0.07812907|11.6525%|
|full|0.07373697|0.08230706|11.6225%|

full/image的增益由120轮5.3758%变为300轮5.3476%；full/BPR由82.6086%变为82.2917%。延长预算后文本项优势保留，BPR没有明显缩小相对差距；这削弱“优势完全来自120轮预算”的解释，但不证明无限预算下的最终优势。

## 曲线末段

|seed|方法|260–279最高R|280–299最高R|末窗口最高差|末窗口平均差|
|---|---|---:|---:|---:|---:|
|2022|bpr|0.04533224|0.04556634|0.00023410|0.00035627|
|2022|full|0.08189152|0.08232826|0.00043674|0.00039554|
|2022|image_matched|0.07787984|0.07865002|0.00077017|0.00051468|
|2023|bpr|0.04474268|0.04519355|0.00045087|0.00043304|
|2023|full|0.08172882|0.08205020|0.00032138|0.00018341|
|2023|image_matched|0.07814600|0.07838477|0.00023878|0.00020277|
|2024|bpr|0.04454535|0.04469400|0.00014865|0.00013665|
|2024|full|0.08202751|0.08254274|0.00051522|0.00076311|
|2024|image_matched|0.07709491|0.07735241|0.00025751|0.00016313|

九组最佳epoch285–299，最后20轮最高值和均值均超过前20轮。仍有边界和残余学习趋势，不能称充分收敛；本次不自动延长预算。

## 解释边界与唯一下一步

已支持：固定实际图像系数、固定教师和300轮预算下，文本监督在三种子上带来稳定正向Validation增益；完整方法明显高于BPR。尚未证明：正式Test泛化、教师种子稳健性、统计显著性、文本语义因果机制、相对PromptMM的新颖性或效率优势。图像目标来自多模态教师，去文本不是纯视觉教师。

Sports官方特征语义来源和冷物品限制保留；九组paper_ready_eligible=false（Validation-only），教师原false及Baby历史seed2023按审计例外接受而原manifest仍false均不改写。第二创新点未确定。前后台和系统负载未控制，不从耗时日志推断速度优势。

**下一步：准备原始PromptMM在统一Sports数据协议下的公平对照与RTX5060资源检查方案，核对官方实现、预提取特征、Validation选模、调参预算和受控效率指标，再提供用户手动命令；不在本任务启动。** 不因追求复杂度直接加门控；门控须有问题证据、任务学习信号及固定/全局学习权重对照。

本次归档2026-09-17交接计划；后续训练均由用户手动执行。保存原始run声明，追加结果。不建立最终结果冻结、稳定里程碑或物理备份声明；Git不保护忽略的原始产物。

## 审计复核与产物身份

审计命令：`D:/miniconda/envs/run_5060/python.exe -B exp/audit/audit_sports300.py`（只加载现有产物，无模型前向或Test评估）。
串行sidecar：`exp/runs/sports/serial_validation300_seed2022_2024.json` SHA256 `91c50dd3c277f1397fea3399fe36b9a88397a015ca7d624d0524f63de0dc7dea`。
审计脚本SHA256 `1df478caca3e296a2a75e394fe139e118e32c3f8104749c97e2c4b58367d8cde`；明细JSON `exp/audit/sports300_audit.json`。

### sports_student_bpr_seed2022_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2022_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-17 11_50_42.452019_sports_light_init_pid21216.json` SHA256 `78eceaa381efccdd5f157e6d78f2d5ab0041780e3f62fef861af227ede422254`
- `D:\Download\PromptMM\logs\2026-09-17 11_50_42.452019_sports_light_init_pid21216` SHA256 `4ecd61349af67b624df893e011ec99374378e7f92d20f5632b858b49abc8131f`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 11_50_42.452019_sports_light_init_pid21216.pkl` SHA256 `ae461c674f1fbb457e8528fc1f3b434050d2ffaddb6e8074d0e17fd734ca9473`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-17 11_50_42.452019_sports_light_init_pid21216.json` SHA256 `c01f47b55a2ef8a279c8111cda1301347548500b46b5bc114c2937abf48e4c04`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 11_50_42.452019_sports_light_init_pid21216.pth` SHA256 `6ca67b9c6d2b00e939fafcc0c5296c6710a43c8ccdcc80d6c8d53c3c0472ad0d`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 11_50_42.452019_sports_light_init_pid21216.pth` SHA256 `9902e8dfc6ec9c39680004ea3fa31839ff27874cea4cf71bdc3dd59e5be8c16a`

### sports_student_full_seed2022_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2022_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-17 17_08_55.469360_sports_light_init_pid47952.json` SHA256 `17088e5a6e58f6f33db99a800341e4e0d038537b2e721a4ad0c283a013ccc836`
- `D:\Download\PromptMM\logs\2026-09-17 17_08_55.469360_sports_light_init_pid47952` SHA256 `f8f7f9653e92a7cd9d7882069e60239d6953e50bdf2ac6728246ad16d8c54ce5`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 17_08_55.469360_sports_light_init_pid47952.pkl` SHA256 `2274c95de4adaa651e8e1d8ff4c2ee2720f9b4eee985f6abbd7c4833e5820685`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-17 17_08_55.469360_sports_light_init_pid47952.json` SHA256 `8e2ab17c96f8fc9ec39edecc42c5166cadb2de9d4c3ba46b287d85523ffc9d14`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 17_08_55.469360_sports_light_init_pid47952.pth` SHA256 `54288e702d6f2e94bcb088bf94b12e50227f7e055b6808cd94ab83d52acd126e`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 17_08_55.469360_sports_light_init_pid47952.pth` SHA256 `04acfc443747c68846f534bff4e80ab8107dcc265e13628fda00e4c83eacf6b9`

### sports_student_image_matched_seed2022_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2022_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-17 20_34_43.591646_sports_light_init_pid30528.json` SHA256 `447e97e12797b85e8eb676f894db84c40f641910fe42d7d7ac1c20d3ab95e24f`
- `D:\Download\PromptMM\logs\2026-09-17 20_34_43.591646_sports_light_init_pid30528` SHA256 `f5abf3bf2046326ff24b4f3cd5265e560fdd76b1507707b3ce68c73985ff9e43`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 20_34_43.591646_sports_light_init_pid30528.pkl` SHA256 `d61caaf9dd38dffdea12a49397e8d8e49ec24b61d9dd1063033a1ad6d2c0805c`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-17 20_34_43.591646_sports_light_init_pid30528.json` SHA256 `ebbf178cdbe69acac29224b64f97d9eaf9e52072595d76d3192245f55d4f6acc`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 20_34_43.591646_sports_light_init_pid30528.pth` SHA256 `dd3fe607b62649c5e40c465361e75549a604a77dc429a3202ad6a6f74a65fde7`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 20_34_43.591646_sports_light_init_pid30528.pth` SHA256 `d5c7fb8c35d36ad65b3f3f1a69caae2913fc6cd4c8f3c495033118c8f6f1f781`

### sports_student_bpr_seed2023_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2023_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-17 23_49_20.915423_sports_light_init_pid48412.json` SHA256 `3a30b6481db36b0580a5364438b72fba0bbd0e2766a633add93d36acc1269be6`
- `D:\Download\PromptMM\logs\2026-09-17 23_49_20.915423_sports_light_init_pid48412` SHA256 `44909d6399075a58003c40ee24e5395fea08c0c211c6ae2a41b42845581d73d2`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 23_49_20.915423_sports_light_init_pid48412.pkl` SHA256 `a42609935b0159d13c80da6208192b9d931852bf70656dfa950c3ad6c300411f`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-17 23_49_20.915423_sports_light_init_pid48412.json` SHA256 `d84e08c4724ff1bbb446126dc251b50ce0ceefca2d8368fc9d08a470d3e27f7a`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 23_49_20.915423_sports_light_init_pid48412.pth` SHA256 `6278b05d08120b0400cb1ea5efff23311ee390921fb9458bcb809530aa404d49`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 23_49_20.915423_sports_light_init_pid48412.pth` SHA256 `859e94dac1b33a8084b2fc8822ea751f7da73c2ddcefc4ac11778b080072f18f`

### sports_student_full_seed2023_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2023_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-18 02_55_57.358892_sports_light_init_pid38584.json` SHA256 `04b4258411e079add8992a38382bc7339a71bba19072926b9e0d90020fed99c2`
- `D:\Download\PromptMM\logs\2026-09-18 02_55_57.358892_sports_light_init_pid38584` SHA256 `772c98d249ce0cc86acd209e481703209474d409e29d9609500333c5afe02d63`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-18 02_55_57.358892_sports_light_init_pid38584.pkl` SHA256 `c83af19bcb8e4f69ea38a59516bc73fab370a281994ca9b612da4596dd0f56c5`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-18 02_55_57.358892_sports_light_init_pid38584.json` SHA256 `43c20fed8893e1050f1a08b79e4e1b72639129c4c8d50b2044ce091c27d9a628`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-18 02_55_57.358892_sports_light_init_pid38584.pth` SHA256 `d0a491b539482747f2ad69dcdc8c4dd3d0c9fb1662c5c767e8f4fa8a8f51fd34`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-18 02_55_57.358892_sports_light_init_pid38584.pth` SHA256 `c3c9a7917ae870cc6b0641da16a5b65a53166b04c7cd7380188b54c75aff5afb`

### sports_student_image_matched_seed2023_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2023_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-18 06_05_32.090309_sports_light_init_pid43032.json` SHA256 `58994307bddb67ac0b500e4d01ee208a394fc15f1ff9a1abb13db0f0fc7c79e0`
- `D:\Download\PromptMM\logs\2026-09-18 06_05_32.090309_sports_light_init_pid43032` SHA256 `7366e7a4a2545044af251b3bb2b70d063be70ee2f01861179cac48386293a62a`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-18 06_05_32.090309_sports_light_init_pid43032.pkl` SHA256 `def9495daa3c550ac771b1656f80579894a6d13b704b484b2ef9f8b4a650c00e`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-18 06_05_32.090309_sports_light_init_pid43032.json` SHA256 `b1fb7528eca82f7382b1804434b41a171e500a26f806cdca829063c717dcc80a`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-18 06_05_32.090309_sports_light_init_pid43032.pth` SHA256 `9fd7ffea66be23d687ec422f16571ba4da27a11be3911efcf49b158c6d2d3b72`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-18 06_05_32.090309_sports_light_init_pid43032.pth` SHA256 `6c5107e831000e673b69ff14279e88598f2408fa92dfdea4f02545ca157e7eb3`

### sports_student_bpr_seed2024_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2024_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-18 09_13_44.833581_sports_light_init_pid40228.json` SHA256 `844d339aade57867f32434d75939d7971b19a975903c14489598be88bff23d90`
- `D:\Download\PromptMM\logs\2026-09-18 09_13_44.833581_sports_light_init_pid40228` SHA256 `37b6ef2e047fa087e905a40227d334c77ec2f6a5d656b5302f02a1516e2c3fd2`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-18 09_13_44.833581_sports_light_init_pid40228.pkl` SHA256 `5b989fe1b726e72549c2de9f6e75cd3ff4887d6c3f520f565e294ff819e36688`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-18 09_13_44.833581_sports_light_init_pid40228.json` SHA256 `83d9d77e2563e26a0df9709961d177a9c0c65723edc47ac232ac2146c097f2a5`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-18 09_13_44.833581_sports_light_init_pid40228.pth` SHA256 `d1a8073af9ef1933704086882125b6e956282bee448993de04e4d8f1a7b57daa`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-18 09_13_44.833581_sports_light_init_pid40228.pth` SHA256 `46097fd5e34d5cb12fed4bc84e40e04d12a729025d38f37ffc195262badfa9f5`

### sports_student_full_seed2024_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2024_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-18 12_21_00.822710_sports_light_init_pid47072.json` SHA256 `d5ec82574f10c058d4e4cb5b6c588e9357bda2f1feb27e3320e560e5a4d732a4`
- `D:\Download\PromptMM\logs\2026-09-18 12_21_00.822710_sports_light_init_pid47072` SHA256 `db2be966832c249c317a9ff2926a8d89a5a8ca90f2636f7f8e7a2202636c6139`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-18 12_21_00.822710_sports_light_init_pid47072.pkl` SHA256 `73c2e87cdb9a268cdd742a6f8782d669c4b02aebf9a0128d2c0bb0ae8cf7bdd1`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-18 12_21_00.822710_sports_light_init_pid47072.json` SHA256 `dfc13e7e6e1f7f7855a2e50003d1e1b3987e2ee2c596b3279167173d58a152fa`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-18 12_21_00.822710_sports_light_init_pid47072.pth` SHA256 `900fb30ae273b46c4697d62e4b15a7a625ec3cfa8a0d0848151e46e286c3bd1b`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-18 12_21_00.822710_sports_light_init_pid47072.pth` SHA256 `381fcb2376486a512d5af1c82ea36446660fd982d29c1560d91a5dc34e14b264`

### sports_student_image_matched_seed2024_val300_v1

命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2024_val300_v1 --gpu_id 0`
- `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-18 15_59_59.130995_sports_light_init_pid33376.json` SHA256 `59d7b783f20eda3bfb309b7e656426883360fc69ddb740cf98d738d57388dc27`
- `D:\Download\PromptMM\logs\2026-09-18 15_59_59.130995_sports_light_init_pid33376` SHA256 `91789405573c8242a89f93dc067abab6d14a28489c7fd583acc2ed949517feb0`
- `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-18 15_59_59.130995_sports_light_init_pid33376.pkl` SHA256 `80f9c13a888b064bfc36204b97bf73402cc544d2339dca4b5205074159725527`
- `D:\Download\PromptMM\exp\runs\sports\dataset_preflight__2026-09-18 15_59_59.130995_sports_light_init_pid33376.json` SHA256 `e0986407f0063667117dbf6c6585228c9532455aaee4d97c43f0ff8a59659656`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-18 15_59_59.130995_sports_light_init_pid33376.pth` SHA256 `dbe1e67c495388afbdc8a7bc8a66247585c9b9f5d88d4d4bfc4387800fce0cc8`
- `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-18 15_59_59.130995_sports_light_init_pid33376.pth` SHA256 `a1c9186b93c931bf4eae987cbab6c048e1ef970daad2a5623223403f8dafbf05`

## Preflight警告核对

九份preflight均为warning，唯一警告是已声明retain_official冷物品：Validation 4个训练未见物品/9条交互，Test结构统计5个/17条。没有把warning改写成passed，也未新增Test评估。报告生成器最初误要求status=passed而停止（未写报告/结果段），核对九份原警告后改为精确接受该已声明警告；训练本身没有失败或重跑。
