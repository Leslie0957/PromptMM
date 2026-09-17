# Sports 三种子固定120轮 Validation 汇总（2026-09-17）

新增六组均通过执行审计，连同seed2022共九组。每组120轮，Validation Recall@20选模、恢复最佳后结束，零教师/学生Test评估。此为固定教师、变化学生种子的验证诊断，非最终Test结果冻结。

## 单次结果

|seed|方法|最佳Epoch|Recall@20|同轮NDCG@20|
|---|---|---:|---:|---:|
|2022|bpr|118|0.0405241995|0.0197346992|
|2022|full|115|0.0737743396|0.0332764874|
|2022|image_matched|119|0.0697015452|0.0317525009|
|2023|bpr|118|0.0410379072|0.0196904229|
|2023|full|119|0.0738383825|0.0333438054|
|2023|image_matched|119|0.0705027763|0.0320580113|
|2024|bpr|119|0.0395772980|0.0191380282|
|2024|full|117|0.0735982004|0.0333559010|
|2024|image_matched|119|0.0697213653|0.0318151017|

## 三种子均值与样本标准差（ddof=1）

|方法|Recall@20 均值 ± SD|NDCG@20 均值 ± SD|
|---|---:|---:|
|bpr|0.04037980 ± 0.00074093|0.01952105 ± 0.00033244|
|image_matched|0.06997523 ± 0.00045698|0.03187520 ± 0.00016138|
|full|0.07373697 ± 0.00012437|0.03332540 ± 0.00004279|

## 配对差值

|seed|完整−去文本 Recall绝对差|Recall相对增益|NDCG相对增益|
|---|---:|---:|---:|
|2022|0.0040727944|5.8432%|4.7996%|
|2023|0.0033356062|4.7312%|4.0108%|
|2024|0.0038768351|5.5605%|4.8430%|

均值之比：完整方法相对去文本Recall提升5.3758%、NDCG提升4.5496%；相对BPR Recall提升82.6086%。这与“各seed百分比的均值”是不同统计量，本文明确采用均值之比。

## 可以与不可以推出的结论

- 已支持：固定120轮、相同实际图像系数0.3/1.3时，加入文本项在三个学生种子上都改善Validation；完整方法相对BPR也保持正向。
- 尚未支持：充分收敛后的优势、正式Test泛化、统计显著性、教师随机种子稳健性、文本语义机制。去文本的图像目标仍来自多模态教师；不能称为纯视觉教师对照。
- 预算限制：九组最佳均在115–119；所有组最后20轮窗口最高Recall仍超过80–99轮窗口。patience120没有实际提前终止，因此不是组间训练轮数不同，但仍存在未收敛与学习速度差异的解释空间。
- 本次不是效率实验，前后台窗口状态和系统负载未控制；不根据日志时间声称速度优势。

## 曲线后段诊断

|seed|方法|100–119窗口最高Recall − 80–99窗口最高Recall|
|---|---|---:|
|2022|bpr|0.0017211458|
|2022|full|0.0020327297|
|2022|image_matched|0.0017761610|
|2023|bpr|0.0016077682|
|2023|full|0.0019795120|
|2023|image_matched|0.0021780250|
|2024|bpr|0.0017332434|
|2024|full|0.0022941364|
|2024|image_matched|0.0025268273|

## 验证边界与下一步

六组新增完整/推理checkpoint均检查：最佳轮次与指标一致，权重/优化器张量有限，两个导出ID表与完整权重逐元素一致。九组曲线与日志连续120轮，指标/目标有限，无参数覆盖，数据/教师身份一致，用户项0；BPR语义项0，去文本text项0。日志确认恢复最佳后跳过Test；无selected-versus-tested比较。每份新run源码指纹与当前1e6f305文件一致；运行时未保存Git HEAD/dirty快照这一追溯缺口仍保留，不逆推启动时干净。

下一步优先准备seed2022三组240轮、仅Validation、不早停的同预算收敛诊断，从相同初始化规则重新运行，保持既有120轮结果原样；不假设现有最佳checkpoint可精确续训。240轮是有限诊断窗口，不保证收敛；准备时先检查120轮前缀可比性要求与固定配置。当前任务不声明或启动新训练。

仍保留Sports冷物品、官方特征语义来源限制；教师原paper_ready_eligible=false、Baby历史seed2023审计例外及其原manifest false均不改写。第二创新点未确定。当前不是最终论文结果冻结，未作稳定里程碑/物理备份声明；Git仅保存报告与日志。

## 新增六次运行身份


### sports_student_bpr_seed2023_val120_v1

- exp\runs\sports\run_manifest__2026-09-16 23_27_49.601262_sports_light_init_pid40628.json SHA256 54c87cacfcdd316d881ce12475e41639a437171f5b1c43ee95f60b7f1524f7a8
- logs\2026-09-16 23_27_49.601262_sports_light_init_pid40628 SHA256 d21a57e3408f27ff47c06c188ab0632326ebedd63f13dae004f697a063c51ecd
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-16 23_27_49.601262_sports_light_init_pid40628.pkl SHA256 4aa44cdc186ef9b9f78d10237d13095e252d3a876e8f0b80bfd288bfd707ede0
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-16 23_27_49.601262_sports_light_init_pid40628.pth SHA256 c3f487a02e9e7da8a4c1d848f3ad9b892e37d4f1349dafa6288fa8c23f9dbeee
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-16 23_27_49.601262_sports_light_init_pid40628.pth SHA256 3cd333d390083dc412bd232b8103fd8d0dbd6dbacfd25af5aaaafe4e20c616d2

### sports_student_full_seed2023_val120_v1

- exp\runs\sports\run_manifest__2026-09-17 00_43_06.905567_sports_light_init_pid9324.json SHA256 5fdf1e34e0bec607b5555597579a1622c2222f83f5aff51ae187b7f65a89e053
- logs\2026-09-17 00_43_06.905567_sports_light_init_pid9324 SHA256 17613ccf0f417c0b45960d669297b7b1bb602391e2f3712e22172d4f0debcc90
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 00_43_06.905567_sports_light_init_pid9324.pkl SHA256 9d11fe4c7000d5c8ecd7831526fe2aa65c6210183d8854b39e0aa377b102c1c0
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 00_43_06.905567_sports_light_init_pid9324.pth SHA256 9ad9899d0bed02bf0fbb46234e21ac45d69ff66347d0798ae43147f73489a1b3
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 00_43_06.905567_sports_light_init_pid9324.pth SHA256 7a58431cb60dbf710526ab4f0bb527ba3b86c18dca17ac6c137a572ac33d8198

### sports_student_image_matched_seed2023_val120_v1

- exp\runs\sports\run_manifest__2026-09-17 01_59_31.370287_sports_light_init_pid45812.json SHA256 15f275c7041644c94301e5fcbee73dca933d9468c854bed998ce16c5d194d93e
- logs\2026-09-17 01_59_31.370287_sports_light_init_pid45812 SHA256 06f06ccbba5a9f3c86d67a9276d13affad1d61c4bab78d1cd3028d920c6e3e83
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 01_59_31.370287_sports_light_init_pid45812.pkl SHA256 c5264d13b244e94f27c837f5ef08e18a7bb687b6768d1d42dcb20a8298d1d88e
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 01_59_31.370287_sports_light_init_pid45812.pth SHA256 330077600cce1468e0a7f6f25314e221edd42a65716d956223852e2a3e920478
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 01_59_31.370287_sports_light_init_pid45812.pth SHA256 7fa9dbf7b05d90e7bb504a5c0796b95799f73f2736e84bad114c1b8e12281bf4

### sports_student_bpr_seed2024_val120_v1

- exp\runs\sports\run_manifest__2026-09-17 03_15_48.676117_sports_light_init_pid48292.json SHA256 446877a9edbd47c5a8a4c54aa063130eda46299dce911a70dbfcfcac30b08a72
- logs\2026-09-17 03_15_48.676117_sports_light_init_pid48292 SHA256 cc8b006b42c70bcaf072a561c296bbcb4d8da1719adfbb63738a1db9e9d61bbd
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 03_15_48.676117_sports_light_init_pid48292.pkl SHA256 2ffeeddd889ffaf8e552bfeaf27b754e713f9c7cd5560cd75b71927c48a6ca69
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 03_15_48.676117_sports_light_init_pid48292.pth SHA256 a2ff42ae10e8e383258bd78ea3f816b75630e4fd3267a57393df28bd65bec45f
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 03_15_48.676117_sports_light_init_pid48292.pth SHA256 03ffa6d63ddf43c00cf9c6116a3f95ae447709e093a1fbf900d2ade99d3df3e1

### sports_student_full_seed2024_val120_v1

- exp\runs\sports\run_manifest__2026-09-17 04_31_26.826865_sports_light_init_pid29552.json SHA256 1ab8c07b6ae2ab45c1dcd75c059444562dacc1a71f6d28b9201a61f1e7c9a3f6
- logs\2026-09-17 04_31_26.826865_sports_light_init_pid29552 SHA256 c16e31603affe64515edeb3734c064049334d8c17a54ca576cd2af01cea0c792
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 04_31_26.826865_sports_light_init_pid29552.pkl SHA256 1ab1efa1bf10ef2a572c1f27b62e164248c7805690d52b75eaa567e8eb7ecfe7
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 04_31_26.826865_sports_light_init_pid29552.pth SHA256 ab3b762a2df0560b99c89c3cd88a09d5f8d96a1758291f571b755282956906ec
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 04_31_26.826865_sports_light_init_pid29552.pth SHA256 081e3fea5a8c208c9faf7e23eb34555e4a09268f7e0404c6a05dfc13cdb2ec13

### sports_student_image_matched_seed2024_val120_v1

- exp\runs\sports\run_manifest__2026-09-17 05_47_26.495792_sports_light_init_pid25684.json SHA256 1c8296bdcf0500dec4b7a73ddc2b2a5ac6987b7cceb4072c6510a348214fe932
- logs\2026-09-17 05_47_26.495792_sports_light_init_pid25684 SHA256 5f31f78845b0e917b5c8e7f91c987bdf160933c7029dffd8220da51c3b66cbf2
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-17 05_47_26.495792_sports_light_init_pid25684.pkl SHA256 1dc8e89a01312a1990b50e146c1965de763dfdfccdfa5074fe06af4cffe33e11
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-17 05_47_26.495792_sports_light_init_pid25684.pth SHA256 1f492c188e0f2e9b320b74925d0453362d72191f5b14e9fa5e9901da806c266c
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-17 05_47_26.495792_sports_light_init_pid25684.pth SHA256 79450074e85d00ecadc530d8bae6a522e01735d912c4dca1c119b1eadb6b80dc
