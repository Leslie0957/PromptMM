# Sports 随机初值严格配对三种子审计（2026-09-25）

## 结论与边界

用户在 `codex/experiment/baby-teacher-baseline` 的干净提交 `37d3901f6c5904602c8814dadbc72c04526acdfa` 手动启动一次 `tools/run_sports_paired_cold_three_seed.py`。六臂按 seed2022/2023/2024 的 Full→image-matched 顺序完成，批次报告 `exp/paired_coldinit/sports_three_seed_v1/batch.json` SHA256 `b254d9e969edd3e001a4a85c2b14b69f8eb68b3d279d667100973c62fe7797b8`；时间为 2026-09-24 18:22:47 至 2026-09-25 12:26:56 +08:00。没有失败后重试、延长轮次或新阶段启动的记录。

这次比历史三种子比较更严格：每个种子的两臂从同一个保存的随机 user/item ID 表出发，第二臂逐个 epoch/batch 回放首臂实际采样的全部训练三元组，且 image 蒸馏梯度系数按现有加权平均目标匹配。三对 Full 的最佳 Validation Recall@20 全部高于 image-matched；平均差 `+0.004177997837`，Full 均值 `0.082307064899`，image-matched 均值 `0.078129067062`，均值之比 `+5.3476%`。这支持在该 Sports/教师/固定预算下，加入当前 item-text 蒸馏项与更高的验证指标相关，并排除了初值与训练三元组不同这两个混杂。它仍未逐位保存并核对六个进程各自计算出的教师中间语义，也不能仅凭三种子说明文本语义的唯一因果机制、统计显著性或 Baby 论文主结果。最佳轮次都接近 300 上限，不能声称充分收敛。

## 协议与审计证据

共同身份：SPORTS_CONVERTED_20260916 conversion SHA `3772a17c8b70fa4739653534dca8d542e67e0649f1f44ccba4194fec17bc1104`；训练矩阵 SHA `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`；固定 epoch37 教师 SHA `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。环境 `D:/miniconda/envs/run_5060/python.exe`、GPU0；各臂 seed 如下，epoch=patience=300，214 batch/epoch，batch1024，ID维度64，student_lr=6e-5，AdamW weight_decay=0.01，随机初始化且无教师拷贝；Full alpha0.3、item image/text rates1/0.3，image-matched alpha0.3/1.3、rates1/0；两个 user rate均0。所有解析参数与各自历史 profile 默认值一致且无覆盖。

独立只读复核重新执行现有 `verify()` 并比对六项批次报告、manifest/checkpoint/initial/tape SHA、teacher/train_mat SHA、源码指纹、数据身份、最早最大 Validation Recall@20 选模、300 个有限 objective/Recall/NDCG 值、完整 checkpoint 有限权重、推理导出的 user/item ID 张量与所选完整 checkpoint 逐元素相等。每臂完成 64,200 个训练 batch；每对的 initial.pt 与 triplets.npy 路径及 SHA 相同，tape 形状 `[300,214,3,1024]`。六臂 `status=validation_completed`、`selection_split=validation`、`candidate_exclusion_policy=train_only`、`run_final_test=false`、`teacher_final_test_performed=false`、`final_test_performed=false`、`paper_ready_eligible=false`。既有加载器会结构性读取 Test 矩阵，但没有教师或学生 Test 排名；selected-versus-tested 不适用。没有额外前向、优化、验证或 Test 评分。

|Seed|臂|最佳 epoch（从0计）|Recall@20|同轮 NDCG@20|Full−image Recall|
|---:|---|---:|---:|---:|---:|
|2022|Full|293|0.082328259836|0.037207407800|+0.003678244039|
|2022|image-matched|299|0.078650015797|0.035730006129|—|
|2023|Full|291|0.082050198914|0.037094821672|+0.003665425535|
|2023|image-matched|287|0.078384773380|0.035534242301|—|
|2024|Full|292|0.082542735948|0.037306458146|+0.005190323937|
|2024|image-matched|285|0.077352412011|0.035409652479|—|

Full−image NDCG@20 差分别为 `+0.001477401671`、`+0.001560579372`、`+0.001896805667`，均值 `+0.001644928903`。Recall 配对差的样本标准差为 `0.000876723547`；仅三对，不据此宣布显著性。新六臂的最佳 Recall、同轮 NDCG 和最佳 epoch 与 2026-09-18 历史同名臂相同；完整 checkpoint 权重并非逐位相同，曲线有小幅数值差异，故不宣称历史训练轨迹逐位复现。历史臂原本没有初值和训练三元组资产，本次配对保证只针对新六臂。

## 原始产物身份

以下 manifest、checkpoint、曲线、推理导出均为 Git 忽略的运行产物。表中 SHA256 为独立复核值；manifest 内可定位完整绝对路径，批次报告也保存这些路径。

|Seed|臂|Manifest SHA256|完整 checkpoint SHA256|曲线 SHA256|推理导出 SHA256|
|---:|---|---|---|---|---|
|2022|Full|d31cf835fd68b575b7312e1328f8e439ad0d459302e65c9587b524ce760d6ddf|d2648d9f1f6c593b8720b27ad226e81152373537f7cb4d3fa18de07acf20b4a2|e425f897fa131b0097133952a9308d754097e8e597b7ffcfe307a79ce3506a5d|a9d76fd2d9b108097fa1f89e0cbd7c3b5d69eb5539320ffcc74c5b7881a606ef|
|2022|image|4c502922168f6e71e4f8ed8939d007297421ee4bc5c21de9b38aa14f9c47aaf7|b35cf068217e34df7da89711c78e93fa10d0e72659b571af23e0e497dcb0e76c|c108b2acc9ec715be31acd3aba3db1bb123e0e73e36802d446d6f365502a61db|23e2cb645e826a5bf9a80a43020176cee442c0fd39afb155f543525c198e31cc|
|2023|Full|d9785ae45b916c5b99ec1501342b8175c74631474e8b4fad216a2386593479a3|d52756660a06c61c012ea9354ced9d3ae6175f55358df72d6ee68156951d6f9b|59318ab16ddb31f7d894c829c9341a67a0089b56ab6bb842fc6925ac6514e787|62afda7871f0a5c71c2537f80fc9caf763dc0bdcff9f4453ee0c04763492031b|
|2023|image|0ad40f7e895445ea256711d540b6c0e69d26f24167b14ecd1c032a719bc79603|222f07b3c1bde206b0629ea4221d57f973e6eabca2eb82036014c5e1722b3ebd|9020da74a4db3679fd9e608715e3ffea5bd300a591eded490f17b2e1d011dea8|fb7323b132364851521b298b5584fb4fbf693dc8ddbd3d34ae20fe950b0b9915|
|2024|Full|81d88af57ca1224ad634170c5cbf702ce27bc509a9a54c44d2f1eb2737e9230d|a9565b2de2cef1de46a28194f8b49d49c018f609727d6f8b843c76acc399ace3|5864d266ccc174e506690b2b3fdab662749ce004062f007a3905f8f3f6e2fa37|401387dba69a6bab22fe60f029f92891aa6d66e53f2661e9a390974ddf1e1824|
|2024|image|0bf4067bfc86e85996a313b888df3cf4be097ea43b6941f5a4f62eb11b50215d|32532f5e4f7351a78365ed3a63fdb19bc5f617b54005e5818b2347fcb9f473d8|9c755cf04db772724c329811ec9b007431181a47fc270200ad0e82706107a45c|f9200646c1e42d8aa51f867e94b9da52faacb43491f9a301cb36e0524aa64e33|

|Seed|初值 SHA256|训练三元组 tape SHA256|
|---:|---|---|
|2022|2357868cefeb086dba14c3808415eac97936f67cf5e6bef8382d95ff43bcffff|e8d982c6a2d1be20a934502d4a1ad6ba7e2b699b21098668343341f10db3d77b|
|2023|bb5429aa8a1edfe53b36fed977146460d1101ae79539d2badf856d5514ef9d85|3763128ba003228d60c09924b9370b4cab0b8c16679366742835fc4559d89085|
|2024|e2c6885debdec09a8ff8401fe4f4f96ea5f46dcc1fd90f24dcc4ecdbd2188e37|131cc14ebe6853e79216183f06923f3689d47d26c69eac2521d7f93b1b314bc7|

Git 只保护本审计文档和源码，不保护上述忽略资产。当前没有物理或异机备份证据。本阶段下一步是依据严格配对证据重述 Sports 文本项优势及其边界，先不自动发起新训练或 Test 评估。
