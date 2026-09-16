# Sports 教师 120 轮运行复核（2026-09-16）

运行正常完成。120 轮（0–119），最佳 Epoch 37，Validation Recall@20 精确值 0.09418449519085098；该轮 NDCG@20=0.04332（日志五位小数）。末轮 Recall@20=0.08436、NDCG@20=0.04009（均为日志舍入值）。耗时约 2 小时 59 分 42 秒。

## 曲线与结论

|轮次窗口|最高 Recall@20（舍入）|轮次|
|---|---:|---:|
|0–19|0.08967|18|
|20–39|0.09418|37|
|40–59|0.09401|44|
|60–79|0.09242|62|
|80–99|0.09031|81|
|100–119|0.08717|102|

最佳之后82轮未刷新，窗口后半段下降，支持选择早期最佳权重；不支持继续延长本次训练。可描述为后期泛化退化/可能过拟合，不能仅凭曲线断定机制。有限窗口也不能证明全局收敛或超参数最优。此处是教师可用性证据，不是学生创新方法在第二数据集上优于基线的证据。

## 复核与边界

- 120条连续验证记录，记录中的目标/指标均有限；checkpoint 所有张量有限。
- checkpoint 内 best_epoch=37、best_selection_recall 与 manifest 精确一致；实际文件 SHA256 与 manifest 一致。日志确认恢复该最佳权重。没有 Test，因此 selected-versus-tested 等同性不适用。
- 零教师/学生 Test 排名，零学生训练，未生成学生权重或共享教师别名。Test 矩阵只在既有结构加载/预检中读取；本次审计未进行模型前向或新评估。
- paper_ready_eligible=false，唯一 blocker 为 final test evaluation is disabled，符合声明；不篡改原 manifest。
- 配置覆盖为空；seed2022、PCA64 seed2022、lr0.00055、batch1024、epoch/patience120，源自 sports_teacher_validation120_v1。命令及完整参数见该配置文档与运行 manifest。
- 当前干净 HEAD=0b3b673，运行记录的11份源码指纹与当前文件一致；运行 manifest 未记录 Git HEAD/dirty 状态及完整环境快照。不能声称 manifest 已证明启动时干净或自动捕获提交号。准备阶段环境记录仍可引用，但不是运行时快照。此前说明中“runtime manifest records exact launch commit”不符合现有实现，特此更正；不回写旧记录。
- 数据转换身份与既定 Sports 锚点一致；PCA缓存和教师文件重新校验通过。原始模态语义来源与5个冷物品限制仍保留。

## 下一步

准备 Sports 学生验证阶段：先明确这个“仅 Validation、原标记 false”的教师如何安全复用，并准备 seed2022 的 BPR-only、完整方法、固定图像权重去文本对照的配对配置。保持同一教师、种子和训练预算，仅 Validation；先检查并记录复用门禁，不直接关闭校验或添加 Test。具体命令在下一阶段验证后提供，本次未启动任何新实验。

seed2023 历史审计例外及原 manifest false 不变，第二创新点未确定。

## 产物身份

- manifest: exp\runs\sports\run_manifest__2026-09-16 12_38_48.480698_sports_light_init_pid30708.json; SHA256 11636306eac6c8144c753bb091966f75660070d259412edf9a87be562b36235a
- log: logs\2026-09-16 12_38_48.480698_sports_light_init_pid30708; SHA256 b187d7dfd458bd8855190ee1b16b2f7b4ae9635ab2abc786da9910d25f724ef1
- checkpoint: D:\Download\PromptMM\Model\sports\runs\teacher_model_val_test_once_v1__2026-09-16 12_38_48.480698_sports_light_init_pid30708.pt; SHA256 57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea
- image PCA cache: D:\Download\PromptMM\data\sports\hard_token_image_pca_v2_1544b94d9df2b5f1.pkl; SHA256 b18fd8b32baeb894c84f6af28db8b383a6ed6dd325c69d7a811f4b2ebe26b37e
- text PCA cache: D:\Download\PromptMM\data\sports\hard_token_text_pca_v2_801e7998ca2e5b46.pkl; SHA256 9574b8342bb24236620e87e9a997c34ac6e522ea47e32cb8d39fc08728d7ad3d
