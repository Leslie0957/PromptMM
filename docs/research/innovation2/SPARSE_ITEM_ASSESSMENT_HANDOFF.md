# 稀疏物品统计交接

2026-09-29。用户明确授权的一次统计已完成，执行方自检通过；尚非独立审计。[21行矩阵与解释](SPARSE_ITEM_ASSESSMENT_RESULTS.md)。启动源码`cd053ecb32ac073f403ac94371feb37ee8d910b9`，分支codex/experiment/baby-teacher-baseline；结果提交见最终交接。

已执行一次命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sparse_item_assessment.py`。七份已有top20、Train/Val仅作标签与频次；无checkpoint/新打分/训练/Test/NDCG。原输出不覆盖，禁止重跑该namespace。

输入精确SHA和来源见原manifest.json；输出目录exp/innovation2/sparse_item_saved_lists_v1/被Git忽略，提交不能备份。验收检查含标签/Train排除、七原Recall和分组G/L一致性；合成测试先验证分母、贡献、曝光与空组。exit completed/0，acceptance passed。逐文件指纹如下：

| 文件 | SHA256 |
|---|---|
| acceptance.json | `22255c3eed53cbf4e2a6cab892b2b74abef6f83185c1a6e06e6154ad48d1e501` |
| exit.json | `30b08f80648dd83ef3609f4d63b2368cfd9d16b16cad0cf3f28f0c8d40efee56` |
| manifest.json | `7ff162a174fe5617eda6151675ec9d716d68c18fdefc8bad9157ce20fecbc958` |
| report.json | `8069ea1ce0dd6f770c18fd856584a194f9df0fc9d584343a1e94c2089ec828b1` |
| resource.json | `abe7508fba8c1c2cc7226458ef4adf7996bec60541a7e0418d60accb7160c373` |

结论：低频组师生均弱，教师仅10次命中，R1为15/15/17、T0为14/15/16；不支持“学生遗失教师低频能力”。共同不足值得评审，但信息利用不足的因果解释仍未验证。频次5和9的边界按ID切组，须保留限制。

记录路由：RESULTS为唯一21行矩阵/本次执行自检，HANDOFF指纹与复核入口；TRAINING_LOG追加outcome，root/第二项/实验族导航同步，生成目录刷新。第一项正文/矩阵、总体目标、原始产物、旧审计、zhuanli与评审材料不变；未改历史NDCG。无tag/merge/新阶段。

唯一下一步：独立复核本次保存列表聚合与问题解释，再决定是否另行立项；不自动追加实验。
