# Sports 原始数据审计（2026-09-16）

结论：原始数据结构检查通过，存在已明确记录的冷物品；适合进入格式接入阶段，尚未达到训练就绪状态。本结论不涉及模型效果或跨数据集泛化结果。

## 来源与审计边界

- 原始目录：`data/_incoming/mmrec_sports`。
- 官方入口：https://github.com/enoche/FREEDOM
- 官方 Sports 文件夹：https://drive.google.com/drive/folders/1iJtyDmgeYdZsvO5297dNafyPDya21e8D
- 下载身份及五个文件的 SHA256 见 TRAINING_LOG.md 的 2026-09-16 acquisition completed 记录。此次重新校验五个文件的字节数及 SHA256，全部与下载清单一致。
- download_manifest.json SHA256：`af3265a8e450f9e0efca6f279a6176d3899c831b02d502b1c7b17f8ecaa088ac`。保留其原始下载状态，不覆盖为训练验收状态。
- 只读调用现有 tools/convert_mmrec_baby.py 的映射、交互、特征验证及冷启动报告函数，未调用 convert_baby，未导入训练入口。
- 官方 MMRec split() 按 0/1/2 顺序生成 train/validation/test，与仓库转换器一致。核验来源（访问于本日）：https://raw.githubusercontent.com/enoche/MMRec/master/src/utils/dataset.py

## 实测结果

用户 35,598，物品 18,357，共 296,337 条交互。两个映射均为一一映射，整数 ID 从 0 连续；所有交互 ID 有效，交互全集覆盖映射全集。字段可解析，rating 有限，timestamp 为整数，标签仅为 0/1/2。

| 官方标签/集合 | 交互数 | 用户数 | 物品数 |
|---|---:|---:|---:|
| 0 / Train | 218,409 | 35,598 | 18,352 |
| 1 / Validation | 37,899 | 35,598 | 13,342 |
| 2 / Test | 40,029 | 35,598 | 13,738 |

全集重复用户–物品对为 0；Train–Validation、Train–Test、Validation–Test 的交集均为 0。此处是数据身份与泄漏检查，并非 Test 排名评估。

| 特征 | 形状 | 类型 | 非有限值 | 全零行/列 |
|---|---|---|---:|---|
| image_feat.npy | 18,357 × 4,096 | float64 | 0 | 0 / 0 |
| text_feat.npy | 18,357 × 384 | float32 | 0 | 0 / 0 |

两路特征维数及指纹不同，没有本地 AmazonBook/Yelp 的“两路文件实际相同”问题。行数与物品映射一致；具体每行是否对应正确商品、原始图像/文本缺失是否被上游编码器填充，不能仅由 NPY 和映射文件独立证明。非零特征不等于原始模态全部真实可用，仍依赖官方预处理来源。

## 冷启动情况与解释限制

- 没有冷用户：Validation 和 Test 的所有用户均在 Train 出现。
- Train 缺失的物品共 5 个：6500、9814、11472、13739、15279。
- Validation 冷物品：6500、11472、13739、15279，涉及 9 条交互。
- Test 冷物品：上述全部 5 个，涉及 17 条交互。
- 后续接入建议采用 retain_official，与 Baby 保留官方划分的原则一致，不过滤、不重新分割。当前审计没有修改任何数据。
- 对只依靠训练交互学习物品表示的学生，这些物品缺乏正样本训练信号。记录其存在即可；不能将其自动当作错误，也不能据此选择性删去评估样本。

## 单一下一阶段

将已审计 Sports 无损接入仓库格式：保留 ID、官方标签及冷物品，生成独立 data/sports 的 train_mat、val_mat、test_mat 和转换 manifest；复制原始特征而不拟合 PCA，做转换前后计数/指纹与 dataset preflight 验证。现有转换器文件名和清单固定为 Baby，需要显式支持 Sports，不能把 sports.inter 伪装成 baby.inter。

该阶段不启动训练、不计算 Test 指标。通过后才另行冻结 Sports 专用教师、学生配置及预算；Baby 专用 profile/checkpoint 不可直接视为 Sports 已验收配置。已有 Baby 正向结果不等于 Sports 效果已得到证明。

历史约束不变：seed-2023 原 image-only 按审计例外接受，原 manifest 的 paper_ready_eligible 仍为 false；第二创新点尚未确定。
