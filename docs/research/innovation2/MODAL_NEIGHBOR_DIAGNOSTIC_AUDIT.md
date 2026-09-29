# 模态邻居诊断执行与产物核验

2026-09-29。此文为**执行方自审**；另一个脚本做保存产物核验，不冒称独立外部审计。未重新构图、训练、重采样或访问Test。

## 身份、执行与验收

- 准备源279c217；声明及启动源`0ffa72596c13a76481ba493a1e8fe0c500b3c853`，分支`codex/experiment/baby-teacher-baseline`。启动受跟踪源码干净，只允许既有未跟踪archive/reviews/。运行期间只新增独立产物审计脚本，运行核心/profile未修改；不是从新审计代码重新启动。
- 冻结profile：`MODAL_NEIGHBOR_PROFILE_V1.json`，SHA `e32bac877273a5be7c7eda69b09c11b6a0a9157d20da8fb27d9cc1d964a02e6b`；相对准备状态只激活status。k20、100随机组seed20260929、两种1000次bootstrap种子20260930/31、0.005描述性门槛、CPU4线程均保持。
- 命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_modal_neighbor_diagnostic.py' --formal`。一次启动、exit0、无重试；无训练/选模/学生或教师Test，检查点选择/测试相等项不适用。
- Runtime依次核验固定Train与教师cache的SHA，生成两个模态及随机图，持久化graph_seal后才hash/load Validation；guard拒绝早读与Test。Train/Val形状、有限性、重复/重叠及低组分母检查通过。资产锚点见冻结profile。
- 耗时464.922秒（7.75分钟）；父子合计采样RSS峰值734605312字节（0.684GiB），最终产物103385531字节；30min/4GiB/512MiB及磁盘余量门槛通过。0.5秒采样不能排除瞬时峰值。
- Test拒绝尝试0，Validation早读拒绝尝试0；代码路径和日志支持Test0，不是OS级全访问证明。

## 保存产物核验

`tools/audit_modal_neighbor_saved.py`通过297项检查：exit/acceptance/seal文件SHA、profile SHA、全部图逻辑SHA与有限值、查询/候选/自身排除/唯一性、200组随机图精确频次多重集、可变槽位重算、保存用户/物品汇总的关联率和差值/宏平均、随机均值、预设screen、资源及时间顺序。

局限：没有重新读Train/Validation独立重建原始交集，没有重新计算教师余弦TopK或bootstrap区间；这属于保存产物一致性与执行路径核查，不能消除实现方共有错误。合成端到端和独立集合复算已在准备阶段通过。正式运行后未重跑准备测试（其中启动拒绝测试只适用于准备状态），未调用正式命令验证拒绝。

| 模态 | 真实邻居关联率 | 频次匹配随机均值 | 差值（百分点） | 用户簇区间（百分点） | 物品簇区间（百分点） | 预设筛查 |
|---|---:|---:|---:|---|---|---|
| 图像 | 7.8731% | 1.4213% | +6.4519 | [5.8180, 7.0607] | [5.7477, 7.1237] | candidate_signal |
| 文本 | 13.0009% | 1.7198% | +11.2812 | [10.4859, 12.0805] | [10.3982, 12.1245] | candidate_signal |


两种差值均超过事前0.5个百分点门槛，用户/物品簇区间下界均大于零；99.8582%覆盖、100%匹配支持满足规则，因此是有效的候选信号。不是算法成功、显著性保证或因果收益。

最强反对意见：教师分支含Train协同学习信息，观察到的关联可能主要来自协同结构；随机对照只控制邻居频次，不能排除这个解释。即使关联真实，也可能简单邻居评分已获得全部价值，复杂模块未必必要。多数正例依然没有邻居历史交集，不能声称全面解决稀疏性。

## 指纹与记录路由

原始产物保留`exp/innovation2/modal_neighbor_v1/`，Git不备份该忽略目录。本次不创建标签/里程碑备份，未覆盖任何基线。

| 文件 | SHA256 |
|---|---|
| `manifest.json` | `75db4db1af391028099412e3669a753e412812830d7c29a1a197565224c0fcaf` |
| `report.json` | `6ad0d56d0382c428e4190e2e7a1c6bdeddb6a3957dcb87beb7e8e0623f525ce7` |
| `graph_seal.json` | `42b6352974c18ace650db5ae56c8b820a582311e2361fd53f17401a7fbbfa45b` |
| `image_graphs.npz` | `5e811a19f547ff08bef1db4385fc10990d9ab6367d75ab012c9cea7dbcbf2ccd` |
| `text_graphs.npz` | `d3eb7e936f2457042897daec8087a8c7811667aee006e1f7965a9d5afa526a58` |
| `image_summaries.npz` | `98711f6b2411a8a8ee10a867fcc98628eeb7255fdefb18c145c4f3a741cd1ef5` |
| `text_summaries.npz` | `639a3b47f91f8625005b0784341c779565739fc483a400552b58093fb4cefe3f` |
| `acceptance.json` | `1a835650de0d5403c2e9d6b6144ab10db42f25f715600607202bb6b27d1b66a4` |

路由：本审计、RESULTS两行、HANDOFF、TRAINING_LOG outcome及当前状态、root/第二项入口、实验族行已更新；六个生成导航刷新。第一项数值/正文/缺口、THESIS_ROADMAP、AGENTS和历史审计不变：本次没有新模型结果或总体目标变化。docs/README稳定入口无需改变。唯一下一步：交给另一执行者独立只读审计这次产物与结论，不训练、不补跑。
