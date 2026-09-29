# 最小排序监督三seed独立结果审计

2026-09-29。用户委托核验已有三个子目录，非启动授权。**执行硬验收通过；固定Recall方案三seed均screen_stop；次指标标准性有重要限定。** 起始/启动源427459f60416a44ef00c859e835f0313546ccb36，分支codex/experiment/baby-teacher-baseline。工作区仅archive/reviews/未跟踪，原始产物保持原样。

## 身份和核验

[独立CPU核验程序](../../../tools/audit_minimal_ranking_three_seed.py)执行289项检查，[机器证据](MINIMAL_RANKING_THREE_SEED_VERIFICATION_2026-09-29.json)含三目录全部文件SHA。无训练/评分、数据分割加载或Test；CPU weights_only读取18份best/final。

- 从启动Git blob读取cohort及base profile，独立重建三个effective_profile，与各manifest逐字段相同；源码六文件SHA匹配运行manifest，规范化Git换行后与启动blob一致。共享teacher/candidate/Train/Val锚相同，仅seed、tape、原manifest、B回归值和输出路径作预定变化。
- 三个原anchor manifest SHA重核；51份acceptance引用产物（每seed17份）SHA全匹配，另对所有现存输出记录SHA。每份候选副本与预先固定文件SHA一致。稳定大tape/Train/Val/教师未再次读取/重哈希，依赖已提交锚、运行源码加载前哈希校验和完成记录；不冒称本次重新验证所有输入现存字节。
- 九条curve各300连续epoch，与report完全一致；九臂各64200更新，合计577800更新。每seed904次记录Validation，合计2712（903训练期评价+1混合评价/seed）。final top20额外导出不加载标签，不计新的Validation指标调用；本次不重跑。
- 三个B末轮Recall分别.09515169996514843/.09434317853046820/.09461550968515230，与原T0回归一致；九臂epoch0满足共同教师.094184495190847733的1e-6容差。
- 18个checkpoint的arm/epoch/source/profile/metric绑定通过；best为epoch1–300 earliest最大Recall，final为epoch300；两张FP32表形状35598×64、18357×64且有限。AdamW lr6e-5、wd.01、betas(.9,.999)、eps1e-8，两状态moments形状/有限性/二阶矩非负，step=epoch×214。
- 九臂300条epoch诊断和全部报告浮点有限；所有曲线指标位于[0,1]。逐步有限性依靠启动源码的loss/gradient/parameter强制检查，未保存每一步完整张量，不能称离线重验每一步。12份top20形状/ID范围/无重复通过；本次不读Train/Val，未独立重验过滤或命中标签。
- 每个exit completed/0、acceptance passed，cohort completed/0，三子目录均完整。stderr/stdout为空；runner不打印进度，空stdout并非证据缺失，曲线与report是执行记录。

## 资源、Test与边界

| seed | 实际墙钟秒 | CUDA allocator峰值B | 采样RSS峰值B | 最终输出B | 执行/科学筛查 |
|---|---:|---:|---:|---:|---|
| 2022 | 7128.188 | 224395264 | 2061283328 | 268710888 | passed / screen_stop |
| 2023 | 6719.000 | 224395264 | 1028030464 | 268695255 | passed / screen_stop |
| 2024 | 4398.359 | 224395264 | 1004003328 | 268487493 | passed / screen_stop |

cohort总18245.890秒，约5.07小时。三seed各低于8小时，cohort低于24小时；各CUDA<2GiB、采样RSS<6GiB、输出<2GiB，cohort<6GiB，采样盘余量≥4GiB。本次实际通过资源验收，不反推旧6–11条smoke具有完整负载代表性。资源为协作式检查，可能漏短暂峰值，评价内部无逐批超时检查；本次未触发上限不证明精确硬隔离。该运行耗时包含评价/记录等，不支持算法间纯训练速度结论。

Test拒读hook在run资产加载前安装，专用Train/Val loader；report/acceptance记录被拒Test打开尝试0，不等于OS级访问审计。无Test指标、无selected-versus-tested绑定。三seed同进程串行且有source/namespace/failure-stop保护；历史授权以用户已写入日志的三seed手动声明为准，早期单seed提案不覆盖后续明确修订。此次没有重试或下一阶段。

## 主结论及次指标勘误

详见[12格矩阵](MINIMAL_RANKING_SUPERVISION_RESULTS.md)。R−B依次−.001704605752、−.000731169577、−.001260381580，全部低于预定+.001；A−B与M−B也均负。R best位于14/10/1轮，三个best均低于对应B best。固定候选KL未产生预期净收益，原oracle互补机会未被此配方转化为改进。保留有效负结果，停止本配方；没有理由由此自动追加权重搜索、复杂门控或长尾模块。

**次指标发现：**初始化评价器使用ideal=_dcg(sorted(bits,reverse=True)[:k])。本轮ks=(20,)，IDCG依赖已命中个数而非min(全部留出正例数,20)，因此原字段ndcg20不是标准NDCG@20。原初始化ks含更大K，分母还可能不同，所以B的Recall回归通过但NDCG不同不构成B训练回归失败。此次仅加勘误、保留原值，不修代码、不重评。所有标准NDCG及跨旧表NDCG推断暂不成立；主Recall验收及screen_stop仍有效。

仅能说明同一Sports教师/候选/预算和三套tape下此固定干预方向一致为负，不能宣称显著性、收敛最优、所有排序蒸馏无效或信息无价值。R/A剂量、候选访问不匹配，单配置没有排除弱/强约束等优化因素；M存两套表且只测0.5。三seed共享教师与Val，不是三个独立数据集。负结果不自动支持长尾备选。

## 记录路由与停止点

新增本审计、机器核验、CPU辅助程序、12格RESULTS与HANDOFF；追加日志outcome并更新当前状态、根/第二项入口/实验族行，刷新六个导航。原profile/代码/manifest/checkpoint/曲线、历史声明、第一项矩阵/正文、THESIS_ROADMAP及政策不改。docs/README通用入口无状态依赖，无需修改。无tag/bundle/merge；忽略运行资产不由Git保护。

唯一下一步：由用户审阅本次HANDOFF并决定是否另行评审备选问题；本任务在结果归档交接处停止，不启动任何下一阶段。
