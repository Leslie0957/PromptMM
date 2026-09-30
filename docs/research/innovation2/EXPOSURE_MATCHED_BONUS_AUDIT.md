# 曝光匹配加分诊断独立收尾审计

2026-09-30。**执行验收通过；按预定描述标准，简单全局加分对照不足。** 不等于第二算法或模态语义机制成立。

## 来源与协议

用户运行`tools/run_exposure_matched_bonus.py --formal`，来源6dbdc400cb029340d2c40bdc65e8f4a442941108，profile SHA0708f1596b18f9a55c9cf649747a19d67fa3b7afd161499b76e847fe06601932。分支codex/experiment/baby-teacher-baseline，运行前source_gate要求干净跟踪源码，原archive/reviews/例外。固定B seed2022/2023/2024 epoch300，batch256、CUDA FP32、TF32关闭；R仅用保存Top20。执行命令、范围、参数见[准备](EXPOSURE_MATCHED_BONUS_PREPARATION_2026-09-30.md)及[冻结方案](EXPOSURE_MATCHED_BONUS_PLAN_2026-09-30.md)。无训练、best选择或Test指标。

原始目录exp/innovation2/exposure_matched_bonus_v1/。父进程exit0，termination null，worker completed，54.406秒；采样峰值RSS1000292352B、CUDAallocator75497472B，最终输出36441455B，剩余盘通过。时间/RSS/allocator/输出均低于固定上限。采样不是OS硬隔离，父进程RSS不计；allocator不是整卡显存。

## 200项核验及证明范围

[核验JSON](EXPOSURE_MATCHED_BONUS_VERIFICATION.json)包含输出completeness全哈希、原manifest/completeness及15份输入SHA、profile/源提交、校准封存和beta二分轨迹、单调性/端点选择、曝光零误差、B/R原Top20逐位相等、B/R指标回归、列表ID/唯一性、分组曝光与micro/contribution、逐用户/逐物品配对差及决策重算。原始completeness SHA bc8603c4cd4ae10ba345e1b0c26453319d9b1f0b8e06a626bfc8183d29bf50ab。

审计没有重新调用评分器、生成排名、读取Train/Validation/Test标签或载入模型tensor。因此绝对命中、候选Train排除、原始模型分数有限及参数不变主要由成功运行时硬检查支持；本次只独立验证保存证据的一致性。未从候选分数重算每次二分曝光，只核对记录轨迹与最后列表曝光。不能将200项称为独立完整复现。

三个seed β分别1.1999290771200322、1.192203172831796、1.1884095493733184；S曝光分别22159/22067/22065，Q与R精确相等。S仍为6114个查询，不改成6119低组。当前两者低组曝光与S曝光恰好相同是实际结果，不是协议预设。

## 标签顺序与Test0

calibration seal SHA9744da852f86e1a6bfab18044dab2d47a43697cf8f890d1397fb0a161ec6797a，三个seed的候选cache、β和列表均完成后封存；封存时间1790745958349231300 ns，首次Validation open1790745958417704500 ns，顺序正确。运行记录Train open5次、Val open2次，拒绝尝试0。源码先限制Train，封存后才开放Val，最终回验seal未变；校准函数没有Val输入。旧checkpoint包含历史metric字段随对象加载，但未读取用于β选择，因此不是严格历史结果盲态。

Test0由明确读取路径、白名单hook和0拒绝尝试共同支持，不是OS级访问证明。零更新由inference_mode、无optimizer/backward及参数前后相等支持，不能只凭parameter_updates常量。此次审计自身未访问交互标签。

## 科学结论与最强反对意见

已证实（所测固定协议）：相同全局目标曝光下R低组比Q多命中10/4/9个；2022总体R比Q略低约0.0000211，另外两seed高约0.000391/0.000434。三个seed达到预定“R至少多4低组命中且总体不低于Q超过0.0002”的描述筛查。简单全局加分不能完全复现R的低频收益。

同样应强调Q已取得R相对B低频增益的约79%–91%。这一比例不是因果分解，不能说其余9%–21%必然是语义价值。最强反对意见：总曝光相同不等于给同一用户相同机会；R与Q逐用户槽位差仍明显，R和B的用户/物品参数也来自不同训练轨迹。剩余差异可来自把低频推荐分配给更合适用户，而不是更准确的低组内部物品排序。它们均未被当前对照区分。

合理推测是R含有超出一个全局常数偏移的作用。不能宣称真实图文邻域传递成功、教师语义因果作用、统计显著、正式非劣、低频问题解决或第二创新点成立；原N输随机R和C整体screen_stop仍有效。重复Sports Validation仍是探索证据。

## 收尾与唯一下一步

已更新RESULTS（六个B/Q及三R参照）、AUDIT、HANDOFF、verification，TRAINING_LOG及根/第二项/实验族入口，刷新生成导航。原矩阵/旧审计、第一项论文/成本/Test表、总体路线及原始产物保持；docs/README通用入口无需改。无tag/里程碑备份，Git不保护忽略资产。

唯一下一步：设计一个固定检查点、逐用户目标槽位匹配的最小对照，用R每用户目标槽位数和B组内排序区分用户分配与物品选择；只做设计，不运行或恢复训练。这是进一步排除解释的控制，不是拟定新算法。
