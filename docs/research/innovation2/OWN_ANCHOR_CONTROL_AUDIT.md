# 自身锚点六状态诊断：完成审计

2026-09-30。结论：**执行验收通过，科学筛查整体screen_stop，得到有边界的机制支持。** 本审计未重新排名、训练或读取Train/Validation/Test标签。

## 身份与执行

用户按[准备命令](OWN_ANCHOR_CONTROL_PREPARATION_2026-09-30.md)运行`tools/run_own_anchor_control.py --formal`。manifest源提交bd775f55dca2ee6470364e50c3afd57a28d9e396，profile SHA882ff1099579addea6f0aca0ee10415847472ef18a77a85d1bf7382156ccfcc0。独立目录`exp/innovation2/own_anchor_intervention_v1/`，三个seed×R/C，固定既有R epoch300、batch256/CUDA FP32/TF32关闭，零参数更新；没有best轮或Test选模。

文件时间约2026-09-30 13:05–13:06本地，父进程记录26.203秒；manifest未保存绝对开始时间，文件时间只是补充，不冒充精确起止。父进程exit0、无termination，worker completed和acceptance通过。manifest的status字段保持launched是代码设计，最终完成依据report及supervisor，不能单独读该字段。

## 核验覆盖与限制

[95项保存产物核验](OWN_ANCHOR_CONTROL_VERIFICATION.json)通过：全部completeness文件SHA、绑定原manifest/completeness与12份输入文件SHA、提交与profile一致，六个Top20形状/范围/用户/唯一性，三份R与原Top20逐位相等，分组曝光独立重计、micro与contribution算术、逐用户/物品gain/loss总量、重叠、分组净差、Recall差及筛查重算。

未重新载入交互标签，因此不声称独立重算绝对命中、Train排除或C原始分数；它们由已核对源码中的运行时检查和保存明细支持。teacher/Train/Val等共同资产沿用绑定anchor及运行时SHA，不在本审计重复读取交互数据。C公式、全体参数不变、非目标行不变、有限分数由成功执行时的硬断言支持，本次没有重新加载tensor重做代数证明。审计不是独立另写评分器的复现。

采样峰值RSS1094811648 bytes、CUDA allocator96468992 bytes，输出最终8392967 bytes，墙钟及剩余盘均通过既定上限。外部监控每秒采样，有瞬时漏测可能；RSS针对worker，CUDA allocator不是整卡占用。原始completeness SHA：efcddbd9e739ef4b9322dc7521319aa2ef484060bd7e6491d0a80d1bc2e3e877。

Test0：记录允许Train和Validation各3次open（哈希及加载），拒绝数据open尝试0；代码先安装data白名单、只显式加载Train/Val，未调用旧Test自动加载路径。parameter_updates=0常量不能独自证明零更新，结合inference_mode、无optimizer/backward及前后参数相等检查支持。Python hook不是OS访问取证，不声称OS级完备证明。审计自身未访问任何交互标签。

## 研究判断

已验证：C在三个既有检查点上保留R相对B低组增益的97.87%/106.38%/93.75%；逐目标邻居差异并非保留这些低组收益的必要条件，限于这个固定参数替换及当前Val。2024总体Recall损失−0.00033710越过事前−0.0002界限，不能用均值或事后改阈值宣布整体通过。

合理解释：共享自身线性变换加共同偏移可以解释当前R低频收益的相当部分，具体随机邻居中心在评分端作用有限。最强反对意见是W、q、用户表已在随机图上训练；C还保留从图计算的均值与尺度，因而邻居信息可以间接留在参数中。此实验不能证明从头训练不需要邻居、语义冗余、共同偏移和自身变换各自的因果份额，更不能把保留率称为机制贡献百分比。

当前没有稳定总体替代效果，原真实模态N低组输R的证据不变。因此不推进C从头训练，不新增门控/注意力或强度扫描，不把R/C改名为第二创新算法。反复使用Sports Validation使结论仍是探索性的；无泛化、显著性或学位充分性保证。

## 记录路由

新增RESULTS六状态、AUDIT、HANDOFF、verification；追加TRAINING_LOG，更新根和第二项入口及实验总览，刷新生成导航。原始目录、旧12臂矩阵/审计、第一项论文/缺口/成本/Test表、THESIS_ROADMAP、政策均未修改：这不是新算法或第一项证据变化。无标签/里程碑备份，忽略资产仍需用户保管。

唯一下一步：做一次第二项候选问题的收束评审，以现有失败和混合结果为约束，决定是否转向新的明确问题；只做问题与最简单替代方案评审，不启动实验。当前随机邻域/自身替代配方暂不进入训练队列。
