# PromptMM 学习率对照审计（2026-09-21）

执行成功，结果要求修正此前对照的概括：原来的完整方法三种子优势仅针对发布版lr2e-5配置。lr6e-5在seed2022上已经超过当前完整方法，不应继续声称完整方法普遍优于PromptMM。

|seed2022，300轮|选中轮次|Recall@20|同轮NDCG@20|
|---|---:|---:|---:|
|PromptMM发布版适配，lr2e-5|299（1起算）|0.0769788614|0.0348083482|
|PromptMM发布版适配，lr6e-5|294（1起算）|0.0876345563|0.0401254519|
|现有完整方法，lr6e-5|293（原日志0起算）|0.0823282598|0.0372074078|

新配置相对旧配置Recall+13.8424%、NDCG+15.2754%；相对完整方法同种子分别+6.4453%、+7.8426%。不混用新配置单种子与旧配置三种子计算所谓统一均值。相同lr不代表两种结构/目标获得完全公平调优。

## 协议与产物核验

- 启动b6c801e3e8a09073d3f1c4e6b62fde45cd38917b，分支codex/experiment/baby-teacher-baseline，launch_dirty=false，运行结束源码未变；当前全部报告源码指纹一致。
- 300轮、64200更新、301次学生Validation完整；教师/学生Test0，Test划分未读。本次仅CPU加载已保存权重和报告，无模型前向或新评估。
- 对照旧report配置，仅learning_rate由2e-5变6e-5。数据哈希及initial_validation完全一致，共享SPORTS_CONVERTED_20260916、固定epoch37教师。初始Recall0.0473432010。
- 全部曲线loss/指标与保存权重有限；严格Recall改善标记、最佳epoch与指标一致；best.pt配置/输入/source匹配，tensor digest及共享别名副本核验通过。runtime教师/prompt未变、学生更新、alias及最佳权重roundtrip通过。selected-versus-tested不适用。
- 命令：`D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2022 --epochs 300 --student_lr 6e-5 --gpu_id 0`。
- 其余参数引用TRAINING_LOG.md的PROMPTMM_RELEASE_SPORTS_VALIDATION300_BATCH_V1；batch1024、214步/轮、dim64、layer1、AdamW decay0.01、embedding decay1e-5、pair/list1e6、feature0.1、SCE2、negative10、foreach=False、teacher drop0.2/prompt0。Validation每轮、初始不参选、Recall@20严格改善/较早同分，窗口内不早停。
- 目录`exp/promptmm_release/sports_promptmm_release_validation300_seed2022_lr6e5_v1/`：report.json411456bytes，SHA256 `84b6a90f2f5510ff22df24dfa2e267d634a4fbb6a260f0500f3062bf48b9ad03`；best.pt27629405bytes，SHA256 `e32f9907d68874f09f4327a6ccfba8b4c86e1b7c73c7d0a06713f990b6e94e27`。
- 对照旧report SHA256 `a0872a978fd0087190b1bb431897ca7d6dd73e840a12712d533c7933df657330`已复核。原始产物保留且Git忽略，不构成物理备份/论文结果冻结。

## 验证曲线

|前N轮最佳Recall|lr2e-5|lr6e-5|
|---|---:|---:|
|30|0.06579641|0.06857338|
|120|0.06954956|0.07874371|
|200|0.07292355|0.08473913|
|300|0.07697886|0.08763456|

新配置在120轮已经超过旧配置300轮最佳，支持原配置优化进展受学习率影响，不能把此前差距全归因于算法。相同seed不保证DGL完全相同轨迹，AdamW实际衰减也随lr变化；不将其宣称为排除全部随机和正则化因素的因果证明。

新配置最终Recall0.0871794746/NDCG0.0400561524，低于294轮最佳，但281–300轮平均Recall仍比261–280轮高0.0004795447（旧配置0.0008338639）。最后数轮回落不足以确认持续过拟合；也未证明充分收敛。不要自动继续加轮数。

目标约−1.577e8到−4.018e8，BPR0.12070046到0.15972474，feature0.78322985到0.77336167；旧配置最终BPR0.59955430。目标取舍随优化率明显变化，单看总负loss不能判断推荐质量。

全程13.0008小时，训练/验证平均16.50/138.89秒每轮，训练allocator峰值1.6484GiB。与前实验系统负载未控制，不作为效率对照结论。

## 解释与下一步

已支持：发布版适配对学习率敏感；seed2022下，lr6e-5超过当前完整方法。旧三种子结果仍是有效固定配置证据，但不再支撑广义“超过PromptMM”。已有完整方法相对BPR和去文本的消融证据没有因此失效。

尚未证明：新配置在其他种子仍占优、合理调优后的最终差距、Test泛化、效率收益、论文创新性。此结果不能单独推出必须重做课题；也不能以简化模型为理由，在未测受控成本前宣称轻量化贡献已成立。

唯一下一步：准备seed2023、2024同一lr6e-5/300轮/仅Validation复核，固定其余配置、独立目录、串行失败停止，由用户手动执行。先完成新配置三种子配对，再决定是否转向受控效率/初始化机制诊断或增强方法；不马上加门控、不自动追加调参或Test。此处只建议，未实现或启动。

所有eligibility=false保留，Baby历史seed2023审计例外已接受且原manifest仍为false；第二创新点未定。
