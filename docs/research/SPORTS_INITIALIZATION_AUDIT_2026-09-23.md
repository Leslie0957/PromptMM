# Sports seed2022 初始化对照审计（2026-09-23）

两新增臂均完整完成300轮并通过审计；这是固定配置、单种子Validation诊断。已有完整方法随机初始化并不是其可达到的最好配置，初始化差异足以改变本次方法比较的方向。但尚不能将继承教师表示的收益归因于蒸馏。

|方法|初始化|最佳轮次（统一1起算）|Recall@20|同轮NDCG@20|
|---|---|---:|---:|---:|
|TD完整方法（已有）|随机|294|0.0823282598|0.0372074078|
|TD完整方法（新增）|教师|283|0.095438076853|0.043780603134|
|PromptMM release lr6e5（已有）|教师|294|0.087634556329|0.040125451936|
|PromptMM release lr6e5控制变体（新增）|随机|1|0.042124376895|0.019498707032|

TD新臂相对原随机初始化Recall提高15.9238%；相对原教师初始化release提高8.9046%、NDCG提高9.1093%。仅seed2022，不能推广为三种子稳定优胜或Test优势。随机release是机制控制变体，不替换原较优基线；较差结果仍是有效完成证据，不回退或删除。

## 初始化前后必须分开解释

TD未训练初始Recall0.094184495191、NDCG0.043320782869，已经高于release训练后最佳值。训练到最佳后，较自身初始Recall仅增加0.001253581662（相对1.3310%），NDCG相对增加1.0614%。这证明当前继承教师表示非常有效，不证明这些小幅后续增益由方向蒸馏独有地产生；BPR微调也可能获得。教师原始向量与release图传播后的评分不同，不能称两者初始评分受控相同。

TD第1/30/120/200/283/300轮Recall分别0.094254723855/0.093701589517/0.093864720672/0.094338764157/0.095438076853/0.095060402701；前30与前120最佳均0.094254723855，后期才超过早期最佳。末20轮均值较此前20轮增加0.000284191996，不能声称完全收敛或自动加轮数。

随机release初始Recall0.001046407102，第1轮达到最佳0.042124376895，最终0.028967993557，末20轮均值增量-0.000038355655。当前固定损失权重/学习率等配置在随机起点下表现差；不能证明所有随机初始化PromptMM都会差，也不能仅凭Validation下降就断言具体原因是过拟合。所有记录目标和指标有限，运行并未失败。

## 运行身份与审计

手动批次命令：`D:/miniconda/envs/run_5060/python.exe -B tools/run_sports_initialization_pair.py`。完整分臂命令见TRAINING_LOG的SPORTS_INITIALIZATION_PAIR_SEED2022_V1声明及docs/SPORTS_INITIALIZATION_PAIR_SEED2022.md。干净启动源码6af8f5132d76e73799fdfd76179e186b336e7c90，分支codex/experiment/baby-teacher-baseline。两臂seed2022、lr6e-5、batch1024、dim64、300轮，固定Sports数据及epoch37教师；各自损失/采样/优化器语义不变，TD只启用教师初始化，release只改变初始向量来源并保留alias。

批次2026-09-22 20:57:35至2026-09-23 04:58:44，约8小时1分。TD约3小时8分，release约4小时53分；不同评估路径，这些时长不是受控方法效率结论。

重新执行只读完成门槛核验，检查300轮完整有限曲线、严格Recall选模、配置、初始来源、源码指纹与原始文件身份。CPU加载TD权重/优化器确认有限，推理导出ID表与完整checkpoint逐元素相等；CPU加载release权重确认有限、alias副本相等、选模指标及初始化元数据匹配。运行时release还验证教师/prompt不变、64200更新、301Validation、checkpoint roundtrip及源码不变。TD初始Validation不参选，后续300轮选模；TD最佳原始epoch282为0起算。

教师和学生Test排名评估均为0；TD保留历史加载器Test结构读取，release无Test文件读取。没有新增训练、前向或Validation/Test排名。selected-versus-tested不适用。不同代码路径的教师提取/图传播、初始化尺度、参数共享和调参历史仍限制跨方法因果归因。

## 产物指纹

- batch: exp/initialization_checks/sports_initialization_pair_seed2022_val300_v1/batch.json；SHA256 d8f65d96122156294ce1c54cff9b0fe397f671102fde7ca0a5913e8f430f0d9a。
- TD manifest: exp/runs/sports/run_manifest__2026-09-22 20_57_40.716760_sports_light_init_pid8964.json；SHA256 7f863a7eec52d5af7eec2e6a611d05820653d32790e3a344813d536f795c9e59。
- TD checkpoint: Model/sports/td_distill/td_distill_full__val_test_once_v1__2026-09-22 20_57_40.716760_sports_light_init_pid8964.pth；SHA256 3173f2372a34e568fb894a650b494514f58da43bcc6232973c73c1601a0595aa。
- TD curve位置见manifest artifacts.converge_run；SHA256 320dd9899dd3d574363d58ab63cbb4ed693a3ce5a578275e1c30420ba842987b。
- release目录: exp/promptmm_release/sports_promptmm_release_validation300_seed2022_lr6e5_randominit_v1/；report SHA256 fcb4cf713f15a86647efcf105f742b895ef55b5c29ba2839c0ceac5d51b842e3；best.pt SHA256 ab7487e77591cc34fe4e6d25875ed6893da2ef3d00dde46ef67692502e7c318b。
- 教师SHA256 57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea。TD初始用户向量SHA81dfece426261cd6920ff64d79e8fe4b0579286185f6e7e7bb2c29ff28b76791、物品向量SHAe61376f4a0ed29e9c96a7d62083732ae0c991fdfcc023c51efc478b373fdd9c2，独立复制；release随机向量身份见report。

## 唯一下一步

先准备seed2022、相同教师初始化的BPR-only对照，沿用TD本次其余参数和300轮Validation预算，关闭方向蒸馏；同时核验实际初始向量指纹一致。与本次TD教师初始化及已有未训练初始指标对照，回答“继承表示后，方向蒸馏是否比单纯BPR微调更有价值”。本次不准备或启动该运行，也不自动补种子。若要将初始化/方法优胜作为稳定结论，再按原计划补齐两初始化控制臂的2023/2024，保留所有结果。

paper_ready_eligible=false不改；Babyseed2023仍按审计例外接受且原manifest=false；第二创新点未确定。忽略目录产物保留，本次Git提交不代表物理备份或论文最终冻结。
