# Validation 精确提速：准备结果与一次手动核验声明

状态：实现及21项合成CPU测试通过；本页真实GPU核验**尚未执行**，正式四臂仍不可启动。此前资源smoke已完成，不再使用旧命令。当前源码基线 `a51bcce4a2fdf0d5cd8335c31a9fe96166a96ea9` 加本次准备提交；启动为包含本声明的干净提交HEAD，分支 `codex/experiment/baby-teacher-baseline`，manifest记录实际HEAD。

## 改了什么，保留什么

新增 `codes/initialization_kd_fast_eval.py`。用NumPy分区找第K大分数的边界，收全边界以上项，再按物品ID升序填边界同分项，最后按分数降序/物品ID升序排这K项。没有给分数加epsilon，没有使用任意的argpartition并列顺序，没有改变精度。有限分数下，此顺序与本阶段旧评价器按升序候选ID调用heapq.nlargest一致。Train物品在副本中标为负无穷，原始分数不改；全部候选不足K或非有限分数立即拒绝。

`initialization_kd_adapter.evaluate_validation`只增加可选ranker参数；默认仍走原heapq路径。新旧路径共享**相同的GPU点积矩阵乘、batch256、用户遍历、Train排除、K10/20/40/50、浮点精度、DCG/NDCG公式及逐用户求和顺序**。不减少用户、物品、评价频率或训练轮数，不修改训练器/损失/采样器。快路径尚未接入四臂训练，也未凭合成测试宣称真实数据等价或24小时预算成立。

合成验证覆盖：随机分数和离散密集同分、K边界并列、全同分、正负零、负数、相邻可表示浮点值、float32/64极值、Train排除、可用项恰好K/不足K、非有限输入、原分数/模型/矩阵不变、仅有Val正例的用户、不同用户batch尾块、完整指标逐字段完全相等；人为打乱快路径排名时核验必须失败。保存排名文件损坏时父进程验收拒绝。

CPU合成rank-only计时：seed20220927，128×18357 float32标准正态分数，排除ID0–29、K50，各路径预热1行后各3遍。旧路径秒数0.430219/0.427955/0.431953；快路径0.007373/0.006698/0.006748，中位数比值约63.75。每遍有序ID完全相等。**不含GEMM、指标、真实数据、GPU或完整评价，不能直接当成真实加速比。** 无真实资产/分割加载，AI没有执行本页命令。

## 唯一用户手动命令

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_eval_parity.py'
```

固定声明JSON：[eval parity config](INITIALIZATION_KD_EVAL_PARITY_CONFIG_2026-09-27.json)，继承[smoke环境/资源配置](INITIALIZATION_KD_RESOURCE_SMOKE_CONFIG_2026-09-27.json)与原共同配置；不调用smoke训练代码。新独占目录 `exp/initialization_kd_interaction/sports_init_kd_eval_parity_seed2022_v1/`，一次尝试，无恢复/重试/自动下一阶段。入口无参数调参，内部worker经pipe token、PID、source/config与独占claim绑定；父进程监督和完整产物验收复用已通过的框架。

输入仅为：

- 已审计smoke的 `T1/final.pt`，SHA256 `ec18349231ded3a4b1dcd400bff31b80a2e23b1c819519ad6c6fd0eb64ddbcc7`，来源 `42336f5355dd76f4f17f91009deb8a4477ab2e44`，T1/epoch1/8步后的固定状态。仅使用其中两张ID表；不创建或恢复优化器，不作任何更新。
- 固定Train/Val，SHA分别 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8` / `1b4224c3fb091ad23a59e0f8a7b58c7b0c959863a7189cfc8b89eeb90a467e03`。沿用[只读预检锚](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)与[smoke结果](INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md)，只对本次实际输入复核hash；不读取旧teacher/cache/tape资产内容。

工作量固定为**先reference、后fast两个完整Val遍历**，都覆盖35598用户、同一两表、batch256和相同评分配置。reference遍历保存每用户全分数行的SHA及有序Top50；fast遍历逐用户检查分数行字节完全相同、有序Top50完全相同；任一不符立即失败，不用容差或挑选用户。完整指标字典必须完全相等。保存两份35598×50 int32排名表，父进程再次检查文件hash、shape、全数组相等和统一排名digest。无Test文件打开/排名，无teacher forward、BPR/KD计算或训练；不会重新执行8步smoke。

环境固定Python3.10.20/torch2.11.0+cu128/numpy2.2.6/scipy1.15.3/psutil7.2.2，原run_5060解释器；cuda0、float32、TF32off、deterministic algorithms、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=2022、torch CPU线程4。比较只解释此固定状态与此实现，不构成新科学性能结果。

## 资源、输出、验收和停止

父进程900秒（约1秒轮询，含输入检查/评价/保存/验收），CUDA allocator2GiB、workerRSS4GiB、输出256MiB、可用盘至少4GiB、attempt1。CUDA allocator不含驱动/桌面总占用；RSS为采样检查。预计耗时尚未实测：旧完整Val约122秒可作参考，但本次还含逐行hash/比较，不能承诺完成时间。15分钟是停止限额。

输出：`launch_manifest.json`、`worker_claim.json`、`worker.log`、`telemetry.json`、`reference_top50.npy`、`fast_top50.npy`、`report.json`、`acceptance.json`、`supervisor.json`；失败时尽可能记录`failure.json`，保留所有部分产物。无新checkpoint。

通过条件：exit0且supervisor completed；source/config和输入身份一致；35598个分数行与有序Top50全部一致、K50、所有指标完全一致；保存数组SHA及数组相等复核通过；训练步数0/Test0，资源未触发限制。**慢或没有加速本身是有效结果，不自动标失败或重试。** 任一身份/协议/数值/排序/预算错误，停一次尝试；不删目录、不调参数、不扩大预算。

计时分别包含完整GEMM/传输/排名/指标以及各自的hash/核验开销，固定顺序reference→fast，无重复、ABBA或稳态统计。后跑路径可能受缓存影响；记录的是单次、带核验开销的观测，不是稳健效率结论。只有结果返回审计后才能重新估计1204次Val与训练/保存的总体预算，仍不得把rank-only63.75倍或本次一次速度当作整个训练加速比。

## 记录更新目标

结果审计 `docs/research/INITIALIZATION_KD_EVAL_PARITY_AUDIT.md`；TRAINING_LOG本pending单独completed/failed及当前交接；初始化×KD实验族和root/docs入口按状态更新；六项生成导航刷新。四臂结果矩阵、旧18格、论文/缺口不变（仅实现等价性与计时）；历史资产/日志/manifest/check/保留，不提交排名原始数组。非正式稳定里程碑，不做tag/bundle/合并/推送。正式四臂候选阈值与24小时候选预算不在本次改变。

唯一下一步：用户只执行上面的新命令一次，完成或失败后返回审计；不重跑旧smoke，不自动启动四臂。
