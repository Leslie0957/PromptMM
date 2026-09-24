# Sports 正样本采样加权方向诊断（2026-09-24）

状态：一次 CPU 零更新诊断完成。命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B tools/diagnose_sports_weighted_directions.py`。运行前合成概率与自动求导自检通过。使用当前分支 `codex/experiment/baby-teacher-baseline` 的 `ab56ba921d4b99caeb93304122f7a103b751048f` 为范围基点；诊断脚本与 pending 日志在执行时尚未提交，因此本次不是正式训练启动。脚本 SHA256 `a02c51166cc00761af0d6e221548a37f9a246de85bf1f645bc7b1bf558cd616d`；输出源文件指纹与执行脚本相同。

相关未改源码 SHA256：`codes/td_distill_model_no_projection.py` 为 `13fd9e55c2ab6bbe7c87aa478935e902feee12b1b3aee953b7f66698e0fdaf1b`，`codes/utility/load_data.py` 为 `d5c3c0f7ccc9ef85cefb3b6c1ab83080a22d1ed7e2975f7ba807561f6a6e4746`，`codes/Models_mmlight.py` 为 `ec993272ccf07d24a70b41bb9fb87273d95e01d2de6eb1dfb1950c10172933dc`。

## 输入与实际采样权重

- 共享初值六张张量：`exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt`，SHA256 `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`。
- 仅训练交互：`data/sports/train_mat`，SHA256 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`；35,598 个有训练交互的用户，218,409 条交互，18,352 个正概率物品。
- 固定批次：`exp/gradient_checks/sports_sharedinit_seed2022_v2/batches.json`，SHA256 `d0489b97de4733f14b9dbd10bbea82dffdd1cbbc4420181a91681cf0ccfa75f0`；原零更新梯度报告 SHA256 `5219ef3e3ccb7399bb484f42dd7edb7bc0dfe0281f7e3a555accbb68c48b74ff`。不重新采样，也不重新计算 BPR 梯度。

`Data.sample` 每次先在有训练交互的用户中均匀选用户，再从该用户的训练正物品中均匀选一个。因此单个采样位置抽到物品 `i` 的精确边际概率为

\[
p_i=\frac{1}{|U_+|}\sum_{u:i\in I_u^+}\frac{1}{|I_u^+|}.
\]

同一批内用户不放回不会改变单个位置的边际概率。`sum_i p_i=1`。五个零模态目标物品的总 `p_i=0`；下面方向均值排除零目标并将有效 `p_i` 归一化。该公式与均匀按交互、均匀按物品计数不同。

## 方向与梯度

记 `F_i` 为保存的教师最终物品向量，`v_i=N(T_i^image)`、`t_i=N(T_i^text)`。教师 `F_i=C_i+0.55v_i+0.55t_i`，其中 `C_i` 包含 ID、提示和图传播路径。学生实际目标组合为 `q_i=(v_i+0.3t_i)/1.3`，并没有再次归一化；表内方向余弦使用 `N(q_i)`。对非零 `F_i`，alpha=0.3 时一个正样本的解析梯度是

\[
g_i=-\frac{2(0.3)}{64\|F_i\|_2}(I-N(F_i)N(F_i)^\top)q_i.
\]

这是单个正样本对平均损失的贡献在除以批大小**之前**的值。期望物品行梯度为 `p_i g_i`；实际有限批次还会重复物品并按批大小聚合。梯度与 `F_i` 径向正交，下降方向是目标在 `F_i` 切平面上的投影。

|测量|结果|
|---|---:|
|采样加权 `cos(F,N(q))`|0.4839051465|
|采样加权 `cos(F,N(v+t))`|0.6917894593|
|无权、非零目标 `cos(F,N(q))`|0.5396549106|
|无权、非零目标 `cos(F,N(v+t))`|0.7295938730|
|`cos(F,N(q))<0.5` 的采样概率质量|0.5101775183|
|采样加权 `||F||`|5.1347299222|
|采样加权 `||q||`|0.8125410614|
|采样加权目标切向分量范数|0.6993817393|
|采样加权单正样本梯度范数（批均值前）|0.0014114051|
|期望物品梯度矩阵 Frobenius 范数|0.00001416155|

第一固定批次重新用当前 `directional_distillation_loss` 做 CPU 自动求导：物品梯度范数 `4.8754929594e-5`，原 GPU 报告 `4.8754935838e-5`；解析与自动求导的最大逐元素绝对差 `4.62e-13`。八批次解析范数与原报告的最大相对差 `3.63e-7`。原八批次记录的物品蒸馏/BPR 梯度范数比均值 `1.045023%`、余弦均值 `0.010416`。学生参数和 `.grad` 缓冲未改变。

## 解释与边界

采样权重让方向不一致更明显：按真实正样本边际概率，`F` 对 1:0.3 目标的平均余弦低于无权物品均值；等权图文方向更接近 `F`，但仍不等于 `F`，因为 `C` 仍存在。这验证了**结构和数据分布上的方向差异**，以及实际损失确实沿相应切向方向求梯度。它不是代码错误的证明，也不是 alpha3 退化的因果证明。按物品的期望梯度与 AdamW 更新不同；八个固定批次仅是已有初始状态诊断，不代表后期轨迹。没有做新的推荐指标计算、Checkpoint 选择或结果比较。

运行仅加载训练矩阵、共享张量及既有训练批次/报告；优化器步数0、教师前向0、Validation/Test 文件加载和排名均0。无新选模指标，selected-versus-tested 不适用。报告 `exp/gradient_checks/sports_weighted_directions_seed2022_v1/report.json`，SHA256 `ca3c2a4c3d69794b2c70db35875c9f09538b802b4abc7cff135f6737819c93d1`。该原始报告和输入是 Git 忽略资产，提交文档不构成物理备份。Baby seed2023 审计例外和原 manifest=false 保留；第二创新点未确定。

唯一下一步：先设计一个固定初值、固定训练批次的零更新对照，单独改变目标组合方向并匹配梯度尺度，检查方向差异是否能与强度差异分开；本次不准备或启动新训练。
