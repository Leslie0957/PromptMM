# Sports 真实文本残差与几何匹配伪残差：手动串行验证合同

状态：**准备完成，正式六臂训练未启动**。本合同检验当前 Full 的 Sports Validation 收益是否依赖物品对应的真实文本残差方向。主比较是在每个 seed 内真实目标减伪残差目标的最佳 Validation Recall@20。历史严格配对 Full/image-matched 结果仅作背景锚，不代入新配对统计。

## 固定身份及唯一差别

- 数据：SPORTS_CONVERTED_20260916，`data/sports/train_mat` SHA256 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`；冻结教师 epoch37 checkpoint SHA256 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。
- 六臂蒸馏语义共同读取已保存的 `SHARED_TD_TENSORS_PROCESS0_V1`，SHA256 `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`。这份资产是先前 teacher process0 的图文输出，不宣称与历史冷初值六臂各自进程中计算的语义逐位相同；本次新六臂均从它读取相同图像与真实文本语义。训练入口仍加载冻结教师，但其另行计算的语义被该资产替换。
- 从已完成严格配对批次 SHA256 `b254d9e969edd3e001a4a85c2b14b69f8eb68b3d279d667100973c62fe7797b8` 读取每个 seed 的同一随机 ID 初值与完整 300×214×3×1024 训练 tape；不生成新采样、不覆盖旧产物。每个 seed 的真实/伪臂从同一初始表开始并逐批回放同一 triplet。
- 三个 seed 为 `2022/2023/2024`，伪目标 RNG 种子事先固定为 `9022022/9022023/9022024`。所有臂均沿用历史 Full 参数：300 epochs、patience300、batch1024、ID64、student_lr6e-5、AdamW weight_decay0.01、BPR + `0.3*(Dimage+0.3 Dtext)/1.3`、用户蒸馏 rates0、随机 ID 初始化、Validation Recall@20 选模、跳过最终 Test。新 profile 只更换身份；伪臂只更换**物品文本残差方向**。

单位化图像与真实文本向量为 `v_i,t_i`，令 `c_i=v_i·t_i`。用固定 RNG 产生并正交化随机单位向量 `s_i ⟂ v_i`，令 `t_i^sham=c_i v_i+sqrt(1-c_i²)s_i`，再恢复原始文本目标的行范数。这样逐物品保留文本范数、图文夹角和 `||v_i+0.3t_i||`。图像或文本零范数、或文本残差范数 `≤1e-6` 的退化行保持原目标；若训练相关有效行或几何误差检查不满足硬门槛则不启动训练，不依据 Validation 重抽伪目标。

## 手动启动及硬门槛

从干净的 `codex/experiment/baby-teacher-baseline` 提交执行**一次**：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sham_residual_three_seed.py
```

串行 child profile 顺序固定：`sports_student_real_textresidual_seed2022_val300_v1`、`sports_student_sham_textresidual_seed2022_val300_v1`、`sports_student_real_textresidual_seed2023_val300_v1`、`sports_student_sham_textresidual_seed2023_val300_v1`、`sports_student_real_textresidual_seed2024_val300_v1`、`sports_student_sham_textresidual_seed2024_val300_v1`。每条 child 命令为同一 run_5060 Python 执行 `-B codes/main_mmlight.py --dataset sports --student_profile PROFILE --gpu_id 0`。启动器记录 launch HEAD、dirty=false、完整命令及脚本 SHA；任何失败停止，不自动重试、恢复或进入下一 seed。

第一条 child 前，启动器检查 clean HEAD、旧批次报告/初值/tape、训练矩阵、教师 checkpoint、共同教师语义资产 SHA；为三个 seed 一次性构造独占伪目标。合成几何误差的每项最大值不得超过 `2e-5`，有效行须至少 `18000`。然后仅取每个已存 tape 的 epoch0 前八个 batch，对保存的冷初始 item ID 表求真实与伪蒸馏目标的**零更新原始 item 梯度**。每 seed 的 `RMS(sham)/RMS(real)` 必须落在 `[0.95,1.05]`，同时记录图像及 BPR item 梯度 RMS；不调整 alpha，也不写参数或 `.grad`。这是初始尺度门槛，不保证全程 AdamW 更新等强。若门槛失败，报告保留失败和伪目标资产，正式 child 数为零；修改门槛、目标或重跑须重新声明和授权。

预检通过才写 `exp/sham_residual/sports_three_seed_v1/preflight.json` 与每 seed 的 `sham_text.pt`，随后按序启动六臂。所有路径为 Git 忽略产物，预检及训练报告在 `exp/sham_residual/sports_three_seed_v1/batch.json`；每臂生成独立 timestamped manifest、曲线、完整与推理 checkpoint。每臂验证解析参数、相同初值/tape/共同语义、300 个有限 Validation 曲线值、最早最大 Recall@20 选中 checkpoint、推理 ID 表与所选完整 checkpoint 相等、零教师/学生最终 Test 排名。原加载器的结构性 Test 读取保留；无 Test 排名，不选 Test checkpoint。

## 预设判读

- 主读数：三对 `best Validation Recall@20(real) − best Validation Recall@20(sham)`；同轮 NDCG@20 辅助，既有 image-matched 三种子只作背景。
- 若三对均为正且平均差 `≥0.00209`（此前 Full−image 平均差约一半），支持正确物品文本残差比几何相似的随机残差更有用。
- 若平均绝对差 `<0.001` 或伪目标更好，则“文本身份对应是增益必要条件”的解释受削弱；其他结果记为未定。低指标是有效结果，不触发自动回退。
- 仅能解释此 Sports 数据、此教师、此预算；三 seed 不自动构成显著性、充分收敛、Baby 主结果、最终 Test 泛化或推理效率结论。下一阶段需要另行授权。
