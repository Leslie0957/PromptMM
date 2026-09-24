# Sports 随机初值三种子已选表征方向诊断（2026-09-24）

一次 CPU 只读诊断完成：`& 'D:\miniconda\envs\run_5060\python.exe' -B tools/diagnose_sports_coldinit_selected_geometry.py`。九份 Sports 300 轮 Validation-only Full、image-matched、BPR 的唯一 manifest 与已选完整 checkpoint 均通过[原审计](SPORTS_THREE_SEED_VALIDATION300_2026-09-18.md)中的 SHA256 校验；只从 checkpoint 读取物品 ID 表。共同几何参照是后续保存的 process0 教师最终物品、图像和文本张量（SHA256 `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`），不是已证明逐位等同于九次历史运行各自进程内目标的记录。训练矩阵 SHA256 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`。

## 测量定义

每个物品向量先单独单位化。精确训练正物品边际权重 `p_i`：均匀抽一个有训练交互的用户，再从该用户正物品中均匀抽一个物品。另定义文本独有方向 `r_i = normalize(t_i - (t_i·v_i)v_i)`，其中 `v_i`、`t_i` 为单位化图像、文本方向。对学生已选物品向量分别计算 `cos(student,image)`、`cos(student,text)`、`cos(student,r)`、`cos(student,teacher final)`，再按 `p_i` 求均值。18352 个有正采样概率的物品全部有效，保留概率质量约 1.0；图文参照自身的加权余弦为 `0.045375`。该统计没有使用 Validation/Test 交互或重新计算推荐指标。

## 结果

|已选模型，三 seed 均值|图像余弦|文本余弦|文本去图像残差余弦|教师最终向量余弦|
|---|---:|---:|---:|---:|
|BPR|−0.002005|−0.016425|−0.016886|−0.019956|
|image-matched|0.723993|0.045425|0.014695|0.265282|
|Full|0.683934|0.326221|0.298963|0.489743|

同 seed 的 Full − image-matched 配对差值：

|seed|图像|文本|文本去图像残差|教师最终向量|既有最佳 Val Recall@20 差值|
|---|---:|---:|---:|---:|---:|
|2022|−0.041057|+0.280586|+0.284102|+0.223858|+0.0036782440|
|2023|−0.039764|+0.280872|+0.284334|+0.224527|+0.0036654255|
|2024|−0.039356|+0.280930|+0.284366|+0.224997|+0.0051903239|

这些数值表明：Full 的已选 ID 物品方向在三个 seed 上均有更多**不属于图像方向的文本分量**，同时图像方向余弦稍低；这与三个 seed 的 Full Validation Recall@20 高于 image-matched 同时出现。它是目标确实影响所学方向的描述性证据，不能由此断言文本残差导致了推荐增益。教师最终向量余弦较高也不能替代排名效果。

## 解释边界与协议

- 历史随机初始物品向量未保存，不能计算同一运行的精确“训练前→已选状态”位移；同 seed 也不能证明各臂训练三元组逐批一致。
- BPR 没有固定教师坐标的约束；其用户和物品表可以共同旋转而不改变点积排名，因此 BPR 与教师参照的原始余弦不具坐标旋转不变性。它只作描述，不当作 BPR 缺乏推荐信息的证据。
- 公共 process0 教师张量是相同几何标尺，不保证每次旧运行当时的教师语义张量逐位相等。九份 checkpoint 经过 SHA 校验且 selected epoch/Recall、数据 train_mat、教师指纹及 Test 标志与已审计 manifest 相符；没有再次运行模型或计算 Validation/Test 排名。
- 结果不是因果中介分析、显著性检验或 Test 结论。没有优化器步、模型前向、参数修改、Validation/Test 文件读取或排名。原九臂仍是 `paper_ready_eligible=false`；本诊断没有 selected-versus-tested 比较。

报告 `exp/representation_checks/sports_coldinit_three_seed_selected_v1/report.json`，SHA256 `0431eaf9d34e7ccee3634f053ded95a7e0a3bf3fd1685f27af2a717069935183`；脚本 SHA256 `88c12766d4993e9b8f67bbdbc0745e735ab5a652bb0943f3b8f7ef9c9e957195`。诊断报告和九份模型是 Git 忽略资产，尚无物理/异地备份。

**唯一下一步**：设计一组严格配对的随机初值 Full 与 image-matched Validation-only 对照合同，预先固定并保存相同初始 ID 张量与每轮训练三元组，保持其余预算、教师、数据、选模协议一致，再由用户单独决定是否手动执行。当前不准备启动命令、不运行新训练或 Test。
