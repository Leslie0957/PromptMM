# Sports 教师 Validation 120 轮配置

配置：sports_teacher_validation120_v1；seed-2022；仅教师。

这是首次 Sports 教师参考运行：明确迁移 Baby 的结构和优化参数，独立使用 Sports 数据重新训练，不加载 Baby 权重，也不声称是 Sports 最优超参数。120 轮、patience=120，让有限预算内的验证曲线完整可见；这不是收敛保证。

## 固定配置

```json
{
  "Ks": "[10, 20, 40, 50]",
  "allow_teacher_alias_overwrite": false,
  "batch_size": 1024,
  "cf_model": "light_init",
  "dataset_preflight": true,
  "drop_rate": 0.2,
  "duplicate_modalities_policy": "error",
  "early_stopping_patience": 120,
  "embed_size": 64,
  "epoch": 120,
  "eval_protocol": "val_test_once_v1",
  "feat_reg_decay": 1e-05,
  "feat_soft_token_rate": 1.0,
  "hard_token_seed": 2022,
  "hard_token_type": "pca",
  "if_train_teacher": true,
  "layers": 1,
  "lr": 0.00055,
  "mess_dropout": "[0.1, 0.1]",
  "model_cat_rate": 0.55,
  "prompt_dropout": 0.0,
  "regs": "[1e-5, 1e-5, 1e-2]",
  "run_efficiency_benchmark": false,
  "run_final_test": false,
  "seed": 2022,
  "smoke_train_batches": 0,
  "soft_token_rate": 0.005,
  "sparse": 1,
  "t_feat_mf_rate": 1.0,
  "t_prompt_rate1": 100.0,
  "t_prompt_rate2": 1.0,
  "t_prompt_rate3": 1.0,
  "t_weight_decay": 0.001,
  "teacher_checkpoint": "",
  "teacher_only": true,
  "teacher_reg_rate": 1.0,
  "test_flag": "part",
  "weight_size": "[64, 64]"
}
```

配置由 codes/utility/dataset_profiles.py 定义；--dataset sports 自动选择。命令行覆盖会记录到 dataset_config_overrides；本次声明要求为空。其余参数及环境由运行 manifest 捕获；不激活 student_profile。

数据锚点：TRAINING_LOG.md「2026-09-16 Sports conversion onboarding (completed)」，转换 manifest SHA256 为 3772a17c8b70fa4739653534dca8d542e67e0649f1f44ccba4194fec17bc1104。原始特征完整保留，训练启动时按现有硬 token 路径执行 PCA（64 维、随机种子 2022，缓存与 Sports 绑定）。这属于使用全物品内容的传导式设置，不使用验证/测试交互拟合图；图仅由 Train 构建。保留官方冷物品。

每轮使用 Validation Recall@20 严格改善保存最佳，结束恢复最佳 checkpoint。候选集排除 Train 交互。run_final_test=false，因此不做 Test 排名；teacher_only=true，因此不进入学生训练。预检/加载器仍读取 Test 矩阵做结构校验，这不属于 Test 指标评估。

当前实现会将 paper_ready_eligible 标为 false（final test evaluation is disabled），这是本次仅验证运行的预期状态，不是执行失败。不会发布共享教师别名；不能在尚未审计后续复用规则时声称该权重已可直接用于正式学生运行。

## 手动启动（仓库根目录）

```powershell
Set-Location D:\Download\PromptMM
& D:\miniconda\envs\run_5060\python.exe -B codes/main_mmlight.py --dataset sports --gpu_id 0 --if_train_teacher true --teacher_only true --run_final_test false
```

从包含此配置及声明的干净提交运行一次，不重试、不续训、不自动进入学生/其他种子。需要更改参数或运行中失败时保留日志及产物，先审计。此文档和准备任务没有启动训练或拟合 PCA。

产物：Model/sports/runs/teacher_model_val_test_once_v1__<run>.pt、exp/runs/sports/run_manifest__<run>.json 及 logs 下运行日志；具体文件名以启动输出为准。GPU 显存与训练时长尚未经 Sports 实测。运行完成后核对120轮、有限损失/指标、最佳轮次、checkpoint 指纹、零 Test/零 student、曲线是否在预算边界继续上升，再决定能否冻结教师。

历史 seed-2023 审计例外保持不变，原 manifest false；第二创新点未确定。
