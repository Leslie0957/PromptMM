# 逐用户槽位控制：独立产物审计

2026-09-30。用户报告完成后，只读核查保存产物；没有加载Train/Val标签、模型或Test，没有重新打分/评价。原运行完成、硬验收通过；科学判断unresolved，详见[结果与统一判断](PER_USER_SLOT_CONTROL_RESULTS.md)。

## 身份与执行

- 目录：`exp/innovation2/per_user_slot_control_v1/`；source `6a0179e5862e85c1482360186495513f7479b930`，分支`codex/experiment/baby-teacher-baseline`。启动源与当前审计前HEAD相同；仅未跟踪archive/reviews，不涉及运行源。
- 原命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_per_user_slot_control.py' --formal`。固定seed2022/2023/2024，CPU缓存列表构造；模型加载与参数更新零；无epoch/选模/checkpoint或被测checkpoint，本次不适用。
- profile SHA256 `726f66a4448e73a3753e413a53a21a6bfaaae0c4ff64268f985f28a168f58201`。manifest原status=launched是启动记录；report completed、supervisor exit0/no termination、acceptance hard=true才是完成证据，不修改原manifest。
- 全部新产物完整性哈希、9项固定输入及两图资产哈希通过；Train/Val身份沿用已钉定anchor，本次不重新读取数据。输入来源为全局加分提交6dbdc400cb029340d2c40bdc65e8f4a442941108。

## 验收与证据边界

独立保存产物检查148项通过，机器记录见[VERIFICATION](PER_USER_SLOT_CONTROL_VERIFICATION.json)：三seed用户与S身份、H/R逐位目标位置相同、H恰为B缓存组内前缀、每用户20项唯一、封存哈希不变、B/Q/R旧指标逐项回归、曝光及micro算术、配对明细净命中/Recall一致、有限值与原决策复算。

无Train已见项、分母6347/6389/25163及指标对标签正确性来自原运行硬检查和已审计输入，本次不以重新评价来独立复证。缓存排序检查见原construct源码；审计验证其哈希来源与实际前缀。完整资源采样通过：29.203秒、采样RSS峰92430336 bytes、输出4814016 bytes，低于600秒/2GiB/128MiB；磁盘余量满足2GiB。采样不保证捕获瞬时峰值。

源码禁GPU并无模型调用；三份H先封存，再允许Train/Val读取。保存访问记录4次均在封存后，仅train_mat/val_mat、denied0；Test0由Python whitelist、代码路径及记录共同支持，不是OS级全进程访问证明。审计自身没有打开任何数据矩阵。训练选模与Test次数均不适用于本次缓存控制，Test访问为零。

逐seed执行均合格；科学联合规则2022/2024满足、2023因总体Recall超过容忍线不满足，三seed整体unresolved。不是统计显著/等价/正式非劣；不重写先前N/C的screen_stop。

## 记录路由与保留

新增RESULTS/AUDIT/HANDOFF/VERIFICATION；追加TRAINING_LOG，更新根README、第二项入口及实验族；刷新生成导航。第二项无独立活跃数值矩阵，本次RESULTS承担12状态引用和3seed主差值。docs/README链接仍有效无需改；第一项Test/初始化/成本矩阵、正文、缺口及THESIS_ROADMAP未触发变更。旧记录、原始产物、reviews和zhuanli保留；不是冻结/备份里程碑，无tag/bundle/资产迁移。

唯一下一步：个性化低频分配算法立项评审，仅设计并核对近邻方法，不再自动延长此诊断链，不授权训练或Test。

report SHA256 `73752a75640ff1a737b65906586c3f1b36ca1c48694ca207d21ab7aab88de839`；completeness SHA256 `8dd8f7370a026beef533c34062f4b47b167627db67141ff548e9cd7297dc1119`。
