# 标量校准产物审计

2026-09-30。保存产物150项核查通过；只读输入哈希、NPZ/JSON及代码，不加载原Train/Val/Test矩阵或checkpoint张量，不重新拟合、打分、排序或评价。原始产物保持不变。

## 身份和协议

- 分支codex/experiment/baby-teacher-baseline；launch source `61496eeb7e9e09485862d8875f71c55d8fcdde89` 与准备提交一致；启动source_gate要求干净源码，原有archive/reviews例外。manifest为launched启动记录，最终report completed、supervisor exit0/无termination、acceptance true相互一致。
- 命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_scalar_calibration.py' --formal`。CPU，NumPy2.2.6、torch2.11.0+cu128，仅CPU使用。
- 三个既有B final(seed2022/2023/2024、300轮)冻结；本次无B训练/AdamW更新。每用户64对、共同tape8721、λ=.01、G/U各40次二分；只取终态，无Validation选轮。profile完整哈希与manifest一致；全部12项输入及anchor文件哈希通过，包括checkpoint仅字节哈希不反序列化。
- 原始目录`exp/innovation2/scalar_calibration_v1`。report SHA256 `385b9a40f101ce364070ab318344bfdc4d8f2c403a94f2dbfe0d12126caedbdb`；completeness SHA256 `7a95b2edee13d4ff692423e17e7a1afaf4e8b4cf47596e3a4bb093b9298d8b32`。逐文件和封存指纹核对见[验证JSON](SCALAR_CALIBRATION_VERIFICATION.json)。

## 核查结果与限度

三seed执行合格。六拟合状态、九评价状态齐全。独立检查tape形状/范围/delta与S对应、采样计数、offset摘要、保存的40次求解轨迹/区间/有限误差/目标顺序；原B列表逐位与Recall/命中回归；各组前缀及“最差已选不低于最佳未选”的缓存截断条件；列表合法唯一；保存曝光/micro/贡献算术；U−B/U−G/G−B逐用户/物品配对净差与报告一致。复算科学规则为screen_stop。

驻点值来自原运行报告，本审计未用checkpoint重算margin或导数；原Train正负成员关系、已见物品排除及指标对标签正确性依赖运行硬检查和已审计代码/资产来源，本次不重复评价。不能把150项保存产物检查写成150次独立实验。

全部参数/列表/tape封存哈希不变；访问记录前两次Train在封存前，随后Train与两次Val在封存后。没有禁止访问尝试；Test0由源码白名单及运行记录支持，非OS全进程追踪证明。教师Test0、学生Test0；没有selected-versus-tested checkpoint问题，因为无Test且标量终态不选模。

墙钟57.453秒，采样worker RSS峰741142528 bytes，输出21200886 bytes；低于1800秒/4GiB/256MiB，可用盘满足2GiB。采样不保证瞬时峰。该时长包含读取/拟合/列表/评价，不代表B训练成本或线上加速，未做公平完整成本对比。

## 记录路由

新增RESULTS（三seed×三状态矩阵）、AUDIT、HANDOFF和VERIFICATION；追加TRAINING_LOG outcome、更新实验族与根/第二项状态、刷新生成目录。docs/README入口未变；第一项Test/初始化/成本矩阵、正文和缺口、总体路线未触发修改。旧声明、审计、原始产物、reviews、zhuanli原样保留；非冻结/备份里程碑，无tag/bundle/merge。

研究结论见[结果](SCALAR_CALIBRATION_RESULTS.md)：有效低效用结果，停止当前固定配方扩展；不得声称显著性、等价性、无损或第二算法成立。

唯一下一步：第二项路线收束评审，最多两个主题一致候选，只评审证据、差异和价值，不启动新实验。
