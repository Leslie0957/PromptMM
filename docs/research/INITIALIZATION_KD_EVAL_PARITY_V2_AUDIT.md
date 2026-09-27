# 评价等价与计时 v2 结果审计

2026-09-27，**completed / artifact acceptance passed**。用户手动执行一次，审计仅CPU读取保存的JSON、日志和排名数组，未重新加载模型/数据分割、执行排名或使用GPU。

## 来源、命令与协议

启动源码 `42d1fd815ae8964171765ac50024d3fffe5818d2`，分支 `codex/experiment/baby-teacher-baseline`。manifest与report均记录该提交；源码启动/结束门检查干净HEAD（允许用户未跟踪check/），本次审计开始时HEAD相同、已跟踪文件干净。父PID31440、worker23888及claim对应。完整声明见[v2启动记录](INITIALIZATION_KD_EVAL_PARITY_V2_LAUNCH_2026-09-27.md)。已执行命令（只作历史记录，不可重跑）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_eval_parity.py' --attempt v2
```

run_id `sports_init_kd_eval_parity_seed2022_v2`；resolved config与固定v2配置及继承的共同配置完全一致，digest `fdc483e19f8ce2d8ddb37426c3e64be115a069ec07d983d5570b8eda13dba8dc`。与v1只改变新目录/命令选择及已提交的有限JSON发布重试；原评价/科学协议不变。

输入仍为smoke T1/epoch1/8步后的固定两张ID表，checkpoint SHA `ec18349231ded3a4b1dcd400bff31b80a2e23b1c819519ad6c6fd0eb64ddbcc7`，来源 `42336f5355dd76f4f17f91009deb8a4477ab2e44`。Train/Val身份沿用[资产预检锚](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)，worker运行前校验；审计不重复读取输入资产。环境/数值：run_5060 Python3.10.20、torch2.11.0+cu128、numpy2.2.6、scipy1.15.3、psutil7.2.2，RTX5060/cuda0/float32/TF32off/deterministic、CPU线程4，与声明一致。

固定reference→fast两遍完整Val，35598用户，batch256、K10/20/40/50、Train候选排除、分数/指标运算相同。训练步数0、无优化器、无teacher forward、Test读取/排名0，由冻结源码入口守卫和报告支持；没有独立OS文件访问轨迹。没有选模或最终Test，不适用selected-versus-tested checkpoint equality。没有训练objective；全部评价指标有限且在[0,1]。

## 结果与独立验收

supervisor completed、exit0、acceptance passed。审计重新执行保存产物验收函数，并独立确认manifest/config/source/checkpoint声明一致；两份排名矩阵均35598×50 int32、ID范围合法、各行无重复、全数组完全相等。全部保存文件SHA记录如下。

- 报告：35598/35598用户分数行字节哈希一致、有序Top50一致、所有指标字段完全一致。分数行本身未持久保存，不能在不重新计算的前提下独立重验每行分数；该项依赖已提交比较代码和完成报告。保存Top50的相等已独立复核。
- 排名原始数组digest：`f46589ab40d16d451ea14d0d7c9679992414761a357c1232fb3f0ecd1b20292b`。
- 全进程309.781秒；reference291.939965秒、fast10.123664秒，比值28.837381。计时包括GEMM/传输/排名/指标/哈希核验。

两个实现指标逐字段相同：

| K | Recall | NDCG |
|---|---:|---:|
| 10 | 0.063677442003447313 | 0.035237953785412383 |
| 20 | 0.094184495190847733 | 0.043332489261105771 |
| 40 | 0.13527336830115855 | 0.052174802541743469 |
| 50 | 0.15173153386601548 | 0.055301059934687168 |

Recall20/NDCG20也与既有smoke对应值相同。这是固定模型的实现一致性证据，不是新训练质量、KD收益或新算法证据。

## 资源与解释限制

900秒上限内完成；CUDA allocator峰值75497472 bytes（72MiB）<2GiB；报告采样RSS峰值976748544 bytes（约0.910GiB）<4GiB。最终遥测RSS966893568 bytes、可用盘411708551168 bytes>4GiB，最终文件总计14249429 bytes<256MiB。父进程验收通过，未触发预算限制。RSS不是全程连续峰值，CUDA不含驱动/桌面占用；最后遥测wall302.141秒是worker相对计时，完整父进程时间以supervisor309.781秒为准。

这一次不再出现v1文件替换失败；不能据此保证未来没有文件占用，也没有记录实际发生过几次发布重试。v1失败完整保留。

本次reference291.94秒高于此前smoke121.98秒，两者运行时机及核验开销不同，未独立测量差异来源。固定顺序reference先、fast后，仅一次，可能受缓存及系统负载影响。28.84倍是本次带核验的评价路径比值，不是稳健基准，更不是整个训练加速。

若未来1204次评价都维持10.123664秒，则评价部分约3.385803小时。此为条件粗估，不含四臂训练、输入/保存/验收及运行波动，不能单凭此宣布24小时完整预算成立。快路径尚未接入隔离四臂训练入口；本次不修改正式配置、预算、300epoch/每epochVal或候选效应阈值。

## 原始证据指纹与路由

目录 `exp/initialization_kd_interaction/sports_init_kd_eval_parity_seed2022_v2/` 原位保留，无failure.json，无新增checkpoint。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `acceptance.json` | 387 | `0f1167352567895188707f4df36276279164c0615299f16d58c2e3ad6c9525ef` |
| `fast_top50.npy` | 7119728 | `c34fbfe170384095ad6c5b0e549dbb1ad05b95de7c28ffe5925dc197eaec8c0c` |
| `launch_manifest.json` | 7118 | `ea4259c6be5e57a465054b289b2cb6708ab67060fb0e9a2df4a74a8243349791` |
| `reference_top50.npy` | 7119728 | `c34fbfe170384095ad6c5b0e549dbb1ad05b95de7c28ffe5925dc197eaec8c0c` |
| `report.json` | 2050 | `c274dc7227bbef065d53730116e638a760bd79d6ba100f14c870588aafa1e890` |
| `supervisor.json` | 126 | `5105984adfd59f76efb0c92d88315f88c6aa101db28c0d2da0d5c8742b58cf3b` |
| `telemetry.json` | 187 | `31588e70d520b88950b2c0cc3847c3768a2f675cc94e0f1542c489510c2409af` |
| `worker.log` | 70 | `78c03d187585e356f6098d73025a9ec0dca525174cd767fcbfb39ad2eb7343a5` |
| `worker_claim.json` | 35 | `d823b3cf3340cd87fab63797c98edd000fd316bddc4d9d3e20f2ea0798f1d478` |

已更新本审计、TRAINING_LOG独立outcome/短当前状态、root README和初始化×KD实验族导航；六项生成目录在人工编辑后刷新。docs总览无具体parity状态，无需变化；四臂/旧18格矩阵、论文/缺口无新科学证据，不适用。旧声明/审计、smoke/v1/v2全部资产及check/保持原位，不提交原始排名或模型。非科学结果冻结，无tag/bundle/merge/push。

唯一下一步：准备把已核验快评价器接入隔离四臂入口，完成合成一致性检查并核算训练/保存总预算，再判断能否形成用户手动运行声明。不得自动重跑parity/smoke、启动四臂、扩大预算、进入P1b/P2或门控；本审计不提供新实验命令。
