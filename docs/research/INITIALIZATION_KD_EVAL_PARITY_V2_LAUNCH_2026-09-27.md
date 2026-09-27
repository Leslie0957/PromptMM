# 评价等价与计时 v2：一次用户手动运行声明

状态：准备验证通过，真实运行尚未执行。本声明取代当前启动建议；[v1失败及修复审计](INITIALIZATION_KD_EVAL_PARITY_AUDIT.md)和其全部产物保持原样。v2不是恢复v1进度，是独立的一次从头核验。正式四臂仍不可启动。

## 身份与唯一命令

修复基线 `128fecd68527c83592f3a253f79ca5165c7781d4`；本次提交只新增v2配置/声明及显式attempt路由、对应合成测试。启动源码必须是包含本声明的干净提交HEAD，分支 `codex/experiment/baby-teacher-baseline`，manifest记录实际HEAD。只允许用户手动执行一次，AI未执行真实核验；完成或失败均停下审计，不重试、续跑或进入下一阶段。

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_eval_parity.py' --attempt v2
```

固定配置：[v2 JSON](INITIALIZATION_KD_EVAL_PARITY_V2_CONFIG_2026-09-27.json)。相对[v1配置](INITIALIZATION_KD_EVAL_PARITY_CONFIG_2026-09-27.json)，仅run_id、output_namespace及命令的`--attempt v2`变化。旧v1默认命令仍指向旧目录并拒绝复用，不删除旧目录来绕过保护。父/子进程必须同为v2。新独占目录：`exp/initialization_kd_interaction/sports_init_kd_eval_parity_seed2022_v2/`，准备时不存在。

共同身份沿用已提交的[CPU资产预检锚](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)、[smoke结果锚](INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md)及[smoke环境/资源配置](INITIALIZATION_KD_RESOURCE_SMOKE_CONFIG_2026-09-27.json)。原四臂配置SHA256 `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d` 不变；v1共同环境/参数不变。输入仅：

- smoke `sports_init_kd_resource_smoke_seed2022_v1/T1/final.pt` 的两张ID表，SHA256 `ec18349231ded3a4b1dcd400bff31b80a2e23b1c819519ad6c6fd0eb64ddbcc7`，来源提交 `42336f5355dd76f4f17f91009deb8a4477ab2e44`、T1/epoch1/8步。不得将此状态作为正式四臂初值。
- Sports Train/Val，SHA256分别 `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8` / `1b4224c3fb091ad23a59e0f8a7b58c7b0c959863a7189cfc8b89eeb90a467e03`；运行时校验身份。准备阶段不重新读这些资产内容。无缓存/tape/教师内容读取，不创建优化器。

## 固定工作与验收

seed2022，先reference再fast，各完整遍历35598个Val用户；batch256，K10/20/40/50，Train排除、候选集、精度、GEMM及指标求和顺序不变。逐用户全部float32分数行字节SHA、有序Top50完全相同，完整指标逐字段完全相同。保存两份35598×50 int32排名矩阵，父进程复核SHA、shape、数组相等及排名digest。训练步数0、Test读取/排名0、教师forward0；Test打开拒绝守卫先于资产读取安装。

环境：Python3.10.20/torch2.11.0+cu128/numpy2.2.6/scipy1.15.3/psutil7.2.2，指定run_5060解释器；cuda0、float32、TF32关闭、deterministic algorithms开启、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=2022、CPU线程4。

验收须同时满足：exit0、supervisor completed、report completed、acceptance及保存数组验证通过、输入/source/config一致、35598用户全部精确等价、指标和计时有限、零训练/Test、未触发资源限制。无加速或更慢仍可为有效完成，不以速度低为由重试。

本次不是四臂科学比较，不采用统计显著性或实际效应阈值作为工程验收；四臂候选+0.002交互/.001非劣及24小时预算不改变。固定reference→fast顺序、无重复/ABBA，计时含分数哈希/核验，可能受缓存影响，不作为稳健加速或全训练速度结论。

## 资源上限、输出与停止

父进程总预算900秒（15分钟，约1秒轮询，含输入检查/评价/保存/验收）；CUDA allocator2GiB、worker RSS4GiB、输出256MiB、可用盘至少4GiB，一次attempt。CUDA指标不含驱动/桌面，RSS为采样值；上限不是完成耗时承诺。此前完整reference Val约122秒仅作参考，v2真实耗时尚未测得。

输出：launch_manifest.json、worker_claim.json、worker.log、telemetry.json、reference_top50.npy、fast_top50.npy、report.json、acceptance.json、supervisor.json；失败尽可能写failure.json，保留全部临时和部分产物。无新checkpoint。

身份、环境、源码、有限值、排名/指标、完整产物或资源任一门失败即停止；不扩大预算、改参数、清目录或重试。Windows监控文件发布只允许同一临时字节最多10次额外替换（10×50ms等待），不会重算评价或重启worker；持续文件占用仍将失败。真实I/O修复效果须本次运行验证。

## 准备检查与记录更新目标

27项合成CPU测试通过：原精确排名/指标/损坏拒绝/资源与绑定检查、Windows读取占用和永久失败边界、v1→v2只有声明差异、父进程传递v2、worker使用v2、目录复用拒绝。未加载真实模型/分割或运行GPU/训练/排名。提交后只读检查源码/环境门；实际输入字节由手动运行时核验。

- outcome：新增 `docs/research/INITIALIZATION_KD_EVAL_PARITY_V2_AUDIT.md`，范围本次seed2022固定smoke状态的工程核验；不覆盖v1审计。
- TRAINING_LOG：本次v2 pending另行追加completed/failed，更新短当前交接。
- 实验族：`docs/experiments/README.md` 初始化×KD行；root README随状态更新。docs总览无具体parity状态，当前无需改动。
- 活跃四臂/旧18格矩阵、论文/缺口：不适用，无科学结果或新结论；仅在未来确有触发时另行更新。
- 六项生成导航：准备和结果审计后分别刷新，发现新文件/运行元数据。
- v1/v2原始产物、smoke检查点、所有历史资产及check/原位保留，不提交大资产/原始排名。非稳定里程碑，无tag/bundle/merge/push。

唯一下一步：用户执行上面带`--attempt v2`的命令一次，将完成或失败状态交回审计。禁止重跑smoke/v1或启动四臂、P1b/P2、门控。
