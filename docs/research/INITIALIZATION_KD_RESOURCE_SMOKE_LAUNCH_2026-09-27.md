# 初始化 × KD：用户手动资源 smoke 声明

状态：准备与合成检查通过，**尚未执行**。这是一项独立、有界的运行与资源检查，不是正式四臂实验，也不产生可填入四臂结果矩阵的指标。本声明只对应 `sports_init_kd_resource_smoke_seed2022_v1` 一次尝试。

## 唯一命令

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_resource_smoke.py'
```

完整路径可直接复制。由用户执行；AI 本次只完成准备、合成 CPU 验证和提交，没有执行该命令。不要使用四臂入口、旧 P0/P1a 入口或只读预检脚本代替。已存在输出目录时立即拒绝；失败后不得删目录、重试、恢复或改 seed。

## 精确工作量与共同身份

实现基线：`5ebf5dfd257ba103b8b981de7ed999e3e281f3d3`；启动源为包含本声明与实现的干净已提交 HEAD，分支 `codex/experiment/baby-teacher-baseline`。父进程 manifest 记录完整实际 HEAD，worker 开始/结束核对干净源与同一 HEAD。仅容许用户原有未跟踪 `check/`，不得把任何已跟踪修改豁免。不合并 main、不打 tag、不推送。

固定配置见 [smoke JSON](INITIALIZATION_KD_RESOURCE_SMOKE_CONFIG_2026-09-27.json)。引用 [四臂候选配置](INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json)，其 SHA256 必须为 `15dced0bc76461a3868a69c1820fe31b809d6cb89ef473d1234852a82c1b0d0d`；后者仍是不可启动状态。共同资产锚为 [已通过 CPU 预检](INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json)：共享六张缓存表、随机初值、完整 tape、Train/Val 与教师来源 checkpoint 六项 SHA 均固定。运行时再次校验实际使用资产；教师 checkpoint 只计算来源哈希，不实例化教师、不前向。原始资产只读，全部留在原位。

与四臂候选的明确差异：**只运行 T1，只有一次 8 批次的短程更新和一次完整 Validation**。T1 从共同缓存 `users/items` 初始化，seed2022；取同一 tape 的第1轮前8批，每批1024。新建空状态 AdamW，lr6e-5、weight_decay .01、betas(.9,.999)、eps1e-8、amsgrad false。BPR 无额外显式 L2；Full 为 `.3*(image+.3*text)/1.3`，图文方向损失沿用原实现的均值归约，仅正物品，用户KD为0。维度35598用户×18357物品×64。加载旧随机初值只做共用资产结构检查，不训练 R0/R1。

epoch0 **不排名**；8次更新后只对35598个有 Val 正例的用户做一次完整 Val 排名，候选仅排除 Train，K10/20/40/50。主用途是测量 Validation 开销与完整保存/验收链条；保存的 Recall/NDCG 只是功能检查数值，不比较历史结果、不选新门槛、不估计四臂交互。Test 文件打开、Test 排名、teacher forward 均为0。8步产物不作为后续正式实验初值或恢复点；正式实验仍须回到既有固定初值和全新 AdamW。

环境锁定：Python3.10.20，指定run_5060解释器，torch2.11.0+cu128，numpy2.2.6，scipy1.15.3，psutil7.2.2。smoke 数值设置明确为 cuda:0、float32、TF32关闭、deterministic algorithms开启、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=2022、torch CPU线程4。设备名称运行时记录，不伪称已实测硬件表现。以上只作用新隔离进程，旧正式配方不变；四臂正式数值设置尚不因此自动获准。

## 资源上限与停止规则

| 项目 | 上限 / 执行方式 |
|---|---|
| 墙钟 | 父进程从保留新attempt开始计时，900秒；轮询约1秒，含worker预检、8步、一次Val、保存及验收。调度/终止可能带来约一个轮询间隔的超时延迟 |
| CUDA | PyTorch allocator 2GiB；在任何模型转入GPU前设memory fraction，worker记录峰值，末验收再检查。不是整个驱动/桌面显存上限 |
| 内存 | worker RSS 4GiB；父进程轮询。不是可用RAM保证，也不是整机总内存上限 |
| 输出 | 256MiB，包含checkpoint/temp/log/JSON，轮询；低于4GiB可用磁盘即停 |
| 尝试 | 1；独占目录、父进程pipe token/manifest/PID绑定、worker独占claim；不可直接绕过父进程调用worker |

实际耗时尚未测定，不能承诺几分钟。8批训练很短，完整 Val 的Python排名和设备状态可能主导耗时；15分钟是停止预算，不是预计耗时。不会以 smoke 的8步速度承诺300轮×4臂所需时间。

SHA、环境、配置、源、attempt身份、Tensor/指标有限性、保存状态或验收不匹配，CUDA OOM、超时或资源超限均停止且保留部分产物。不要自动增加预算。GPU接口不支持确定性操作也按失败记录，不静默退回其他设置。低 Recall 本身不是失败条件。

## 输出与验收

独立原始目录：`exp/initialization_kd_interaction/sports_init_kd_resource_smoke_seed2022_v1/`。

- `launch_manifest.json`：实际源HEAD、配置/资产、parent身份；`worker_claim.json`：唯一worker。
- `worker.log`：stdout/stderr；`telemetry.json`：最新资源采样。终端每约30秒显示仍在运行。
- `T1/curve.json`、`T1/best.pt`、`T1/final.pt`：仅1个短程epoch、8次AdamW更新；两份checkpoint为同轮不同标签，含完整模型与优化器状态，无正式“最好结果”含义。
- `report.json`：T1状态、恰好1次Val、8步、计时、CUDA峰值及采样RSS峰值；无四臂interaction。
- `acceptance.json`：worker已验模型两表、参数、有限性、AdamW两组moments/step、最早最优及末轮一致性，并绑定文件SHA。
- `supervisor.json`：父进程独立核对完整预期文件集/SHA及资源，只有全部通过且exit0才标completed。缺文件或仅有exit0不算通过。异常时尽可能保存`failure.json`/failed supervisor，硬中断可能只有部分文件，应保留全部。

终端最终应显示 `Completed; artifact acceptance passed. No next stage will start.`。最终验收仍需审计上述文件：来源HEAD一致、Test0、T1/8步/1次Val、checkpoint/curve/header一致、资源未超限。**成功或失败都到此停止**，用户返回“跑完了”后只审计本次产物，不直接重试或进入四臂训练。

本次16项CPU合成测试通过，包括微型数据贯通相同worker的更新/Val/保存/验收（GPU接口模拟），父进程真实超时杀停与部分文件保留、exit0缺产物拒绝、checkpoint损坏/错步数拒绝、parent/token/source/config错配拒绝、重复claim拒绝，以及原有四臂配对/损失/AdamW/Test拒绝。它们验证逻辑，**真实GPU表现由这一次手动smoke检验**。

## 记录更新目标

- 身份：本页 smoke seed2022 T1，非正式结果；未来 outcome 审计 `docs/research/INITIALIZATION_KD_RESOURCE_SMOKE_AUDIT.md`。
- `TRAINING_LOG.md`：本声明pending的单独completed/failed结果，更新当前交接；`docs/experiments/README.md` 初始化×KD实验族；当前入口变化时更新根/docs README。
- 四臂新矩阵与旧18格：均不适用，资源smoke不填实验性能格子。
- 论文/缺口：本次无科学主张变化，不更新；不据短程Recall调候选delta_I=.002/epsilon=.001。
- 审计后刷新六项生成导航；原始产物保留新目录，不提交checkpoint/raw logs，不删除历史资产/check/。非正式稳定里程碑，不打tag/bundle。

唯一下一步：用户手动执行本页唯一命令一次，保留完成或失败状态，返回后审计；无自动执行或后续阶段授权。
