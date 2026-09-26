# Sports 缓存部署效率基准：实现与手动运行

状态：实现准备完成；**真实数据预检、GPU基准均未由助手运行**。依据[效率协议](SPORTS_DEPLOYMENT_EFFICIENCY_PROTOCOL_2026-09-26.md)，一个入口串行执行，失败即停，既有输出不覆盖。

2026-09-26 修订：用户首次 v1 执行在环境信息收集阶段因 Windows NUL 被文件守卫误拦而失败，完成计时条件0/36；CPU预检资产保留。已用独立回归复现并修复，仅放行解析后的平台空设备。当前入口指向全新 v2 目录；是否重新执行由用户决定，助手未重跑。v1 目录不删除、不覆盖，工作负载及教师输出容差不变。

## 唯一手动入口

接通电源，保持当前性能模式，关闭其他训练/GPU负载，终端保持前台可见，不让其他应用全屏。`--conditions-confirmed` 表示你已安排这些条件，不会自动更改系统设置。

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_cached_deployment.py --conditions-confirmed
```

这是同一批次内的重复计时，**没有新训练**。共1个CPU预检进程和36个隔离GPU工作进程：3轮×（3个离线生成条件+3个批大小×3个在线臂）。进程按协议轮换臂和批大小顺序，避免总把某个臂放在最前。

仅查看计划、不打开数据或创建输出：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_cached_deployment.py --dry-run
```

## 固定实现与计时细节

- `codes/cached_deployment_benchmark.py`：原始资产校验、共同表格式、train-only请求、共同GPU scorer、存储去重、教师适配器、离线/在线测量与文件访问守卫。
- `tools/run_sports_cached_deployment.py`：独占结果目录、clean HEAD校验、CPU预检、36个串行子进程、硬超时、原始计时、GPU遥测及轮级汇总。内部 worker 参数只供启动器使用，不是重跑入口。
- `tests/test_cached_deployment_benchmark.py`：CPU合成验证。未对真实数据做预跑；第一次真实教师输出验证就是手动批次里的硬门槛。

预检验证固定manifest、teacher/shared/source/data/modal/cache/学生checkpoint指纹，并核对学生推理表与所选完整checkpoint相等。三臂导出相同格式的双表到本次结果目录。在线worker只加载对应双表、train矩阵与公共请求，不加载教师模型或学生优化器。preflight记录这批导出资产的哈希，完成时再核对源资产与导出资产未变。

线程固定为1（PyTorch intra/inter-op、OMP/MKL/OpenBLAS），CUDA设备0，FP32，无TF32/autocast/compile；报告实际环境。每个在线条件20次预热+100次服务墙钟计时，另100次独立纯GEMM CUDA事件计时。后者不插入服务墙钟计时。排除mask、topk及CPU返回包含在服务时间内；结果摘要、报告落盘和正确性检查在计时之外。因此吞吐是已定义服务路径的总用户数/总计时时间，不包括测试工具写报告及请求间停顿。

每个离线条件有1次输出正确性检查、3次预热、10次生成计时、1次最终导出。教师每轮实际前向15次，共45次；学生仅复制冻结ID表。权重、prompt缓冲、原始模态张量和图在计时前后检查未改；逐次输出检查有限。教师参考比较固定 `atol=1e-4, rtol=1e-4`，失败停止，不调整容差。这里是冻结权重生成成本，不含训练或吸收新数据的成本。

教师源构造器内部直接调用 `.cuda()`，因此模块构造、模态传输和权重恢复合并报告为 `construction_modal_transfer_restore_seconds`，不伪装成纯CPU构造或纯传输。源文件读取连同完整性哈希报告为 `read_and_hash_seconds`，不能当作纯磁盘I/O。图CPU归一化、图传输、模型导入及最终GPU→CPU拷贝/序列化分别记录。公共导出格式的加载成本另列；不同阶段不合成无定义的加速倍数。

## 原始教师路径对应关系

| 主训练源码行为 | 独立适配器 |
|---|---|
| `Trainer` 读取 image/text .npy 和 train_mat | 仅加载这三类固定输入；无 Data、batch_test 或 main_mmlight 导入 |
| `csr_norm(mean_flag=True)` | 逐行 `(degree+1e-8)^(-1/2)` 左乘；合成矩阵与原方法逐位比较 |
| `matrix_to_tensor` | float32 coalesced sparse COO，保持形状与归一化数值 |
| `Teacher_Model` + `PromptLearner` | 直接使用原始类与forward；不复写简化教师 |
| parser全局args | 隔离进程里使用已固定manifest的完整namespace；避免解析训练入口CLI |
| hard-token缓存 | 只读固定缓存；禁止重新PCA/ICA或写回；随后原checkpoint严格恢复prompt缓冲 |
| 教师checkpoint加载 | 原teacher/prompt state严格加载，inference config与manifest及实际参数逐字段核对 |
| eval/no_grad的最终表生成 | eval、requires_grad=false、inference_mode；与固定process0表做容差比较 |

原始 `Models_mmlight.py` SHA固定为 `ec993272ccf07d24a70b41bb9fb87273d95e01d2de6eb1dfb1950c10172933dc`。独立适配器不构造原Trainer的优化器、DGL训练采样对象或评价器；它们不进入此教师forward。CPU合成fixture验证了完整适配器恢复出的两张最终表与原始类的预保存输出逐位相等；真实GPU数据一致性仍须运行时验证。

## 输出、失败及结果边界

当前新尝试独占路径：`exp/efficiency/sports_cached_deployment_seed2022_v2/`。失败的 `sports_cached_deployment_seed2022_v1/` 原样保留。

- `batch.json`：启动commit/分支、源文件指纹、全部条件、每个已完成worker报告及哈希、最终轮级汇总。
- `preflight.json`、`request_order.npy`、`T/F/B_tables.pt`：固定输入校验及共同部署表。
- 每个 `round*_offline/online_*` 目录内 `report.json`：原始wall/GEMM耗时、环境、内存、输出摘要、正确性；离线另有本轮 `tables.pt`。
- 每个worker有独立 `*_console.txt`；`telemetry.jsonl`每约2秒记录nvidia-smi温度/频率/功率/负载等。不可取得的遥测显式记缺失。

CPU预检上限600秒，每个worker上限1800秒。超时/异常/一致性失败均保留已生成内容、标记父报告failed并停止后续条件，不自动重试或恢复。输出目录已存在则拒绝启动。若失败，把batch.json及对应console交回审计，不删除目录后重跑。

只要代码及硬门槛通过，慢、无优势或相近的结果都有效。父报告completed也不自动证明环境无干扰：程序记录了启动条件确认，但无法保证整个桌面过程一直不受干扰。运行完告知是否有全屏切换、休眠或其他GPU负载；审计结合全部读数和遥测解释，不挑选最快值。

不访问Validation/Test矩阵，不计算质量指标，无参数更新。三臂逻辑表体积应相同；Full/BPR速度差不能直接归因于蒸馏。忽略的输出不受Git保护，源提交也不等于资产备份。

唯一后续步骤：用户手动运行本批次一次，完成后只读审计结果；不启动其他训练或评估。
