# 曝光匹配加分对照：启动准备

2026-09-30，固定[方案](EXPOSURE_MATCHED_BONUS_PLAN_2026-09-30.md)实现完成。没有真实数据评价、训练或Test访问；不含算法有效性结果。

## 唯一命令

用户决定执行时，从仓库根目录运行一次：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_exposure_matched_bonus.py' --formal
```

这是固定seed2022/2023/2024完整诊断，不是三seed训练。每seed只计算一次B全候选分数，评价B与选定Q；R只复用已有Top20作为参照。没有best、额外系数、Test或后续阶段。输出`exp/innovation2/exposure_matched_bonus_v1/`，目录存在即拒绝，失败不覆盖/续跑/重试。

分支`codex/experiment/baby-teacher-baseline`，必须从干净committed HEAD启动，原有未跟踪archive/reviews/例外；manifest保存实际来源提交、profile SHA及命令。准备提交身份见本次交接。AI代执行须先记录该一次run pending并提交；用户手动执行仅授权本固定诊断，运行后追加实际记录和审计。

## 实现与冻结身份

- `tools/exposure_bonus_core.py`：纯预测候选合并、最多48次二分、曝光单调检查、确定性端点选择和三类决策；不接收标签。
- `tools/run_exposure_matched_bonus.py`：隔离加载、CUDA FP32评分、缓存、阶段读保护、封存后评价和父进程资源监控；不创建优化器或反向传播，校准前后ID表逐位一致。
- [profile](EXPOSURE_MATCHED_BONUS_PROFILE_V1.json)：原共享残差manifest/completeness、三份B final、B/R status/Top20，以及Train/Validation/两图的SHA绑定。checkpoint的metric字段可能随对象反序列化进入内存，但不读取或用于校准；旧status仅封存后用于回归。
- 准备只读取JSON元数据，不载入真实checkpoint、图张量或交互标签。运行时校验输入SHA；Validation甚至哈希读取也在全部β封存后才允许。

保持6114目标与原低组6119区别。每用户两组各20候选，FP32分数转float64作跨组分差比较，固定组内顺序及ID并列规则。β=0必须回归原B Top20。三seed的候选缓存、B/Q/R列表、β及轨迹全部写入，calibration_seal记录SHA和时间，再开放Validation。后续回验封存文件不变与读取时间顺序。

数据目录只允许Train；封存后加入Validation白名单，其余路径拒绝并计数。Python hook不是OS级取证。程序只有显式`--formal`才执行，`--help`只显示说明。

## 已做验证与剩余风险

6项合成测试全部通过：随机并列/负分/大β的缓存与全候选参考一致；极小差值与不足20的分组；精确/不可达校准；台阶跳变相同误差取小β；封存前Val及始终Test拒绝（虚构open事件，不实际读取）；决策阈值边界。另检查语法、CLI、绑定路径存在、输出目录不存在和历史保留。

单seed候选数组约11.4MB（35598×40×8 bytes），三个seed约34.2MB未压缩；不会保存35598×18357全量分数。batch256评分矩阵约18.8MB，以上不是完整进程显存或RSS峰值。完整运行仍未测，合成测试不能证明1800秒上限或实际Top20回归一定成功。

固定上限：1800秒、worker RSS4GiB、CUDA allocator2GiB、输出256MiB、磁盘余量2GiB。父进程每秒采样并超限终止，worker每批/每轮搜索检查；瞬时峰值可能漏测，父进程RSS未计，allocator不是整卡占用。硬失败保留产物，无自动重试；曝光匹配不到容忍范围是有效的unresolved结果，不判代码失败。

## 收尾目标

运行原始产物含manifest、每seed候选/列表/两份配对明细、calibration及seal、report、acceptance、worker_resources、supervisor、console、completeness。report保存九个指标状态（B/Q六个新评分状态加三个已有R列表参照），不是九次模型评分或训练。

独立审计核对身份、B逐位回归、R指标回归、校准轨迹/封存先后/曝光误差、配对恒等和Test0/资源。按RUN_CLOSEOUT创建本目录`EXPOSURE_MATCHED_BONUS_RESULTS.md`、`EXPOSURE_MATCHED_BONUS_AUDIT.md`、`EXPOSURE_MATCHED_BONUS_HANDOFF.md`，更新TRAINING_LOG、实验族及当前入口/生成导航。原矩阵/历史审计/第一项/THESIS_ROADMAP均保持；没有新结果时不创建空结果文件。

**唯一下一步：用户运行上方命令一次，将输出目录交回审计。** 结果再好也不直接等于语义共享成功，结果不好也不自动搜索更大β或新配方。
