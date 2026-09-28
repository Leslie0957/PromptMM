# Sports 初始化 × KD 四臂 seed2023 完成审计

2026-09-28。用户手动运行已声明的 `sports_init_kd_interaction_seed2023_v1` 一次；本次仅只读核查现有产物，无重训、重排、真实分割加载、GPU或 Test 访问。原始文件原位保存在 `exp/initialization_kd_interaction/sports_init_kd_interaction_seed2023_v1/`。`supervisor.json` 为 `completed`/exit0，`acceptance.json` 为 `passed`；独立 CPU 复核 13 个被验收文件的 SHA、四臂曲线和八个完整模型+AdamW 检查点均通过。没有 `failure.json`。

## 来源、命令和协议

[一次手动运行声明](INITIALIZATION_KD_SEED2023_COHORT_LAUNCH_2026-09-28.md)是执行依据；历史命令已消耗，**不得重跑**：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_initialization_kd_cohort_seed2023.py'
```

manifest、report、acceptance 一致记录干净启动源码 `32c89f119275f3d9001466721452f25e4a35e41e`，分支 `codex/experiment/baby-teacher-baseline`，解析配置与提交中的固定 seed2023 spec 逐字段相同，digest `da3c34aea7f2569cfa99173ebd04fc35ad7a474cee51058f1c9b41f55006dce8`。运行身份 seed2023、run_5060/Python3.10.20/torch2.11.0+cu128、RTX5060、`exact_topk_v1`。本次 manifest 记录启动时资产身份：随机初值 SHA `bb5429aa8a1edfe53b36fed977146460d1101ae79539d2badf856d5514ef9d85`，只读回放 SHA `3763128ba003228d60c09924b9370b4cab0b8c16679366742835fc4559d89085`，共同教师缓存 SHA `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`，Train SHA `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`，Val SHA `1b4224c3fb091ad23a59e0f8a7b58c7b0c959863a7189cfc8b89eeb90a467e03`，教师 checkpoint 来源 SHA `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。本次审计不重新哈希六大输入；来源门和运行时加载门由上述固定源码及 manifest 支持。seed2023 初值/回放与 seed2022 不同且在[预检](INITIALIZATION_KD_SEED2023_PREFLIGHT_2026-09-28.md)中核验，共同缓存保持相同。固定教师不是第二个独立教师样本。

R0/R1 同一 seed2023 随机用户/物品 ID 初值，T0/T1 同一教师最终两表；四臂串行 R0→R1→T0→T1、同一不可写回放，各 300 epoch×214 batch×1024，64200 次更新。R0/T0 为 BPR；R1/T1 加固定图文物品方向 KD，alpha0.3、text rate0.3，无 user KD。每臂新建空状态 AdamW，lr6e-5、weight decay0.01、betas(.9,.999)、eps1e-8；无显式 embedding L2。与 seed2022 比，只有 seed、随机初值、回放和运行身份/输出/命令变化。严格初值逐位相同及回放复用由冻结源码的运行时检查支持，epoch0 的两组指标分别完全相同；本次目录没有另存训练开始时两表，不能单靠最终 checkpoint 倒推出初值。

## 验收与 Validation 结果

各臂曲线为连续 epoch1–300，报告各有 epoch0 描述性 Validation；共 `1204` 次完整 Validation，所有 35598 用户，Test 文件读取计数 `0`。CPU 验收函数逐一反序列化 best/final，检查两表形状、float32/有限值、优化器两个参数状态和有限 moments、参数/步数、epoch/指标/源码/config 身份；独立复核报告与曲线、最早最大 Recall20 选择、末轮交互和文件 SHA。运行时每步拒绝非有限目标，但没有逐步 loss 曲线；本审计不声称已验证优化轨迹或收敛。

| 臂 | seed | epoch0 Recall20 | epoch300 Recall20 | epoch300 NDCG20 | 最早最佳 epoch | 最佳 Recall20 | AdamW 步数 | 状态 | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| R0 | 2023 | 0.0011601775380639362 | 0.0449533652985852 | 0.021515801952744565 | 296 | 0.04519354733128367 | 64200 | 完成 | 未读取 |
| R1 | 2023 | 0.0011601775380639362 | 0.08170139654698551 | 0.03707313980131608 | 292 | 0.08205019891415968 | 64200 | 完成 | 未读取 |
| T0 | 2023 | 0.09418449519084773 | 0.0943431785304682 | 0.04348153376223026 | 290 | 0.09451172732534432 | 64200 | 完成 | 未读取 |
| T1 | 2023 | 0.09418449519084773 | 0.09451406828082871 | 0.0434728568165053 | 293 | 0.09464047987698579 | 64200 | 完成 | 未读取 |

固定主终点 epoch300：delta_R=R1−R0=`+0.03674803124840031`，delta_T=T1−T0=`+0.00017088975036051723`，I=delta_R−delta_T=`+0.03657714149803979`，逐项重算与 report 一致。预定描述性 `I>=0.002` 候选门通过，暖状态 `T0−T1=-0.00017088975036051723<=0.001` 代价门通过。最佳轮分别如表，不能把各臂最佳轮代替末轮主终点。事后最后20轮均值 R0/R1/T0/T1 分别为 0.04493604/0.08172758/0.09435348/0.09448629，仅作曲线敏感性观察，连续轮次不是独立重复。

与[seed2022审计](INITIALIZATION_KD_INTERACTION_AUDIT.md)并看，两次末轮随机初值 KD 增量为 `+0.03648319`、`+0.03674803`，教师初值增量为 `-0.00009130`、`+0.00017089`；交互分别 `+0.03657449`、`+0.03657714`。这削弱“只有 seed2022 随机初值或回放偶然导致大不对称”的解释；**暖状态 KD 正负号不稳定**，不应说它有害。两次教师起点均为约0.09418，远高于随机起点约0.0012；差分描述完整初始化处理与此 KD 配方在固定300轮下的交互，不能分离初值质量、用户表知识、表示尺度和优化轨迹，也不能证明语义冗余、统计显著、等价/非劣、长期最终性能或第二算法。两个 seed 共用教师缓存，不能当作教师不确定性的重复样本。结果仅 Validation，未填旧主 Test 矩阵。

## 资源、Test chronology 与保留

父进程 `8279.407` 秒（约2小时18分），worker `8268.516` 秒；R0/R1/T0/T1 各 `2394.313/2812.515/1537.266/1524.406` 秒。报告 CUDA allocator 峰值 `161480704` 字节、采样 RSS 峰值 `2070241280` 字节；末次输出 `332161781` 字节，可用盘 `411206017024` 字节，均在 2GiB/8GiB/1GiB/至少4GiB门内。没有墙钟硬停止，父进程自然完成。采样 RSS 和 CUDA allocator 并非操作系统/GPU 总占用的独立监测。

Test chronology：预定 Test0；冻结源码在加载分割前安装 Test 文件打开拒绝；report 为 `test_file_reads=0`，selection_split=`Validation`，没有教师 Test、学生 Test、Test 排名或被测 checkpoint。没有独立 OS 文件访问轨迹；不存在 selected-versus-tested checkpoint 相等性或 Test 指标。本次没有重试/恢复/下一 seed；低科学效用不能使已经通过硬验收的运行变成失败。原 seed2022 产物、历史审计、数据、manifest、`check/` 均保持原位。

## 原始文件指纹与记录路由

以下为审计时原位 SHA256；acceptance 内 13 个验收项的指纹均与独立计算相同。忽略的原始产物不会随本次 Git 提交保存。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `acceptance.json` | 1348 | `ab7dde3a016bb17c88b1b521c019d4b925a8efd46c38eedb36b263e624dfa7ce` |
| `launch_manifest.json` | 7504 | `58018597532c414adb102279b1ae0ca0c3dca3803c13d3d92ba9abdaf8df7a3d` |
| `R0/best.pt` | 41441421 | `9bacbfd317f60cd494ba9f3026a77f6affb8f6c964e259ee3ce39c1b96578d7e` |
| `R0/curve.json` | 140211 | `d319b7f079f8df0cdb831666c7a78bda575a1066cce8a149286273038a43e734` |
| `R0/final.pt` | 41441755 | `05815325085ce48a9c5ae0c3bc2eeebcae83e9188bc47d4c25466bd5b37d183e` |
| `R1/best.pt` | 41441421 | `3fba8b1b83da886c6e12018e1358693d09332ed32868b38f95f7036603da8a99` |
| `R1/curve.json` | 139249 | `312751907cafa831132382e771a1cfd5eef74a51d4746a78d1c731fb8593379e` |
| `R1/final.pt` | 41441755 | `d6603a9a1b98719c1a3c4b3a170b626383a00c2f23788841efdce7b215284649` |
| `report.json` | 62736 | `bc8e95e5cca6189d663d40fdf7a7fdd317a031514ad730421308841f625a723d` |
| `supervisor.json` | 127 | `7384f42fd1e6384ce8e109c8aaae09103bf6bfc5ef87fee4f9493b63f9d0079c` |
| `T0/best.pt` | 41441421 | `462509a1a25faf538b57fb9cc6785b64d874b4cd9ffb74c42e147bb50aac5bfb` |
| `T0/curve.json` | 138835 | `b399c1f23e3c20831d6a28e8cbf25e1fdf8abb537f6184859492ae10c84927ad` |
| `T0/final.pt` | 41441755 | `13255e543e6d1a643745d11a20305313f45c8a9fc6ea4e8ae20eb5c5879711db` |
| `T1/best.pt` | 41441421 | `f9386018c5e98035a604e971ca05cf66977bd33965ac24ca6e7f1f2994ea66f5` |
| `T1/curve.json` | 138785 | `391d613e87b079f1d86e5da26506a021b45010a743dd3c120c0e8f0d36790654` |
| `T1/final.pt` | 41441755 | `36721f61be22373da59cf43618cfd29522c74b19ca291caf116704d569646629` |
| `telemetry.json` | 189 | `9e6de20acb4af3e9e335285d700bea8572510446311d93e40ddf886b2bb944e4` |
| `worker.log` | 184 | `5dfc47265e4ccee5a17596a297d65101af5da89314006a63390078096fb9b5a0` |
| `worker_claim.json` | 35 | `d60a92dd01930a4f4b133d87b53e67cffe0009bd53462e3f97513f50ba0851dc` |

记录路由：本审计、TRAINING_LOG outcome/current、独立矩阵 seed2023 四格、实验族导航和 root 短入口更新；六项生成目录刷新。`docs/README.md` 无独立状态入口，未改；旧18格/Test矩阵、立项历史计划、论文正文和缺口表不因第二个 Validation seed 自动改变，未改。无 tag/bundle/push/merge，原始资产及 `check/` 不提交。

唯一下一步：只读综合两 seed 的曲线、起点质量与实际成本，界定第一项工作可支持的机制边界；不自动启动 seed2024、Test、P1b/P2 或门控开发。
