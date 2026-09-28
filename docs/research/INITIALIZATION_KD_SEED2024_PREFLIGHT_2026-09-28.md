# Sports 初始化 × KD：seed2024 只读资产预检

2026-09-28。基于用户要求增加一次独立随机种子，执行一次 CPU 只读检查；机器可读结果见[预检 JSON](INITIALIZATION_KD_SEED2024_PREFLIGHT_2026-09-28.json)。历史 `exp/paired_coldinit/sports_three_seed_v1/batch.json` 中 seed2024 的 full/image_matched 两项引用相同的初值和回放；当前文件逐字节 SHA 与两项来源记录一致，且不同于 seed2022、seed2023。

| 资产 | seed2024 SHA256 | 状态 |
|---|---|---|
| 随机用户/物品 ID 两表 | `e2c6885debdec09a8ff8401fe4f4f96ea5f46dcc1fd90f24dcc4ecdbd2188e37` | 35598×64 / 18357×64，float32、有限 |
| 不可写三元组回放 | `131cc14ebe6853e79216183f06923f3689d47d26c69eac2521d7f93b1b314bc7` | int32 `[300,214,3,1024]` |

先安装 Test 文件打开拒绝，再读取固定共同缓存及 Train/Val；共同缓存、教师 checkpoint 来源和 Train/Val 身份沿用[seed2023预检锚](INITIALIZATION_KD_SEED2023_PREFLIGHT_2026-09-28.json)，本次实际 loader 复核缓存、初值、回放、Train/Val SHA，没有加载教师 checkpoint 或 Test。全部 65,740,800 个三元组的正例不在 Train=0、负例在 Train=0；Train/Val 交集0，35598 个 Val 用户，最少候选18120。运行 16.704 秒，未实例化模型、排名、训练、分配 CUDA 或读取 Test。seed2024 四臂新输出目录不存在。此预检只证明资产与协议形状可复用，不能借旧 full/image_matched 结果填入新四臂，也不能推断第三种子结果。

只读复核前两次保存曲线：epoch300 的随机组 KD 增量为 `+0.03648319`、`+0.03674803`，教师组为 `−0.00009130`、`+0.00017089`；后50轮端点差 R0/R1 在 seed2022 为 `+.00097924/+.00096120`，seed2023 为 `+.00018751/+.00030967`。两次均显示随机组较大收益和暖组近零增量，不能证明随机组已收敛；固定教师初始 Recall20 均约 `.09418`，随机初始分别约 `.00122/.00116`。固定起点质量和优化预算仍是主要解释缺口。第三次重复仅检验随机初值/回放的描述性稳健性，不独立重复教师，不保证显著性或机制归因。

原始初值、tape、batch manifest、缓存、分割、旧结果和 `check/` 原位保留；没有运行正式四臂。下一步仅从干净提交由用户手动执行[独立 seed2024 声明](INITIALIZATION_KD_SEED2024_COHORT_LAUNCH_2026-09-28.md)一次，之后审计；不自动重跑或进入 Test/新算法。
