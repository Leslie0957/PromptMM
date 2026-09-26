# Sports 缓存部署效率审计（2026-09-26）

状态：**v2 有效完成，36/36 条件通过只读复核**。用户执行，助手仅读取已有报告并复算统计与文件指纹；本次没有新模型前向、计时、训练、梯度或质量评估。失败的 v1 原样保留。

## 1. 结论

本次没有观察到学生相对缓存教师的一致在线速度优势。三臂在线表体积和同批大小的 GPU allocated 内存读数相同。学生在冻结权重的离线表征生成阶段省去了教师多模态与图传播前向；这是此实现和此阶段的成本差异，不是训练、动态数据更新或整个推荐系统的加速比。

## 2. 执行与核验

- 手动命令：`& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_cached_deployment.py --conditions-confirmed`。源提交 `4fc77b3e54b345bdba550c445a2b7381838436ea`，分支 `codex/experiment/baby-teacher-baseline`，launch_dirty=false，执行前后源码及资产门槛通过。
- 开始 `2026-09-26T12:37:49.082961+08:00`，结束 `2026-09-26T12:40:25.911278+08:00`；墙钟共 **156.828317秒（约2分37秒）**，含预检、进程启动、校验与报告写入，不是服务延迟。
- 环境：RTX5060，驱动595.97，PyTorch2.11.0+cu128/CUDA12.8，Python3.10.20，Windows；单 CPU intra/inter-op 线程，FP32，TF32/autocast/compile关闭。每个worker环境字段已核对。
- 三臂为固定 epoch37 教师最终表T、暖启动seed2022所选Full表F和BPR表B。35,598用户、18,357物品、64维；全部候选排除训练已见物品，top20返回CPU。固定train-derived请求seed9026026，批大小1/128/1024。
- 3轮×3离线条件，每条件3预热+10计时；3轮×3批大小×3在线条件，每条件20预热+100墙钟计时，另100纯GEMM事件计时。总计90条离线、2700条在线墙钟及2700条GEMM原始耗时。真实教师前向45次均属于用户基准，无参数更新。
- 独立检查36个worker报告哈希、condition/launch身份、各样本有限正值及长度、所有逐worker统计、12组跨轮汇总、同批大小请求张量指纹、固定导出源和九份离线导出文件指纹。结果与batch汇总一致。所有输入表未变；三次教师state/模态/图未变门槛通过。
- 教师输出与固定参考的最大绝对误差不超过9.5367431640625e-7，满足事前atol=rtol=1e-4。个别接近零元素的单独相对误差可超过1e-4，但allclose使用绝对+相对联合容差，不能误判为失败。学生复制输出与参考逐位相同。
- 源代码指纹与当前文件匹配，preflight/request/common导出/九份离线导出/telemetry哈希重新检查；稳定大数据和教师checkpoint引用已验证锚及运行末尾完整性门槛，不为文档审计再加载模型或重复全量资产审计。

## 3. 缓存在线服务

单位是 **毫秒/整批请求**，不是毫秒/用户。点值为三个轮级中位数的中位数；方括号为三个轮级中位数的范围。墙钟包含查表、全物品matmul、训练mask、topk和CPU返回，不包含报告写入、输入预加载或网络传输。

| 批大小 | 缓存教师T | 学生Full F | 学生BPR B |
|---|---:|---:|---:|
| 1 | 0.340300 [0.338450, 0.342600] | 0.345300 [0.341500, 0.387600] | 0.340200 [0.337750, 0.344950] |
| 128 | 0.541800 [0.502050, 0.592950] | 0.514800 [0.507300, 0.570950] | 0.525400 [0.512050, 0.564950] |
| 1024 | 2.108950 [2.094850, 2.109400] | 2.112250 [2.104750, 2.130550] | 2.114050 [2.113000, 2.118600] |

Full相对教师的点估计在不同批大小上有快有慢；batch1024时约2.112250与2.108950ms，十分接近。batch128的Full其中一轮p95为1.708895ms，其他轮约0.635420/0.841505ms；不删除慢样本，也不将短时重复作为训练种子来宣布显著性或统计等价。

轮级吞吐中位数（用户/秒；总用户数/对应总计时，不用median延迟倒数代替）：

| 批大小 | T | F | B |
|---|---:|---:|---:|
| 1 | 2831.7 | 2796.2 | 2799.7 |
| 128 | 229737.3 | 227034.5 | 225252.2 |
| 1024 | 479137.2 | 477813.0 | 475834.7 |

## 4. 冻结权重的离线生成

单位毫秒；教师做原始Teacher_Model+PromptLearner完整前向，学生复制已有冻结ID表。输入已就绪，计时不含源文件读取、校验、模型构造、设备传输或写盘。

| 臂 | 轮级中位数的中位数 | 轮级中位数范围 |
|---|---:|---:|
| T | 15.889450 | 14.964550–19.981600 |
| F | 0.109900 | 0.108300–0.109950 |
| B | 0.112550 | 0.108600–0.170250 |

**教师/Full该阶段耗时之比约144.58**。此比值只比较一次冻结教师生成与一次学生表复制；缓存教师同样可以跳过重新生成，不能把它写成在线查询加速144.58倍，也没有测量吸收新交互后的再训练/适配成本。BPR学生结构相同，离线优势不能独归因于蒸馏损失。

另存了read+hash、CPU图归一化、导入、图传输、构造+模态传输+恢复、GPU到CPU及序列化时间；这些阶段混合了磁盘缓存、首次初始化和校验成本，不合并为算法速度比。三个T的输出GPU→CPU拷贝约10.48–13.51ms、序列化19.41–32.10ms；F分别2.28–2.34ms和9.61–9.96ms，说明实际导出总成本不只包含上表生成内核。

## 5. 体积与内存

- 三臂双表逻辑字节均为 **13,812,480 bytes（约13.1726 MiB）**；相同格式文件均为 **13,814,309 bytes**。维度和dtype相同，因此本对照没有缓存表压缩。
- 离线resident_storage按实际storage去重：原教师实现 **695,457,904 bytes**；学生双表 **13,812,480 bytes**。这比较的是运行完整教师及图/模态/prompt资产与学生表；原教师类还保留forward中未使用的image/text_embedding等参数，不是最小裁剪教师服务的内存下界。
- 离线报告peak allocated：T为1,649,867,776 bytes，F/B为44,221,440 bytes。峰值窗口还包含计时外有限性检查等工具操作，因此是该基准阶段峰值，不能冒充纯forward最小峰值。主机working set还包含Python/库及工具开销。

在线同一批大小三臂和三轮的allocated基线/峰值完全相同：

| 批大小 | GPU基线bytes | GPU峰值bytes |
|---|---:|---:|
| 1 | 22,516,736 | 22,591,488 |
| 128 | 24,029,184 | 33,726,976 |
| 1024 | 35,442,688 | 113,321,472 |

## 6. 解释边界与质量证据

74条GPU遥测均成功，温度40–60°C、SM频率532–2872MHz；这是包括空闲/加载/测量阶段的采样范围，不能据此诊断节流或认定窗口状态造成某个慢值。工具没有独立证明全程无干扰；本次仅给描述性结果，不再向用户追问前台状态。所有原始耗时保留。

结合[机制结论](SPORTS_INTEGRATED_MECHANISM_CONCLUSION_2026-09-26.md)：冷初值文本残差收益仍成立于其Sports Validation范围；暖启动蒸馏额外质量收益尚未观察到。本效率批次证明的是测得的阶段成本及同形状部署事实，不新增准确率、质量无损压缩或Test泛化证据。对外宜写“学生无需在生成其冻结ID表示时再次执行教师图/模态前向；与缓存教师相比，在线打分和表存储成本相近”。

所有36个worker均报告optimizer_steps/validation_accesses/test_accesses/quality_evaluations=0；本次基准连结构性Validation/Test读取也没有。只有基于train的无标签服务打分。无新选模或质量指标；selected-versus-tested N/A，既有paper_ready状态不变。Baby2023审计例外/originalmanifestfalse与Innovation2未定保持原状。

## 7. 产物指纹

所有运行产物位于Git忽略的 `exp/efficiency/sports_cached_deployment_seed2022_v2/`，没有覆盖或提交它们。源提交与审计文档不构成异机备份，当前无此备份证据。相对路径以下述目录为根。

| 文件 | SHA256 |
|---|---|
| batch.json | `2795775ae025b97b9554ff907c0e933713ad9b3d928379a0df3512a51cd6b67a` |
| preflight.json | `14ccee581d3d66823e81fb15cc8c50d8fe5f32e3e55b284096241d30ff0d0fad` |
| request_order.npy | `491dad23f9e1544694fda4b876e26eeefeeee2ae194f841d9d7be25a7f619a9a` |
| telemetry.jsonl | `3ea905d42281f7f7383d66bb23ce2b591ec4d75baa7c02a33df785ef3829a712` |
| T_tables.pt | `077511993f966c4e90db86017e0570bf6c6d3db0238c2c043c84991a21f0f824` |
| F_tables.pt | `07ef527e3a26233e1ad62c86d118644dcd4895aa60e1c8fd08b61fc2c6bd534f` |
| B_tables.pt | `762bdabbc7daa409dc14c54e50188af47e3ea0a879ae249ae90f91aec59b34f5` |

36个报告（每份报告记录原始样本、环境、内存及对应离线export哈希；后者亦已重新核验）：

| 条件 | report.json SHA256 |
|---|---|
| round0_offline_T_b0/report.json | `c05e8e6e3ccab247ce1f9465ac7318ebc7e37eb2ef21a6551ac70049a2dde4d5` |
| round0_offline_F_b0/report.json | `d1d9c274fd1596a78e298431f29ae4a295b843d251ac4921134f6d2cb5cf68b8` |
| round0_offline_B_b0/report.json | `96a7f032fda10c0407a51662d7b24d2fc15bec076bd3bea725a6680639a334a2` |
| round0_online_T_b1/report.json | `c08993e62a43bac22a4635801c8c5312f8634968c7e257514650492e375176ae` |
| round0_online_F_b1/report.json | `a01b800c77037b7ee971af6d7838cf990616b51927ccb01809f2b7ff4e8e51f0` |
| round0_online_B_b1/report.json | `09c4ed0bec386086bdc3a07dca65526bd4b01b93da76761b80c5e8cce09d4e5a` |
| round0_online_T_b128/report.json | `f6a5d990d8bd94d121a9e3d752380e22abdec0fa669e994db507617a34101c67` |
| round0_online_F_b128/report.json | `6b0784871b8f5c9f942bc0980e40be83e91a304839a6de10b14f91f9a574ba43` |
| round0_online_B_b128/report.json | `a5e7c05b51ba3e4c2df6c6c07ac529ea737a6305ba63a51e3859029891015abb` |
| round0_online_T_b1024/report.json | `31c0a457fa74fd2f5f0588b70d6006bfdc3fb23e1ea64c9ef34ea2e5eb9eae68` |
| round0_online_F_b1024/report.json | `1b5b1ea718ddce364a589f515d42e7789d960d2003debc161ab628faaf646141` |
| round0_online_B_b1024/report.json | `37d202bf4aa795399dda0cbfd28f997c42389617c32cd7caefd5971979506708` |
| round1_offline_F_b0/report.json | `fb4e547f12e3e1f17634abd227da058960c1745eabbbd62585d9ec7de33266c3` |
| round1_offline_B_b0/report.json | `df5cff8891ad08c6010212db5a872d91737aa8f658752af1ca5de58c6cb7f8a4` |
| round1_offline_T_b0/report.json | `724288fe780ab1e52c316f134dbe6b4a67a542dcb0645b57680724059ef49c18` |
| round1_online_F_b128/report.json | `c7acd9ba350a8b74ee9d26a2e92d8abdc22e77c46bdef830fe130c464bb9fb84` |
| round1_online_B_b128/report.json | `bd6e675fb16a0677e6f0e30175098bbb519efd32f5a2f70c7d876b8d759f828d` |
| round1_online_T_b128/report.json | `591262c24339e7abbe41cf8c9be216efebbdcc358c5eb818dbf4e81e01121b3b` |
| round1_online_F_b1024/report.json | `256ba25fe44d3906625333a84467d7c31830cb777ccc130ea14826992e321700` |
| round1_online_B_b1024/report.json | `eee23c8f51cd7296e67be385dbac2e1157df1ca637516c9f1f7ab37437091f85` |
| round1_online_T_b1024/report.json | `c02687b271cb58148951c766572072cffc175878ec3eede1cd51cf3b16a06448` |
| round1_online_F_b1/report.json | `f5d2610e67c36a83c49a9e92b67f72de96dbfda92ddef13479931830f0446ede` |
| round1_online_B_b1/report.json | `ba45858bfba41e3904da2f4449eab3bbfcbf1911a2fa123cbc52310008cea48e` |
| round1_online_T_b1/report.json | `8fca8c47609864c9ada2f9e47f58eccbe77c85fd2c5cea54d0b6ace664ae41d5` |
| round2_offline_B_b0/report.json | `1d02c251eb62915f3d11441ef91cb4ed1f21117b9e57ca2382f8ec926c239b4c` |
| round2_offline_T_b0/report.json | `29ae8bb0d3e22be11c922f8375563172eba9d87c566aad89802880b82071382a` |
| round2_offline_F_b0/report.json | `292b003c6665b18fa065ad9ed0531738e488c1dcf9eaff99488e3bbc959472b5` |
| round2_online_B_b1024/report.json | `61bfc3356383a54b92cd681c214717e5fc7e1b65517fbcb9d17b37ba1e4b04a5` |
| round2_online_T_b1024/report.json | `8243705d24241caf60331b22eb65293b439af8f6b604a24cf7ee1ffd92bb7826` |
| round2_online_F_b1024/report.json | `2b4224d334f8dbb86fd74f02fcb0b130671616129b2662d79b2002af712a436e` |
| round2_online_B_b1/report.json | `292ae5ee8ee8121387abc2cdf94d2316c01b9424600cb55e7ec5da496bd85bd7` |
| round2_online_T_b1/report.json | `abddc4993cc5190f10f1d2eb5857789964499711ca9402d820db4779c6e1029b` |
| round2_online_F_b1/report.json | `87e51df19747cd52affd262bed5a8f260c666a2c7e0cf3c3f165f9e9352c564e` |
| round2_online_B_b128/report.json | `6039bb780cea831b3ec846895f6845a3c07078453279559002a288b7517d1cc0` |
| round2_online_T_b128/report.json | `dace2cc8f1a4ee0f08196524b037260b0fe381bc7602a08f373450454b85941e` |
| round2_online_F_b128/report.json | `73fbdfb232574c1785519b7111058f0b0ced833151c70bf7c47f4f2d88e34c7a` |

## 唯一下一步

整理一版论文可用的结果与局限段落，将冷初值文本方向收益、暖启动增量收益边界、离线生成成本和缓存在线成本分别表述；暂不追加训练或效率重跑。
