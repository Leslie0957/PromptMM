# 质量约束前沿 v1：保存产物审计与收尾

2026-09-30，当前agent自审，不是另一评审者独立认证。用户报告“跑完了”后只读核验既有输出。**执行产物硬验收通过，completed；科学simple_control_sufficient_in_development。** 终端进程退出码未保存，不能声称独立验证exit0。batch completed、完整产物与一致性检查是此次完成证据。

## 运行来源、命令与固定范围

分支`codex/experiment/baby-teacher-baseline`；launch `0d2eafdef81aae6fc58e781fe688ecf332ddf1b6`，收尾前HEAD `161117818baa83abd3cfd29ea33acfb4c283fc2a`（此后只有用户手动交命令政策更新）。启动声明及两runtime/profile的canonical Git内容与launch一致，两runtime的实际文件SHA等于source_manifest。历史启动门槛代码拒绝dirty tracked source、未提交声明或已存在输出；该次实际launch来源与authorized declaration相符。没有独立OS或外部启动现场快照；untracked archive/reviews不在tracked门槛内且未纳入提交。

用户手动命令（已经执行一次，**仅留存来源，不再运行**）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_quality_constrained_frontier.py --execute --declaration docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_RUN_DECLARATION_V1.json
```

共享锚点：原launch07eaeee8db0fc62b8fc998f7ebed1c30dd024e78、原结果6714bf14880b636261a03584f731bac70e776d09、路线1dea1c88393c67e2644cddc5bb3dde440291853f；原batch SHA9d4861d0089a230db2b76613443b6b5e3c1f1173dfda0f4e9db25356af026764。profile SHAa665ed386d58c47067fc33d04304f06752d814b07b0eb8077c524367dfd986cd，resolved_profile完全一致。runner SHAe97a58221177fe9efe160a525ec79cd8bcf20b1864a844126f726881e164398f；core SHA922a2b96f0bdd56fcce18225572f015fa5c94d76394529b2bfb5b38b832efa20。

六固定模型身份/epochs/N query锚点与profile一致，保存PCA/CF P/Q复用，模型、教师和PCA重拟合0，重新选checkpoint0。参数delta只限此前声明的float32基础/float64仿射排名与6×63开发校准；K20/batch128、a=2^(j/2),j=-4..4、b=[-1,-.5,-.25,0,.25,.5,1]、主ε.001/次ε0,.0005,.002、2000bootstrap/seed20260930不变。新训练0，不导入旧probe-capable Roles，不读取T_MM tables。warm CF_W_ONLY、overall D/raw共同基准，完整数值及24格见[RESULTS](QUALITY_CONSTRAINED_FRONTIER_RESULTS.md)。

环境保存记录：run_5060，Python3.10.20、numpy2.2.6、torch2.11.0+cu128、psutil7.2.2，实际设备cpu/线程4。scipy1.15.3、sklearn1.7.2是原声明环境，不冒充本轮source_manifest重新记录的版本。本轮无GPU训练或完整训练成本结论。

## 保存证据核验

使用独立的一次性只读审计脚本（`.codex_tmp/frontier_closeout.py`，临时文件不入Git）核查2536项，全部通过；脚本无项目runtime导入，不加载原始输入/模型，不重新排名或重跑bootstrap。

| 验收项 | 实际证据与界限 |
|---|---|
| 产物 | 所有13类必需文件及8缓存齐全，共21文件；batch以外20文件bytes/SHA与batch一致，batch本身另取SHA；目录无额外未登记文件 |
| 来源 | launch/source_commit/profile/声明一致；两runtime当前SHA与保存记录一致；14输入保存身份与冻结锚点一致，未为审计重哈希旧大资产 |
| 角色 | 13011用户、13930暖/917冷候选、fit84695、暖12671、冷5351/4067用户/881物品及原pair SHA全部匹配；保存用户数组SHA匹配 |
| 数值与范围 | 378唯一网格点+2非搜索参考=380；全部evaluated、0 skipped；汇总R/N有限且范围正确；6冷向量范数有限；各批次全网格严格冷排序证书为runtime证据，没有重新证明全候选分数 |
| 质量选择 | 独立核对24格可行点数、完整确定性tie选择、缺失和ε嵌套；六臂三维非支配点集合完整且成员指标一致 |
| 逐用户 | 19不同状态/248字段；Top20形状、唯一ID及属于逐用户缓存候选、同分母、总体=暖+冷的命中/分母代数、Recall代数与R/N汇总、冷槽位/曝光集中度全部相符；两暖参考等于暖缓存 |
| 标签/屏蔽局限 | 审计不重新恢复fit/select标签；fit屏蔽和命中位置/DCG正确性依据启动runtime的检查、已通过合成测试及保存来源；逐用户NDCG本次核对有限/范围/均值，未从原标签重新计算 |
| 区间 | 36记录，27 completed、9因ε0缺少可行点跳过；差值与已保存配对逐用户Recall相等，用户数和区间有限/有序；不重跑重采样，不作选择后显著性认证 |
| 科学判据 | 主ε最佳KD−最佳简单=.000122940742562085≤.002；样本足够、sigma不缺失，simple_control_sufficient_in_development与预定分流相符；原v1不变 |
| 资源 | 335个单调采样均RSS≤6GiB/输出≤2GiB/free≥12GiB；最低free397280571392bytes；总1822.579s及各stage低于上限；peak RSS689827840bytes与采样相同；最终输出21749257bytes |
| 执行限制 | 记录training_updates0、automatic_retry false、repeated ranking0；agent没有执行formal命令、恢复角色、模型推理或新实验 |

## Test与访问时间线

1. 原手动启动：用户明确批准新originalTrain partition-only接触；运行实际先SHA核验再加载完整源容器。这会接触来源容器的probe/lock字节，不能写“原Train全程未重开”或“源字节零接触”。
2. 角色恢复只导出fit/warm_select/cold_select；暖rank1未保留，probe/lock标签未构造。恢复manifest记录全源与临时pairs在评分前释放。特征数值仅warm_active/cold_select；源feature容器mmap，未重新整文件hash。
3. 应用事件中14次允许payload load、所有role events均在allowlist；原batch元数据SHA先验证。旧Val/Test读取0、旧probe评价0、sealed payload读取/哈希0、锁定确认0，无新的seal。
4. 评分、选择及描述性bootstrap只开发select；本次没有`val_test_once_v1`最终Test，所以“恰好一次student Test”及selected-versus-tested checkpoint相等不适用。固定模型身份相等已验证，未改选或测试其他checkpoint。
5. 收尾审计仅读新输出/源文本/冻结声明并哈希新产物，旧Train/feature/model/probe/Val/Test/sealed payload接触0，排名/训练/重采样重执行0。导航刷新仅元数据，sealed文件名deny guard另行验证。

这些是应用guard/event和源代码的证据，不是OS完整I/O跟踪；不存在额外独立无访问证明。禁止以审计为由打开或哈希sealed_lock。

## 输入身份保留

以下是source_manifest保存的实际运行核验方式，审计只核对与冻结锚点的一致性。两feature为可信原batch锚点+stat，不是新整文件完整性证明；其他12输入为运行加载前SHA验证。

| 既有输入身份（仅核查保存manifest） | bytes | SHA256 | 核验方式 |
|---|---:|---|---|
| `data/sports/train_mat` | 1890060 | `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/teacher_selection.json` | 1698200 | `0668df48fb97db898b5916e5765c69734085777929d63a852b22c5250f6d9740` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/student_selection.json` | 14236589 | `1cb24b2cab2f8c1982b2f585ec34d7d1f9b5ffef846c37852ab547aad2779a84` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/split_manifest.json` | 1410 | `06c1509d572f23c9b8969fbf9cbbcfcb5640c4e5afa4bf12513477e208092efd` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/source_manifest.json` | 1472 | `73aa1a7697209627fcd4de12a1f25fae08ad4381b216b0da97db5f94edda4170` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/T_CF_tables.npz` | 6371655 | `c6afaafcda0be55c22929cae5d44fa1391859ed209d6dd9aea8b0f52db527b52` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/transform.npz` | 2233807 | `349047a3e398cfa2d63b7aa3d4d22c571d589c8655bdf7c359b1bc44b6a231c2` | runtime_sha256_before_load |
| `data/sports/image_feat.npy` | 601522304 | `222f924a0694b6c7e2bc26ea4bc2ef4ec89f7045aa5105ef463af8455c7a7695` | cited_anchor_plus_stat_no_feature_rehash |
| `data/sports/text_feat.npy` | 28196480 | `27d6087f0d8644b8245c60052425530c6446dffe61024b8126db0009ad94de84` | cited_anchor_plus_stat_no_feature_rehash |
| `exp/innovation2/sports_unseen_transfer_v1/students/R_lambda0.001/selected.npz` | 63731 | `81f97ceef4668e57965b25d175a36df3a507069f65fa7cd66944243cbf285e11` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/students/V_lr0.001_beta0.0/selected.pt` | 101415 | `ca132d4cc47af4585055f693679cb5aa25f6f7ddc7942cd0b0c0b0c0a04c91a3` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/students/D_lr0.001_beta0.0/selected.pt` | 101415 | `5f8999de0145244dc9c5d9e15ffa63be5a1422f7743fb882b3d64f47c3d8b89d` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/students/K_CF_lr0.0003_beta0.1/selected.pt` | 101415 | `97695d3984ccda2dc9d1a01ff56e015ac823b51f58e555653b0abcc22cb474f7` | runtime_sha256_before_load |
| `exp/innovation2/sports_unseen_transfer_v1/students/K_MM_lr0.0003_beta0.1/selected.pt` | 101415 | `3e846289ec593f0e40419b048d99a44eb33356ee1fd7a7000a8bb908c6d7bd6a` | runtime_sha256_before_load |

## 新产物指纹

位置`exp/innovation2/sports_quality_frontier_v1/`。全部忽略资产留在本机，无删除、覆盖或重新生成；Git只保护源码/审计。没有声称off-device备份、恢复训练能力或正式结果冻结里程碑；本次不创建标签/bundle。

| 本轮产物 | bytes | SHA256 |
|---|---:|---|
| `access_audit.json` | 5242 | `0854e4aacc2e7b7d8274e32bc0146a3b1681326b73694d746c3291bc498bca0b` |
| `budget_selection.json` | 17204 | `16d66e2549d07662815d84fe677999aef878fcbf7565c563f2ece521273c7a86` |
| `frontier.json` | 46248 | `c267a9966677ff6f64c23b8dd21a84a0ebaa565d69315431ae70246348b712bf` |
| `grid_metrics.json` | 298214 | `9ab962b641314d65d5813240311d48320b06432577fe10f6aae3ca8a58a4fd97` |
| `model_manifest.json` | 2401 | `c78b3c4de0cb3f53eb610954b427c9f5f442d3c4632f655085942cbeebc2e0b2` |
| `per_user_selected.npz` | 9477439 | `07719e32094f49ba7bb94c7ffa2915dcdf86f9cd1541dc6964d6c8175e4c7113` |
| `report.json` | 13294 | `92fd6997e60f52aba259b873b4ac266712bd0251c95241351c2dc01e2dc8da6a` |
| `resolved_profile.json` | 17742 | `08d9aeed2cf0767e5e6e7d95c52bbfb5f8f3b499d98990e3d9475301c1d1c4f3` |
| `resources.jsonl` | 45868 | `5a8afd250568f30131c3674dc10beeca87aa6dae09316001cc53a49473c1f914` |
| `restoration_manifest.json` | 1103 | `1734e9c4b369b7c58664313cf7d04d131b615c2bef105b27f5f0d527955e3aca` |
| `score_cache/CF_warm.npz` | 1476955 | `dcd8d035b5aa0a8b33f9982ba08117328db0feb1313e66fb9f59ef326c041fc8` |
| `score_cache/D_cold.npz` | 1468076 | `bd25ce742bde01c737ac37a29f6938960370d59ba819755c8dbe221e039f98c7` |
| `score_cache/K_CF_cold.npz` | 1454905 | `e87c646be3e16680bbb44b7723221d62519bb3b025de48c10e497b761a974bba` |
| `score_cache/K_MM_cold.npz` | 1458512 | `9353df104274f547555ae1574d6eb40d8cca3f4495d11ec308aacabbdbec4da0` |
| `score_cache/N_cold.npz` | 1496488 | `fe85a28fef6b0e7a2403a8e499d1f8a4185b2ce3d5beaf055ebcde61d8c03a4e` |
| `score_cache/N_warm.npz` | 1640901 | `d18bbe1a9247d5b76f0f10ecb1429ab3d0aa6b80c9dd3076a78a5632ae2d26e1` |
| `score_cache/R_cold.npz` | 1398463 | `4f64c91a2cd439e9eca234de9ff31fd60fc107c4a3eee8fea99c69a4fc039209` |
| `score_cache/V_cold.npz` | 1420124 | `7cdd195fe30d1cefc2d217afa67cf452f925f618c3523db258d6276376b208f3` |
| `score_cache_manifest.json` | 1646 | `787c36dd1284909a054dbb3c6947bbe2df174c70537a29fdcdc4773d0c97f274` |
| `source_manifest.json` | 4407 | `11edb0166edd7a999d025151eca3b8b81ff1ea927a5117907fb3253751db872a` |
| `batch.json` | 4025 | `1b80a99342317efcb2ae02736be56e6c3d7078ab054ffee0e35ff1e86adda1db` |

## 记录路由与风险

| 目标 | 处理 |
|---|---|
| TRAINING_LOG | 保留原pending/授权/政策记录；新增closeout pending和completed，当前摘要移除“待启动” |
| RESULTS | 新增真实24格、6raw/2reference、辅助项与区间；全标开发select |
| AUDIT/HANDOFF | 新增本页与独立交接，来源/指纹/门槛/访问/意义/下一步明确 |
| 根README、innovation2 README、实验总览 | 更新当前完成状态/审计入口；实验总览残留旧“评分器未实现”当前文字一并纠正 |
| 六生成目录 | 所有手工编辑后metadata-only刷新；sealed deny guard；实际差异文件纳入同一提交 |
| docs/README | 已核对，链接稳定/无具体运行状态，无需修改 |
| 第一项正文/gap/Test矩阵、THESIS_ROADMAP | 不适用：第一项冻结，本次无Test或总体目标变更；保持不动 |
| 原v1结果/审计/profile/runtime、前沿冻结profile/protocol/declaration/历史launch review | 不改：历史状态保留，可从新RESULTS/AUDIT/HANDOFF看到当前阶段 |

剩余限制：单seed固定模型、有限网格、select反复开发；所选区间描述性，不能证明等效或泛化。最大KD微小优势不证KD全无作用，cold-only/mixed差距不唯一归因尺度；不能将普通全局校准改名认证创新。现有证据不足以让当前MM教师/KD配方成为成立第二贡献；按THESIS_ROADMAPv1.1评价具体差异和价值，不采用基础工具必须原创的额外门槛。

## 唯一下一步

基于两轮诊断做第二项路线重评，筛选相对“直接学习＋相同校准”有明确新增价值的问题，并设计一个最小可证伪对照；仅评审与设计，不启动实验。
