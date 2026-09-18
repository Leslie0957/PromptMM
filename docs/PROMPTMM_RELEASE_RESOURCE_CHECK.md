# PromptMM发布版接入与3步资源检查

## 本次交付范围

已接入`PromptMM-release-Sports-sharedTeacher-v1`的独立学生、实际参与总目标的loss、官方图排列及列表候选处理；在活动入口`codes/main_mmlight.py`最前面提供隔离resource-only分支。普通TD训练路径与已有配置保持不变。

**这是资源检查入口，不是300轮正式训练入口。** 不支持Validation选模、导出或正式长训练；这些留待资源结果核验后接入。源码审计与合成一致性通过不等于5060上已跑通真实数据。

## 用户手动命令（仅执行一次）

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B codes/main_mmlight.py --promptmm_release_resource_check --dataset sports --steps 3 --gpu_id 0
```

固定seed2022、batch1024；成功执行1个预热步和2个测量步后退出，最多3次optimizer.step。没有epoch循环、自动重试或下一阶段；失败直接保存报告并退出。CLI拒绝其他步数、数据集、epoch、Test等参数。命令不输出推荐准确率。

结果：`exp/resource_checks/sports_promptmm_release_resource_seed2022_v1/report.json`。该目录使用排他创建，成功或失败后再次运行均拒绝，避免误重跑。不要为了重试删除结果；把结果交回审计。代码必须已提交且工作区干净才允许启动。

## 数据及副作用边界

- 只读取`data/sports/train_mat`、`image_feat.npy`、`text_feat.npy`和共享教师checkpoint；用checkpoint已有身份核验这三个输入，不打开val_mat/test_mat/val.json/test.json，不调用Data、batch_test或普通preflight。
- 教师沿用SPORTS_CONVERTED_20260916锚的epoch37，SHA256 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`。初始prompt及硬token缓冲区直接从同一checkpoint读取，不重算PCA、不写缓存。
- 创建内存中的学生、执行3步后释放；只写独立JSON报告，不保存或覆盖任何模型checkpoint，不发布教师别名，不改变数据或原manifest。
- 教师初始eval前向用于学生初始化；各训练步按release切为train并在no_grad下前向。教师权重不更新，prompt虽然在优化器中但没有梯度且不更新，这是release行为，不应误判为资源检查失败。
- 报告包含源码commit、文件指纹、输入哈希、GPU/torch/DGL环境、各loss、逐步时间/峰值显存、参数量/共享存储量、更新及不变性检查；失败包含阶段步数、异常和可取的显存峰值。teacher原paper_ready_eligible=false保留，resource本身也为false。

## 此次诊断配置来源

|参数|值|说明|
|---|---:|---|
|学生表示维度|64|与共享教师对齐，不冒充官方默认维度|
|图层数/负采样数|1 / 10|官方release默认族|
|student_lr|0.00002|固定官方parser最后一组默认值，非Sports调参结果|
|AdamW weight_decay|0.01|官方未指定时的默认值|
|BPR embedding decay|0.00001|共享Sports参数锚；不是新增调参|
|pair/list/feature系数|1000000 / 1000000 / 0.1|固定官方parser最后一组默认值，仅用于结构资源诊断|
|SCE幂 / tip_rate|2 / 0|此次只接入该显式配置|
|教师drop_rate / prompt_dropout|0.2 / 0|共享教师配置；保留release训练态前向|

系数很大且release KL输入并非标准概率语义，**有限的负loss不自动表示失败**；不因数值不好看静默修正。非有限loss、梯度、参数或optimizer状态均停止。此配置未经Sports调参，不据3步loss推断模型质量。

## 一致性证据与边界

测试：`python -B -m unittest discover -s codes/tests -p test_promptmm_release.py -v`，10项通过。合成输入仅3个用户/5个物品/4维，CPU执行；不使用真实训练交互、不调用推荐评估。

- 测试参照是官方提交`70da1002a35d6f2c7712c16cd0b2ca24c8813008`的真实源码片段，保存在`codes/tests/fixtures/promptmm_release_reference.json`，记录完整原文件SHA256及总目标片段行号。没有导入官方旧训练入口。
- 比较float64和float32学生前向、每项loss、梯度、连续3步AdamW后的参数及optimizer状态；保留两个不同Parameter对象共用存储的行为。float64容差rtol1e-12/atol1e-10（参数atol1e-10上限），float32 rtol2e-5/atol2e-5（参数atol1e-7）；不声称逐位相等。
- 图矩阵保留官方`[UI,zeroUU;zeroII,IU]`的非标准列顺序，不静默改成常规二部图；候选保留DGL全局采样后丢弃neg_row的处理方式。
- 采样适配使用CPU DGL，同一随机种子重置仍观察到非逐位确定性，因此检验同一组已采样pair的处理与调用语义，不假装跨设备RNG相同。实际小型CPU DGL调用已成功；真实Sports调用尚待用户检查。
- Windows DGL分布式/GraphBolt导入兼容已验证；教师模块以隔离args命名空间导入，避免普通parser/data的副作用。小型测试确认不存在held-out文件也能完成train-only加载。
- 显式使用AdamW `foreach=False`保证逐参数的别名更新语义；与作者旧CUDA环境的逐位等价没有证明。
- 为资源探测不计算官方未进入总目标的多余diagnostic loss；它们不影响本次loss/梯度一致性，但省略会影响耗时/内存，**3步报告不能当正式原版效率加速比**。正式效率协议应单独处理冗余计算口径。
- 共享教师是本地既有教师，不是重训官方教师的端到端复现；只有学生参与目标的语义通过上述合成对照。

现有Sports student profile两项回归及serial launcher一项回归通过。普通活动入口仅增加最前面的显式分流；未改写原训练主体、默认parser或已有profile。

开发期间发现并修复：Windows DGL stub需提供可继承的类；测试抽取的文本loss片段需包含完整末行；精确梯度断言需允许浮点累加误差。以上均在合成测试阶段解决，没有真实资源运行失败或重试。

## 运行后下一步

把“资源检查跑完了”或失败输出发回。先检查报告、3步完整性、显存和原版目标稳定性，再决定是否可进入Validation-only正式训练接入/调参声明。不自动启动300轮，不根据本次资源结果断定论文价值。历史Baby seed2023审计例外及原manifest false继续保留；第二创新点未定。
