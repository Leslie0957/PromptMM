# Sports未见物品诊断：独立运行实现与合成预检

2026-09-30。本轮用户授权的是上一阶段唯一下一步：软件实现和合成预检。**实现准备完成，真实Sports资产、CUDA执行与正式效果均未验证，未启动正式诊断。** 第一项继续冻结，创新判断仍遵循[路线v1.1](../../../THESIS_ROADMAP.md)。

## 实现范围

[运行入口](../../../tools/run_unseen_transfer_diagnostic.py)及[独立核心](../../../tools/unseen_transfer_core.py)实现[冻结协议v1](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROTOCOL_V1.md)的Train-only身份核验/划分、封存、暖PCA/图、两类教师、共同CF评分锚点、六类内容对照、校准、混合排名、分组指标、配对区间和筛查规则。没有导入旧Data、parser或main训练入口；第一项源码/默认参数/检查点未改。

[原JSON](UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROFILE_V1.json)保持冻结时的历史准备状态和全部参数，不通过修改它给自己授权。其SHA256为 `8fa483db8ee8fd1a80086b6f7bcbb090803adc4d57ccddf8c85e6cf96ad9ed8e`。当前软件就绪程度以本页及活跃日志为准。

真实模式只接受三种固定载荷，先核验bytes/SHA256再反序列化已知本地Train；图文以mmap打开容器，数值行按角色访问。划分器可接触后来属于lock的Train交互，只分区封存；封存指纹在写入时计算，汇总文件指纹不重开它。PCA仅fit W_active。模型训练不能读取probe；全部raw/校准选择封存后才能进入报告，不能再回到训练或选择阶段。访问记录是程序守卫与事件，不声称OS级全文件访问审计。

两教师ID初值/三元组一致，MM增量只在内容线性输入；学生使用共同P_CF/Q_CF及标准化评分差蒸馏，避免跨教师坐标直接对齐。六臂公开次数：R3/V2/D2/K_CF4/K_MM4/N1；不隐瞒KD搜索多于D/V。R的ridge正则按逐元素mean公式解析；raw配置先冻结，再独立选择三值冷分数系数。单次报告使用同一混合排名，联合正例分母与分组NDCG分开定义，边界同分按原ID顺序。

资源管理包含5s采样和每batch协作检查、阶段/总时间、RSS/CUDA分配/磁盘/输出约束与OS GPU使用查询。超限或异常保留failed产物，目录已存在即拒绝；每次改善选择时保存checkpoint/曲线，禁止自动重试/续跑。该守卫不是OS硬隔离，长库调用的中断时效未验证。总48h是保护上限，CPU合成耗时不能预测真实训练时间。

## 已完成的验证

环境只读导入检查：Python3.10.20、PyTorch2.11.0+cu128、NumPy2.2.6、SciPy1.15.3、scikit-learn1.7.2、psutil7.2.2。现有环境未改装。以下从仓库根执行：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B -m unittest discover -s tests -p test_unseen_transfer_runtime.py -v
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_unseen_transfer_diagnostic.py --preflight
```

[16项针对性测试](../../../tests/test_unseen_transfer_runtime.py)通过：角色/封存边界、冷数值不改变暖PCA、负例合法与回放、教师初值及归一化图的数值恒等式、MLP初值、教师旋转不改变评分KD、ridge驻点与截距、精确TopK边界同分、已见项mask与联合/分组分母、选择顺序、禁止旧载荷、bootstrap人群对齐、支持/停止/未判定、目录拒绝覆盖、封存文件不重读、注入失败保留且只尝试一次。开发时曾出现list传入set的类型错误，已在正式合成贯通前修正；没有真实运行失败或重试。

`--preflight`只读配置、转换manifest和三个允许文件的stat，未重算数据哈希或加载其载荷；正式资产身份还须在真实执行中核验。依赖存在不等于真实规模或CUDA kernel已经通过。

最终合成贯通命令（**已执行，输出目录不能重用**）：

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_unseen_transfer_diagnostic.py --synthetic-check --output exp/innovation2/synthetic_unseen_preflight_20260930_v2
```

- 人工16用户/160物品；两模态各8维，PCA各4维，教师8维，MLP8→16→8，K=5，各fit最多2epoch、每epoch选择，尺度tape256、bootstrap40，强制CPU、总时间保护120s。学习率/beta臂数保留，用于遍历路径；这些覆盖只属于fixture，真实JSON未改变。
- 六教师fit、十二神经学生fit、三ridge及一个内容参考完成；14个教师/学生variant报告状态齐全。CUDA allocated全程为0；报告内cohort总时间约1.843s（不含解释器依赖导入），采样RSS峰约643.75MiB，不能外推Sports资源。
- 140项保存产物核查通过：文件指纹（封存文件使用写入时指纹，不重开）、预定fit次数、14报告、空真实输入访问清单、选择封存先于probe、逐用户联合分母/命中及报告均值、CPU遥测。fixture的undetermined来自缩小样本/轮次，不作为研究方向负证据。
- 最终输出：`exp/innovation2/synthetic_unseen_preflight_20260930_v2/`。report SHA256 `1a70b207e47f0b4082f795ec52dd6f8dc7d360b43505e61be9e8348fab260b6a`；batch SHA256 `9efe4310ef727542ac1d6d660b5754ba03f69a4e63ecb0e33751af734ed62467`。完整源/环境及文件身份在其source_manifest/batch中，合成源commit是基线参考而非干净正式launch身份；实际未提交实现以文件SHA标识，最终提交在任务交接报告。
- 第一份开发合成产物 `synthetic_unseen_preflight_20260930_v1/`保留；随后发现其清单哈希会重开封存文件，已修复并增加拒绝重开测试，最终以v2为准。注入异常的`synthetic_unseen_failure_gate_5367db349a/`亦保留，用来验证failed记录/无重试，不是实际研究失败。

真实Train/图文/检查点载荷、旧Val/Test、真实lock均零访问；未建真实划分或暖教师，没有正式cohort、Test指标或第二创新点成立结论。

## 真实启动接口与剩余边界

已提供可审查的[运行声明草稿](UNSEEN_ITEM_TRANSFER_RUN_DECLARATION_V1.json)：实际输入锚点、seed2022、6教师/12神经fit、R/N、校准、48h上限、比较锚点、禁止Test/重试及完整outcome/matrix路由都已填写；`explicit_user_authorization=false`，**草稿不是执行许可**。

未来获授权并完成pending正式声明/干净源提交后，固定接口是：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_unseen_transfer_diagnostic.py --execute --declaration docs/research/innovation2/UNSEEN_ITEM_TRANSFER_RUN_DECLARATION_V1.json
```

目前此命令会被拒绝：声明未授权。运行守卫要求声明已提交、源干净、参数身份及路由一致；只允许一个固定输出目录，不支持改seed/epochs、resume或重试。GPU算法的确定性开关不保证所有平台位级复现；有关开关的限制见[PyTorch2.11原文](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html)，本轮没有执行GPU操作以验证实际kernel。

未验证项包括真实数据划分后有效样本、PCA时长、GPU算子/峰值、教师优化强度、质量与成本。若真实执行缺少样本/尺度/优化或教师额外能力，须记undetermined；执行异常/资源超限为failed，保存产物；有效负证据为completed/screen_stop，随后与用户讨论。任何情况下都不能自动改配方/增seed/回到第一项。

记录路由：本轮只更新此准备页、活跃日志/短入口及生成导航。真实AUDIT/RESULTS/HANDOFF尚未生成，矩阵无数值；第一项正文/结果、THESIS_ROADMAP、旧协议参数/历史报告不变。没有tag、bundle、推送、合并、删除或资产备份里程碑。

**唯一下一步：授权并声明一次固定seed2022的Sports串行诊断cohort（含协议规定的真实资产重建），从该声明提交的干净HEAD启动一次，结束后审计；不追加seed、重试或下一阶段。** 本轮到软件准备止步。
