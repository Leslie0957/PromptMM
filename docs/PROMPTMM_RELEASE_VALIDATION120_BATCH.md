> 2026-09-19更新：用户已改为三种子各300轮。当前请使用 [300轮批次说明](PROMPTMM_RELEASE_VALIDATION300_BATCH.md) 的命令；以下120轮说明仅保留历史，不要与300轮批次重复运行。

# PromptMM Sports 三种子 × 120 轮手动批次

本批次按用户2026-09-19明确要求扩大为seed2022、2023、2024各120轮，按此顺序串行，共360轮。每组从同一固定epoch37教师重新初始化，batch1024，每轮214步，单组25680步；每轮Validation、不早停，另有初始Validation，共121次/组。仅seed及轮数改变，其余系数和发布版算法保持原诊断配置。共享教师保持固定，不是三组独立教师实验。

运行一次：

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B codes/run_promptmm_validation120.py
```

预计单组约5小时，总计约15小时，取决于机器负载。保持供电、禁用睡眠，运行期间不要改项目文件或切换commit。只关闭显示器一般不影响计算；不要关闭终端。

主训练入口仍为main_mmlight.py。批次脚本依次调用：

```text
python -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2022 --epochs 120 --gpu_id 0
python -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2023 --epochs 120 --gpu_id 0
python -B codes/main_mmlight.py --promptmm_release_validation --dataset sports --seed 2024 --epochs 120 --gpu_id 0
```

不要在总命令之外再运行这三条。脚本开始前要求三个输出目录均不存在、Git干净；每组退出码为0且完整报告、种子、25680更新、121验证、选模、最佳checkpoint哈希、有限指标、零Test等检查通过才继续。跨种子公共配置和数据哈希必须相同。任何失败立即停止，已成功的结果保留，不自动重试/跳过/续训。源码HEAD须全批次固定。

每组结果在`exp/promptmm_release/sports_promptmm_release_validation120_seed{2022,2023,2024}_v1/`，各有report.json和best.pt。汇总状态在`exp/promptmm_release/sports_promptmm_release_validation120_three_seed_v1/batch.json`。不要删除目录绕过重复运行保护。完成或失败后把状态发回统一审计，无需逐组确认。

已完成30轮结果保留。本批次不是从best.pt续训；CPU DGL随机性意味着不保证与此前30轮前缀逐位相同。Recall@20严格改善保存最佳、同分保留较早轮次；初始Validation不参与选模。指标和目标的既有语义/限制见原诊断说明。仍是固定、未调参的发布版适配，多种子不能替代合理调参，不能直接证明论文创新性或公平效率优势。

所有运行保持Validation-only、零Test数据读取/评估，paper_ready_eligible=false；历史Baby seed2023审计例外及原manifest false保留，第二创新点未定。下一步仅统一审计本批次：三种子均值/波动、初始化至最佳/最终变化、30/120轮窗口、末段曲线，并与已有TD同种子120轮前缀谨慎比较；不自动延长300轮或新增Test。
