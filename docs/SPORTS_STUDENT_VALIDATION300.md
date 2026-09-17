# Sports 三种子三组 Validation300

用户指定300轮，取代此前240轮建议。九组从头训练，保留120轮原件；不是断点续训。只改变每个已验收120配置的epoch/patience为300及名称/source。教师Epoch37、PCA seed2022、数据/图像实际权重、随机初始化策略、学生lr6e-5/decay0.01/batch1024全部不变。学生seed2022/2023/2024，每个依次BPR/full/image_matched。

启动（由用户手动）：

```powershell
Set-Location D:\Download\PromptMM
& D:\miniconda\envs\run_5060\python.exe -B tools/run_sports_validation300.py
```

窗口内不早停，仅Validation Recall@20选模，零Test/零教师重训；从新随机初始化开始，不加载旧学生。每组一次，串行九组；源树不干净、已有300轮manifest或批次记录、进程非零退出、异常完成manifest均停止，禁止自动重试。中断时保留所有文件并告知助手，不删除记录或重跑整个批次。普通低指标不是失败。自动完成状态检查不代替跑后曲线/权重审计。

批次记录exp/runs/sports/serial_validation300_seed2022_2024.json保存实际启动commit、dirty=false、Python、命令和已完成列表。每组前复查commit/工作区；运行中不要修改源码。原run manifest不变，由此新sidecar补充来源。记录使用独占创建，二次执行不会自动恢复。dry-run仅打印命令，不写批次记录或训练。

60项测试通过；九组实际parser/教师hash/冻结元数据验证通过；全部旧120配置与59b56a3逐项相等。未执行GPU训练、模型前向或PCA fit。GPU非确定性可能造成前120轮差异，跑后先核对曲线前缀再分析预算变化；300轮不是收敛保证。

数据锚SPORTS_CONVERTED_20260916，转换manifest SHA3772a17c8b70fa4739653534dca8d542e67e0649f1f44ccba4194fec17bc1104；教师SHA57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea，原false不修改。保留Baby历史seed2023审计例外/原manifest false，第二创新点未确定。

## 展开的九条命令

- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2022_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2022_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2022_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2023_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2023_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2023_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_bpr_seed2024_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_full_seed2024_val300_v1 --gpu_id 0
- D:/miniconda/envs/run_5060/python.exe -B codes/main_mmlight.py --dataset sports --student_profile sports_student_image_matched_seed2024_val300_v1 --gpu_id 0
