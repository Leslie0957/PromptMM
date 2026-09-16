# Sports BPR seed2022 Validation120 结果复核

运行正常完成（manifest status=validation_completed，符合仅验证流程），120轮0–119；没有重训教师，没有教师或学生Test评估。原paper_ready_eligible=false及教师原件保持不变。

最佳 Epoch 118；Validation Recall@20=0.04052419949026864；同轮 NDCG@20=0.019734699161345955。末轮 Recall@20=0.040425879359924234。

各20轮窗口最高Recall@20持续上升（约0.02156、0.02961、0.03377、0.03642、0.03880、0.04052），最佳在倒数第二轮。说明该预算内基线仍在学习，不能视为充分收敛；后续完整方法即使更好，也先解释为同预算优势。保持现有配对预算，暂不单独延长BPR以免改变比较条件。单种子Validation结果不等于最终Test泛化证据。

核验：所有曲线数组120项且有限；语义损失及四个分量全为0；checkpoint最佳轮次/指标与manifest及曲线一致，完整权重张量有限，推理导出的两个ID表与完整checkpoint逐元素相等。日志确认恢复最佳后跳过Test，selected-versus-tested比较不适用。数据身份与教师锚点一致，教师指纹匹配；源码指纹与准备提交b500215当前文件一致。runtime缺少Git HEAD/dirty快照的历史限制仍存在，不能倒推启动时干净。

下一步：运行已声明的 sports_student_full_seed2022_val120_v1，一次、仅Validation、120轮。当前仅完成BPR，不能判断完整方法或文本项在Sports上的收益。

用户提到前后台it/s差异；当前时间记录未控制窗口状态及系统负载，因此本次耗时不用于方法效率结论。

- exp\runs\sports\run_manifest__2026-09-16 17_18_56.750542_sports_light_init_pid25008.json; SHA256 d8f4e9e29b6f61a4ef47a35c5a58a3598955a2eeea473319ec6eeeb05b0a51f3
- logs\2026-09-16 17_18_56.750542_sports_light_init_pid25008; SHA256 6c9edaeb7900457b09b1ff108815a260c2bf41addae47d0f289efc7b1c9ee8ec
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-16 17_18_56.750542_sports_light_init_pid25008.pkl; SHA256 3bffed2588fc3bbb8d410613e234b6763ae1968fff5c7fdc6598072cfe906f4b
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-16 17_18_56.750542_sports_light_init_pid25008.pth; SHA256 72b6a56ca12e106bd748fdcaf8645b621bd72e83690b18345ab108b380ececf7
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-16 17_18_56.750542_sports_light_init_pid25008.pth; SHA256 71677150e2ec06851b5183a93c6ab361fb772637a7c48044336a5973473eec85
