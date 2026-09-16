# Sports 完整方法 seed2022 Validation120 复核

正常完成120轮0–119；没有教师重训、教师或学生Test评估。恢复最佳权重后结束；原manifest false保留。

|方法|最佳轮次|Validation Recall@20|同轮NDCG@20|
|---|---:|---:|---:|
|BPR|118|0.04052419949026864|0.019734699161345955|
|完整方法|115|0.07377433958487237|0.03327648738012842|

Recall相对提升 82.05008491923034%；同选中轮次NDCG相对提升 68.61917735896657%。NDCG按Recall选中轮次报告，不另行挑选NDCG峰值。

完整方法各20轮窗口最高Recall依次为 0.06446736942335919, 0.06724073282793802, 0.06857944723761156, 0.07028225825695732, 0.07174160990592647, 0.07377433958487237；最后一轮 0.07364324607774649。两组峰值均接近预算边界，不能认为已充分收敛。这是单种子、同预算Validation正向证据；尚不能断言多种子Test泛化、最终收敛优势或文本独立贡献。

核验：120条连续日志和所有曲线数组长度/有限值；checkpoint最佳轮次/指标等于曲线与manifest，checkpoint张量有限，推理导出ID表与完整权重一致。两组数据身份、教师指纹和12份源码指纹一致；解析参数差异仅profile名称及alpha/image/text三项，未发现其他参数混杂。用户语义项为0，图像/文本项均有效。Teacher reuse为sports_pinned_validation_only_v1；零Test，selected-versus-tested不适用。

当前源码与b500215实现一致，ca07f4d只增加记录；运行时Git HEAD/dirty快照缺失仍是追溯限制。窗口状态与系统负载未控制，不用本次耗时评价效率。

下一步：既定sports_student_image_matched_seed2022_val120_v1，alpha=0.3/1.3、image=1、text=0；与完整方法的实际图像系数一致，其他预算/锚点不变。无新训练或Test在审计中执行。保留seed2023审计例外及原false，第二创新点未确定。

- exp\runs\sports\run_manifest__2026-09-16 18_39_26.957343_sports_light_init_pid24140.json; SHA256 e46e32104ec676c86f51d4c5ff4bf53f37db55265360c1d679a7c0156a51314d
- logs\2026-09-16 18_39_26.957343_sports_light_init_pid24140; SHA256 653b20a664242f954350a7b90bacbe7f0a6c02915682250ef5f3fdba1ac081bf
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-16 18_39_26.957343_sports_light_init_pid24140.pkl; SHA256 f5c43a04be9368388fd0148ac3d2f4ce9eba8e6a77f0793da083c0447e51cb09
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-16 18_39_26.957343_sports_light_init_pid24140.pth; SHA256 ab1ea77b3566caef6edd8c5e0cf4fd2310539b0c6d21d30451f6b83a2bcb102a
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-16 18_39_26.957343_sports_light_init_pid24140.pth; SHA256 599ba5ca908dc4dbbfe037d1ee0f11086f2e998429d3f76442b25ed1e8920012
