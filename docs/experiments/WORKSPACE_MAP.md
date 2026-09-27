# 实验前工作区地图（2026-09-27）

本次只整理入口，不移动或删除文件。当前优先级以 [跨交互路线](../research/CROSS_INTERACTION_ROUTE_2026-09-27.md) 和 [日志最新追加记录](../../TRAINING_LOG.md) 为准。日志中旧“Current authorized task / handoff”描述过去恢复阶段，不是当前任务。

| 分类 | 原位路径与解释 |
|---|---|
| 活跃正式实现 | codes/main_mmlight.py、codes/utility/parser.py 及其模型/协议依赖；td_distill_model_no_projection.py 是维护中的对照，不是临时文件 |
| 手动入口及验证 | tools/、tests/、codes/tests/；旧启动器保留，可查源码但不可当自动队列 |
| 当前文档 | 根规则/日志；跨交互路线；docs/experiments/ 导航；docs/paper/ 已有证据及缺口 |
| 历史参考 | docs/research/ 和 docs/research_notes/ 中旧候选、RGCS、旧总览；论文路线.txt；codes/run_patent.py 独立线；其他 main、4.9数据集Photo代码/、LATTICE/、zhuanli/ 不属于当前训练入口。保留，不逐个改写历史审计 |
| 实验资产 | data/、Model/、exp/、logs/ 的数据、缓存、checkpoint、manifest、原始输出全部原位；完整状态与 infer 表不能互换 |
| 历史与环境 | archive/training/ 不可变快照，archive/catalog/ 可刷新导航；environment/、backups/ 保留。导航不是资产备份 |
| 用户评审 | check/ 原文保留，不纳入本次提交 |
| 待人工判断的临时候选 | .codex_tmp/、.codex_review_user/、_qa_current_doc/、Python 缓存；未逐项确认用途，因此不处理。.agents/ 等工具配置也不动 |

没有发生搬迁，因此无旧→新路径映射。未将历史 pending、旧文档建议或评审意见转换成执行任务。当前只准备 P0，不开发完整门控，不进入 P1/P2。
