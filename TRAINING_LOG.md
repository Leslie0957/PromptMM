# Training Log — 当前状态与近期记录

更新：2026-09-29。此文件与已核验历史快照共同构成实验记忆。先读AGENTS.md及README.md，历史pending不是执行队列。

## 当前状态

- 第一项：有限实验、独立审计及正文整合完成，暂停扩展；[结论与未验证边界](docs/paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md)。待跑实验为零。
- 第二项：进入问题探索准备，尚无已选算法或运行声明；[探索入口](docs/research/innovation2/README.md)。
- 约束：旧Test不选新方法；P1a已筛查停止；旧RGCS/门控不是队列。未知项不自动补跑。
- 唯一下一步：明确第二项要解决的问题、区别于第一项的新增价值及最小可证伪条件；尚未授权训练。

## 证据导航

- [粗粒度实验总览](docs/experiments/README.md)：先按问题找实验族，再进审计/原始产物。
- [历史章节索引](archive/training/ENTRY_INDEX_2026-09-26.md)：按主题定位旧命令、参数与声明。
- 最新完整快照：[TRAINING_LOG_ARCHIVE_FULL_2026-09-29.md](archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-09-29.md)，286218 bytes，SHA256 `905ec732cdca38e737b79155a81edda832b8549a7853da766175cd2945238413`。复制后逐字节核验；Git -text保留混合换行及历史控制字符。
- [旧入口快照清单](archive/navigation/2026-09-29/manifest.json)：旧导航按原路径结构保留，仅供历史查询；其中相对链接须按source原目录解析。
- 最近实质结果：成本独立审计b56b23a；正文整合4243dea。数据、检查点及运行目录未移动；Git不备份忽略资产。

## 本次整理记录

## 2026-09-29 Navigation and active-log cleanup (pending)

- User authorizes repository organization: shorten current entrances/log, rough experiment-family overview, first-work conclusions/open boundaries, second-work exploration entrance. Base4243dea; preserve check/, all raw assets, historical audits, source and result matrices. Scope: README/docs entrances, current first-work ledger, RUN_CLOSEOUT current pointers, immutable byte snapshots plus index and -text Git attributes; no training/evaluation/Test or algorithm decision.
- Before shortening, copy complete active log including this pending record byte-for-byte and verify size/SHA; preserve replaced navigation documents in a separate path-mirrored snapshot. Risks: broken historical links, lost commands/control characters, old pending mistaken for queue, EOL changes. Acceptance: byte identities including staged Git blobs, reachable evidence, short current pages, generated navigation refreshed and one scoped commit. Recovery is preserved snapshot/reference, never automatic reset. Next step only second-work problem exploration, not an experiment.

## 2026-09-29 Navigation and active-log cleanup (completed)

- Replaced accumulated status banners with concise root/docs/paper entrances. Experiment overview lists ten coarse families; detailed prior family navigation remains one level deeper. First-work ledger now contains conclusions, unknowns and restart conditions only; second-work page records no selected algorithm/run.
- Full old active log including pending preserved verbatim at the snapshot above (1035 lines, 286218 bytes); all six replaced navigation documents also byte-preserved with manifest. Git -text protects snapshots. Existing research audits, result matrices, raw assets and check/ unchanged; no tensor/data/Test access, experiment, source/model change or asset relocation.
- Verified snapshot byte lengths/SHA and current local links; refresh generated catalogs and check staged snapshot identities before scoped commit. Current log is deliberately short; future outcomes append here and detailed runs stay in family audits. This is navigation maintenance, no tag/backup milestone claim.
- Unique next step: second-work candidate problem comparison with novelty/value hypotheses and minimal falsification conditions; no historical gate or pending automatically resumes.
