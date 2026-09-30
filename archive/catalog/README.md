# 本机实验资产路径目录

导航版本：2026-09-26；可用 `python -B tools/build_experiment_navigation.py` 刷新。
这些生成清单可刷新，**不是不可变的实验身份记录，也不是物理备份**。正式身份查原manifest及审计。

- [全部文件路径和字节数](WORKSPACE_FILES_2026-09-26.jsonl)：3522项；仅文件元数据，不读数据集/模型内容。
- [exp JSON元数据](RUN_RECORDS_2026-09-26.jsonl)：1034项；对应[可点击记录表](../../docs/experiments/RUN_RECORDS.md)。
- [源代码与文档导航](../../docs/experiments/FILES.md)；[实验族导航](../../docs/experiments/README.md)。
- [历史章节索引](../training/ENTRY_INDEX_2026-09-26.md)：424个章节定位。

范围：当前本机可达文件；包含data、Model、exp、logs、backups及历史代码目录的路径，不读取大文件载荷。
排除目录（所有层级）：`.agents, .codex, .codex_review_user, .codex_tmp, .git, .mypy_cache, .pytest_cache, .ruff_cache, .venv, __pycache__, _qa_current_doc, node_modules, venv`；不跟随符号链接/目录联接。
六个生成清单自身不进入资产字节目录，避免自引用；它们仍列在源文件导航。
`source_catalog`表示路径同时出现在源文件导航，不能等同于已经提交。
JSON仅读取exp下最多32MiB的记录，空缺/读取失败照实显示；不会根据路径猜测验收结果。
目录保留原始路径；找不到的历史产物仍可查不可变训练日志，目录不保证过去全部资产仍在本机。
本工具不会移动、删除、重命名产物，不会重写任何完整训练日志快照。
