"""Refresh navigation using paths, file sizes and saved JSON metadata only.

No project imports, model/data loading, evaluation, training or archive rewriting.
The dated catalogs are mutable navigation; the full training snapshots are immutable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-09-26"
EXCLUDED = {".git", ".agents", ".codex", ".codex_tmp", ".codex_review_user",
            "_qa_current_doc", "__pycache__", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", "node_modules", ".venv", "venv"}
OUTPUTS = ["docs/experiments/FILES.md", "docs/experiments/RUN_RECORDS.md",
           f"archive/training/ENTRY_INDEX_{STAMP}.md", "archive/catalog/README.md",
           f"archive/catalog/WORKSPACE_FILES_{STAMP}.jsonl",
           f"archive/catalog/RUN_RECORDS_{STAMP}.jsonl"]


def write(path, content):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


def link(path, origin, label=None):
    relative = os.path.relpath(ROOT / path, ROOT / origin).replace("\\", "/")
    return f"[{escape(label or path)}]({quote(relative, safe='/.-_')})"


def escape(value):
    return str(value).replace("|", "\\|").replace("[", "\\[").replace("]", "\\]").replace("\n", " ").replace("\r", " ")


def tracked_paths():
    result = set()
    for args in (("ls-files", "-z"), ("ls-files", "--others", "--exclude-standard", "-z")):
        raw = subprocess.check_output(["git", *args], cwd=ROOT)
        result.update(p for p in raw.decode("utf-8").split("\0") if p)
    return result | set(OUTPUTS)


def files():
    """Do not follow symlinks/junctions or read payloads; paths/sizes only."""
    for current, directories, names in os.walk(ROOT, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in EXCLUDED
                                and not is_link(Path(current) / d))
        for name in sorted(names):
            path = Path(current) / name
            if not is_link(path):
                yield path


def is_link(path):
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def scalar_metadata(data, name):
    if not isinstance(data, dict):
        return None
    # First recorded spelling wins; absent values are not invented.
    aliases = {"profile": ("profile", "student_profile", "student_config_profile", "run_profile"),
               "seed": ("seed",), "status": ("status",)}[name]
    for obj in (data, data.get("args"), data.get("resolved_args"), data.get("resolved_arguments"),
                data.get("parameters"), data.get("config")):
        if isinstance(obj, dict):
            for key in aliases:
                value = obj.get(key)
                if isinstance(value, (str, int, float, bool)):
                    return value
    return None


def json_record(path, size):
    """A sealed holdout is inventoried, never opened for metadata parsing."""
    relative = path.relative_to(ROOT).as_posix()
    row = {"path": relative, "bytes": size, "status": None, "profile": None, "seed": None}
    if path.name.lower() == "sealed_lock.json":
        row["read_status"] = "sealed_payload_not_loaded"
        return row
    try:
        if size > 32 * 1024 * 1024:
            raise ValueError("metadata read capped at 32 MiB")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        row.update({key: scalar_metadata(data, key) for key in ("status", "profile", "seed")})
        row["read_status"] = "parsed"
    except (ValueError, OSError) as exc:
        row["read_status"] = str(exc)
    return row


def main():
    paths = tracked_paths()
    source = ["# 文件导航\n\n按路径列出Git已跟踪文件和当前未忽略的新文件。大型运行资产见",
              "[本机资产目录](../../archive/catalog/README.md)。这是导航，不是启动时必读列表。\n"]
    groups = {}
    for path in sorted(paths):
        if (ROOT / path).is_file() or path in OUTPUTS:
            groups.setdefault(path.split("/")[0] if "/" in path else "根目录", []).append(path)
    for group, entries in sorted(groups.items()):
        source.append(f"\n## {group}（{len(entries)}）\n")
        source.extend(f"- {link(p, 'docs/experiments')}" for p in entries)
    write(OUTPUTS[0], "\n".join(source) + "\n")

    records = []
    inventory = []
    for path in files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in OUTPUTS:
            continue  # Generated inventories excluded to prevent self-size recursion.
        size = path.stat().st_size
        inventory.append({"path": relative, "bytes": size,
                          "category": relative.split("/")[0] if "/" in relative else "root",
                          "source_catalog": relative in paths})
        if relative.startswith("exp/") and path.suffix.lower() == ".json":
            records.append(json_record(path, size))
    records.sort(key=lambda r: r["path"])
    inventory.sort(key=lambda r: r["path"])
    run_doc = ["# 运行与诊断记录导航\n",
               f"目录版本：{STAMP}。共{len(records)}个现存exp JSON记录；数量不是独立实验数。",
               "保存状态按JSON原文摘录；—表示该字段未记录/未在约定层级找到，不表示成功或失败。",
               "批报告、子任务、资源检查与正式运行混列供定位，不能相加作为样本数。",
               "历史已做但本机没有JSON的实验，请查[完整历史章节](../../archive/training/ENTRY_INDEX_" + STAMP + ".md)。",
               "验收、协议例外和结论以[实验族审计](README.md)及原始日志为准。\n",
               "| 记录路径 | 保存status | profile | seed | JSON读取 |", "|---|---|---|---|---|"]
    for row in records:
        values = [escape(row[k]) if row[k] is not None else "—" for k in ("status", "profile", "seed", "read_status")]
        run_doc.append("| " + link(row["path"], "docs/experiments") + " | " + " | ".join(values) + " |")
    write(OUTPUTS[1], "\n".join(run_doc) + "\n")

    index = ["# 历史训练记录章节索引\n",
             "先搜索标题，再按源文件行号读取相关段落。快照互有重叠；声明/失败/完成不是独立实验计数。",
             "旧pending和旧‘下一步’不是当前授权。当前状态见[短日志](../../TRAINING_LOG.md)。\n"]
    section_count = 0
    archives = sorted((ROOT / "archive/training").glob("*.md"))
    for path in archives:
        if path.name.startswith("ENTRY_INDEX_") or path.name == "README.md":
            continue
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        relative = path.relative_to(ROOT).as_posix()
        index.extend([f"\n## {path.name}\n", link(relative, "archive/training"),
                      f"\n{len(raw):,} bytes；SHA256 `{hashlib.sha256(raw).hexdigest()}`\n",
                      "| 起止行 | 原始章节标题 |", "|---|---|"])
        headings = [(i + 1, line[3:]) for i, line in enumerate(lines) if line.startswith("## ")]
        if not headings:
            headings = [(1, "完整历史交接文件")]
        for n, (start, title) in enumerate(headings):
            end = headings[n + 1][0] - 1 if n + 1 < len(headings) else len(lines)
            index.append(f"| {start}–{end} | {escape(title)} |")
            section_count += 1
    write(OUTPUTS[2], "\n".join(index) + "\n")
    for dest, rows in ((OUTPUTS[4], inventory), (OUTPUTS[5], records)):
        write(dest, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    excluded = ", ".join(sorted(EXCLUDED))
    write(OUTPUTS[3], f"""# 本机实验资产路径目录

导航版本：{STAMP}；可用 `python -B tools/build_experiment_navigation.py` 刷新。
这些生成清单可刷新，**不是不可变的实验身份记录，也不是物理备份**。正式身份查原manifest及审计。

- [全部文件路径和字节数](WORKSPACE_FILES_{STAMP}.jsonl)：{len(inventory)}项；仅文件元数据，不读数据集/模型内容。
- [exp JSON元数据](RUN_RECORDS_{STAMP}.jsonl)：{len(records)}项；对应[可点击记录表](../../docs/experiments/RUN_RECORDS.md)。
- [源代码与文档导航](../../docs/experiments/FILES.md)；[实验族导航](../../docs/experiments/README.md)。
- [历史章节索引](../training/ENTRY_INDEX_{STAMP}.md)：{section_count}个章节定位。

范围：当前本机可达文件；包含data、Model、exp、logs、backups及历史代码目录的路径，不读取大文件载荷。
排除目录（所有层级）：`{excluded}`；不跟随符号链接/目录联接。
六个生成清单自身不进入资产字节目录，避免自引用；它们仍列在源文件导航。
`source_catalog`表示路径同时出现在源文件导航，不能等同于已经提交。
JSON仅读取exp下最多32MiB的记录，空缺/读取失败照实显示；不会根据路径猜测验收结果。
目录保留原始路径；找不到的历史产物仍可查不可变训练日志，目录不保证过去全部资产仍在本机。
本工具不会移动、删除、重命名产物，不会重写任何完整训练日志快照。
""")
    print(json.dumps({"source_paths": sum(map(len, groups.values())), "asset_paths": len(inventory),
                      "json_records": len(records), "history_sections": section_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
