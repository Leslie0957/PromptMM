"""Inventory and copy the already selected Innovation1 fixed-result assets.

Metadata/file operations only: never imports model code or evaluates a split.
Use prepare before copy; verify reads the pinned manifest without source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/research/INNOVATION1_REUSE_ASSETS_2026-09-26.json"
TABLE = ROOT / "docs/paper/INNOVATION1_FIXED_MAIN_TABLE_2026-09-26.md"
DEFAULT_MANIFEST = ROOT / "docs/research/INNOVATION1_RESULT_FREEZE_MANIFEST_2026-09-26.json"
COHORT = "innovation1_eval_recovery_v1"
EXPECTED_LEDGER_SHA = "b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c"
EXPECTED_BATCH_SHA = "fd4d1bf7eb57678db25e3ba32a818d17ab512c5eaebc6c48ac57d20547f79e0f"
SLOTS = [f"{arm}_{seed}" for arm in ("b3", "s1", "s2", "s3") for seed in (2022, 2023, 2024)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: str | Path) -> str:
    value = Path(path)
    resolved = (value if value.is_absolute() else ROOT / value).resolve(strict=True)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Asset outside repository: {resolved}") from exc


def collect() -> dict:
    if sha256(LEDGER) != EXPECTED_LEDGER_SHA:
        raise ValueError("Pinned shared asset ledger changed")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    files: set[str] = set()
    expected: dict[str, str] = {relative(LEDGER): EXPECTED_LEDGER_SHA}

    def add(path: str | Path, fingerprint: str | None = None) -> None:
        name = relative(path)
        files.add(name)
        if fingerprint:
            previous = expected.get(name)
            if previous and previous != fingerprint:
                raise ValueError(f"Conflicting pinned SHA256 for {name}")
            expected[name] = fingerprint

    def add_tree(path: Path) -> None:
        for child in sorted(path.rglob("*")):
            if child.is_file():
                add(child)

    add(LEDGER, EXPECTED_LEDGER_SHA)
    for dataset in ("baby", "sports"):
        add_tree(ROOT / "data" / dataset)
    add_tree(ROOT / "environment")
    for folder in ("formal_closeout_cohort", "formal_closeout_preflight", "formal_closeout_eval"):
        add_tree(ROOT / "exp" / folder)

    old: dict[tuple[str, int], tuple[float, float]] = {}
    for item in ledger["baby_existing_formal"]:
        add(item["manifest"], item["sha256"])
        report = json.loads((ROOT / item["manifest"]).read_text(encoding="utf-8"))
        if not (report["status"] == "completed" and report["paper_ready_eligible"] and report["final_test_performed"]):
            raise ValueError(f"Old Baby result is not completed/eligible: {item['manifest']}")
        method = "BPR" if report["resolved_arguments"]["td_distill_alpha"] == 0 else "Full"
        old[(method, item["seed"])] = (
            report["final_test_result"]["recall"][1], report["final_test_result"]["ndcg"][1]
        )
        for key in ("converge_run", "preflight_report"):
            add(report["artifacts"][key])
        for key in ("td_full_checkpoint", "td_infer_checkpoint"):
            add(report[key])
        add(report["teacher_checkpoint"], report["teacher_checkpoint_fingerprint"]["sha256"])

    for item in ledger["sports"]:
        add(item["teacher"]["path"], item["teacher"]["sha256"])
        for asset in item["assets"]:
            add(asset["path"], asset["sha256"])

    new: dict[str, tuple[float, float]] = {}
    for slot in SLOTS:
        report_path = ROOT / "exp/formal_closeout_eval" / COHORT / slot / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not (report["status"] == "completed" and report["paper_ready_eligible"]
                and report["student_test_attempts"] == 1 and report["student_test_evaluations"] == 1
                and report["teacher_test_evaluations"] == 0
                and report["selected_checkpoint_sha256"] == report["evaluated_checkpoint_sha256"]):
            raise ValueError(f"Final report qualification failed: {slot}")
        if report["ledger_sha256"] != EXPECTED_LEDGER_SHA:
            raise ValueError(f"Final report ledger mismatch: {slot}")
        add(report["source_path"], report["source_sha256"])
        add(report["selected_checkpoint_path"], report["selected_checkpoint_sha256"])
        new[slot] = (report["test_result"]["recall"][1], report["test_result"]["ndcg"][1])

    batch = ROOT / "exp/formal_closeout_cohort" / COHORT / "batch.json"
    add(batch, EXPECTED_BATCH_SHA)
    batch_record = json.loads(batch.read_text(encoding="utf-8"))
    if batch_record["status"] != "completed":
        raise ValueError("Recovery batch not completed")
    steps = batch_record["steps"]
    if len(steps) != 24 or [step["phase"] for step in steps] != ["no_test_validation"] * 12 + ["final_test_once"] * 12:
        raise ValueError("Recovery batch has wrong phase/order/count")
    last_preflight_finish = max(datetime.fromisoformat(step["finished_at"]) for step in steps[:12])
    first_final_start = min(datetime.fromisoformat(step["started_at"]) for step in steps[12:])
    if last_preflight_finish >= first_final_start:
        raise ValueError("Final Test began before all preflights finished")
    for slot in SLOTS:
        preflight = json.loads((ROOT / "exp/formal_closeout_preflight" / COHORT / slot / "report.json").read_text(encoding="utf-8"))
        if not (preflight["status"] == "passed" and not preflight["test_split_loaded"]
                and preflight["student_test_evaluations"] == 0 and preflight["teacher_test_evaluations"] == 0):
            raise ValueError(f"Preflight/Test-zero qualification failed: {slot}")
    if len(old) != 6 or len(new) != 12:
        raise ValueError("Expected 6 original Baby plus 12 recovery Test cells")
    table = TABLE.read_text(encoding="utf-8")
    for dataset, method, prefix in (
        ("Baby", "BPR", None), ("Baby", "Full", None),
        ("Baby", "PromptMM 发布版共享教师适配", "b3"),
        ("Sports", "BPR", "s1"), ("Sports", "Full", "s2"),
        ("Sports", "PromptMM 发布版共享教师适配", "s3"),
    ):
        pairs = [old[(method, seed)] if prefix is None else new[f"{prefix}_{seed}"]
                 for seed in (2022, 2023, 2024)]
        line = next(line for line in table.splitlines() if line.startswith(f"| {dataset} | {method} |"))
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            raise ValueError(f"Malformed paper row: {dataset}/{method}")
        for index, (recall, ndcg) in enumerate(pairs):
            if cells[index + 2] != f"{recall:.6f} / {ndcg:.6f}":
                raise ValueError(f"Paper Test cell mismatch: {dataset}/{method}/{index}")
        for index in (0, 1):
            values = [pair[index] for pair in pairs]
            if cells[index + 5] != f"{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}":
                raise ValueError(f"Paper summary mismatch: {dataset}/{method}/{index}")

    entries = []
    for name in sorted(files):
        path = ROOT / name
        actual = sha256(path)
        if name in expected and actual != expected[name]:
            raise ValueError(f"Pinned SHA256 mismatch: {name}")
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": actual})
    return {
        "identity": "innovation1_fixed_results_18_cells_2026-09-26",
        "source_root": str(ROOT),
        "paper_table": relative(TABLE),
        "shared_asset_ledger_sha256": EXPECTED_LEDGER_SHA,
        "recovery_batch_sha256": EXPECTED_BATCH_SHA,
        "test_cells": {"old_baby": 6, "recovery": 12, "total": 18},
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }


def verify(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = destination / "assets" / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Backup differs or is missing: {path}")
    copied = destination / "manifest.json"
    if not copied.is_file() or sha256(copied) != sha256(manifest_path):
        raise ValueError("Backup manifest differs or is missing")
    print(f"VERIFIED {len(manifest['files'])} files, {manifest['total_bytes']} bytes: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "copy", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    if args.mode == "prepare":
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to replace existing freeze manifest: {manifest_path}")
        manifest = collect()
        manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(f"INVENTORY {len(manifest['files'])} files, {manifest['total_bytes']} bytes: {manifest_path}")
        return
    if args.destination is None:
        parser.error("--destination is required for copy/verify")
    destination = args.destination.resolve()
    if args.mode == "verify":
        verify(manifest_path, destination)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to write into occupied destination: {destination}")
    if destination == ROOT or ROOT in destination.parents and destination.parts[-1] == "assets":
        raise ValueError("Unsafe backup destination")
    for entry in manifest["files"]:
        source = ROOT / entry["path"]
        if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
            raise ValueError(f"Source changed before copy: {source}")
        target = destination / "assets" / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(manifest_path, destination / "manifest.json")
    verify(manifest_path, destination)


if __name__ == "__main__":
    main()
