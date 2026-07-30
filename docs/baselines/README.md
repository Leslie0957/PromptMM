# PromptMM Recovery Baseline

The first recoverable source baseline is the state that passed the one-epoch validation-protocol smoke on 2026-07-29.

- Git branch: `main`
- Stable tag: `baseline-protocol-smoke-20260729`
- Baseline commit message: `baseline: preserve protocol-smoke state`
- Local repository identity: `PromptMM Research <promptmm@local>`
- Canonical experiment memory: `TRAINING_LOG.md`

## What Git Protects

Git tracks source code, tests, small historical source trees, research notes, documentation, environment manifests, and the asset fingerprint manifest. It intentionally does not store generated checkpoints, datasets, run logs, experiment outputs, patent binaries, installers, or archives.

The exact ignored-asset inventory is recorded in `ASSET_MANIFEST_2026-07-29.csv`. Each row contains the repository-relative path, byte length, UTC modification time, and SHA256 digest.

The manifest detects missing or changed assets but cannot restore them. Copy `data/`, `Model/`, `logs/`, `exp/`, `zhuanli/`, and `archive/run_patent_results/` to another physical disk for protection against deletion or disk failure.

## Safe Recovery

Inspect the stable baseline without modifying `main`:

```powershell
git switch -c inspect/baseline baseline-protocol-smoke-20260729
```

Restore one source file from the baseline after first committing or otherwise preserving current work:

```powershell
git restore --source baseline-protocol-smoke-20260729 -- codes/main_mmlight.py
```

Verify all fingerprinted large assets:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\verify_baseline_assets.ps1
```

A standalone local Git bundle is stored at `backups/PromptMM_baseline_protocol_smoke_20260729.bundle`. It is intentionally ignored by Git and can recreate the source repository if `.git` is damaged. Because it is on the same disk, it is not a substitute for an external backup.

## Baby Adapter Milestone

The first audited real-multimodal Baby adapter state is preserved separately after its one-batch teacher smoke:

- Task branch: `codex/experiment/baby-data-adapter`
- Stable tag: `baby-adapter-smoke-20260730`
- Standalone bundle: `backups/PromptMM_baby_adapter_smoke_20260730.bundle`
- Canonical conversion and smoke record: `TRAINING_LOG.md`

This milestone includes the converter, protocol checks, tests, and experiment record. The raw and derived Baby data, hard-token caches, checkpoint, log, and run manifests remain ignored assets; the tag and bundle can verify their recorded hashes but cannot recreate those files after physical deletion.

## Future Work

Create a branch before the next code change:

```powershell
git switch -c experiment/<short-name>
```

Commit each confirmed code or protocol change and append its training consequence to `TRAINING_LOG.md`. Completed training runs must also remain recorded there under the repository guidance.
