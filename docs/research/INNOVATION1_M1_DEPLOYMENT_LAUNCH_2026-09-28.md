# M1 serial same-checkpoint deployment measurement declaration

2026-09-28. **Prepared for one user-manual launch; not executed.** This is the single M1 stage in the [bounded roadmap](INNOVATION1_COST_ROADMAP_2026-09-28.md). M0 passed its [real smoke audit](INNOVATION1_COST_M0_AUDIT.md). M2 is outside this declaration.

## Identity, scope and comparison

- Run ID `innovation1_cost_v1/m1/serial_v1`; exclusive output `exp/efficiency/innovation1_cost_v1/m1/serial_v1/`. Stop if it exists. The launch source must be the clean committed HEAD on `codex/experiment/baby-teacher-baseline`; the exact hash of this declaration commit is reported in the handoff, while the runtime records its own HEAD and source SHA. Only the pre-existing unrelated untracked `check/` is permitted.
- Shared anchor: [M0 binding map](INNOVATION1_COST_M0_BINDINGS.json), SHA256 `5d38b4d91760d776d46b587ebd19c1574d1db7365485bf0791badf26dddb1148`; [M0 audit](INNOVATION1_COST_M0_AUDIT.md), launch source `aa7eddd9cc707e6cac8781316c818fb566a6e2db`; selected Sports BPR/Full/release checkpoints for seeds 2022/2023/2024 and one shared teacher. Runtime checks M0 worker report and export SHAs, selected checkpoints and Train SHA. Prior formal Test quality is linked only through committed JSON; no new Test ranking.
- M1 delta from M0: 3 measurement rounds and 30 serial isolated workers. Each round includes all 9 students and the one shared teacher. Order rotates by three positions per round. Each worker measures one identity's original representation path and serves its selected float32 table under batch sizes 1/128/1024. There is no new model state, optimizer update, checkpoint selection or quality evaluation.
- Full/BPR original preparation is the native ID table clone after verifying full selected checkpoint equals the inference-only table. Release original preparation restores `ReleaseStudent` with its storage aliases and runs `release_graphs(train)` forward. Shared teacher is loaded once per worker via the original teacher/prompt adapter and common SHA; it is reported as one identity in each round, not triplicated per student. The original warmed-student benchmark output is never used.

## Measurement and hard gates

- Common Train-only request seed `9026026`, same eligible-user order, masks, item universe and `torch.topk(k=20)` for all identities. Per identity/round: 3 GPU generation warmups and 10 timed generation calls; one actual CPU transfer and file write; 20 cached-service warmups and 100 timed requests for each of batch 1/128/1024. CUDA is synchronized at timing boundaries. A separate 100-call CUDA event GEMM-only loop is labeled diagnostic, separate from the full mask/top-k service wall timer. Raw samples, median/p95 and round ranges are retained for audit; 100 requests are not independent model replicates.
- The reference table is the M0 exported same-checkpoint table. The worker requires original output versus reference `atol=rtol=1e-4` before timing, and exports its generated table to a new round-specific file. Offline totals include Train load, original setup, one representation generation, device-to-host copy and write; repeated generation timing is separate. Original setup may include OS-cache hits and model/checkpoint loading, so do not label it cold disk. Storage inventory reports model-resident and logical/file table bytes separately. Service timing excludes network transport.
- One process per identity/round, single GPU, sequential only. Parent cap 3600 s; 4 GiB PyTorch CUDA allocated, 8 GiB process-tree RSS, 2 GiB new output, at least 10 GiB free disk. The source records device/driver/PyTorch/threads and samples GPU temperature/clock/utilization/memory. Before launch use AC power/fixed performance mode and ensure no competing GPU training; the runner does not terminate user processes. If environmental conditions are unsuitable, revise the declaration before launching.
- Test and Validation file I/O denied by audit hook. Counters must remain Test0, Validation0, optimizer steps0. Any identity, SHA, parity, finite, cap, output or worker failure stops the batch. Do not retry, resume, skip a condition or launch M2. Preserve partial files. A weak or absent speed advantage is valid evidence if all hard gates pass.

## Exact command and acceptance

Check that the output directory does not exist, the declared Python/CUDA device is available, free disk exceeds 10 GiB, and the source is committed. Then run **once**:

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_cost_m1.py'
```

The dry route `--dry-run` is synthetic and does not open model/data assets or create the M1 output. It is for review, not a second measurement. The real command saves parent `batch.json`/`report.json`, per-condition report/raw samples and table exports, stdout/stderr, and GPU telemetry in its exclusive directory. Only a completed parent with exactly 30 completed condition reports, all checks and resource limits passing, constitutes an accepted M1 run. Audit actual exports, summaries, condition balance, quality-to-checkpoint binding, Test0 and environmental disturbances before interpretation. No result is assumed in this declaration.

## Record-update targets

- Outcome audit: `docs/research/INNOVATION1_DEPLOYMENT_COST_AUDIT.md`; all 10 identities × 3 rounds, with per-method/seed preparation and batch 1/128/1024 serving cells. `TRAINING_LOG.md`: preserve this declaration/pending text and append completed, failed or partial outcome.
- Active matrix: M1 row and its method/seed/batch cells in [cost results](INNOVATION1_COST_RESULTS.md). Experiment-family navigation: cost row in `docs/experiments/README.md`; root/docs current pointers only if stage changes. Current goal ledger `docs/paper/INNOVATION1_CURRENT_GOAL_EVIDENCE_GAPS.md`: C1/C3 only if the measured evidence changes its status; paper result prose/gap consumers only if scientific claims actually change. The old 18-cell quality/Test and initialization matrices are not updated.
- Refresh six generated navigation outputs after manual edits. Preserve raw files under the exclusive M1 directory and all historical audit/manifests/checkpoints/`check/`; no tag, bundle, push, merge or baseline overwrite. This is not a stable milestone backup. After audit and one coherent outcome commit, the sole next decision is whether to separately authorize M2 preparation. M1 launch permission does not extend to M2.
