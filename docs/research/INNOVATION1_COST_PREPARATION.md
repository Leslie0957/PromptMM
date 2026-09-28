# Innovation1 bounded cost work: M0 preparation

2026-09-28. This is preparation for the M0 correctness/resource preflight in the [bounded roadmap](INNOVATION1_COST_ROADMAP_2026-09-28.md). It is not M1 or M2 cost evidence. The fixed status matrix is [here](INNOVATION1_COST_RESULTS.md).

## Identity and semantic gates

- [Nine-slot binding table](INNOVATION1_COST_M0_BINDINGS.json) was derived from the final `innovation1_eval_recovery_v1/batch.json` `final_test_s*` verified reports, original source JSON, and the existing asset inventory. Each slot carries method/seed/config, teacher/initialization, source path/SHA, selected checkpoint path/SHA and epoch/Validation Recall@20, saved quality-report path/SHA and final metrics. It checks selected versus evaluated checkpoint SHA equality. One teacher SHA is common to all slots.
- The original Full/BPR manifests record source-file fingerprints but no source Git commit. The binding table leaves `source_commit=null` for these slots. It does not infer a commit from dates. Release report commits are recorded. The new runner's launch commit is separately captured.
- The static check verifies existing file sizes and inherited SHA declarations, source-selection fields and report links. Payload SHA is rechecked during the real smoke. Test JSON is read as historical evidence only; no Test file is opened or evaluated.
- Full/BPR exports use the selected model's ID embedding weights and require equality with its inference-only checkpoint. Release export instantiates the original `ReleaseStudent`, restores its aliased state and runs `forward` on `release_graphs(train)`; raw IDs are not treated as final release embeddings. The shared teacher uses the existing `cached_deployment_benchmark.load_teacher` adapter, which loads the original teacher and prompt classes; it references an older warm manifest only to reconstruct the identical teacher identified by the shared SHA. The old benchmark output namespace and warm student checkpoints are not used.
- Float32 tables and fixed Train-only masked requests are exported, reloaded and compared to the same original model output: one request per student seed (three total per method) and three for the shared teacher. Table tolerance is `atol=rtol=1e-4`; fixed score absolute delta must be at most `1e-4`, with identical exclusion positions. The release graph, table and masked score route passed a synthetic CPU check. This tolerance is a preflight fidelity criterion, not a quality threshold.

## One manual M0 smoke command

Prerequisites at launch: clean committed `codex/experiment/baby-teacher-baseline` HEAD, `D:\miniconda\envs\run_5060\python.exe`, CUDA device 0, at least 10 GiB free on the output volume, and no competing training/GPU workload. The source gate permits only the existing unrelated untracked `check/`; all tracked modifications and other untracked files block launch. Confirm that `exp/efficiency/innovation1_cost_v1/m0/smoke_v1/` does not exist. The current read-only check found CUDA available on an RTX 5060 with 8151 MiB total, about 6452 MiB free, and over 410 GiB free on D:; verify conditions again immediately before running.

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\innovation1_cost_m0.py' --smoke-real
```

The command exclusively claims the smoke directory and launches one isolated worker. It has no retry or resume path and never proceeds to M1/M2. The parent monitors wall time (600 s), process-tree RSS (8 GiB), output (2 GiB) and free disk (10 GiB); the worker caps the CUDA allocator at 4 GiB and checks resource bounds between identities. It performs zero optimizer updates, no Validation/Test ranking, three fixed requests per method across its seeds, and three for the shared teacher. It writes `batch.json`, `worker.json`, `report.json`, stdout/stderr, and ten exported table files. The Python audit hook denies named Validation/Test split I/O. A failure preserves all partial output and consumes this attempt; recovery requires a separately declared command/new namespace. Power loss or forceful termination may leave an incomplete report, which is not a pass.

M0 acceptance requires a completed parent and worker, nine student rows plus one teacher, one matching request per student and three teacher requests, source/checkpoint SHA and original/export fidelity, Test0/Validation0, zero updates, limits passed and retained raw reports. Synthetic and static checks do not substitute for this real preflight. No timed M1/M2 claim is made from smoke measurements.

## Future stages and routing

M1 and M2 remain separate declarations and separately authorized stages. Their planned multi-arm/seed work is single-GPU serial, with balanced order as fixed in the roadmap; there is no concurrent run. The M0 smoke is one command containing all ten identities. If M0 succeeds, audit its saved artifacts first, append an M0 outcome, update [the cost matrix](INNOVATION1_COST_RESULTS.md), refresh navigation and commit. Only then prepare the M1 declaration. M2 follows an independently audited/authorized M1 stage. There is no authorization in this preparation to run any of them.

Record targets for the real M0 outcome: `docs/research/INNOVATION1_COST_M0_AUDIT.md` (new); `TRAINING_LOG.md` outcome/current state; M0 row in `INNOVATION1_COST_RESULTS.md`; cost experiment family row in `docs/experiments/README.md`; root current pointer if stage changes; six generated navigation files. Existing 18-cell formal Test and initialization matrices remain unchanged; paper/gap text changes only if an actual conclusion changes. Preserve all original results and `check/`.
