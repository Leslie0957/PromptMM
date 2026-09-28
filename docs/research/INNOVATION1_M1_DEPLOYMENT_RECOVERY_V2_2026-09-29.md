# M1 serial_v2 recovery declaration

2026-09-29. **Prepared, not run; new authorization required.** This supersedes only the consumed v1 launch command/namespace in [the original M1 declaration](INNOVATION1_M1_DEPLOYMENT_LAUNCH_2026-09-28.md). The [v1 failure audit](INNOVATION1_DEPLOYMENT_COST_AUDIT.md) is retained.

- Run ID/output: `innovation1_cost_v1/m1/serial_v2`, exclusive `exp/efficiency/innovation1_cost_v1/m1/serial_v2/`; it must not exist before launch. The v1 raw directory is never overwritten or read as a result. Clean committed HEAD on `codex/experiment/baby-teacher-baseline`, allowing only pre-existing unrelated `?? check/`; parent records exact launch hash and source fingerprints. The source recovery anchor is v1 launch commit `978254941ae06255224a69a1c0328a77d99ecf26` for comparison, not an automatic reset.
- Exact implementation delta: keep M0 saved reference tables on CPU for `compare_reference`, create a separate CUDA copy for the common cached service, and add a synthetic parity API-contract check. No change to original model outputs, checkpoints, teacher, graph, Train-only requests, methods/seeds, order, dtype/precision, 3 rounds, 3+10 generation iterations, 20+100 service iterations, batch 1/128/1024, timing boundaries or analysis. Same M0 binding SHA256 `5d38b4d91760d776d46b587ebd19c1574d1db7365485bf0791badf26dddb1148`; nine formal Sports student checkpoints and one shared teacher. Existing quality JSON only; new Test0/Validation0/optimizer0.
- Same 3600 s, 4 GiB CUDA allocated, 8 GiB process-tree RSS, 2 GiB new output and 10 GiB free-disk hard caps; single GPU, no competing training, AC power/fixed performance, no silent limit change. Thirty serial workers must all complete with original/M0 parity, full raw samples, identical per-batch Train request fingerprints, export roundtrip and SHA, finite times, source and asset gates. First failure stops; no retry/resume/skip/M2. A negative speed comparison is accepted if these gates pass.

After explicit authorization, manually run **once**:

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_cost_m1.py'
```

The script's committed v2 output constant distinguishes this command from the consumed v1 attempt. Check the launch commit reported in the preparation handoff, the empty v2 namespace and conditions before executing. No real measurement was performed to verify the repair.

Record-update targets after this new attempt: append a v2 section to [the M1 audit](INNOVATION1_DEPLOYMENT_COST_AUDIT.md) covering all 30 conditions or the exact failure/partial state; append separate outcome in `TRAINING_LOG.md`; update M1 method/seed/batch cells and stage status in [cost results](INNOVATION1_COST_RESULTS.md), experiment-family and current pointers; refresh six generated navigation files. Update current goal C1/C3 and paper/gap consumers only if accepted evidence changes those claims. Preserve old v1 audit/output, M0 assets, formal 18-cell quality and initialization results. One outcome commit; M2 remains separately declared and authorized.
