# Innovation1 M0 bounded real-smoke audit

2026-09-28. **Outcome: completed, within the declared M0 correctness/resource scope.** This is a non-formal preflight. It supplies no M1/M2 deployment or update cost result and performs no new quality evaluation.

## Launch and source identity

User manually executed once:

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\innovation1_cost_m0.py' --smoke-real
```

The exclusive [smoke directory](../../exp/efficiency/innovation1_cost_v1/m0/smoke_v1/) contains the preserved [parent report](../../exp/efficiency/innovation1_cost_v1/m0/smoke_v1/report.json), [worker report](../../exp/efficiency/innovation1_cost_v1/m0/smoke_v1/worker.json), `batch.json`, stdout/stderr and ten table exports. Parent and worker report `completed`; worker exit code 0. Launch source is `aa7eddd9cc707e6cac8781316c818fb566a6e2db` on `codex/experiment/baby-teacher-baseline`, the committed M0 preparation source. The parent was created at 2026-09-28 19:57:29.850986 +08:00 and finished at 19:57:43.826000 +08:00; 13.953 seconds measured parent wall time. No second attempt is present in the declared `smoke_v1` namespace.
`batch.json` is the initial launch receipt and retains `status=started`; `report.json` is the final parent status. The generated run-record catalog lists each JSON separately, so its `batch.json` row does not indicate an unfinished worker.

The parent records [binding map](INNOVATION1_COST_M0_BINDINGS.json) SHA256 `5d38b4d91760d776d46b587ebd19c1574d1db7365485bf0791badf26dddb1148`, independently matched to the committed map; its worker-report SHA256 `5eca2051656617346e3e4c9c7749ce3e9dfdf34143be791f63e250b231b70437` matches the saved worker JSON. `batch.json` carries the same launch source and fixed caps. The Train file SHA256 was independently checked as `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8`. The nine original source JSON, nine selected checkpoints, six TD inference files and one shared teacher payload were independently hashed against the binding table: 25/25 match. The release source commit is carried by each original report; Full/BPR original manifests give file fingerprints but no Git commit, so that field remains unresolved and is not inferred.

## Fidelity and bounded-resource evidence

| Check | Declared limit / rule | Saved evidence and independent audit |
|---|---|---|
| Identity coverage | BPR/Full/release × seeds 2022/2023/2024; one teacher | Exactly nine expected slots and one shared teacher; no duplicate slot |
| Export shape and dtype | User 35598×64, item 18357×64; float32 | All ten exports report these shapes and `torch.float32`; each has 13,812,480 logical bytes and 13,814,309 file bytes |
| Original/export parity | Tables `atol=rtol=1e-4`; masked score absolute delta ≤1e-4 | One Train-only request per student slot, three per method; three shared-teacher requests. All twelve saved request score deltas are 0.0. Runtime checks reject table mismatch, nonfinite/wrong-shape tables and exclusion mismatch |
| Export integrity | Saved file identity | All ten output file SHA256 and byte counts independently match `worker.json`; nine student and one teacher exports retained |
| Wall time | ≤600 s | Parent 13.953 s |
| CUDA allocated | ≤4 GiB | Worker configured a 4 GiB allocator fraction; largest saved post-condition allocation 741,928,448 B at teacher. This is not an independently traced instantaneous peak |
| Process-tree RSS | ≤8 GiB | Parent sampled maximum 3,047,157,760 B; worker post-condition maximum 1,545,637,888 B |
| Output | ≤2 GiB | All 15 files together 138,154,483 B |
| Free disk | ≥10 GiB | Lowest saved worker post-condition free space 410,685,374,464 B |
| Model update | Zero | Worker `optimizer_steps=0`; source has no optimizer route in M0 |
| Validation/Test | Zero new accesses/evaluations | Parent/worker counters both 0; source audit hook denies named split files. No independent OS file-access trace was recorded |

The worker generated the release tables by restoring the original `ReleaseStudent` selected state and running its graph `forward`, and generated teacher tables through the original teacher/prompt adapter. TD full selected weights were required to equal their inference-only tables. The worker's completed status means its source/checkpoint hashes, finite/shape checks, export reload comparisons and fixed masked score comparisons returned normally. The only stderr content is PyTorch's sparse invariant warning from the unmodified release graph adapter; there is no exception. Stdout printed historic hard-token cache hit flags, not new cache-cost evidence.

The runner did not save a launch-time software/driver snapshot or periodic GPU telemetry. A read-only post-run check found Python 3.10.20, PyTorch 2.11.0+cu128 (CUDA 12.8), NumPy 2.2.6, SciPy 1.15.3, psutil 7.2.2, RTX 5060 (8151 MiB) and driver 595.97. These values describe the machine at audit time and cannot certify every launch-time condition. M1 preparation must include explicit launch-time environment and telemetry capture before its separate declaration.

Export SHA256, in slot order BPR 2022/2023/2024, Full 2022/2023/2024, release 2022/2023/2024, teacher:

```text
183353d851eb94ce7ec559b306ec7380df093f4d39c1da716996fda3353c3be2
a02df545f7c2ae41b5f40580297a16297e462f0cefa751f77a5e6ac5c56a9439
b2860dbeeae2ed58191d481a923d376fe9013943cd28cfbd74040abcf37f08cc
5350c4b7d7622f8a0f20a06c0cc5419f0c052ff26b93eac885169f1baa49aa1e
e0cdbfcec96cb9c37096d9fff859cdc107231617fbcfa83a0a19a390b5fca8af
8346c53c5ae4b59834109d6b5042ddb65db288d0fefc61e9741fbeb2ca6a1cc4
85f2c3019a2274cad36f383f2b03f4901780fa7551b5fdecac2ca0d7a4287cac
79876ed9253f759aaae7e91447f0b3de47fbeb616e55da9a62a16a8b4f48db6d
69211f7070cf8f15113faa252bde124a295f0ad6d65c3b7b036856289ce98e73
a73c0158e704ff2f53c92e2c11ba0acb37102785f6486c845e5d7bbe3626daf0
```

## Interpretation, limitations and routing

M0 establishes that the declared nine formal Sports student assets and common teacher can be loaded and exported into same-shape float32 tables, and that those exports reproduce the original model tables and fixed Train-masked scores in this smoke. It does not measure a stable preparation latency, cached serving latency, update-step cost, 300-epoch cost or time to equal quality. Existing final Test quality is cited only through the saved JSON map; this smoke has no new Test access. The scripted split guard and counters support Test0 within this process; absent an OS trace they do not prove that no unrelated process opened Test. The CUDA/RSS samples support bound compliance as observed, with allocator limiting additional allocation; they are not full-resolution independent peak traces.

Record routing: this audit, the M0 row of [cost status](INNOVATION1_COST_RESULTS.md), active `TRAINING_LOG.md` and current root/experiment-family pointers are updated. Generated navigation is refreshed. M1/M2 rows remain unrun. Existing 18-cell Test matrix, initialization matrix, paper result text and gap status are unchanged because M0 adds no quality or cost outcome. Raw smoke files, old audits/manifests and `check/` are retained; no tag, merge, push or artifact overwrite.

**Single next action:** prepare a separately declared, serial M1 same-checkpoint deployment measurement with its exact command and acceptance gates for user review. Do not launch M1 or M2 from this audit.
