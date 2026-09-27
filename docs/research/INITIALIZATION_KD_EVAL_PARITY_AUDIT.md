# Evaluation parity v1 failure audit / Windows publication repair

2026-09-27. Status: **failed**, partial first reference pass only. No accepted parity result or timing comparison. Source `06edf407a79be83228e791538a7f6d3ba80abd45`, branch `codex/experiment/baby-teacher-baseline`. User executed the sole command in [v1 declaration](INITIALIZATION_KD_EVAL_PARITY_LAUNCH_2026-09-27.md); that historical declaration is consumed, not current launch authority.

## Observed failure

`worker.log` records `PermissionError: [WinError 5]` at `_json`: replacing `telemetry.json.tmp` with `telemetry.json`, during `compare_evaluators`' reference callback. Worker exit1; supervisor failed. No report, acceptance or either saved ranking array exists. This is an I/O execution failure, not evidence of unequal rankings, bad model quality or out-of-memory. No complete reference/fast time or validation metrics can be reported.

The last published telemetry records worker-relative wall38.093s, RSS964091904 bytes and CUDA allocator75497472 bytes; preserved temporary telemetry is wall39.093s, RSS964149248 bytes. These are samples, not full-run elapsed time or certified peaks. Both are within configured caps. The parent cap remains900s, CUDA2GiB/RSS4GiB/output256MiB/free disk4GiB; no cap increase.

The parent periodically opens telemetry while the worker replaces it. A synthetic Windows test holding the old target open reproduces replacement denial, then verifies publication succeeds after releasing that reader. This validates a plausible race in this implementation; logs alone cannot identify which process held the original file (external file scanners are also possible).

## Identity and protocol

Fixed v1 config and inherited smoke/common configuration remain unchanged. Input anchor remains completed smoke T1 final checkpoint SHA `ec18349231ded3a4b1dcd400bff31b80a2e23b1c819519ad6c6fd0eb64ddbcc7`, and pinned Train/Val identities in the declaration. Manifest records launch source/config; the worker reached reference evaluation after input gates. This audit reads only existing small logs/JSON metadata, not input checkpoint or split bytes. No rehash of unchanged large assets.

Training updates0: evaluation entry creates no optimizer. Test reads/ranking0 according to the guarded entry and observed failure path; no independent OS-level file-access trace exists. No selected/tested checkpoint or teacher Test event applies. Partial reference Validation access occurred in the user's run; no new real evaluation, model load, split read or GPU work occurred in this repair task.

Failed output remains in `exp/initialization_kd_interaction/sports_init_kd_eval_parity_seed2022_v1/`, including `.tmp`; it is not overwritten, resumed or retried. Fingerprints:

| File | Bytes | SHA256 |
|---|---:|---|
| `failure.json` | 98 | `ed8722b127307e1b07dfbd774ccc7fb1340095622f23fd8fc21fa4f7e28abcdd` |
| `launch_manifest.json` | 7104 | `b688308c24fcef067e2ba232a76ee7f9081d19137461e1e435002b8a6ecf5e28` |
| `supervisor.json` | 85 | `db5cc5c53abd382c3070605e98cdfb79658b362a1a3e63e0205e87cb1b9a7769` |
| `telemetry.json` | 184 | `77877a04e4112b3451442f9bcf41e2334541dd05f5d757fa4c90c84288591c4f` |
| `telemetry.json.tmp` | 184 | `36996c60e8bcc06c3c580cc39e6c3a0cc030c240600c10b79230eb8869bdccab` |
| `worker.log` | 1623 | `9336d6be4004e1df21b0fa16d48778b5f7563d11bb75806811266bb0dfd9c7a6` |
| `worker_claim.json` | 34 | `61e253643cd099aea97b02113cf4e2b40834422fdf1b68caf3d4188be797514d` |

## Repair and verification

Shared `_json` retains same-directory temporary write and atomic replacement. Only Windows errors5/32/33 trigger up to10 further publication attempts, each after50ms (0.5s requested wait total). The payload is serialized once. Permanent denial propagates and preserves old target/temp bytes; unrelated errors propagate immediately. No worker, model, optimizer, evaluation or experiment retry is added. Ranking/scoring/configuration and scientific thresholds are untouched. The parent budget supervision continues during publication waits.

24 synthetic CPU tests passed, including real Windows reader-lock reproduction/release, persistent denial bound and byte preservation, unrelated-error propagation, existing evaluator exactness/artifact corruption/launch/resource checks. Synthetic smoke-shaped test messages are fixtures, not a real smoke run. Real GPU parity remains unverified.

## Routing and single next step

Updated this audit, active TRAINING_LOG current state plus failure/fix outcome, root README and experiment-family README; regenerated six navigation files. `docs/README.md` has no parity-specific stale status and needs no change. Four-arm matrix/old18-cell matrix/paper/gaps unchanged: no new scientific result. Historical declaration/config/log entries and all original datasets, caches, checkpoints, manifests, raw logs and user `check/` remain in place. No tag, bundle, push or merge.

Next step: separately authorize and declare one user-manual parity check in a new namespace using the repaired committed source. Do not run the consumed v1 command, delete its directory, retry automatically, or start four arms/P1b/P2/gating. There is no verified new launch command in this task.
