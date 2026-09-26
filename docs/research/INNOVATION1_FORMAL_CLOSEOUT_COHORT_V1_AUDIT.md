# Innovation1 `innovation1_fixed_v1` failed-at-preflight audit

Date: 2026-09-26. This audits the user's single manual launch from clean commit `c96b8885aa045adbb9bd7e8550a5781dfdd4afab` on `codex/experiment/baby-teacher-baseline`. It does not certify any new training, Validation replay or Test result. The original [formal declaration](../../TRAINING_LOG.md) and [fixed protocol](INNOVATION1_FORMAL_CLOSEOUT_PROTOCOL_2026-09-26.md) remain unchanged.

## Attempt identity and preserved artifacts

- Command: `& 'D:\miniconda\envs\run_5060\python.exe' -B 'D:\Download\PromptMM\tools\run_innovation1_formal_closeout.py' --conditions-confirmed`.
- Shared asset anchor: [selected-source ledger](INNOVATION1_REUSE_ASSETS_2026-09-26.json), SHA256 `b34ee802a104fdfc926479e5123da7aaa064e3f615940fa48c6fd6983bfb717c`. Baby planned delta: seeds 2022/2023/2024, release student lr6e-5, cap1000/patience7 and strict earliest post-epoch Validation Recall@20 selection. No runtime parameters were resolved by a Baby training report because its body never ran.
- Preserved ignored [batch report](../../exp/formal_closeout_cohort/innovation1_fixed_v1/batch.json), SHA256 `0ce638d4a963de18e521730ee67766da8e8e5643ec7d71b8a140de7886a6fbef`, and [failed B3 preflight](../../exp/formal_closeout_preflight/b3_2022/report.json), SHA256 `6bf4cc712f7185b1edd99c58127f3f5763493a0bd7dfba071cb3c45c392a6fed`. Batch ran from 14:34:56 to 14:34:59 Asia/Shanghai. These ignored artifacts have not been edited, moved or deleted; they are not protected by Git commits.
- The three `train_b3_2022/2023/2024.log` files in the cohort directory are each zero bytes. Their subprocesses exited 0 in approximately 0.09 s, but no `exp/promptmm_release_baby/` directory, `report.json` or checkpoint was created. The batch's `completed` label on these three child entries records exit code only and is **false as a training completion assertion**. No Baby student training, optimizer steps or Validation selection occurred.
- Step 4 `preflight_b3_2022` exited 1 when its source `report.json` could not be opened. The preflight report was saved with `status=failed`, `test_access_started=false`, `test_split_loaded=false`, zero student and teacher Test evaluations. It contains Python 3.10.20, Torch 2.11.0+cu128, CUDA 12.8 and RTX 5060 environment metadata. The cohort stopped immediately; no other preflight and no `final_test_*` step started. `exp/formal_closeout_eval/` does not exist. New student Test attempts **0/12**, teacher Test evaluations **0**; selected-versus-tested checkpoint and Test metrics are inapplicable.

| Slots | Training / selected Validation | New final student Test |
|---|---|---|
| B3 2022 | Trainer no-op; preflight failed before source report load; no selected metric | 0 attempts; no metric |
| B3 2023, 2024 | Trainer no-op; preflights never launched; no selected metric | 0 attempts each; no metric |
| S1 2022, 2023, 2024 | Existing Sports selected assets remain; new preflights never launched | 0 attempts each; no metric |
| S2 2022, 2023, 2024 | Existing Sports selected assets remain; new preflights never launched | 0 attempts each; no metric |
| S3 2022, 2023, 2024 | Existing Sports selected assets remain; new preflights never launched | 0 attempts each; no metric |

## Cause and interpretation

The Baby trainer defined `main()` but its file ended at the function return without `if __name__ == '__main__'`. Thus direct `python -B codes/promptmm_release_baby_formal.py ...` parsed nothing, trained nothing and exited successfully. The earlier synthetic test called `baby.main()` directly, so it could not detect this entry-point fault. The serial runner accepted exit 0 without requiring the declared report/checkpoint; the first evaluator preflight exposed the missing artifact. This is an execution-flow failure, not a low-quality model result or a failed Validation parity comparison. All B3 Test cells and nine Sports Test cells remain missing; historical Baby B1/B2 results and Sports Validation records are unaffected.

## Repair and verification

The repair adds the Baby direct-file guard and verifies each child report before the runner advances: Baby `validation_completed` plus seed/Test-zero/checkpoint hash, preflight `passed` plus Test-zero, and final Test `completed` plus one student attempt/evaluation, zero teacher Test and selected/evaluated hash equality. It also adds a direct subprocess `--describe` regression and synthetic exit-0/no-report and report-family checks. Ten no-Test tests passed; no real data/model Test split, formal training or real Validation replay was run during repair. This source change has its own commit; the old launch remains anchored to `c96b888`.

## Record routing and next stage

Updated: active log, this audit, root/docs/family navigation, formal matrix status and launch warning. Thesis gap and paper result prose have no new metrics or coverage, so their scientific statements remain unchanged. The six generated metadata catalogs are refreshed after manual edits. Original batch/preflight outputs, pending declaration and existing checkpoints remain untouched. A future recovery requires a **new cohort identity, exclusive artifact paths, clean committed repair source, amended declaration and explicit user authorization**. The old command cannot be rerun. The next concrete step is to prepare that separately authorized recovery declaration and its collision-free command; it does not imply permission to launch it.
