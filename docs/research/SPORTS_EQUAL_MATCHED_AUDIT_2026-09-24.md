# Sports equal-target matched-scale arm: outcome audit (2026-09-24)

## Identity and protocol

- User launched once with `D:\miniconda\envs\run_5060\python.exe -B tools/run_sports_sharedinit_equal_matched.py` from clean branch `codex/experiment/baby-teacher-baseline`, source commit `83bc9bfcd620b3ec94c51b20807838ce2eb54bce`. Run: 2026-09-24 12:36:52 to 17:28:01 +08:00. Report status `completed`.
- Profile `sports_student_equal_matched_sharedteacherinit_seed2022_val300_v1`, seed 2022, 300 epochs, patience 300, batch 1024, student LR 6e-5, dimension 64, AdamW weight decay 0.01. Objective is BPR + 0.4111064309069839 × (Dimage + Dtext)/2, with user distillation rates zero. No CLI profile overrides. Relative to the existing shared Full arm, only item-text rate 0.3→1.0 and alpha 0.3→0.4111064309069839 changed.
- SPORTS_CONVERTED_20260916 data and pinned teacher SHA256 `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea`; shared six-tensor SHA256 `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20`. Paired Full/BPR report SHA256 `4c2e0ba811f8085a69247a1b4339ed2e120b295097db79ed2f901404aed3ac26`. Calibration report SHA256 `b990d361fa56ae94a3b20480d0db9c1dac3a672bf20a36c2d409d66453972cc2`. Its eight fixed train-only batches matched initial raw item-gradient RMS, not each training step.
- The existing CPU verifier passed: 300 finite loss and selection series; strict earliest maximum Validation Recall@20; saved full checkpoint epoch, finite weights, resolved defaults, source fingerprints and shared initialization identity. Initial user/item hashes and initial Validation were exactly equal to the prior paired Full arm. Test matrices were structurally read by the existing loader/preflight; teacher/student Test ranking flags are both false. Initial Validation was not selected. Paper-ready eligibility remains false; selected-versus-tested checkpoint equality is not applicable because no Test evaluation occurred.

## Validation-only comparison

| Shared initialization, seed 2022, 300 epochs | Target and alpha | Best epoch (1-based) | Best Recall@20 | NDCG@20 at selected epoch | Final Recall@20 |
|---|---|---:|---:|---:|---:|
| BPR-only | alpha 0 | 286 | 0.095477873096 | 0.043805670602 | 0.095151699965 |
| Full | image:text 1:0.3; alpha 0.3 | 283 | 0.095438076853 | 0.043780603134 | 0.095060402701 |
| Equal matched | image:text 1:1; alpha 0.4111064309069839 | 299 | 0.095383454558 | 0.043722460516 | 0.095299180161 |

Equal matched best Recall@20 is `0.000054622295` below Full (relative `0.05723%`) and `0.000094418538` below BPR-only. Its best first-30 and first-120 Recall@20 were both `0.094254723855`; the first-200 best was `0.094451130021`. The best at epoch 299 is close to the fixed budget endpoint; the final epoch Recall@20 is `0.095299180161`. Low metrics are valid completed evidence, not an execution failure.

## Interpretation and limits

At the tested seed, the equal target with an initial matched raw-gradient scale did not improve the selected Validation Recall@20 or NDCG@20 over the 1:0.3 Full arm. The numerical difference is small, so this does not establish a general target ranking or a mechanism for the earlier alpha3 outcome. The calibration matched aggregate initial item-gradient RMS on eight fixed batches; optimizer updates and later gradients were not matched. The training code used the same seed and sampler, but did not persist every sampled training batch for a batch-by-batch equality audit. Window/fullscreen-dependent throughput also varied during this run, so wall-clock speed is not used for method comparison. No extra seed, extension, Test ranking, or new arm is authorized by this audit.

## Artifacts

- Run report: `exp/initialization_checks/sports_sharedteacherinit_equal_matched_seed2022_val300_v1/run.json`, SHA256 `3bc80044b5f5a99c718fb38ca02fb5eae0f7a84720a35f2513d4f3989ddb0381`.
- Manifest: `exp/runs/sports/run_manifest__2026-09-24 12_37_07.329601_sports_light_init_pid39320.json`, SHA256 `adfe536c6dc277150e90626fb2e729f456621698159367a0cdd18fdea2e9f353`.
- Full checkpoint: `Model/sports/td_distill/td_distill_full__val_test_once_v1__2026-09-24 12_37_07.329601_sports_light_init_pid39320.pth`, SHA256 `197fd1741e3d303bc41f88ba4c3a723448b7ac0db9c179c09e5927e2f4f1e836`.
- Curve: `exp/converge/sports/auto__2026-09-24 12_37_07.329601_sports_light_init_pid39320.pkl`, SHA256 `1bdcaa1f916227bcc50873199fd528998a9ed6540b355fa286aefd38df360310`.

Generated artifacts are ignored by Git and have not received a physical/off-device backup. The single next step is a read-only synthesis of the existing shared-initialization arms and the gradient diagnostics before deciding whether another experiment is warranted.
