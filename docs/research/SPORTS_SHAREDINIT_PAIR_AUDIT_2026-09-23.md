# Sports shared-initialization full/BPR audit — 2026-09-23

Both runs completed and passed CPU-only artifact re-verification. No training or new Validation/Test ranking was performed in this audit.

## Protocol and identity

Manual command: `& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sharedinit_pair.py`. Exact child commands are preserved in batch.json and the preceding formal declaration. Clean launch 0b53f1a79f06ec8b85a302a9ce343813bef77766 on codex/experiment/baby-teacher-baseline. Seed2022,300epochs,patience300,lr6e-5,batch1024,dim64,AdamW decay0.01; shared Sports data/epoch37 teacher anchor unchanged. Only behavior argument difference is alpha0.3 versus0; profile names also differ. Initial vectors and all initial validation metrics exactly match. Initial metrics are excluded from checkpoint selection.

Both initial Recall20=0.09418449519085098, NDCG20=0.043320782868578635. Shared asset SHA256=e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20. BPR distillation curve is identically zero and total equals BPR. Both curves contain300 finite values; selected checkpoint metadata, finite weights, earliest maximum Recall20 and current source fingerprints reverified. Both teacher/student final Test flags false; existing Test structural reads remain. Selected-versus-tested checkpoint equality is not applicable. paper_ready_eligible remains false.

| Arm | Best epoch (1-based) | Recall20 | NDCG20 at selected epoch | Last Recall20 | Last20 mean Recall20 |
|---|---:|---:|---:|---:|---:|
| sports_student_full_sharedteacherinit_seed2022_val300_v1 | 283 | 0.095438076853 | 0.043780603134 | 0.095060402701 | 0.095234881917 |
| sports_student_bpr_sharedteacherinit_seed2022_val300_v1 | 286 | 0.095477873096 | 0.043805670602 | 0.095151699965 | 0.095232970136 |

Full minus BPR: Recall20 -0.000039796243 (-0.041681% relative); NDCG20 -0.000025067468 (-0.057224% relative). Full versus initial Recall gain 1.330985%; BPR gain 1.373239%.

## Interpretation

This seed/configuration does not demonstrate incremental benefit from directional distillation after teacher initialization. BPR alone reproduces the high validation performance; neither statistical equivalence nor a significant BPR advantage is established. Earlier teacher-initialized full superiority over release cannot be credited to directional distillation. Cold-initialized ablation evidence remains valid in its own initialization regime; do not generalize this warm-start outcome to all regimes. Full and BPR last20 means are nearly equal. Both prefix120 best Recall20=0.09425472385538271, with later gains and best epochs283/286:300 epochs exposed late improvement, but this is not proof of convergence or grounds for automatic extension.

Observed serial batch elapsed about8h17m; full about5h04m, BPR about3h13m. Uncontrolled system load and legacy evaluator preclude attributing this difference to method overhead. No controlled efficiency conclusion.

Single recommended next stage: prepare a zero-update gradient diagnostic on the pinned initialization and identical fixed training batches, measuring BPR versus alpha-weighted image/text gradient norms, cosine alignment and expected update contribution on the shared embedding parameters. Verify effective objective scaling before proposing stronger weights, gates, more seeds or longer runs. No diagnostic was launched or prepared as an executable in this audit.

## Artifacts

Batch: D:\Download\PromptMM\exp\initialization_checks\sports_sharedteacherinit_pair_seed2022_val300_v1\batch.json; SHA256 4c2e0ba811f8085a69247a1b4339ed2e120b295097db79ed2f901404aed3ac26

Arm: sports_student_full_sharedteacherinit_seed2022_val300_v1
- Manifest: D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-23 11_07_51.352217_sports_light_init_pid2784.json; SHA256 825029a48243ac8780b366f088a754d0f9f3f0738e6e69449aaaaf3cc05530fa
- Checkpoint: D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-23 11_07_51.352217_sports_light_init_pid2784.pth; SHA256 bd651d78ab8853ba987c407a23cd485581ccf319a1b6c9202c5413091fc6879c
- Curve: D:\Download\PromptMM\exp\converge\sports\auto__2026-09-23 11_07_51.352217_sports_light_init_pid2784.pkl; SHA256 78ec30cc9220da995d1de9280e905d74228d21681e5c75166f73b94599a35453

Arm: sports_student_bpr_sharedteacherinit_seed2022_val300_v1
- Manifest: D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-23 16_11_54.129842_sports_light_init_pid36040.json; SHA256 300b02c678b9e6e2c5988d79aaa7baa7cd2e2e517b17b9ccdd9b68751edd66a9
- Checkpoint: D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-23 16_11_54.129842_sports_light_init_pid36040.pth; SHA256 2c35884bb4d1552fb346689732e3dcda9cbb6f59d50e8113495d62ae1e396e18
- Curve: D:\Download\PromptMM\exp\converge\sports\auto__2026-09-23 16_11_54.129842_sports_light_init_pid36040.pkl; SHA256 134255732027ffc7045bb39630571c0adee5f78f1037afcdf0ecaf5050b1ce28

Historical failed attempt and original full remain preserved. Baby seed2023 remains accepted by audit exception with original manifest false; Innovation2 undecided. Ignored raw artifacts are not backed up by this documentation commit.
