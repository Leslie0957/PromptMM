# Baby Item-Image-Only Ablation Profiles V1

Date implemented: 2026-08-02

Declaration commit:
`5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`

This document pins the implementation contract for three matched Baby
item-image-only profiles. It does not declare or authorize a smoke, formal
training, Validation, Test, or efficiency run.

## Canonical identities and source profiles

| Seed | Image-only profile | Scope | Source identity | Copied full candidate |
| ---: | --- | --- | --- | --- |
| 2022 | `baby_td_item_image_only_no_projection_seed2022_v1` | `student_ablation` | `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022` | `baby_td_asymmetric_no_projection_v1` |
| 2023 | `baby_td_item_image_only_no_projection_seed2023_v1` | `student_ablation` | `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2023` | `baby_td_asymmetric_no_projection_seed2023_v1` |
| 2024 | `baby_td_item_image_only_no_projection_seed2024_v1` | `student_ablation` | `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2024` | `baby_td_asymmetric_no_projection_seed2024_v1` |

Each defaults map is an independent copy of its corresponding full candidate.
Its key set is identical, and the only value change is:

```text
td_item_text_rate: 0.3 -> 0.0
```

No existing baseline or full-candidate defaults map changes. The canonical
sorted compact-JSON defaults SHA256 values are:

| Seed | Full candidate | Image-only |
| ---: | --- | --- |
| 2022 | `018534ec9cedfe4c1d4f5fc1d23552173d3f8a33843c3e71be38f69ea8c46113` | `0669b57e57a400b2eb1eda1ff47970eb6929894b2affab12003e465e8f3418f6` |
| 2023 | `1c4c89f253d3adcd167211481dd599591012b8881c9c590c68a6e55d797fe61c` | `2ab921262cc34c4b35c817a200267ab1047200c184f7f8ee766d93f80b5ecf64` |
| 2024 | `f7ed2a940a5547afdfd3d242c6dc1832dd7e447b2824aaf633b5dd94c3bead6e` | `67b274963044d9ca7e05bad0a87f7c4fe594a69008493844beea9d9b284ca861` |

## Frozen controls

The three profiles retain their respective student initialization, training,
and maintained pairwise-sampling seeds `2022`, `2023`, and `2024`.
`hard_token_seed=2022` remains dataset-owned and identical for every profile.

All three retain:

- `td_distill_alpha=0.3`, `td_item_image_rate=1.0`,
  `td_item_text_rate=0.0`, and both user-side rates `0.0`;
- random initialization, no student checkpoint load,
  `td_init_from_teacher=false`, `td_distill_no_projection`, embedding
  dimension `64`, and no teacher warm start;
- AdamW with `student_lr=0.00006`, student weight decay `0.01`, batch size
  `1024`, maximum `epoch=1000`, early-stopping patience `7`, and
  `smoke_train_batches=0`;
- unchanged pairwise sampling and all `116` batches per completed formal
  epoch;
- audited Baby data, official splits, PCA/cache identities, duplicate-modality
  policy `error`, and the frozen read-only teacher checkpoint
  `Model/baby/teacher_model_val_test_once_v1.pt`;
- `val_test_once_v1`, selection and early stopping by Validation Recall@20,
  `train_only` candidate exclusion, restoration of the Validation-best
  checkpoint, and exactly one final student Test evaluation only in a later
  separately declared formal run;
- inference export containing only user/item ID embeddings and dimension,
  count, and no-projection deployment metadata. Teacher, modality, prompt,
  semantic cache, graph, projection, and optimizer state remain excluded.

Canonical invocations must not repeat any profile-owned default on the CLI.
Static parser resolution must produce the matching student seed,
`hard_token_seed=2022`, `dataset_config_overrides={}`,
`student_config_overrides={}`, and `paper-ready blockers=[]`. Unknown profile
names, mismatched scope/source identity, and any pinned-value override remain
blocked.

## Interpretation boundary

The maintained semantic loss divides the weighted sum by the sum of active
rates. Consequently, the compared total objectives are:

```text
image-only: L_BPR + 0.3 * L_item_image
full:       L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3
```

This ablation holds the total distillation coefficient alpha at `0.3`, but it
does not hold the effective coefficient on `L_item_image` fixed. That
coefficient is `0.3` for image-only and `0.3 / 1.3` for the full mixture.
Therefore a later paired difference describes switching between these two
normalized supervision mixtures. It is not the pure marginal causal effect of
text under a fixed image weight.

## Future execution order

Every run requires a separate committed formal-run declaration. The fixed
order is seed `2022`, then `2023`, then `2024`; each run and outcome must be
completed separately. A low or direction-reversed result remains valid
evidence and cannot change, cancel, or reorder a later profile.

After all three image-only outcomes are committed, compare them with the
existing same-seed full-candidate outcomes. Report each arm by seed, arithmetic
means, `ddof=1` sample standard deviations, and paired deltas defined as
`full candidate - image-only`. This summary must use stored outcomes only.

The future canonical commands are documentation, not run authorization:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_td_item_image_only_no_projection_seed2022_v1 `
  --gpu_id 0

D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_td_item_image_only_no_projection_seed2023_v1 `
  --gpu_id 0

D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_td_item_image_only_no_projection_seed2024_v1 `
  --gpu_id 0
```

Implementation verification is static and synthetic only. It must not import
or execute `codes/main_mmlight.py`, load Baby data, or access Validation or
Test.
