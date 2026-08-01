# Baby Seed-2024 Matched Replication Profiles V1

Date pinned: 2026-08-01

Declaration commit:
`f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c`

This document pins the third matched Baby student replication pair before
either seed-2024 arm is run:

- baseline: `baby_student_reference_seed2024_v1`;
- candidate: `baby_td_asymmetric_no_projection_seed2024_v1`.

The two profiles are implemented together so evidence from the seed-2024
baseline cannot change the already fixed candidate. This implementation does
not authorize profile smoke, training, validation ranking, or Test evaluation.

## Cross-seed contract

Each seed-2024 profile has the same default keys and values as both its
seed-2022 and seed-2023 counterparts. The only changed default is the student
training and maintained pairwise-sampling `seed`:

| Method | Seed-2022 profile | Seed-2023 profile | Seed-2024 profile | Only default delta |
| --- | --- | --- | --- | ---: |
| ID-only BPR baseline | `baby_student_reference_v1` | `baby_student_reference_seed2023_v1` | `baby_student_reference_seed2024_v1` | `seed: 2022/2023 -> 2024` |
| asymmetric candidate | `baby_td_asymmetric_no_projection_v1` | `baby_td_asymmetric_no_projection_seed2023_v1` | `baby_td_asymmetric_no_projection_seed2024_v1` | `seed: 2022/2023 -> 2024` |

`hard_token_seed=2022` does not change. It belongs to the frozen teacher
preprocessing and cache identity, while `seed=2024` controls student random
initialization and the maintained pairwise sampling stream.

Formal commands must not pass `--seed` or repeat profile-owned values. The
selected profile carries seed `2024`, and both `dataset_config_overrides` and
`student_config_overrides` must resolve to empty maps.

## Frozen matched controls

Both seed-2024 arms retain:

- audited MMRec Baby, official split identities, and `val_test_once_v1`;
- read-only teacher `Model/baby/teacher_model_val_test_once_v1.pt`, SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
- `hard_token_seed=2022`, existing PCA caches, and unchanged modality features;
- `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`;
- `td_distill_no_projection`, embedding dimension `64`, and random
  initialization with `td_init_from_teacher=false`;
- AdamW `student_lr=0.00006`, weight decay `0.01`, batch size `1024`, maximum
  `epoch=1000`, early-stopping patience `7`, and `smoke_train_batches=0`;
- unchanged `data_generator.sample()` pairwise BPR sampling and all `116`
  batches per formal epoch;
- validation Recall@20 selection, `Ks=[10,20,40,50]`, `test_flag=part`, and
  `train_only` candidate exclusion;
- inference export containing only user/item ID embeddings and deployment
  metadata; no efficiency benchmark.

The baseline retains `td_distill_alpha=0.0` and all four component rates at
`0.0`. The candidate retains `td_distill_alpha=0.3`, item-image `1.0`,
item-text `0.3`, user-image `0.0`, and user-text `0.0`. Therefore the actual
value differences between the matched arms remain alpha, item-image, and
item-text only.

## Execution order

Execution remains sequential and requires separate committed declarations:

1. Declare, commit, and later run the seed-2024 baseline alone.
2. Commit its completed or failed outcome without changing the candidate.
3. Only then declare, commit, and later run the seed-2024 candidate alone.
4. After both outcomes, summarize matched seeds 2022/2023/2024 using means and
   sample standard deviations without newly accessing any dataset split.

The two arms must not launch concurrently or in one task. A finite low or
reversed result remains valid evidence and cannot tune, cancel, or alter the
candidate.

## Future formal commands

After a separate run declaration and explicit authorization, the baseline
command is:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_student_reference_seed2024_v1 `
  --gpu_id 0 `
  --if_train_teacher false `
  --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt `
  --run_final_test true
```

Only after the baseline outcome commit, a separately declared candidate may
use:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_td_asymmetric_no_projection_seed2024_v1 `
  --gpu_id 0 `
  --if_train_teacher false `
  --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt `
  --run_final_test true
```

Each formal run trains on `train_mat`, selects and early-stops only on
validation Recall@20, restores the validation-best checkpoint, and then
performs exactly one final student Test evaluation. Merely documenting these
commands does not authorize either execution.
