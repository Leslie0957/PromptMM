# Baby Student Reference Protocol V1

Date pinned: 2026-07-30

Profile ID: `baby_student_reference_v1`

Scope: first paper-ready ID-only student reference on the audited MMRec Baby
split. This profile defines the baseline arm, not the proposed distillation arm.

## Research role

The reference is a randomly initialized user/item ID embedding model optimized
with BPR. It establishes how much recommendation quality is available without
semantic transfer, teacher warm start, graph propagation, modality input, or
projection heads. Later directional-distillation runs must improve on this
anchor under the same student capacity and optimization budget.

The implementation reuses the maintained `td_distill_no_projection` path with
all semantic weights disabled. That path is used because the older generic
`lightgcn` student branch still activates legacy KD losses and is therefore not
a clean BPR-only control. With the settings below, the trainable and exported
student state contains only user and item ID embeddings.

No Baby student test metric was inspected when this profile was selected.

## Fixed identities

- Dataset: audited MMRec Baby under `data/baby/`.
- Conversion manifest SHA256:
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`.
- Evaluation protocol: `val_test_once_v1`.
- Selection metric: validation Recall@20.
- Candidate exclusion: training interactions only for both validation and test.
- Teacher input: `Model/baby/teacher_model_val_test_once_v1.pt`, read-only.
- Teacher checkpoint SHA256:
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.

Dataset preflight is expected to report a documented cold-item warning for
official evaluation-only item IDs `240`, `1212`, and `6115`: `11` validation
interactions and `7` test interactions, with no cold users. The official rows
remain retained. This warning is not a protocol failure; duplicate modalities,
split overlap, invalid values, or identity mismatches are failures.

The teacher is loaded to validate the frozen checkpoint contract and the
semantic dimension, but it contributes no initialization or loss to this
baseline. It must never be retrained or overwritten by a student run.

## Pinned values

| Group | Parameter | Value |
| --- | --- | ---: |
| profile | `student_profile` | `baby_student_reference_v1` |
| run | `seed` | `2022` |
| run | `batch_size` | `1024` |
| run | `epoch` | `1000` maximum |
| run | `early_stopping_patience` | `7` |
| evaluation | `eval_protocol` | `val_test_once_v1` |
| evaluation | `Ks` | `[10, 20, 40, 50]` |
| evaluation | `test_flag` | `part` |
| evaluation | `dataset_preflight` | `true` |
| evaluation | `duplicate_modalities_policy` | `error` |
| evaluation | `run_final_test` | `true` for formal run |
| teacher | `if_train_teacher` | `false` |
| teacher | `teacher_only` | `false` |
| student | `student_model_type` | `td_distill_no_projection` |
| student | `student_embed_size` | `64` |
| student | `student_lr` | `0.00006` |
| student | `student_weight_decay` | `0.01` |
| student | optimizer | AdamW |
| initialization | `td_init_from_teacher` | `false` |
| semantic loss | `td_distill_alpha` | `0.0` |
| semantic loss | `td_item_image_rate` | `0.0` |
| semantic loss | `td_item_text_rate` | `0.0` |
| semantic loss | `td_user_image_rate` | `0.0` |
| semantic loss | `td_user_text_rate` | `0.0` |

The `6e-5` learning rate is a predeclared transfer hypothesis from the old
exploratory line, not a Baby-tuned optimum. It is fixed before the first Baby
student run. Any later learning-rate study requires a new profile ID and must
select only on validation metrics.

`student_reg_rate`, `student_n_layers`, the legacy KD weights, and modality
parameters outside the table do not participate in this execution path. The
resolved argument namespace is still recorded in every run manifest.

## Non-formal smoke gate

Run a one-epoch, one-batch smoke only after the protocol source is committed:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_student_reference_v1 `
  --epoch 1 `
  --smoke_train_batches 1 `
  --run_final_test false
```

This smoke is intentionally not paper-ready. It must:

- pass dataset and frozen-teacher checkpoint validation;
- initialize a 64-dimensional no-projection ID-only student;
- log that teacher warm start and semantic loss are disabled;
- complete one BPR optimization batch and validation evaluation;
- restore the validation-selected checkpoint;
- record `final_test_performed=false` and no final-test metrics.

The runner rejects a capped smoke when `run_final_test=true`. Neither teacher
nor student may be evaluated on `test_mat` during the smoke.

## Formal reference run

After the smoke passes and its outcome is recorded, declare the exact committed
source and launch the uncapped profile without parameter overrides:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_student_reference_v1
```

The formal run must start from a clean committed tree. It selects and
early-stops only on validation Recall@20, restores the best validation
checkpoint, and evaluates the student test split exactly once. A nonempty
`student_config_overrides` field makes the run ineligible for this reference
identity.

## Comparison contract

The first proposed directional-distillation candidate must keep these values
identical to the reference:

- dataset, split hashes, candidate exclusion, teacher checkpoint, and seed;
- 64-dimensional no-projection user/item ID student;
- random initialization with no teacher warm start;
- batch size, maximum epochs, patience, optimizer, learning rate, weight decay,
  negative sampler, and validation selection rule.

Only a separately predeclared semantic-loss profile may change
`td_distill_alpha` and the four component rates. The initial method hypothesis
to test after the reference is the old asymmetric item-dominant direction, but
its Baby values must be declared before execution and must not be chosen from
Baby test results.

## Acceptance and artifacts

The reference is accepted as a valid baseline when the formal run:

- resolves the profile with no overrides or paper-ready blockers;
- passes data and teacher identity checks;
- produces finite training and validation values;
- restores a validation-selected checkpoint;
- performs one final test evaluation after restoration;
- records the full and inference-only student checkpoints, convergence record,
  run manifest, logs, and hashes in `TRAINING_LOG.md`.

Acceptance is about protocol validity, not a minimum quality threshold. The
quality result becomes the comparison anchor even if it is weak. Generated
checkpoints, manifests, and raw logs remain outside normal Git history.
