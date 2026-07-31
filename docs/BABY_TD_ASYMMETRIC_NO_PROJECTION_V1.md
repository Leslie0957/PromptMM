# Baby Asymmetric No-Projection Directional Candidate V1

Date pinned: 2026-07-31

Profile ID: `baby_td_asymmetric_no_projection_v1`

Scope: first predeclared directional-distillation candidate against the formal
`baby_student_reference_v1` ID-only BPR anchor on audited MMRec Baby.

## Research role

This profile tests whether moderate, item-dominant semantic alignment from the
frozen multimodal teacher improves the same randomly initialized deployable ID
student. It does not add projection heads, teacher warm start, graph propagation,
modality encoders, or any additional inference state.

The five semantic settings were fixed from the archived pre-Baby research
hypothesis before this candidate was implemented. The completed Baby baseline
and teacher Test metrics did not select or refine them. Those Test results are
reference evidence only and must not be used for candidate tuning.

## Fixed identities

- Dataset: audited MMRec Baby under `data/baby/`.
- Conversion manifest SHA256:
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`.
- Evaluation protocol: `val_test_once_v1`.
- Selection metric: validation Recall@20.
- Candidate exclusion: training interactions only for validation and Test.
- Frozen teacher: `Model/baby/teacher_model_val_test_once_v1.pt`, read-only.
- Frozen teacher SHA256:
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
- Declaration commit:
  `161e72a018dc21bbebdb39e4306d374f7473e20a`.

The official evaluation-only items and existing preflight warning remain
unchanged. Duplicate modalities, split overlap, invalid data, profile overrides,
or dataset/teacher identity mismatches remain eligibility failures.

## Pinned values

| Group | Parameter | Value |
| --- | --- | ---: |
| profile | `student_profile` | `baby_td_asymmetric_no_projection_v1` |
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
| teacher | `allow_teacher_alias_overwrite` | `false` |
| student | `student_model_type` | `td_distill_no_projection` |
| student | `student_embed_size` | `64` |
| student | `student_lr` | `0.00006` |
| student | `student_weight_decay` | `0.01` |
| student | optimizer | AdamW |
| initialization | `td_init_from_teacher` | `false` |
| semantic loss | `td_distill_alpha` | `0.3` |
| semantic loss | `td_item_image_rate` | `1.0` |
| semantic loss | `td_item_text_rate` | `0.3` |
| semantic loss | `td_user_image_rate` | `0.0` |
| semantic loss | `td_user_text_rate` | `0.0` |

The sampler is unchanged from `baby_student_reference_v1`. Every optimization
batch samples 1024 distinct training users, one training-positive item per
user, and one item absent from that user's training interactions as the
negative. A formal epoch uses the same 116 independently sampled batches.

## Exact comparison delta

The candidate defaults and `baby_student_reference_v1` defaults have identical
keys and are identical outside this five-field semantic contract:

| Parameter | Reference | Candidate |
| --- | ---: | ---: |
| `td_distill_alpha` | `0.0` | `0.3` |
| `td_item_image_rate` | `0.0` | `1.0` |
| `td_item_text_rate` | `0.0` | `0.3` |
| `td_user_image_rate` | `0.0` | `0.0` |
| `td_user_text_rate` | `0.0` | `0.0` |

The two zero-valued user rates are included in the declared semantic contract.
They keep user semantic heads inactive. Profile name, scope, and source metadata
independently identify this candidate.

For each BPR batch, the maintained implementation computes:

```text
L_semantic = (1.0 * L_item_image + 0.3 * L_item_text) / 1.3
L_total = L_BPR + 0.3 * L_semantic
```

The no-projection model passes the 64-dimensional item ID embedding directly to
both active semantic comparisons. There is no trainable projection state.

## Paper-ready eligibility

The Baby eligibility gate recognizes this exact candidate metadata and the
existing baseline metadata. It does not accept arbitrary student profiles.
Formal candidate eligibility additionally requires:

- empty dataset and student profile override maps;
- the exact audited dataset and frozen teacher identities;
- `val_test_once_v1`, validation Recall@20 selection, and `train_only`
  candidate exclusion;
- no smoke batch cap and `run_final_test=true`;
- dataset preflight enabled with non-duplicate modalities;
- no teacher training or shared-alias overwrite.

Any undeclared CLI change to a pinned profile field creates a student profile
override and a paper-ready blocker. The candidate profile must carry its
semantic settings, so formal commands must not repeat them as CLI overrides.

## Validation-only smoke gate

The first execution stage, after a separate pending declaration and clean
source commit, is a one-epoch, one-batch, validation-only smoke:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_td_asymmetric_no_projection_v1 `
  --gpu_id 0 `
  --if_train_teacher false `
  --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt `
  --epoch 1 `
  --smoke_train_batches 1 `
  --run_final_test false
```

The smoke is intentionally non-paper-ready because of its declared cap,
profile overrides, and disabled final Test. It must validate the frozen teacher,
exercise one optimization batch and validation ranking, restore the selected
checkpoint, and record no teacher or student Test result. Merely documenting
this command does not authorize it.

## Formal run gate

A formal run requires a separately committed declaration after the validation-
only smoke is completed and recorded:

```powershell
D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py `
  --dataset baby `
  --student_profile baby_td_asymmetric_no_projection_v1 `
  --gpu_id 0 `
  --if_train_teacher false `
  --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt `
  --run_final_test true
```

The formal run must launch from clean committed source with empty profile
override maps. It trains on `train_mat`, selects and early-stops only with
validation Recall@20, restores the validation-best checkpoint, and evaluates
the student Test split once after restoration. Test quality is not a tuning or
technical-acceptance condition.

## Artifact requirements

Every future smoke or formal execution must use a new timestamp/PID run name and
isolated paths for:

- dataset preflight report and structured run manifest under `exp/runs/baby/`;
- convergence record under `exp/converge/baby/`;
- full training checkpoint and inference-only checkpoint under
  `Model/baby/td_distill/`;
- raw log under `logs/`.

The full checkpoint may contain optimizer state, but the inference-only export
must contain exactly the user/item ID embedding tensors and deployment metadata
(`embedding_dim`, user/item counts, and no-projection variant identity). It must
not contain the teacher, prompt module, modality features, semantic caches,
projection heads, graph state, or optimizer state. All generated artifacts stay
outside normal Git history and must be fingerprinted in the run outcome.
