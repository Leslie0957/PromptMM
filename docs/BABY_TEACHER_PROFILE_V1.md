# Baby Teacher Reference Profile V1

Date pinned: 2026-07-30

Profile ID: `baby_teacher_reference_v1`

Scope: teacher-only reference baseline on the audited MMRec Baby split.

## Why this is a reference profile

PromptMM's official experiments cover Netflix, TikTok, and Amazon Electronics,
not Baby. Therefore, there is no official PromptMM Baby optimum to copy. This
profile is fixed before inspecting any formal Baby test result and is intended
to establish a reproducible teacher baseline, not to claim validation-optimal
hyperparameters.

The repository's dormant Baby parser block supplies the Baby-specific core:
batch size 1024, patience 7, model fusion 0.55, a two-entry propagation
configuration, 64-dimensional embeddings, dropout 0.2, and learning rate
0.00055. Prompt fields absent from that old block use the existing PromptMM
profile whose teacher structure matches the same `model_cat_rate=0.55` and
`weight_size=[64, 64]` family. The active teacher AdamW decay remains `0.001`,
which is shared by all three active PromptMM parser blocks. The dormant Baby
`weight_decay=0.0001` was labeled for a different, inactive optimizer and is not
misapplied here. No test metric was used in this choice.

## Pinned values

| Group | Parameter | Value |
| --- | --- | ---: |
| run | `seed` | `2022` |
| run | `sparse` | `1` |
| evaluation | `Ks` | `[10, 20, 40, 50]` |
| evaluation | `test_flag` | `part` |
| run | `cf_model` | `light_init` |
| run | `batch_size` | `1024` |
| run | `epoch` | `1000` |
| run | `early_stopping_patience` | `7` |
| optimization | `regs` | `[0.00001, 0.00001, 0.01]` |
| teacher | `embed_size` | `64` |
| teacher | `weight_size` | `[64, 64]` |
| teacher | `mess_dropout` | `[0.1, 0.1]` |
| teacher | `model_cat_rate` | `0.55` |
| teacher | `drop_rate` | `0.2` |
| teacher | `lr` | `0.00055` |
| teacher | `t_weight_decay` | `0.001` |
| teacher | `t_feat_mf_rate` | `1.0` |
| teacher | `feat_reg_decay` | `0.00001` |
| teacher | `layers` | `1` |
| prompt | `hard_token_type` | `pca` |
| prompt | `hard_token_seed` | `2022` |
| prompt | `soft_token_rate` | `0.005` |
| prompt | `feat_soft_token_rate` | `1.0` |
| prompt | `t_prompt_rate1` | `100.0` |
| prompt | `t_prompt_rate2` | `1.0` |
| prompt | `t_prompt_rate3` | `1.0` |
| prompt | `prompt_dropout` | `0.0` |

Other resolved values remain visible in every run manifest. CLI values still
take precedence, but any semantic difference from this table is recorded under
`dataset_config_overrides` and prevents the run from being labeled as the fixed
Baby reference baseline.

In the current teacher-only implementation, only the first `regs` value is
consumed; `mess_dropout` and `t_prompt_rate2/3` are parsed and pinned but do not
currently contribute to the teacher forward/loss path. They remain explicit so
that later code activation cannot silently change the profile's identity.

## Formal baseline rule

Use `val_test_once_v1`, dataset preflight, non-duplicate modalities, seed 2022,
and `smoke_train_batches=0`. Select the teacher only by validation Recall@20,
restore the selected checkpoint, then evaluate the test split once. Under v1,
both validation and test candidates exclude training interactions only, matching
the official PromptMM/MMRec convention documented in the masking audit.
`test_flag=part` omits the otherwise expensive AUC calculation; Recall, NDCG,
Precision, and Hit Ratio still use full-item candidate ranking.

## Artifact safety

Each run uses a microsecond-qualified run name and keeps its best teacher under
`Model/baby/runs/`. The shared `teacher_model_val_test_once_v1.pt` alias is not
written during training: it is published atomically only after the best
validation checkpoint has been restored and the final test has completed.
Formal training refuses to start if that alias already exists unless
`--allow_teacher_alias_overwrite true` is explicitly supplied after review.
Once a run manifest has been initialized, model/CUDA initialization and training
share the same failure boundary, so an exception records `status=failed` rather
than leaving the run indefinitely marked `initialized`.
