# Final-Test Candidate Mask Audit

Date: 2026-07-30

## Question

When a model is selected on a validation split and is then evaluated once on a
held-out test split, should validation interactions be excluded from the final
test ranking candidates as known positive history?

## Local behavior before this audit

`codes/utility/batch_test.py` selected validation or test positives according to
`is_val`, but constructed candidates as `all_items - train_items` for both
stages. Therefore, `val_test_once_v1` had the following candidate policy:

- validation: exclude training interactions;
- final test: exclude training interactions only.

The existing `val_test_once_v1` name and behavior must remain unchanged because
earlier Amazon and Baby smoke runs and their checkpoints already use it.

## External implementation evidence

There are two established conventions rather than one universal implementation:

- The official [MMRec quick-start implementation](https://github.com/enoche/MMRec/blob/master/src/utils/quick_start.py)
  passes only the training dataset as additional history to both validation and
  test loaders. Its [evaluation loader](https://github.com/enoche/MMRec/blob/master/src/utils/dataloader.py)
  builds the mask from that additional dataset, and its
  [trainer](https://github.com/enoche/MMRec/blob/master/src/common/trainer.py)
  masks those scores. The old local behavior is compatible with this convention.
- The official [RecBole sampler](https://github.com/RUCAIBox/RecBole/blob/master/recbole/sampler/sampler.py)
  accumulates used items by phase (`train -> valid -> test`). Its
  [full-sort loader](https://github.com/RUCAIBox/RecBole/blob/master/recbole/data/dataloader/general_dataloader.py)
  removes the current phase positives from the accumulated set to form history,
  and its [trainer](https://github.com/RUCAIBox/RecBole/blob/master/recbole/trainer/trainer.py)
  masks that history. This yields `train` for validation and `train + validation`
  for final test.

## Baby impact audit

For the audited Baby split:

- all 19,445 test users also have validation interactions;
- the validation split contributes 20,559 interactions for those users;
- each test user has between 1 and 12 validation interactions;
- keeping the old rule therefore leaves 20,559 known validation positives in
  final-test candidate sets, where they are not counted as test positives.

This can create false-negative competition in the final ranking. The size and
direction of the metric change are model-dependent, so results produced under
the two policies must not be compared as if they used the same protocol.

## Decision

Do not change the primary Baby baseline to a different candidate convention:

- keep `val_test_once_v1` as the paper-ready, PromptMM/MMRec-compatible protocol;
- make its `train_only` candidate-history policy explicit in run manifests and
  teacher checkpoint metadata instead of leaving it implicit in evaluator code;
- retain `legacy_test_best` only for historical result reproduction;
- reserve a separately named `train + validation` protocol for a later
  sensitivity analysis if required. It is not part of the present baseline.

This decision prioritizes like-for-like comparison with PromptMM/MMRec and the
published Baby benchmark family. Introducing a new mask in the main baseline
would change the evaluation task at the same time as the dataset and would make
published reference numbers less directly comparable.

## Reporting consequence

All methods in a formal comparison must use the same protocol. Results compared
with MMRec/PromptMM tables should use and disclose the v1 `train_only` mask. If a
future sensitivity experiment also excludes validation history, it must use a
new protocol identifier, rerun every compared method under that rule, and remain
in a separate table or be explicitly labeled. Metrics from the two conventions
must not be mixed.
