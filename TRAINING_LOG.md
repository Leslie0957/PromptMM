# Training Log

This is the canonical active experiment record for PromptMM.
It starts the paper-ready MMRec Baby experiment line after the July 2026 data
and evaluation audit. The complete pre-reset record remains immutable at
`archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-07-30.md`.

## Read This First

- Active experiment entry: `codes/main_mmlight.py`
- Default argument source: `codes/utility/parser.py`
- Canonical active dataset: audited MMRec Baby under `data/baby/`
- Canonical protocol: `val_test_once_v1`
- Active student profile: `baby_student_reference_v1` (formal seed-2022
  reference completed)
- Current completed stage: frozen Baby teacher reference and ID-only BPR
  student reference, both seed `2022`
- Current incomplete stage: Baby directional distillation, ablation,
  multi-seed, and efficiency experiments
- `codes/run_patent.py` is standalone and outside this experiment line.
- CLI flags override parser defaults. In particular, an explicit
  `--student_lr` overrides the parser value.

## Logging Rules

1. Read this complete active log before any experiment recommendation, edit, or
   project-code execution.
2. Before editing anything that can affect experiment behavior, interpretation,
   or reproducibility, append a pending entry with scope, risks, acceptance
   criteria, rollback point, and planned verification.
3. Preserve that pending entry. After implementation and verification, append a
   separate completed or failed entry with actual changes, evidence, unresolved
   risks, and the next action.
4. Append every confirmed parameter or protocol change before the affected run.
5. Before a formal or long run, record the exact committed source, branch, full
   command, environment, data identity, teacher checkpoint, acceptance rule,
   and intended artifacts.
6. After every completed or failed run, append its status, resolved arguments,
   best validation metrics, one-time final test metrics when applicable, and
   artifact paths. Never silently replace a pending declaration.
7. Keep teacher and student selection metrics separate. Select checkpoints only
   with validation Recall@20 under `val_test_once_v1`; restore the selected
   checkpoint before the single final test evaluation.
8. Frozen-teacher comparisons must reuse an explicitly identified checkpoint
   with `--if_train_teacher false`.
9. Keep generated checkpoints, raw logs, datasets, and run outputs out of Git.
10. Do not copy metrics across protocol labels, dataset identities, teacher
   checkpoints, or candidate-exclusion policies.
11. Update `## Current State` only when a completed result becomes the new active
   reference. Preserve superseded detail in the append-only run history.
12. Historical snapshots are immutable. Add corrections to this file instead of
   editing archived evidence.

## Evidence Validity Boundary

### Paper-ready evidence

Only results produced on audited, non-duplicate multimodal data under the
validation-selected `val_test_once_v1` protocol may enter the main thesis result
tables. The completed paper-ready Baby anchors are now the seed-2022 teacher
reference and seed-2022 `baby_student_reference_v1` ID-only BPR reference.

### Historical exploratory evidence

All Amazon results completed before the July 2026 audit remain useful for
method motivation, debugging, and experiment design, but are not formal thesis
results because:

- the local `data/amazon` identity was Amazon-Book rather than the intended
  PromptMM Amazon-Electronics benchmark;
- its image and text arrays were byte-identical copies of one embedding source;
- historical training selected or inspected checkpoints repeatedly with test
  metrics rather than a held-out validation selection protocol.

Those runs, commands, metrics, checkpoints, alpha sweeps, component ablations,
three-seed summaries, and efficiency measurements are preserved verbatim in
the 2026-07-30 full archive. They must not be mixed with Baby results.

Transferable exploratory findings, to be retested rather than assumed on Baby:

- directional semantic transfer helped the randomly initialized lightweight
  student in the old environment;
- stronger distillation was not monotonically better;
- asymmetric item-dominant supervision outperformed naive four-head symmetry;
- removing projection heads recovered quality in the old structure study;
- teacher warm start was a major confounder;
- the ID-only inference export produced a large efficiency advantage.

## Research Objective

Build a defensible multimodal recommendation thesis around a lightweight
student that receives multimodal knowledge during training but serves without
the teacher, modality encoders, prompt modules, semantic caches, or graph
propagation.

The first method line to validate on Baby is train-infer decoupled directional
semantic distillation with:

- a frozen PromptMM-based multimodal teacher;
- user and item ID embeddings as the deployable student;
- BPR ranking loss plus normalized embedding alignment;
- independently controllable item-image, item-text, user-image, and user-text
  semantic components;
- a no-projection student variant whose embedding dimension directly matches
  the teacher semantic dimension;
- an inference checkpoint containing only user and item ID embeddings.

This is a research target, not a statement that the Baby student method has
already succeeded.

## Active Dataset Identity

### MMRec Baby

- Raw source: `data/_incoming/mmrec_baby/` (preserved read-only)
- Derived dataset: `data/baby/`
- Converter: `tools/convert_mmrec_baby.py`
- Conversion command:
  `D:\miniconda\envs\run_5060\python.exe tools\convert_mmrec_baby.py --source D:\Download\PromptMM\data\_incoming\mmrec_baby --output D:\Download\PromptMM\data\baby`
- Conversion manifest:
  `data/baby/conversion_manifest.json`
- Conversion manifest SHA256:
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`
- Shape: `19,445` users x `7,050` items
- Interactions: train `118,551`; validation `20,559`; test `21,682`
- Pairwise split overlap: `0 / 0 / 0`
- All observed ratings are converted to implicit value `1.0`.
- Official `x_label=0/1/2` train/validation/test assignments are preserved.
- Official evaluation-only item IDs `240`, `1212`, and `6115` are retained:
  `11` validation interactions and `7` test interactions; cold users `0`.

Tracked identities:

| Asset | Shape / type | SHA256 |
| --- | --- | --- |
| `train_mat` | `(19445, 7050)` | `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac` |
| `val_mat` | `(19445, 7050)` | `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2` |
| `test_mat` | `(19445, 7050)` | `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac` |
| image features | `(7050, 4096)` float64 | `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70` |
| text features | `(7050, 384)` float32 | `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4` |

The image and text modalities are finite, nonzero, semantically distinct, and
not byte duplicates.

## Active Evaluation Protocol

Protocol ID: `val_test_once_v1`

- Train on `train_mat`.
- Select and early-stop only with validation Recall@20.
- Restore the validation-best checkpoint.
- Evaluate `test_mat` once after restoration.
- Use `Ks=[10,20,40,50]`; Recall@20 is primary, with NDCG, Precision, and Hit
  Ratio as secondary metrics.
- Validation candidates exclude training interactions.
- Test candidates also exclude training interactions only.
- Candidate-exclusion policy ID: `train_only`.
- This matches the official PromptMM/MMRec result family used for comparison.
- `test_flag=part` omits AUC but still performs full-item ranking for the listed
  top-K metrics.
- A future `train + validation` test mask requires a new protocol name and a
  complete rerun of every compared method.

See `docs/EVALUATION_MASK_AUDIT_2026-07-30.md` for the full decision record.

## Current State

### Frozen Baby teacher reference

- Profile: `baby_teacher_reference_v1`
- Scope: teacher-only reference
- Profile definition: `docs/BABY_TEACHER_PROFILE_V1.md`
- Seed: `2022`
- Verified source commit: `a80235062bd05a2f3175edffa626f2ca91d48ec1`
- Run-declaration commit: `6bdd5c5ae017263facfebe997550c39559797291`
- Result-record commit/tag target:
  `349aea33a05f89b30c87ebf563e689265a157e4d`
- Stable tag: `baby-teacher-baseline-seed2022-20260730`
- Run: `2026-07-30 11_50_42.544441_baby_light_init_pid2480`
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA `12.8`; NVIDIA driver `595.97`; RTX 5060 8 GB;
  GPU `0`
- Resolved profile overrides: `{}`
- Full training batches per epoch: `116`
- Trained epochs: `0-29`
- Validation-best epoch: `22`
- Exact validation Recall@20: `0.08649579399772167`
- Final Test Recall@20: `0.08665691369856875`
- Final Test NDCG@20: `0.04042213264283099`
- Final Test Precision@20: `0.004836718950887038`
- Final Test Hit Ratio@20: `0.09534584726150529`
- Archived checkpoint:
  `Model/baby/runs/teacher_model_val_test_once_v1__2026-07-30 11_50_42.544441_baby_light_init_pid2480.pt`
- Shared read-only alias: `Model/baby/teacher_model_val_test_once_v1.pt`
- Checkpoint SHA256:
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`
- Manifest:
  `exp/runs/baby/run_manifest__2026-07-30 11_50_42.544441_baby_light_init_pid2480.json`
- Manifest SHA256:
  `9b8c080f54a97a1b51ea35d67fd9fd1462030b2d4a1a3d60a2f70df6439f2bf4`
- Recovery bundle:
  `backups/PromptMM_baby_teacher_baseline_seed2022_20260730.bundle`
- Bundle SHA256:
  `facf74dc0a235d34060e7348a2825709a7ca5917cbc9b9d9d67005d004470aad`

Interpretation: this is the frozen seed-2022 teacher anchor for subsequent
student experiments. It is not a multi-seed final teacher estimate.

### Student stage status

- Active reference profile: `baby_student_reference_v1`
- Scope: ID-only BPR student reference; no semantic loss and no teacher warm
  start
- Seed: `2022`
- Protocol implementation commit:
  `58fda059786067768e308f48e10ae864416334f5`
- Run-declaration/source commit:
  `8d8858547a475fd970ae9753496b2a7c06eb472c`
- Run: `2026-07-31 12_33_07.292506_baby_light_init_pid7872`
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA `12.8`; NVIDIA driver `595.97`; RTX 5060 8 GB;
  GPU `0`
- Resolved dataset/student profile overrides: `{}` / `{}`
- Student: `td_distill_no_projection`, dimension `64`, random initialization,
  AdamW `student_lr=6e-5`, weight decay `0.01`
- Loss: BPR only; `td_distill_alpha=0` and item-image, item-text, user-image,
  and user-text rates all `0`
- Full training batches per epoch: `116`
- Trained epochs: `0-170`; natural early stop at patience `7/7`
- Validation-best epoch: `163`
- Exact validation Recall@20: `0.04291303921928788`
- Final Test Recall@20: `0.044279857310199594`
- Final Test NDCG@20: `0.019996624710057805`
- Final Test Precision@20: `0.0024787863203908794`
- Final Test Hit Ratio@20: `0.04906145538698948`
- Full checkpoint:
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-07-31 12_33_07.292506_baby_light_init_pid7872.pth`
- Full checkpoint SHA256:
  `6a9cbe7548ac41a2d362309656923aa4f21f8c4f59a372e510a97535ccb61507`
- Inference-only checkpoint:
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-07-31 12_33_07.292506_baby_light_init_pid7872.pth`
- Inference-only checkpoint SHA256:
  `dbf0bf5ac0fe22ddbea6da5e777a8549b321fe25ae4df9d66f7d97c0e787e206`
- Manifest:
  `exp/runs/baby/run_manifest__2026-07-31 12_33_07.292506_baby_light_init_pid7872.json`
- Manifest SHA256:
  `cd2c5aa1fae79af8f8161dfee4fbb2bf9f451b591607df6c646d1dbf8b67af7d`

Interpretation: this is the accepted single-seed Baby student quality anchor.
Its Test Recall@20 is below the frozen teacher, but the result is valid because
all declared code, flow, and hard acceptance criteria passed. No formal Baby
directional-distillation, ablation, multi-seed, or efficiency result exists yet.

## Next Experiment Gate

The unique recommended next stage is to predeclare, without launching, the
first fair asymmetric no-projection directional-distillation candidate against
the completed `baby_student_reference_v1` anchor. The declaration must keep
seed `2022`, student dimension `64`, random initialization, optimizer budget,
sampling, evaluation protocol, and frozen teacher identity fixed; it must name
the directional component rates and `td_distill_alpha` before any new Baby test
access. Treat teacher warm start as a later separate control.

After that declaration is committed, wait for explicit user authorization
before any smoke or formal run. Do not start broad tuning, another test
evaluation, a tag, or a merge as part of the declaration task.

## Archive Index

- `archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-07-30.md`
  - exact pre-reset canonical log through the completed Baby teacher recovery
    milestone;
  - `118685` bytes;
  - SHA256 `2699931c8621309d6f09127629645990216b3ed19bae18dc7cf3f56a0b6dd5c8`.
- `archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-04-17.md`
  - earlier immutable snapshot containing the detailed April exploratory run
    history before the first log slimming pass.
- `archive/training/TRAINING_HANDOFF_2026-04-10.md`
- `archive/training/TRAINING_HANDOFF_2026-04-11.md`

Raw per-run output remains under `logs/`; generated manifests and convergence
records remain under `exp/`; checkpoints remain under `Model/`.

## Entry Template

### YYYY-MM-DD | Short label (pending/completed/failed)

- Purpose and hypothesis:
- Status:
- Branch and exact commit:
- Dataset and protocol identity:
- Teacher checkpoint identity:
- Full command:
- Resolved parameters and overrides:
- Environment:
- Validation selection result:
- One-time final test result:
- Artifact paths and hashes:
- Acceptance decision:
- Unresolved risks:
- Next action:

## Active Change And Run History

### 2026-07-30 | Baby experiment-line log reset (completed; documentation-only)

- Purpose: start a concise active log for the new Baby experiment cycle without
  deleting or rewriting historical evidence.
- The prior `TRAINING_LOG.md` was copied verbatim to
  `archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-07-30.md` before this reset.
- Source and archive were both `118685` bytes with matching SHA256
  `2699931c8621309d6f09127629645990216b3ed19bae18dc7cf3f56a0b6dd5c8`.
- No dataset, code, parameter, protocol, checkpoint, raw log, manifest, metric,
  or generated artifact changed.
- Historical Amazon evidence was retained but explicitly classified as
  exploratory; the audited Baby line is now the only active paper-ready line.
- Next action: define and review the Baby student baseline/proposed protocol
  before changing parameters or launching training.

### 2026-07-30 | Repository experiment-governance hardening (pending)

- Purpose: make pre-read, before/after logging, committed-source execution, and
  durable Git preservation mandatory for every future AI working on experiments.
- Scope: repository policy in `AGENTS.md` plus this documentation trace only;
  no training code, data, parameters, protocol, checkpoint, or result changes.
- Planned policy:
  - read the complete active `TRAINING_LOG.md` before any experiment-related
    recommendation, edit, or command;
  - append a pending entry before every modification that can affect experiment
    behavior or reproducibility;
  - append a completed or failed entry after implementation and verification;
  - prohibit formal/long training from dirty or uncommitted source;
  - require every material repository change to be preserved in a coherent Git
    commit, with tags and standalone bundles added at meaningful stable
    milestones rather than routine commits;
  - preserve ignored datasets, checkpoints, manifests, and raw logs through
    explicit fingerprints and separate physical backup decisions.
- Risks: redundant policy wording or a rule that is impossible to satisfy due
  to self-referential commit hashes.
- Acceptance criteria: rules state clear timing and scope, distinguish commits
  from milestone archives, protect unrelated changes, and remain operational.
- Verification plan: focused content search, `git diff --check`, and final Git
  status/diff review; no training tests are required for a policy-only change.

### 2026-07-30 | Repository experiment-governance hardening (completed; policy-only)

- Actual changes:
  - added a mandatory startup gate requiring every AI to read `AGENTS.md`, read
    the complete active `TRAINING_LOG.md`, and inspect Git status before edits or
    project-code execution;
  - added mandatory separate pending and completed/failed records around every
    experiment-affecting or reproducibility-affecting modification;
  - prohibited formal or long training from dirty or uncommitted source and
    defined the narrower conditions for pre-commit non-formal smoke runs;
  - required every material change to have a coherent branch commit, with
    annotated tags and standalone bundles reserved for stable milestones;
  - clarified that ignored datasets and run artifacts require fingerprints and
    explicit physical/off-device backup, because Git cannot protect them;
  - required a verified verbatim snapshot before any future canonical-log
    slimming or replacement.
- Scope control: no source code, data, parameter, protocol, checkpoint, metric,
  or generated experiment artifact changed.
- Verification:
  - policy keyword and section checks passed;
  - the active log rules now mirror the repository policy;
  - `git diff --check` passed before this completion entry;
  - focused documentation review found no contradictory formal-run or archive
    instruction.
- Unresolved operational issue: the local Git approval service still returns a
  404 during `.git` write escalation, so this coherent documentation change is
  not yet staged or committed. The files remain preserved in the working tree.
- Next action: once Git write approval is available, stage only `AGENTS.md`,
  `TRAINING_LOG.md`, `archive/training/README.md`, and
  `archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-07-30.md`, then create one
  documentation commit before beginning any student experiment work.

### 2026-07-30 | Baby student reference protocol v1 (pending)

- Purpose and hypothesis: establish the first auditable Baby student reference
  before observing any new Baby student test metric. The reference hypothesis is
  that a randomly initialized, deployable 64-dimensional user/item ID student
  trained with BPR alone provides the fair lower anchor for later directional
  semantic-distillation comparisons.
- Status: pending; no smoke or formal student run has started.
- Branch and rollback point:
  `codex/experiment/baby-teacher-baseline` at commit
  `b15b4fd` (`docs: reorganize training log and repository guidance`).
- Dataset and protocol identity: audited MMRec Baby from
  `data/baby/conversion_manifest.json`, SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  `val_test_once_v1`; candidate exclusion `train_only`; validation Recall@20
  selects the checkpoint.
- Teacher checkpoint identity: reuse
  `Model/baby/teacher_model_val_test_once_v1.pt` read-only with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  set `if_train_teacher=false` and never publish or overwrite a teacher alias.
- Declared reference scope and parameters:
  - profile ID `baby_student_reference_v1`;
  - `student_model_type=td_distill_no_projection` so the trainable/deployable
    state is only user and item ID embeddings;
  - `student_embed_size=64`, `td_init_from_teacher=false`,
    `td_distill_alpha=0`, and all four directional component rates `0`;
  - seed `2022`, batch size `1024`, maximum `1000` epochs, early-stopping
    patience `7`, AdamW `student_lr=6e-5`, and explicit student weight decay
    `0.01`;
  - the `6e-5` learning rate is a predeclared transfer hypothesis from the old
    exploratory line, not a Baby-confirmed optimum; it must not be described as
    tuned on Baby test data.
- Planned implementation scope:
  `codes/utility/dataset_profiles.py`, `codes/utility/parser.py`,
  `codes/main_mmlight.py`, focused tests, a new
  `docs/BABY_STUDENT_REFERENCE_V1.md`, and this append-only log.
- Test-isolation requirement: a non-formal capped smoke must restore/check the
  frozen teacher and exercise training plus validation, but must not evaluate
  either teacher or student on `test_mat`. A formal uncapped run is the only
  path allowed to restore the validation-best student checkpoint and evaluate
  its test result once.
- Comparison contract: the first directional-distillation candidate must keep
  seed, dimension, initialization policy, optimizer, batch/sampling budget,
  validation selection, and teacher identity equal to this reference; only the
  predeclared semantic-loss settings may differ.
- Risks: profile defaults could be silently overridden by CLI values; the
  current Baby eligibility gate is teacher-only; the existing student reuse
  path re-evaluates the teacher test split; AdamW weight decay is currently an
  implicit library default; and a smoke could consume test metrics unless the
  code prevents it.
- Acceptance criteria: profile identity and overrides are recorded in every run
  manifest; formal eligibility rejects an unpinned Baby student configuration;
  the BPR-only profile has no active semantic heads or teacher warm start;
  optimizer weight decay is explicit; smoke mode records no final-test result;
  focused unit tests and `git diff --check` pass; no training is launched in
  this change.
- Planned verification: parser/profile unit tests, protocol unit tests, static
  command/profile resolution checks, teacher/data fingerprint confirmation,
  and focused diff review. The next run after the verified commit is a
  one-batch, one-epoch, validation-only non-formal smoke.

#### Scope amendment before implementation continuation

- Focused verification found that frozen-teacher validation currently compares
  the active child run's `paper_ready_eligible` and `paper_ready_blockers`
  fields for exact equality with the teacher checkpoint. A correctly isolated
  smoke is intentionally non-paper-ready, so it cannot reuse the paper-ready
  frozen teacher under that rule.
- The scope therefore expands to `codes/utility/experiment_protocol.py` and its
  focused tests. Teacher reuse under `val_test_once_v1` will require the teacher
  checkpoint itself to record `paper_ready_eligible=true` and an empty blocker
  list, while active student/smoke eligibility remains independently recorded
  in the run manifest.
- This does not relax dataset, teacher inference configuration, protocol,
  selection split, primary K, candidate policy, or checkpoint identity checks.
  Acceptance additionally requires tests proving that a paper-ready teacher can
  be reused by a non-formal child smoke and that a blocked teacher checkpoint is
  still rejected.

### 2026-07-30 | Baby student reference protocol v1 (completed; no run)

- Purpose and outcome: implemented and documented the predeclared Baby ID-only
  BPR student reference without launching training or inspecting any new Baby
  student test metric.
- Status: protocol implementation completed and verified; non-formal smoke and
  formal reference run remain pending.
- Branch and source identity: implementation is based on clean predecessor
  commit `b15b4fd` on `codex/experiment/baby-teacher-baseline`. The resulting
  protocol commit is pending because this Codex session cannot write `.git`;
  record the user-created commit hash before declaring the smoke.
- Actual profile: `baby_student_reference_v1` in
  `codes/utility/dataset_profiles.py`, with an explicit parser selector and
  run-manifest identity/override fields.
- Resolved reference contract:
  - frozen read-only teacher alias with `if_train_teacher=false`;
  - `td_distill_no_projection`, 64-dimensional random user/item ID embeddings,
    teacher warm start disabled, `td_distill_alpha=0`, and all four component
    rates `0`;
  - seed `2022`, batch size `1024`, at most `1000` epochs, patience `7`, AdamW
    `student_lr=6e-5`, and explicit `student_weight_decay=0.01`;
  - `val_test_once_v1`, validation Recall@20 selection, `train_only` candidate
    exclusion, dataset preflight enabled, and duplicate modalities treated as
    an error.
- Test-isolation changes:
  - added `run_final_test`; capped smoke is rejected unless it is explicitly
    false;
  - teacher reuse and student finalization now skip `test_mat` when final test
    is disabled, restore the validation-best checkpoint, and record
    `final_test_performed=false` without a final-test metric;
  - a no-test teacher training path does not publish a shared teacher alias;
  - formal uncapped runs retain restore-then-evaluate-once behavior.
- Frozen-teacher validation correction: a child smoke may be non-paper-ready
  while reusing a checkpoint that is itself paper-ready. The validator now
  requires the checkpoint's own `paper_ready_eligible=true` and empty blockers,
  while preserving exact dataset, teacher inference configuration, protocol,
  split, K, candidate-policy, and hard-token provenance checks. A blocked
  teacher checkpoint is still rejected.
- Documentation: `docs/BABY_STUDENT_REFERENCE_V1.md` records rationale, pinned
  values, expected official cold-item warning, smoke/formal commands,
  comparison controls, acceptance rules, and artifact requirements.
- Verification evidence:
  - all `27` tests under `codes/tests/test_*.py` passed;
  - Python compilation passed for all changed Python modules;
  - formal profile resolution produced no dataset or student overrides;
  - smoke resolution produced only the declared `epoch`,
    `smoke_train_batches`, and `run_final_test` student overrides;
  - real read-only Baby preflight plus frozen-teacher checkpoint validation
    passed for the smoke context;
  - image/text modalities remained non-duplicate; the only preflight warning
    was the already documented `3` official evaluation-only items (`11`
    validation and `7` test interactions, no cold users);
  - teacher and conversion-manifest SHA256 values re-matched the active log;
  - `git diff --check` passed, with only existing CRLF-to-LF Git notices.
- Metrics and artifacts: none; no training, validation ranking, final test,
  checkpoint, run manifest, convergence record, or raw experiment log was
  generated by this protocol-definition task.
- Acceptance decision: implementation acceptance criteria passed. This is a
  protocol milestone only, not an accepted Baby student result and not yet a
  tag/bundle milestone.
- Unresolved risks: the real GPU execution path has not yet been exercised;
  `6e-5` remains a predeclared cross-dataset transfer hypothesis rather than a
  Baby-confirmed optimum; the actual protocol commit hash is still pending.
- Next action: create one coherent local commit, record its hash in the smoke
  declaration, then run exactly one epoch and one batch with
  `--run_final_test false`. Do not launch the uncapped reference until that
  smoke is logged as completed or failed.

### 2026-07-30 | No-test teacher self-restore correction (pending)

- Purpose and rationale: commit-readiness review found that a teacher trained
  in the active run with `run_final_test=false` is saved as intentionally
  non-paper-ready, but the new restore branch applies the stricter frozen
  paper-ready teacher reuse gate to that run-local checkpoint. The restore
  therefore fails before it can record validation completion and confirm that
  no shared teacher alias was published.
- Scope: distinguish run-local teacher checkpoint restoration from external
  frozen-teacher reuse in `codes/utility/experiment_protocol.py` and
  `codes/main_mmlight.py`, with a focused protocol unit test. No profile value,
  optimizer, dataset, checkpoint, evaluation metric, or formal-run command
  changes.
- Risks: weakening frozen-teacher validation or allowing a child run to reuse
  an ineligible checkpoint. The external reuse path must retain the strict
  requirement that the frozen checkpoint is paper-ready and blocker-free.
- Acceptance criteria: a run-local checkpoint must still match the active
  dataset, inference configuration, protocol, selection split, primary K,
  candidate policy, eligibility flag, and blocker list; external frozen
  teacher reuse must continue to reject an ineligible checkpoint.
- Rollback point: `b15b4fd` plus the existing uncommitted
  `baby_student_reference_v1` implementation reviewed in this entry.
- Planned verification: focused validator tests, all tests under
  `codes/tests/test_*.py`, Python compilation of changed modules,
  `git diff --check`, and final staged-diff review. No training will run.

### 2026-07-30 | No-test teacher self-restore correction (completed; no run)

- Actual changes: added an explicit run-local restore mode to frozen-teacher
  metadata validation. External teacher reuse still requires a paper-ready,
  blocker-free checkpoint; run-local restoration instead requires eligibility
  and blockers to exactly match the active run. The no-final-test teacher path
  uses that mode only for its own run-specific validation-best checkpoint.
- Scope control: no profile value, dataset, optimizer, sampling behavior,
  checkpoint asset, metric, command, or generated experiment artifact changed.
  No training, validation ranking, or test evaluation was run.
- Verification evidence:
  - all `27` tests under `codes/tests/test_*.py` passed, including acceptance
    of an exact run-local ineligible checkpoint, rejection on blocker mismatch,
    acceptance of a paper-ready teacher by a non-formal child smoke, and
    rejection of an ineligible external teacher;
  - Python compilation passed for every changed Python module;
  - real Baby preflight in the declared smoke context passed with distinct
    modalities and only the documented `3` evaluation-only item warning;
  - real frozen-teacher validation passed with protocol `val_test_once_v1`,
    `paper_ready_eligible=true`, and an empty blocker list;
  - formal profile resolution remained override-free, and the smoke resolution
    contained only the declared `epoch`, `smoke_train_batches`, and
    `run_final_test` student overrides;
  - conversion-manifest and teacher checkpoint SHA256 values matched the
    identities recorded above;
  - `git diff --check` passed before this completion entry, with only existing
    CRLF-to-LF conversion notices.
- Acceptance decision: correction passed and the complete
  `baby_student_reference_v1` change is ready for one coherent local commit.
- Unresolved risk: GPU execution remains intentionally untested until the
  separately declared non-formal smoke; this task did not launch training.
- Next action: stage only the declared protocol, tests, documentation, and
  append-only log; review the staged diff; create the local protocol commit;
  then record that commit hash in the future smoke declaration.

### 2026-07-31 | Mandatory completion handoff policy (pending)

- Purpose and rationale: make every experiment task self-contained from
  startup through handoff so repository state, rather than chat memory, always
  identifies the active stage, evidence, risks, and single recommended next
  action.
- Status: pending; this is a repository-policy and documentation change only.
- Scope: add a `Mandatory Completion Handoff` section to `AGENTS.md` and
  preserve this change through separate pending/completed entries in the
  append-only `TRAINING_LOG.md`. No experiment code, data, preprocessing,
  parameter, checkpoint, metric, command, or generated artifact will change.
- Planned policy: require repository-led startup/context recovery, append-only
  experiment tracing, proportional verification and scoped automatic commits,
  a complete end-of-task handoff, truthful failure reporting, explicit approval
  before guarded long-running or publication actions, autonomous completion of
  in-scope work, and one traceable experiment stage at a time.
- Rollback point: branch `codex/experiment/baby-teacher-baseline` at protocol
  source commit `58fda059786067768e308f48e10ae864416334f5`.
- Risks: duplicating existing startup/version-control rules, creating
  contradictory authority around formal runs, or making the handoff too vague
  to audit.
- Acceptance criteria: all nine requested collaboration requirements are
  stated as mandatory operational rules; the section requires a single next
  step and copy-ready continuation instruction; it does not authorize automatic
  formal training, broad tuning, test evaluation, `main` promotion, tagging, or
  baseline overwrite; only the two declared documentation files are committed.
- Planned verification: focused section/keyword checks, manual diff review,
  `git diff --check`, exact staged-file review, and
  `git diff --cached --check`. No project code or training will run for this
  policy stage.

### 2026-07-31 | Mandatory completion handoff policy (completed; policy-only)

- Purpose and outcome: established a mandatory repository-led completion
  handoff so every future experiment task begins from canonical state and ends
  with an auditable result plus exactly one authorized-next-step proposal.
- Status: completed; no experiment execution occurred in this policy stage.
- Actual changes: added all nine requested rules under
  `## Mandatory Completion Handoff` in `AGENTS.md`, covering startup recovery,
  log-derived stage selection, append-only before/after traces, proportional
  verification and scoped commits, complete success/failure handoffs, guarded
  action approval, autonomous in-scope completion, and single-stage execution.
- Scope control: only `AGENTS.md` and this append-only log changed. No code,
  dataset, preprocessing, environment, parameter, checkpoint, metric, command,
  raw log, manifest, or generated experiment artifact changed.
- Verification evidence:
  - focused section and keyword checks found all nine numbered requirements,
    including test-access disclosure, all new commit hashes, clean-tree state,
    a single recommended next step, and a copy-ready continuation instruction;
  - manual diff review found no authority conflict with the existing formal-run
    gate or version-control requirements;
  - `git diff --check` passed before this outcome entry.
- Acceptance decision: policy requirements are complete and ready for one
  documentation-only commit. No tag or bundle is warranted for this routine
  policy commit.
- Unresolved risks: none within the declared documentation scope; operational
  compliance depends on future sessions reading and following `AGENTS.md`.
- Next action: commit only `AGENTS.md` and `TRAINING_LOG.md`, then create and
  commit the separate declaration for the one-epoch, one-batch,
  validation-only `baby_student_reference_v1` smoke before execution.

### 2026-07-31 | baby_student_reference_v1 validation-only smoke (pending)

- Purpose and hypothesis: execute the first isolated GPU-path smoke for the
  committed Baby ID-only BPR student profile. One optimization batch should
  complete, validation ranking should select and restore the only checkpoint,
  and the run should exit without teacher or student test evaluation.
- Status: pending; explicitly non-formal, capped, validation-only, and not
  eligible for paper result tables.
- Branch and source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - protocol implementation commit
    `58fda059786067768e308f48e10ae864416334f5`;
  - policy-only successor before this declaration
    `e2ac51db163f3b4a2bef2d04a838897ab5ba1a86`;
  - this declaration will be committed before launch, and its resulting commit
    will be verified as clean and recorded in the outcome entry.
- Dataset and protocol identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  `val_test_once_v1`; selection split `validation`; primary metric Recall@20;
  candidate exclusion `train_only`; `Ks=[10,20,40,50]`.
- Teacher checkpoint identity: read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, pre-run SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`; no alias publication, overwrite, or teacher update
  is authorized.
- Full command:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --epoch 1 --smoke_train_batches 1 --run_final_test false`
- Resolved parameters and expected overrides:
  - seed `2022`, batch size `1024`, exactly `1` epoch and at most `1` training
    batch, early-stopping patience `7`;
  - `td_distill_no_projection`, student dimension `64`, random initialization,
    AdamW `student_lr=6e-5`, weight decay `0.01`;
  - semantic alpha and all four component rates `0`, so the batch is BPR-only;
  - expected `student_config_overrides` are exactly `epoch: 1000 -> 1`,
    `smoke_train_batches: 0 -> 1`, and `run_final_test: true -> false`;
    explicit GPU `0`, frozen-teacher mode, and checkpoint path do not change
    the pinned profile values.
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA GeForce RTX 5060;
  GPU selector `0`.
- Test isolation: dataset loading/preflight may inspect the declared test split
  only for structural integrity and identity, but neither teacher nor student
  ranking may evaluate it and no test metric may be produced or reported.
- Planned isolated artifacts, all keyed by a new timestamp/PID run name:
  `logs/<run_name>`,
  `exp/runs/baby/dataset_preflight__<run_name>.json`,
  `exp/runs/baby/run_manifest__<run_name>.json`,
  `exp/converge/baby/auto__<run_name>.pkl`,
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`,
  and
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
- Risks: the real GPU path is untested; dependency or memory failures may occur;
  validation full-item ranking remains much larger than the one training batch;
  and this capped metric is diagnostic only, never a baseline quality estimate.
- Acceptance criteria: clean committed launch; dataset and frozen-teacher
  validation pass; one BPR batch and one validation selection complete with
  finite values; the validation-selected checkpoint is restored; manifest
  status is `smoke_completed`; `run_final_test=false`,
  `final_test_performed=false`, and no final-test result field is present; the
  teacher is reused without test evaluation; teacher SHA256 is unchanged; all
  outputs use the new run-specific paths; no frozen alias is published or
  overwritten.
- Planned verification: pre/post teacher SHA256, clean Git launch gate, process
  exit status, focused log inspection without test metrics, structured manifest
  field checks, artifact existence/hashes, and post-run Git status.
- Rollback/preservation point: source commit
  `58fda059786067768e308f48e10ae864416334f5`; retain any run-specific failure
  manifest/log/checkpoint for diagnosis rather than altering the frozen teacher.
- Next action: commit this declaration only, verify the exact launch commit and
  clean tree, then run the command once. Do not start an uncapped or formal run.

### 2026-07-31 | baby_student_reference_v1 validation-only smoke (completed)

- Purpose and outcome: the first isolated GPU-path smoke for the Baby ID-only
  BPR reference completed successfully. It exercised frozen-teacher loading,
  one optimization batch, validation selection, checkpoint restoration, and
  no-final-test termination.
- Status: completed successfully; non-formal and not paper-ready.
- Branch and source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - protocol implementation commit
    `58fda059786067768e308f48e10ae864416334f5`;
  - policy commit `e2ac51db163f3b4a2bef2d04a838897ab5ba1a86`;
  - clean run-declaration/launch commit
    `b5863217800b4505109a65a4db016fb9f5677b8b`;
  - `git diff 58fda059786067768e308f48e10ae864416334f5 HEAD -- codes`
    was empty immediately before launch.
- Full executed command:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --epoch 1 --smoke_train_batches 1 --run_final_test false`
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA GeForce RTX 5060;
  GPU selector `0`.
- Run identity:
  `2026-07-31 12_08_26.244081_baby_light_init_pid24060`; completed at
  `2026-07-31T12:08:39.911356+08:00`; process exit code `0`.
- Resolved protocol and parameters:
  - audited Baby, `val_test_once_v1`, validation selection, Recall@20 primary,
    candidate exclusion `train_only`, seed `2022`, batch size `1024`;
  - `epoch=1`, `smoke_train_batches=1`, `run_final_test=false`;
  - `td_distill_no_projection`, 64-dimensional random student, warm start
    disabled, AdamW `student_lr=6e-5`, weight decay `0.01`;
  - semantic alpha and all four component rates were `0`, producing BPR-only
    training with no active semantic heads;
  - student overrides were exactly the declared `epoch`,
    `smoke_train_batches`, and `run_final_test`; the dataset profile separately
    recorded the expected `epoch` override.
- Execution and validation evidence:
  - the cap selected `1` batch instead of the full `116` batches per epoch;
  - epoch `0` completed in `10.3s`; finite BPR loss was `0.69320`, semantic
    component losses were all `0`;
  - validation-best epoch was `0`; exact validation Recall@20 was
    `0.0029656295534413314`;
  - the validation-selected full checkpoint was restored before exit;
  - manifest status is `smoke_completed`, model stage `td_distill`, and the run
    is correctly ineligible with only declared profile/cap/no-final-test
    blockers.
- Test-split handling: dataset loading/preflight inspected the declared split
  only for structural integrity and identity. Neither teacher nor student
  ranking evaluated the test split. Manifest values are
  `run_final_test=false`, `teacher_final_test_performed=false`, and
  `final_test_performed=false`; neither teacher nor student final-test result
  field exists. No test metric was read or reported.
- Frozen teacher preservation: pre- and post-run SHA256 both equal
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  No run-local teacher checkpoint was created, no alias was published, and the
  frozen alias was not overwritten.
- Isolated artifacts and SHA256:
  - manifest:
    `exp/runs/baby/run_manifest__2026-07-31 12_08_26.244081_baby_light_init_pid24060.json`;
    `7823ce0f13aaa0f995fe08b8e301bc89ed4d87ac0dde7170b82c1ded178eca0e`;
  - preflight:
    `exp/runs/baby/dataset_preflight__2026-07-31 12_08_26.244081_baby_light_init_pid24060.json`;
    `a117402c47e8e71f48fddbff40a75e322acb3db6ef7e0bdd43421c582506a8b6`;
  - convergence record:
    `exp/converge/baby/auto__2026-07-31 12_08_26.244081_baby_light_init_pid24060.pkl`;
    `eefabe59f4102a5a0cc43f018494c745dcbf67af0194d6864e64ec57bfa229a1`;
  - full student checkpoint:
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-07-31 12_08_26.244081_baby_light_init_pid24060.pth`;
    `ea8da2fd6c59a0552a9f601352100cfdefa50c2544b92d11a47c7db3dbaeaa4d`;
  - inference-only student checkpoint:
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-07-31 12_08_26.244081_baby_light_init_pid24060.pth`;
    `6c2b741890d9fdcb0f31243a2a6603482ec100b323cfc4d3286b07af8dad71eb`;
  - raw log:
    `logs/2026-07-31 12_08_26.244081_baby_light_init_pid24060`;
    `89d4ba0734c213e284df63cc5c266eed08b64ec3030652691edb6822bd0d059c`.
- Acceptance decision: all declared smoke criteria passed. This confirms the
  execution and isolation path only; the capped validation value is diagnostic
  and is not an accepted Baby student baseline result.
- Unresolved risks: the uncapped optimizer trajectory and early stopping remain
  untested; `student_lr=6e-5` is still a predeclared transfer hypothesis rather
  than a Baby-confirmed optimum; and no formal student final-test evidence
  exists.
- Next action: after this outcome log is committed and only after explicit user
  approval, declare and execute the uncapped seed-2022
  `baby_student_reference_v1` formal reference from a clean tree, with no
  profile overrides. Expected outputs are the validation-selected full and
  inference-only checkpoints, convergence record, run manifest, raw log, and
  the single protocol-authorized final-test evaluation after restoration.

### 2026-07-31 | Low-result preservation and destructive-action guard (pending)

- Purpose and rationale: make experiment preservation independent of whether a
  result is strong, weak, expected, or disappointing. Scientific outcomes must
  remain traceable unless execution itself violates code, process, or hard
  acceptance requirements.
- Status: pending; repository-policy and documentation scope only.
- Scope: add one explicit rule to `AGENTS.md` under
  `Mandatory Completion Handoff` and preserve this change through separate
  pending/completed entries in the append-only `TRAINING_LOG.md`. No experiment
  code, data, parameter, protocol, checkpoint, metric, run command, or generated
  artifact will change.
- Planned rule: low or degraded metrics remain valid completed evidence and
  must not trigger automatic rollback, reset, revert, or deletion; `failed` is
  reserved for code, execution-flow, or hard-acceptance failures; failure state
  and recovery evidence remain preserved; destructive Git/history/file actions
  require explicit user confirmation.
- Rollback point: branch `codex/experiment/baby-teacher-baseline` at commit
  `4d1848de0421352e72e52aa9c7bf6c577799de2e`.
- Risks: ambiguous wording could confuse a scientifically poor result with a
  technically invalid run, or could appear to authorize silent destructive
  recovery after a genuine failure.
- Acceptance criteria: the rule explicitly preserves low or degraded results,
  defines the narrow `failed` boundary, requires failure-site/root-cause and
  recovery-point reporting, and prohibits revert, reset, branch restoration,
  change deletion, or result deletion without explicit user confirmation. Only
  `AGENTS.md` and `TRAINING_LOG.md` may enter the commit.
- Planned verification: focused exact-concept/keyword checks, manual diff
  review, `git diff --check`, exact staged-file review, and
  `git diff --cached --check`. No project code, dataset command, or training
  process will run.

### 2026-07-31 | Low-result preservation and destructive-action guard (completed; policy-only)

- Purpose and outcome: made weak or degraded experiment outcomes explicitly
  preservable evidence and separated scientific disappointment from technical
  or protocol failure.
- Status: completed; policy-only, with no experiment execution.
- Actual changes: added rule 10 under `Mandatory Completion Handoff` requiring
  low or worse-than-expected metrics to remain recorded `completed` evidence
  when code, execution flow, and hard acceptance conditions succeed. The rule
  reserves `failed` for failures in those three areas, requires preservation of
  the failure state plus cause and recommended recovery point, and prohibits
  revert, reset, branch restoration, change deletion, or result deletion
  without explicit user confirmation.
- Scope control: only `AGENTS.md` and this append-only log changed. No source
  code, dataset, protocol, parameter, environment, checkpoint, metric, command,
  manifest, raw log, or generated artifact changed; no training was launched.
- Verification evidence:
  - focused content checks found the low/degraded-result condition, valid
    completed-evidence treatment, narrow `failed` boundary, preserved failure
    state, recovery-point reporting, and explicit-user confirmation guard;
  - manual diff review confirmed the new rule is within
    `Mandatory Completion Handoff` and does not weaken the existing formal-run
    or version-control gates;
  - `git diff --check` passed before this completed entry;
  - working diff contained exactly `AGENTS.md` and `TRAINING_LOG.md`.
- Acceptance decision: all requested policy concepts are explicit and the
  documentation-only change is ready for one coherent local commit.
- Metrics and artifacts: none; policy text and its Git commit are the only
  outputs. No test split or experiment artifact was accessed.
- Unresolved risks: none within the declared policy scope.
- Next action: after this policy commit, retain the existing experiment gate as
  the single recommendation: only with explicit user authorization, predeclare
  and run the uncapped seed-2022 `baby_student_reference_v1` formal reference
  from a clean tree without profile overrides.

### 2026-07-31 | baby_student_reference_v1 formal seed-2022 reference (pending)

- Purpose and hypothesis: establish the first formal Baby ID-only BPR student
  reference under the committed `baby_student_reference_v1` profile. The run
  tests the predeclared `student_lr=6e-5` transfer hypothesis without tuning and
  creates the quality anchor for later directional-distillation comparisons.
- Status: pending; formal, uncapped, validation-selected, single seed `2022`.
- Authorization: the user explicitly authorized this formal run and its
  protocol-required final-test evaluation after the completed validation-only
  smoke. No later distillation experiment is authorized in this task.
- Branch and source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - protocol implementation commit
    `58fda059786067768e308f48e10ae864416334f5`;
  - completed smoke result commit
    `4d1848de0421352e72e52aa9c7bf6c577799de2e`;
  - current preservation-policy commit
    `af4867514b20885d2b97d280cf19ff1b1758f8d7`;
  - `git diff 58fda059786067768e308f48e10ae864416334f5 HEAD -- codes`
    is empty;
  - this declaration will be committed before launch, then its exact clean
    commit will be recorded in the outcome entry.
- Dataset and preprocessing identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/validation/test matrix hashes remain the active identities recorded
  above; hard-token type `pca`, seed `2022`, and frozen cache provenance must
  match the teacher checkpoint.
- Evaluation protocol: `val_test_once_v1`; train on `train_mat`; select and
  early-stop only by validation Recall@20; candidate exclusion `train_only`;
  `Ks=[10,20,40,50]`; restore the validation-best student checkpoint, then
  evaluate its test split exactly once. Teacher and student metrics remain
  separate, and no test metric may influence selection or acceptance.
- Teacher checkpoint identity: read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`; pre-run SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `allow_teacher_alias_overwrite=false`; no teacher
  training, publication, or overwrite is authorized.
- Full command:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Resolved parameters and override gate:
  - formal profile resolution before declaration produced
    `dataset_config_overrides={}` and `student_config_overrides={}`;
  - seed `2022`, batch size `1024`, full `116` batches per epoch, maximum
    `1000` epochs, validation interval `5`, early-stopping patience `7`;
  - `smoke_train_batches=0`, `run_final_test=true`, dataset preflight enabled,
    duplicate modalities policy `error`;
  - `td_distill_no_projection`, student dimension `64`, random initialization,
    AdamW `student_lr=6e-5`, weight decay `0.01`;
  - `td_distill_alpha=0` and all four semantic component rates `0`, so this is
    the declared BPR-only reference with no teacher warm start or semantic loss.
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver `595.97`;
  NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`; preflight free space on drive
  D approximately `404 GB`.
- Planned run-specific artifacts keyed by a new timestamp/PID run name:
  `logs/<run_name>`,
  `exp/runs/baby/dataset_preflight__<run_name>.json`,
  `exp/runs/baby/run_manifest__<run_name>.json`,
  `exp/converge/baby/auto__<run_name>.pkl`,
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`,
  and
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
- Hard acceptance criteria: launch from the clean committed declaration;
  preflight and frozen-teacher validation pass; both profile override maps stay
  empty; losses and validation metrics remain finite; validation Recall@20
  alone selects/early-stops; the selected checkpoint is restored before one
  final student test evaluation; manifest status is `completed`,
  `paper_ready_eligible=true`, blockers are empty, and
  `final_test_performed=true`; teacher SHA256 is unchanged; all artifacts are
  isolated, present, and fingerprinted.
- Quality acceptance rule: there is no minimum metric threshold. Any finite
  result produced by the declared code and protocol is valid completed
  evidence even if low or worse than the smoke/teacher/expectation. Quality
  alone must not trigger rollback, reset, revert, deletion, or a `failed`
  label.
- Failure boundary and preservation: mark `failed` only for code failure,
  execution-flow failure, or a hard acceptance violation. Preserve every
  run-specific log, manifest, and checkpoint that exists; report the root cause
  and recommended recovery point without destructive recovery.
- Risks: uncapped runtime is substantially longer than the smoke; GPU or
  dependency failure may interrupt the run; `6e-5` remains an unconfirmed
  cross-dataset hypothesis; the single seed is an anchor rather than a variance
  estimate; final test is authorized once and cannot be used to tune settings.
- Planned verification: clean Git launch gate, exact profile resolution,
  pre/post teacher SHA256, process exit, full manifest protocol/eligibility and
  selection/finalization checks, finite metric checks, run-log chronology,
  artifact existence/SHA256, and final Git status.
- Rollback/preservation point: committed protocol source
  `58fda059786067768e308f48e10ae864416334f5` and verified smoke result
  `4d1848de0421352e72e52aa9c7bf6c577799de2e`; no rollback action is authorized.
- Next action: commit this declaration only, verify the exact clean launch
  commit and identities, then execute the command once to completion. Do not
  start a directional-distillation run afterward.

### 2026-07-31 | baby_student_reference_v1 formal seed-2022 reference (completed)

- Status: completed and accepted as valid formal evidence. The run finished
  naturally with process exit code `0`; no training or evaluation was
  restarted during the post-run audit, and the result was retained regardless
  of its quality.
- Goal and outcome: establish the first formal Baby ID-only BPR student anchor
  under `baby_student_reference_v1`. All declared code, execution-flow, and
  hard acceptance criteria passed, so the completed result now replaces the
  prior "formal run pending" student state.
- Source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - launch/declaration commit
    `8d8858547a475fd970ae9753496b2a7c06eb472c`;
  - protocol implementation commit
    `58fda059786067768e308f48e10ae864416334f5`;
  - the launch tree was clean and
    `git diff 58fda059786067768e308f48e10ae864416334f5 8d8858547a475fd970ae9753496b2a7c06eb472c -- codes`
    was empty.
- Executed formal command:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver `595.97`;
  NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`.
- Run identity and duration:
  `2026-07-31 12_33_07.292506_baby_light_init_pid7872`; manifest start
  `2026-07-31T12:33:07.299515+08:00`; completion
  `2026-07-31T13:39:50.548154+08:00`; manifest elapsed approximately
  `1:06:43`; epochs `0-170` completed with `116` full training batches per
  epoch, followed by natural patience `7/7` early stopping.
- Resolved protocol and parameters:
  `val_test_once_v1`, selection split `validation`, primary metric Recall@20,
  candidate exclusion `train_only`, `Ks=[10,20,40,50]`, seed `2022`, batch
  size `1024`, maximum epochs `1000`, patience `7`,
  `smoke_train_batches=0`, and `run_final_test=true`;
  `dataset_config_overrides={}`, `student_config_overrides={}`, and both
  resolved-argument override maps are also `{}`.
- Resolved student configuration: `td_distill_no_projection`, embedding
  dimension `64`, random initialization, AdamW `student_lr=6e-5`, weight decay
  `0.01`, `td_distill_alpha=0`, and item-image, item-text, user-image, and
  user-text rates all `0`. The run is therefore the declared BPR-only reference
  with neither semantic loss nor teacher warm start.
- Validation selection:
  - validation-best epoch `163`;
  - exact Recall@20 `0.04291303921928788`;
  - exact Recall@50 `0.08005846771724855`;
  - exact NDCG@20 `0.01879878388702351`;
  - exact NDCG@50 `0.026503690746942317`;
  - the raw-log vectors at K `[10,20,40,50]`, rounded by the logger, are
    Precision `[0.00287, 0.00228, 0.00185, 0.00170]`, Recall
    `[0.02692, 0.04291, 0.06964, 0.08006]`, NDCG
    `[0.01457, 0.01880, 0.02451, 0.02650]`, and Hit Ratio
    `[0.02864, 0.04562, 0.07349, 0.08444]`.
- One-time final Test result after restoring validation-best epoch `163`, with
  exact vectors ordered by K `[10,20,40,50]`:
  - Precision `[0.002993057341218855, 0.0024787863203908794, 0.0019940858832605012, 0.0018441758806892075]`;
  - Recall `[0.026797499663274653, 0.044279857310199594, 0.07085836323043866, 0.08179702900453686]`;
  - NDCG `[0.015145598433562857, 0.019996624710057805, 0.02605878370945137, 0.02827561590531854]`;
  - Hit Ratio `[0.029724865003856682, 0.04906145538698948, 0.0783234764721007, 0.09040884546155713]`;
  - AUC `0.0`, expected because `test_flag=part` omits AUC.
- Test-access audit: the raw log contains exactly one student final-test event,
  after the `7/7` early-stop event, and explicitly states that epoch `163` was
  restored. The post-run closure invoked no training or test command and only
  read the already-recorded result.
- Manifest and eligibility audit: `status=completed`,
  `final_test_performed=true`, `run_final_test=true`,
  `paper_ready_eligible=true`, and `paper_ready_blockers=[]`. The retained
  official cold-item condition remains the declared preflight warning:
  validation/test have `11/7` interactions on three items absent from train.
- Frozen-teacher audit: pre-run and post-run SHA256 are both
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`
  for `Model/baby/teacher_model_val_test_once_v1.pt` (`141098540` bytes).
  Teacher training was skipped and alias overwrite remained disabled.
- Scoped run artifacts and SHA256:
  - manifest, `exp/runs/baby/run_manifest__2026-07-31 12_33_07.292506_baby_light_init_pid7872.json`, `25331` bytes,
    `cd2c5aa1fae79af8f8161dfee4fbb2bf9f451b591607df6c646d1dbf8b67af7d`;
  - raw log, `logs/2026-07-31 12_33_07.292506_baby_light_init_pid7872`, `76795` bytes,
    `2664a00666cd95920f572dca5b1fc4441fd01d8534b1a711e65a45d300f1e256`;
  - convergence record, `exp/converge/baby/auto__2026-07-31 12_33_07.292506_baby_light_init_pid7872.pkl`, `25278` bytes,
    `43c600a5362dd6e7d13e7569eaec2fe2ffd2e48082d33de182a15f8e8ae3876a`;
  - full checkpoint, `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-07-31 12_33_07.292506_baby_light_init_pid7872.pth`, `20354833` bytes,
    `6a9cbe7548ac41a2d362309656923aa4f21f8c4f59a372e510a97535ccb61507`;
  - inference-only checkpoint, `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-07-31 12_33_07.292506_baby_light_init_pid7872.pth`, `6785989` bytes,
    `dbf0bf5ac0fe22ddbea6da5e777a8549b321fe25ae4df9d66f7d97c0e787e206`.
- Artifact-content audit: the convergence record contains `171` finite epoch
  entries and agrees with the manifest on best epoch/Recall and final-test
  metrics. The format-v2 full checkpoint contains finite user
  `(19445,64)` and item `(7050,64)` embeddings plus two optimizer-state
  entries. The inference-only checkpoint contains exactly the two finite
  embeddings and deployment metadata; both tensors are shape-identical and
  bit-for-bit equal to the full checkpoint tensors.
- Verification evidence: original process exit `0`; structured manifest and
  raw-log chronology audit; convergence pickle finite-value/identity audit;
  PyTorch checkpoint structure, shape, finite-value, and full-to-inference
  exact-equality audit; SHA256 of the five scoped run artifacts and frozen
  teacher; focused Markdown/content/diff checks and Git index scope/checks
  passed before the result-record commit.
- Experimental meaning: the reference provides the required fair ID-only
  quality anchor for directional-distillation comparisons. Its final Test
  Recall@20 is materially below the frozen teacher's `0.08665691369856875`,
  but this is valid completed evidence rather than a failure or rollback
  trigger.
- Unresolved risks: this is one seed and not a variance estimate; the one-time
  test result is consumed and must not be used to tune the next candidate;
  `student_lr=6e-5` remains only the fixed reference choice rather than a tuned
  Baby optimum; ignored checkpoints and raw outputs still require explicit
  physical backup at a future stable milestone.
- Unique recommended next action: after this result-only commit, predeclare
  the first fair asymmetric no-projection directional-distillation candidate.
  Keep seed, dimension, random initialization, optimizer budget, sampling,
  protocol, and teacher fixed; name `td_distill_alpha` and directional rates
  before any run. Stop after the declaration and wait for explicit user
  authorization; do not launch it, create a tag, or merge `main` now.

### 2026-07-31 | Baby asymmetric no-projection directional candidate v1 (pending; declaration only)

- Purpose and hypothesis: predeclare the first fair directional-distillation
  candidate against the completed `baby_student_reference_v1` ID-only BPR
  anchor. The hypothesis is that moderate, asymmetric item-dominant semantic
  alignment can improve the same randomly initialized 64-dimensional student
  without projection heads, teacher warm start, additional deployable state, or
  any change to the reference optimization and evaluation budget.
- Status and authorization boundary: pending candidate specification only.
  Candidate/profile ID: `baby_td_asymmetric_no_projection_v1`. This task
  authorizes only this append-only declaration and its independent Git commit.
  No implementation, profile-resolution command, project-code execution,
  training, validation ranking, final-test evaluation, tag, merge, checkpoint
  publication, or baseline-asset overwrite is authorized or performed here.
- Parameter-selection firewall:
  - the semantic settings below are transferred unchanged from the archived
    pre-Baby asymmetric no-projection, no-warm-start research hypothesis and
    the transferable findings already summarized near the top of this active
    log;
  - they were not selected, refined, accepted, or rejected using the completed
    `baby_student_reference_v1` formal Test metrics or the frozen teacher Test
    metrics;
  - those already-recorded Test results remain final reference evidence only.
    They may be reported after a protocol-valid candidate run but may never
    choose this candidate or trigger an undeclared parameter adjustment.
- Branch, source identity, and rollback/preservation point:
  - branch `codex/experiment/baby-teacher-baseline`;
  - clean declaration predecessor and rollback point
    `6f806fa70de1c707dd109741b1c8fd2d3efdc29a`;
  - active student-protocol implementation commit
    `58fda059786067768e308f48e10ae864416334f5`;
  - completed reference launch/declaration commit
    `8d8858547a475fd970ae9753496b2a7c06eb472c`;
  - no rollback action is authorized; preserve this declaration and any future
    candidate evidence even if quality is low.
- Dataset and preprocessing identity: keep the audited MMRec Baby dataset under
  `data/baby/`, conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`,
  and the active train/validation/test matrix and modality identities recorded
  above. Keep the same `pca` hard-token/cache provenance required by the frozen
  teacher. No dataset, split, feature, preprocessing, or cache change is part
  of this candidate.
- Frozen teacher identity: reuse
  `Model/baby/teacher_model_val_test_once_v1.pt` read-only, SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  keep `if_train_teacher=false` and `allow_teacher_alias_overwrite=false`.
  Teacher training, mutation, alias publication, or overwrite is prohibited.
- Fixed fair-comparison controls inherited exactly from
  `baby_student_reference_v1`:
  - seed `2022`; `student_model_type=td_distill_no_projection`; user and item
    embedding dimension `64`; random initialization with
    `td_init_from_teacher=false`;
  - AdamW with `student_lr=6e-5` and `student_weight_decay=0.01`; batch size
    `1024`; maximum `1000` epochs; full `116` optimization batches per epoch;
    validation every epoch; early-stopping patience `7` consecutive
    non-improving validation evaluations; no smoke batch cap for a formal run;
  - unchanged `data_generator.sample()` pairwise BPR sampling: each batch
    samples `1024` distinct existing users without replacement, one uniformly
    selected training-positive item per user, and one uniformly selected item
    absent from that user's training interactions as the negative; users may
    recur across independently sampled batches, exactly as in the reference;
  - `val_test_once_v1`, validation Recall@20 checkpoint selection,
    `Ks=[10,20,40,50]`, `test_flag=part`, candidate exclusion `train_only`,
    dataset preflight enabled, duplicate-modality policy `error`, restoration
    of the validation-best checkpoint, and at most one final Test evaluation
    only after restoration in a separately authorized formal run.
- Predeclared semantic-loss delta, the only allowed behavioral difference from
  the reference:
  - `td_distill_alpha=0.3`;
  - `td_item_image_rate=1.0`;
  - `td_item_text_rate=0.3`;
  - `td_user_image_rate=0.0`;
  - `td_user_text_rate=0.0`.
- Exact objective meaning under the maintained implementation: for each BPR
  batch, the active semantic term is
  `(1.0 * L_item_image + 0.3 * L_item_text) / 1.3`, and the optimized objective
  is `L_BPR + 0.3 * L_semantic`. User-image and user-text heads are inactive;
  the no-projection model aligns the 64-dimensional student embeddings directly
  to the matching frozen-teacher semantic targets. The exported inference state
  must still contain only user and item ID embeddings.
- Rationale fixed before any Baby candidate run:
  - `td_distill_alpha=0.3` preserves the archived no-warm-start moderate-strength
    hypothesis: weaker settings under-transferred semantics, while the stronger
    archived `0.4` point showed non-monotonic collapse; this is a cross-dataset
    prior to retest, not a Baby optimum claim;
  - item-image rate `1.0` is the anchor semantic direction, while item-text rate
    `0.3` retains complementary text supervision at lower strength rather than
    imposing naive image/text symmetry;
  - both user-side rates remain `0.0` because the archived structure study found
    item-dominant supervision stronger than four-head symmetry and found that
    reintroducing user-side supervision did not improve the no-warm-start
    anchor;
  - no projection and no teacher warm start isolate the proposed training-only
    semantic transfer from projection loss and initialization leakage.
- Planned formal command after, and only after, a separately authorized and
  committed candidate-profile implementation plus its required smoke gate:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
  The profile must carry the declared semantic values, so the formal command
  must not use CLI semantic or optimizer overrides.
- Implementation readiness and declared future scope: current source recognizes
  only `baby_student_reference_v1` and treats semantic CLI changes to it as
  reference-profile overrides and paper-ready blockers. Therefore this
  declaration is intentionally not yet executable as a paper-ready candidate.
  A separately authorized next stage must add
  `baby_td_asymmetric_no_projection_v1` as an independent pinned profile,
  update the Baby student eligibility gate, add focused profile/protocol tests
  and candidate documentation, and append its own pending/completed trace. It
  must not change the sampler, model/loss implementation, reference profile, or
  any fixed control declared above.
- Hard acceptance criteria for the future implementation and run path:
  - candidate profile resolution has no overrides and differs from the
    reference resolved arguments only in profile identity plus the five
    semantic settings declared above;
  - focused tests prove fixed-control equality, exact semantic values, frozen
    teacher reuse, no warm start, no projection heads, run-manifest identity,
    and rejection of undeclared overrides;
  - a later validation-only smoke, if separately authorized, performs no Test
    ranking; a later formal run starts from clean committed source, uses
    validation Recall@20 alone for selection and early stopping, restores the
    selected checkpoint, and accesses Test once only after restoration;
  - losses, metrics, and exported embeddings are finite; the teacher hash is
    unchanged; artifacts are isolated and fingerprinted. There is no minimum
    quality threshold, and Test quality is not a parameter-selection or
    technical-acceptance criterion.
- Risks: all semantic weights are transferred from an invalid-for-paper old
  dataset environment and may not transfer to audited Baby; one seed cannot
  estimate variance; the semantic loss is normalized by the sum of active
  component rates, so the four rates encode relative direction balance rather
  than an additional absolute scale; and any accidental CLI override, warm
  start, sampler drift, teacher mutation, or repeated Test access would break
  the fair-comparison contract.
- Verification for this declaration: inspect the append-only diff and exact
  staged-file scope, run `git diff --check` and `git diff --cached --check`,
  create one commit containing only `TRAINING_LOG.md`, and confirm the final
  branch/HEAD/clean-tree state. Metrics and run artifacts are intentionally
  absent because no project code or evaluation is executed.
- Unique next action: only after explicit user authorization, implement and
  verify the independently pinned `baby_td_asymmetric_no_projection_v1`
  profile from this declaration, append its separate pending/completed trace,
  and create one code-and-documentation commit. Stop without training,
  validation ranking, or Test access; the expected artifacts are the profile,
  eligibility update, focused tests, candidate document, log outcome, and
  commit hash.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 profile implementation (pending)

- Purpose and rationale: implement the already predeclared
  `baby_td_asymmetric_no_projection_v1` as an independent pinned Baby student
  profile, so a future formal candidate can resolve without overrides while
  remaining subject to every existing paper-ready data, teacher, protocol,
  test-isolation, and artifact gate.
- Status and authorization: pending implementation only. The user explicitly
  authorized profile code, focused tests, candidate documentation, this
  append-only trace, proportional static/unit verification, and one coherent
  commit. Training, validation ranking, Test evaluation, run-artifact
  generation, tags, baseline overwrite, and merge to `main` are prohibited.
- Branch and rollback/preservation point:
  `codex/experiment/baby-teacher-baseline` at clean declaration commit
  `161e72a018dc21bbebdb39e4306d374f7473e20a`. No rollback action is
  authorized; this commit is the comparison point for scoped diff review.
- Declared implementation scope:
  - extend `codes/utility/dataset_profiles.py` with the independent candidate
    identity/defaults and a focused Baby student paper-ready profile gate;
  - update only the Baby student profile-identity portion of
    `codes/main_mmlight.py` to use that gate, without changing any other
    eligibility condition;
  - extend `codes/tests/test_dataset_profiles.py` with baseline immutability,
    exact profile-delta, override, eligibility, initialization, no-projection,
    frozen-teacher, and deployment-contract checks;
  - add `docs/BABY_TD_ASYMMETRIC_NO_PROJECTION_V1.md` and append the separate
    completed or failed outcome to this active log.
- Explicitly excluded scope: no change to `codes/utility/load_data.py`, either
  TD-Distill model implementation, semantic-loss computation, optimizer or
  training loop, evaluation protocol implementation, dataset/teacher assets,
  `baby_student_reference_v1` defaults/identity, parser defaults, checkpoint
  format, inference exporter, or generated artifacts. A material need to touch
  any excluded area requires a scope amendment before continuing.
- Fixed comparison contract inherited from `baby_student_reference_v1`: seed
  `2022`; `td_distill_no_projection`; dimension `64`; random initialization
  with `td_init_from_teacher=false`; AdamW `student_lr=6e-5`, weight decay
  `0.01`; batch size `1024`; maximum `1000` epochs; patience `7`; unchanged
  pairwise sampler; frozen read-only teacher
  `Model/baby/teacher_model_val_test_once_v1.pt` with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  and unchanged `val_test_once_v1`, validation Recall@20 selection,
  `train_only` candidate exclusion, preflight, finalization, and test-isolation
  rules.
- Only allowed profile-default differences:
  `td_distill_alpha=0.3`, `td_item_image_rate=1.0`,
  `td_item_text_rate=0.3`, `td_user_image_rate=0.0`, and
  `td_user_text_rate=0.0`. Profile name/scope/source metadata must independently
  identify the candidate; every other resolved default must equal the baseline.
- Paper-ready gate requirement: accept exactly the existing baseline identity
  and the new independently pinned candidate identity when metadata matches and
  resolved overrides are empty. Continue to block missing/mismatched identities
  and any undeclared CLI override. Do not weaken dataset profile, duplicate
  modality, protocol, smoke-cap, final-test, preflight, frozen-teacher, or alias
  protections.
- Risks: shared mutable defaults could accidentally change the baseline;
  generalized identity logic could admit unknown profiles; a CLI override could
  escape the blocker; or tests could assert only values without checking the
  direct ID-only deployment contract.
- Acceptance criteria:
  - the baseline defaults and metadata remain byte-for-value equivalent to the
    declaration predecessor;
  - the two default mappings differ at exactly the five declared semantic keys,
    and the formal candidate resolves with empty dataset/student overrides;
  - an undeclared candidate CLI override is recorded and returns a paper-ready
    blocker, while unknown/mismatched profile metadata remains blocked;
  - tests confirm random initialization, no-projection model selection, frozen
    teacher/read-only alias controls, and that the maintained no-projection
    inference exporter contains only user/item ID embeddings plus deployment
    metadata;
  - no sampler, model structure, semantic loss, baseline profile, protocol, or
    run artifact changes; all scoped tests and static checks pass.
- Planned verification: run focused dataset-profile unit tests, any focused
  deployment-contract unit test added within the declared test file, Python
  compilation for changed Python files, static CLI/profile resolution for both
  formal profiles and one undeclared override, source/diff checks proving
  excluded files are unchanged, `git diff --check`, exact staged-file review,
  and `git diff --cached --check`. Do not instantiate the training runner or
  access dataset splits.
- Metrics and artifacts: none expected. This is profile/protocol-enablement
  work only; no checkpoint, manifest, convergence record, raw log, validation
  metric, or Test metric may be generated.
- Next action after a verified commit: stop and request explicit authorization
  for a separately declared one-epoch, one-batch, validation-only candidate
  smoke with `run_final_test=false`.

#### Scope amendment before implementation

- Static inspection found that `codes/utility/parser.py` has three active
  dataset-branch definitions of `--student_profile`, each with argparse
  `choices` hard-coded to the empty selector plus
  `baby_student_reference_v1`. Without updating those choices, the declared
  independent candidate would be rejected by the CLI before profile defaults
  or eligibility metadata could resolve.
- Scope therefore expands narrowly to `codes/utility/parser.py`: import the
  supported Baby student profile-name collection and use it in exactly those
  three `--student_profile` choice lists. The empty default remains unchanged;
  no parser default, parameter type, dataset selection, or unrelated CLI option
  may change.
- Additional acceptance and verification: parser static resolution must accept
  both named Baby profiles, continue to reject unknown names, and preserve the
  baseline selector/default behavior. Include `parser.py` in Python compilation
  and final scoped diff review. No other excluded area is reopened.

#### Acceptance clarification after first focused test

- The first focused test correctly exposed an ambiguity in the pending phrase
  "differ at exactly the five declared semantic keys." All five semantic fields
  are the only allowed behavioral parameter set and must equal their
  predeclared candidate values, but `td_user_image_rate=0.0` and
  `td_user_text_rate=0.0` are intentionally equal to the baseline values.
- The precise acceptance rule is therefore: both profiles have identical keys;
  every field outside the five-field semantic set is identical; all five
  candidate semantic values equal the declaration; and the actual value-change
  set is exactly `td_distill_alpha`, `td_item_image_rate`, and
  `td_item_text_rate`. This clarification changes no parameter, scope, profile
  identity, or authorization boundary.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 profile implementation (completed; no run)

- Purpose and outcome: implemented and verified the independently pinned
  `baby_td_asymmetric_no_projection_v1` profile from declaration commit
  `161e72a018dc21bbebdb39e4306d374f7473e20a`. The profile is now accepted by
  the Baby student paper-ready identity gate with empty overrides, while the
  completed `baby_student_reference_v1` profile remains unchanged.
- Status: completed successfully as profile/protocol-enablement work only. No
  training runner, validation ranking, Test evaluation, or efficiency benchmark
  was started.
- Branch and source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean implementation predecessor
  `161e72a018dc21bbebdb39e4306d374f7473e20a`; the resulting coherent commit is
  to contain only the files listed below and will be reported in the task
  handoff and recorded by the next formal run declaration.
- Actual changes:
  - `codes/utility/dataset_profiles.py` now defines the candidate name, scope,
    source, independent copied defaults, explicit two-profile registry, CLI
    name collection, exact paper-ready identity collection, and a pure blocker
    helper used by the runtime gate;
  - `codes/utility/parser.py` now accepts either explicit Baby student profile
    name in each of its three active `--student_profile` choice lists; the empty
    default and all parser parameter defaults are unchanged;
  - `codes/main_mmlight.py` replaced only the hard-coded baseline student
    identity comparison with the focused blocker helper. Dataset identity,
    dataset overrides, duplicate modalities, teacher-only mode, smoke cap,
    final-Test enablement, preflight, frozen-teacher validation, and alias
    protections remain in place and unchanged;
  - `codes/tests/test_dataset_profiles.py` locks the complete baseline default
    mapping, exact candidate semantic contract, formal zero-override resolution,
    override/unknown-profile blockers, fixed random/no-projection/frozen-teacher
    settings, and ID-only inference export structure;
  - `docs/BABY_TD_ASYMMETRIC_NO_PROJECTION_V1.md` records the profile identity,
    data/teacher identities, pinned parameters, exact objective, comparison
    contract, validation-only smoke and formal gates, and artifact requirements.
- Resolved profile comparison:
  - both profiles have the same default-key set and match on every field outside
    `td_distill_alpha`, `td_item_image_rate`, `td_item_text_rate`,
    `td_user_image_rate`, and `td_user_text_rate`;
  - candidate values are exactly `0.3 / 1.0 / 0.3 / 0.0 / 0.0` in that order;
  - because both user-side rates are intentionally `0.0` in both profiles, the
    actual changed-value keys are alpha, item-image, and item-text only;
  - fixed controls resolve to seed `2022`, `td_distill_no_projection`, dimension
    `64`, random initialization, AdamW `student_lr=6e-5`, weight decay `0.01`,
    batch size `1024`, maximum `1000` epochs, patience `7`, frozen teacher mode,
    and the same protocol/preflight/finalization settings.
- Paper-ready and override verification:
  - real `utility.parser` static resolution gave `{}` for both dataset and
    student overrides for the baseline and candidate formal configurations;
  - the candidate resolved as scope `student_candidate`, semantic values
    `0.3 / 1.0 / 0.3 / 0.0 / 0.0`, random no-projection 64-dimensional student,
    `if_train_teacher=false`, and `allow_teacher_alias_overwrite=false`, with no
    profile-identity blocker;
  - an explicit candidate `--student_lr 0.001` resolved as an override from
    `6e-5` and produced the existing Baby student override blocker;
  - an unknown CLI profile name was rejected by argparse with exit code `2`,
    and unknown/mismatched metadata remains blocked by the pure gate helper.
- Verification evidence:
  - focused `test_dataset_profiles.py`: `15/15` tests passed;
  - full `codes/tests/test_*.py` suite: `35/35` tests passed in the declared
    environment, including protocol/preflight and converter tests on temporary
    synthetic fixtures only;
  - Python compilation passed for `dataset_profiles.py`, `parser.py`,
    `main_mmlight.py`, and `test_dataset_profiles.py`;
  - frozen teacher remained `141098540` bytes with SHA256
    `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  - static documentation checks found the candidate identity, all five semantic
    values, both smoke/formal final-Test controls, and inference-only export
    requirement;
  - `git diff 161e72a... --exit-code` was empty for
    `codes/utility/load_data.py`, `codes/td_distill_model.py`,
    `codes/td_distill_model_no_projection.py`, and
    `codes/utility/experiment_protocol.py`, proving no sampler, model,
    semantic-loss, exporter, or protocol implementation change;
  - manual scoped diff review passed; `git diff --check` passed with only the
    repository's existing CRLF conversion warnings for two touched files.
- Test-split and protocol audit: the real Baby train/validation/Test matrices
  were not loaded or accessed. No teacher or student ranking was evaluated.
  Mandatory reading of already-recorded Test evidence in this active log did
  not influence any parameter. The executed unit tests used temporary synthetic
  fixtures and complied with the no-run/no-Test authorization.
- Metrics and generated artifacts: none. No validation/Test metric, checkpoint,
  manifest, preflight report, convergence record, raw log, dataset derivative,
  teacher alias, or efficiency artifact was generated or changed. The only
  artifacts are scoped source, tests, documentation, this log outcome, and the
  pending Git commit.
- Acceptance decision: all corrected hard acceptance criteria passed. The
  implementation is ready for one coherent local commit; this is not a smoke,
  formal result, stable milestone, or claim that directional distillation works
  on Baby.
- Unresolved risks: the real runner/GPU path for the new candidate identity has
  not yet been exercised; the semantic settings remain cross-dataset hypotheses
  rather than Baby-validated choices; and a single future seed cannot estimate
  variance. These risks require future staged experiments, not broader changes
  in this implementation task.
- Unique next action: after this implementation commit and only with explicit
  user authorization, append and commit a separate declaration for one
  candidate validation-only smoke using `epoch=1`,
  `smoke_train_batches=1`, and `run_final_test=false`, then execute it once from
  clean committed source. Expected run-specific artifacts are the preflight
  report, manifest, convergence record, full/inference checkpoints, and raw log;
  no Test metric may be produced.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 validation-only smoke (pending)

- Purpose and hypothesis: exercise the first isolated GPU execution path for
  the independently pinned asymmetric no-projection candidate. One optimization
  batch should apply finite BPR plus item-image/item-text directional loss,
  validation Recall@20 should select and restore the only checkpoint, and the
  run should terminate without teacher or student Test ranking.
- Status and authorization: pending; explicitly non-formal, capped to one epoch
  and one batch, validation-only, and ineligible for paper result tables. The
  user authorized exactly one execution of the command below plus complete
  run-artifact/Test-isolation audit and a separate outcome commit. No uncapped
  or formal training, final Test evaluation, efficiency run, tag, baseline
  overwrite, or merge to `main` is authorized.
- Branch and source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - candidate declaration commit
    `161e72a018dc21bbebdb39e4306d374f7473e20a`;
  - verified candidate profile implementation commit
    `64739bbfa75e9baa657f666a78162f0b25aa590b`;
  - this pending declaration must be committed alone before launch. Its exact
    clean commit will be the smoke launch/source identity recorded in the
    outcome; `git diff 64739bb... <launch-commit> -- codes docs` must be empty.
- Dataset and preprocessing identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/validation/Test matrix and modality hashes remain the active identities
  recorded above; hard-token type/cache provenance and seed remain pinned to
  the frozen teacher contract. Dataset preflight remains enabled with duplicate
  modalities policy `error`.
- Evaluation protocol: `val_test_once_v1`; training uses `train_mat`; checkpoint
  selection uses validation Recall@20 only; candidate exclusion is `train_only`;
  `Ks=[10,20,40,50]`; `test_flag=part`. The one validation evaluation is a
  smoke diagnostic only and cannot select parameters or enter paper tables.
- Frozen teacher identity: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, expected size `141098540`
  bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `allow_teacher_alias_overwrite=false`. No teacher
  optimizer step, run-local teacher checkpoint, alias publication, or overwrite
  is authorized.
- Full command, to execute exactly once only after this declaration is
  committed and the tree is clean:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --epoch 1 --smoke_train_batches 1 --run_final_test false`
- Resolved fixed profile values: seed `2022`; batch size `1024`;
  `td_distill_no_projection`; embedding dimension `64`; random initialization
  with `td_init_from_teacher=false`; AdamW `student_lr=6e-5`, weight decay
  `0.01`; `td_distill_alpha=0.3`; component rates item-image `1.0`, item-text
  `0.3`, user-image `0.0`, user-text `0.0`; frozen teacher mode; no efficiency
  benchmark.
- Declared cap and expected override maps:
  - `epoch=1`, `smoke_train_batches=1`, `run_final_test=false`;
  - expected `dataset_config_overrides` exactly contains
    `epoch: 1000 -> 1`;
  - expected `student_config_overrides` exactly contains
    `epoch: 1000 -> 1`, `smoke_train_batches: 0 -> 1`, and
    `run_final_test: true -> false`;
  - explicit GPU, frozen-teacher flag, and checkpoint path equal maintained
    settings or fields outside the profile override map. No semantic, seed,
    optimizer, sampler, batch-size, protocol, or teacher override is allowed.
- Sampling and objective: retain `data_generator.sample()` exactly as the
  reference, drawing `1024` distinct existing users, one training positive and
  one non-training item negative per user. The cap must select one batch instead
  of the full `116`. The maintained objective is
  `L_BPR + 0.3 * ((1.0 * L_item_image + 0.3 * L_item_text) / 1.3)`;
  both user-side heads remain inactive and no projection head exists.
- Test isolation: dataset loading/preflight may read Test split structure and
  identity to enforce the audited dataset contract. Neither the frozen teacher
  nor candidate student may rank/evaluate Test users. Required manifest fields
  are `run_final_test=false`, `teacher_final_test_performed=false`, and
  `final_test_performed=false`; teacher and student final-Test result fields
  must both be absent. No Test metric may appear in the run outcome.
- Planned run-specific artifacts, all keyed by one new timestamp/PID run name:
  - raw log `logs/<run_name>`;
  - preflight `exp/runs/baby/dataset_preflight__<run_name>.json`;
  - manifest `exp/runs/baby/run_manifest__<run_name>.json`;
  - convergence record `exp/converge/baby/auto__<run_name>.pkl`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
- Hard acceptance criteria: launch commit and working tree are clean; profile
  identity and only the declared cap overrides resolve; dataset and frozen
  teacher validation pass; exactly one optimization batch and one validation
  selection finish with finite total/BPR/distillation/component values; active
  heads are exactly item-image and item-text; validation-best epoch `0` is saved
  and restored; manifest status is `smoke_completed`; final-Test fields satisfy
  the isolation contract; teacher SHA256 is unchanged; no teacher artifact or
  shared alias is created/updated; all six run-specific artifacts exist,
  correspond to the same run, and are fingerprinted; the inference-only export
  contains only finite user/item ID embeddings plus deployment metadata and is
  exactly equal to the full checkpoint embeddings.
- Quality acceptance rule: there is no minimum validation threshold. Any finite
  diagnostic metric from a protocol-valid one-batch smoke is accepted as
  execution evidence regardless of quality and must not trigger rollback,
  deletion, or parameter changes.
- Failure boundary and preservation: mark failed only for process/code failure,
  Test-isolation violation, non-finite values, identity/override mismatch,
  teacher mutation, missing/cross-run artifacts, or another hard criterion
  failure. Preserve every generated run-specific file for diagnosis; do not
  reset, delete, overwrite, or retry the run automatically.
- Planned verification: before launch, record environment, exact resolved
  profile/overrides, teacher SHA256, declaration/source diff, clean Git status,
  and existing run-artifact inventory. After process exit, identify the single
  new run; audit raw-log chronology and absence of Test evaluation; parse the
  manifest/preflight/convergence structures; validate finite metrics and exact
  profile/cap fields; inspect checkpoint keys/tensor shapes/finiteness/full-to-
  inference equality; fingerprint all artifacts and teacher; confirm no alias
  mutation and clean Git status; then append and commit one completed or failed
  outcome without starting another run.
- Rollback/preservation point: profile implementation commit
  `64739bbfa75e9baa657f666a78162f0b25aa590b`. No rollback action is authorized.
- Next action after the outcome commit: if and only if all smoke hard criteria
  pass, recommend a separately predeclared uncapped seed-2022 formal candidate
  run from clean committed source; do not execute it without new explicit user
  authorization.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 validation-only smoke (completed)

- Purpose and outcome: executed exactly one independently declared, non-formal
  GPU smoke for `baby_td_asymmetric_no_projection_v1`. The run completed one
  capped optimization batch with finite BPR and the two declared item-side
  directional losses, selected and restored epoch `0` by validation Recall@20,
  wrote all six expected run-specific artifacts, and did not rank or evaluate
  either teacher or student on Test.
- Status: completed successfully. All hard acceptance criteria in declaration
  commit `904cf39b04d7f11e75b1248f77792591210122dc` passed. The validation values
  below are execution diagnostics only: they were not used to change, select,
  or justify any candidate parameter and are ineligible for paper tables.
- Branch and source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - candidate declaration commit
    `161e72a018dc21bbebdb39e4306d374f7473e20a`;
  - candidate implementation commit
    `64739bbfa75e9baa657f666a78162f0b25aa590b`;
  - clean smoke launch/declaration commit
    `904cf39b04d7f11e75b1248f77792591210122dc`;
  - `codes/` and `docs/` had no difference between `64739bb...` and the launch
    commit, so the launch source differed only by the committed pending record.
- Executed command, exactly once:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --epoch 1 --smoke_train_batches 1 --run_final_test false`
- Process and environment: exit code `0`; run name
  `2026-07-31 15_56_00.884183_baby_light_init_pid15900`; manifest duration
  `13.823375` seconds; Python `3.10.20`; PyTorch `2.11.0+cu128`; CUDA `12.8`;
  NVIDIA GeForce RTX 5060 with driver `595.97`.
- Resolved profile and override audit:
  - fixed controls remained seed `2022`, batch size `1024`, model type
    `td_distill_no_projection`, student dimension `64`, random initialization
    with `td_init_from_teacher=false`, `student_lr=6e-5`, weight decay `0.01`,
    frozen teacher mode, `val_test_once_v1`, and the maintained sampler and
    `train_only` candidate-exclusion contract;
  - semantic values remained `td_distill_alpha=0.3`, item-image `1.0`,
    item-text `0.3`, user-image `0.0`, and user-text `0.0`;
  - dataset overrides were exactly `epoch: 1000 -> 1`; student overrides were
    exactly `epoch: 1000 -> 1`, `smoke_train_batches: 0 -> 1`, and
    `run_final_test: true -> false`; no undeclared override resolved;
  - the manifest recorded only the four expected paper-ready blockers: dataset
    profile override, student profile override, capped batches, and disabled
    final Test. No profile, data, teacher, protocol, or semantic blocker arose.
- Optimization evidence: exactly one batch ran instead of the uncapped `116`.
  Active semantic heads were exactly item-image and item-text; warm start was
  disabled and no projection module was used. All observed losses were finite:
  total `0.702596127986908`, BPR `0.6931980848312378`, aggregate semantic
  `0.03132675215601921`, item-image `0.031335148960351944`, item-text
  `0.031298764050006866`, user-image `0.0`, and user-text `0.0`.
- Validation-only evidence: best and restored epoch were both `0`. Exact
  validation Recall@20 was `0.0029656295534413314`, Recall@50 was
  `0.007874346447244367`, NDCG@20 was `0.0010943905658394775`, and NDCG@50 was
  `0.0021342817970126104`. Rounded values for `Ks=[10,20,40,50]` were Recall
  `[0.00145, 0.00297, 0.00632, 0.00787]`, Precision
  `[0.00015, 0.00015, 0.00017, 0.00017]`, Hit
  `[0.00149, 0.00309, 0.00679, 0.00843]`, and NDCG
  `[0.00070, 0.00109, 0.00184, 0.00213]`. There was no minimum quality gate.
- Manifest and preflight audit:
  - manifest status was `smoke_completed`, model stage was `td_distill`, and
    `run_final_test=false`, `teacher_final_test_performed=false`, and
    `final_test_performed=false`; no teacher or student final-result property
    was present;
  - split-overlap counts were all zero and modality arrays were finite and
    non-duplicate;
  - the only preflight warning was the already known official-split condition:
    three cold items cover 11 validation and 7 Test interactions.
- Test-split isolation: dataset loading/preflight accessed Test matrix structure
  and identity only to enforce the audited split contract. Neither teacher nor
  student performed Test ranking or metric evaluation. The raw chronology
  explicitly records frozen-teacher reuse without Test evaluation, checkpoint
  restore at epoch `0`, and final Test skipped. No Test metric was produced or
  used, so execution complied with the declared validation-only protocol.
- Run-specific artifacts, all present under the same run name and preserved as
  ignored experiment outputs:
  - raw log `logs/2026-07-31 15_56_00.884183_baby_light_init_pid15900`, `5835`
    bytes, SHA256
    `4a86df27ed85c8be27f0965aa5efc76d81b2865107c2891ac120922e047a40ba`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-07-31 15_56_00.884183_baby_light_init_pid15900.json`,
    `3399` bytes, SHA256
    `ddd61930a67b0b8c49733dc2446197dabaa6e22e16cd5cd37bbc449a3f8d535b`;
  - manifest
    `exp/runs/baby/run_manifest__2026-07-31 15_56_00.884183_baby_light_init_pid15900.json`,
    `25102` bytes, SHA256
    `d901c6dc2680e8dc65511ca46882d8ac266deacc106e1d3061d2f704eb5fd42e`;
  - convergence record
    `exp/converge/baby/auto__2026-07-31 15_56_00.884183_baby_light_init_pid15900.pkl`,
    `1245` bytes, SHA256
    `e9382896833a17da68605f2ca9d931db62b661cc24a914fcd5cd06d054516e7a`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-07-31 15_56_00.884183_baby_light_init_pid15900.pth`,
    `20354975` bytes, SHA256
    `34a56b386537f1c96ad888498e833a75a19f6406518e21c17aa573b64de775ec`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-07-31 15_56_00.884183_baby_light_init_pid15900.pth`,
    `6785997` bytes, SHA256
    `11e57f900da36de87e38d156a944aafa53ca5d331d8143aff0a8851baccbc34f`.
- Checkpoint/deployment audit: the full student model contained only user/item
  ID-embedding parameters with shapes `(19445, 64)` and `(7050, 64)`, plus two
  optimizer-state entries; every tensor was finite. The inference export
  contained exactly those two embeddings plus deployment metadata. Full and
  inference embeddings were bit-for-bit equal. No projection or warm-start
  state, run-local teacher checkpoint, generic student alias, or shared model
  overwrite was created.
- Frozen-teacher preservation: the declared teacher remained `141098540` bytes
  with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  No teacher optimizer step or teacher artifact was created.
- Verification performed after process exit: structured JSON inspection of the
  manifest and preflight; raw-log chronology review; convergence-record and
  checkpoint loading; tensor-key, shape, finiteness, optimizer-state, metadata,
  and full-to-inference equality checks; SHA256/size inventory for all six
  artifacts and the teacher; before/after inventory review for forbidden
  aliases and teacher artifacts; and Git scope/status review.
- Acceptance decision and experimental meaning: the candidate has passed its
  isolated runner, finite-loss, validation-selection, deployment-export, and
  Test-isolation smoke gates. This proves execution compatibility only. It does
  not establish recommendation quality, improvement over the reference,
  variance, paper-ready eligibility, or authorization for a formal run.
- Unresolved risks: one capped batch cannot predict formal validation quality;
  the asymmetric rates remain a predeclared cross-dataset hypothesis; and one
  future seed will not estimate variance. The official cold-item warning also
  remains part of the fixed audited split identity.
- Unique next action: with separate explicit user authorization, append and
  commit an uncapped seed-2022 formal-run declaration for this exact candidate
  from clean committed source, retaining the frozen teacher,
  `val_test_once_v1`, `epoch=1000`, patience `7`, no batch cap, and one-time
  final Test only after validation checkpoint selection. Do not execute that
  formal run as part of this task. Expected artifacts are a new formal
  preflight, manifest, convergence record, raw log, full/inference checkpoints,
  and the permitted one-time final Test evidence.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 formal seed-2022 uncapped (pending)

- Purpose and hypothesis: predeclare one fair, uncapped formal run of the
  independently pinned `baby_td_asymmetric_no_projection_v1` candidate against
  the completed Baby ID-only BPR reference. The hypothesis is that the fixed
  item-dominant directional semantic loss can improve the deployable student
  while leaving inference state, teacher identity, sampler, and all evaluation
  controls unchanged. This declaration does not claim that the hypothesis is
  true.
- Status and authorization boundary: pending; formal, uncapped, single seed
  `2022`, and eligible for one execution only after this declaration is in a
  clean committed tree. This task authorizes only the declaration and its Git
  commit. No training, validation ranking, Test evaluation, checkpoint
  creation, tag, merge to `main`, or baseline overwrite is authorized or
  performed here.
- Branch and source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean declaration predecessor and
  rollback/preservation point
  `179ac0fb2eb96de0b546c99833fce6f87feaa191` (the preceding smoke-result
  commit). The clean commit containing this pending declaration is the only
  authorized launch source identity for the later run; no code or documentation
  diff is permitted between that launch commit and the declared candidate
  implementation commit `64739bbfa75e9baa657f666a78162f0b25aa590b`.
- Dataset and preprocessing identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`; retain
  the active train/validation/Test matrix and modality hashes recorded above,
  the `pca` hard-token/cache provenance, and the audited `train_only`
  candidate-exclusion contract. Dataset identity, split assignment,
  preprocessing, and feature caches are frozen for this run.
- Evaluation protocol: `val_test_once_v1`; train only on `train_mat`; evaluate
  validation every epoch and select/early-stop only by validation Recall@20;
  use `Ks=[10,20,40,50]`, `test_flag=part`, and validation/Test candidates
  excluding training interactions only. Restore the validation-best student
  checkpoint before executing exactly one final student Test evaluation. Test
  metrics must not influence selection, parameter changes, or technical
  acceptance.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, expected size `141098540`
  bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, run-local
  teacher checkpoint, shared alias publication, or overwrite is allowed.
- Full formal command, to execute exactly once only after explicit user
  authorization and a clean committed launch source:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Resolved fixed parameters and override gate: seed `2022`; batch size `1024`;
  `td_distill_no_projection`; student embedding dimension `64`; random
  initialization with `td_init_from_teacher=false`; AdamW;
  `student_lr=6e-5`; student weight decay `0.01`; maximum `epoch=1000`;
  validation every epoch; early-stopping `patience=7`; and
  `smoke_train_batches=0` (all `116` batches per epoch). The profile carries
  `td_distill_alpha=0.3`, item-image rate `1.0`, item-text rate `0.3`,
  user-image rate `0.0`, and user-text rate `0.0`. Resolved
  `dataset_config_overrides={}` and `student_config_overrides={}` are required;
  no CLI profile, semantic, optimizer, seed, sampler, epoch, patience, batch,
  protocol, teacher, or final-Test override may be added.
- Objective and sampling: preserve `data_generator.sample()` exactly, with
  `1024` distinct existing users per batch, one training positive and one
  non-training negative per user. Optimize
  `L_BPR + 0.3 * ((1.0 * L_item_image + 0.3 * L_item_text) / 1.3)`; user-image
  and user-text heads remain inactive, and the no-projection student aligns
  directly in the 64-dimensional teacher semantic space.
- Planned run-specific artifacts, all keyed by one new timestamp/PID run name:
  raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  Generated artifacts remain ignored and must be fingerprinted in the later
  outcome; none is created by this declaration task.
- Hard acceptance criteria for the later run: launch source and working tree
  are clean; exact candidate identity and both override maps resolve as
  declared; dataset preflight and frozen-teacher validation pass; all training
  losses and validation metrics are finite; validation-best checkpoint is
  saved and restored; manifest reports formal completion with
  `run_final_test=true` and `final_test_performed=true`; exactly one final Test
  evaluation occurs after restoration; teacher SHA256 is unchanged; no teacher
  artifact or shared alias changes; and all run-specific artifacts are present,
  same-run, and fingerprinted with an inference export containing only finite
  user/item ID embeddings plus deployment metadata.
- Test-split isolation and evidence boundary: this task accessed no dataset
  split and produced no validation/Test metric or artifact. During the later
  formal run, Test structure may be read for preflight, but teacher and student
  Test ranking must occur once only after validation-best restoration. A finite
  protocol-valid result is accepted regardless of quality; no Test result may
  be used to tune or retroactively alter this declaration.
- Risks and preservation: the uncapped trajectory is long and single-seed;
  the asymmetric rates remain a cross-dataset hypothesis; GPU/dependency or
  identity/override drift could fail execution. Preserve all generated files
  and the pending declaration on any failure; do not reset, delete, retry, or
  roll back automatically.
- Planned verification after explicit authorization: record environment,
  resolved arguments and empty override maps, teacher SHA256, clean launch
  commit, and pre-run artifact inventory; then audit one new run's chronology,
  manifest/preflight/convergence data, validation selection and restoration,
  exactly-one final Test event, finite metrics, checkpoint keys/shapes/finiteness
  and full-to-inference equality, teacher/alias preservation, artifact hashes,
  and final Git status before appending a separate completed or failed outcome.
- Declaration verification and next action: this append-only diff is limited to
  `TRAINING_LOG.md`; no project code, data, checkpoint, validation, or Test
  command is run. After this pending declaration commit, wait for explicit user
  authorization such as `continue` before launching exactly the command above
  once; do not start another experiment stage in this task.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 formal seed-2022 uncapped declaration (completed; run not started)

- Status and scope: completed the requested declaration-recording change only;
  the formal run declared immediately above remains `pending`. No training,
  validation ranking, Test evaluation, project-code execution, checkpoint
  creation, tag, merge, or baseline overwrite was performed.
- Verification evidence: the append-only diff adds only the independent pending
  declaration and this outcome record to `TRAINING_LOG.md`; `git diff --check`
  passed; the pre-edit branch was clean; and no unrelated file is in scope.
- Metrics and artifacts: none. No test split was accessed, no run command was
  executed, and no validation/Test metric, manifest, preflight, convergence
  record, checkpoint, raw log, or asset hash was created or changed.
- Experimental meaning and unresolved risks: this commit establishes the
  reproducibility contract for the uncapped seed-2022 candidate but provides no
  evidence about recommendation quality or runtime. The declared semantic
  rates remain a cross-dataset hypothesis, and the single seed will not estimate
  variance. Preserve the pending record even if the later run is low-quality or
  fails a hard execution gate.
- Next action: after explicit user authorization, launch the exact declared
  command once from this clean commit and append its separate completed/failed
  run outcome with resolved parameters, validation selection, one-time final
  Test evidence, and artifact fingerprints.

### 2026-07-31 | baby_td_asymmetric_no_projection_v1 formal seed-2022 uncapped (completed)

- Purpose and outcome: executed exactly one declared uncapped formal run of
  `baby_td_asymmetric_no_projection_v1` against the completed Baby ID-only BPR
  reference. The process completed naturally, selected epoch `73` only by
  validation Recall@20, restored that checkpoint, evaluated the student Test
  split exactly once, and produced all six declared run-specific artifacts.
- Status and acceptance: completed successfully. Process exit code was `0`,
  every code/execution-flow/hard-acceptance criterion in declaration commit
  `6948baca05008a8adcb65fe0e83d65c0878e74d4` passed, and the result is valid
  formal single-seed evidence regardless of its quality. No retry, parameter
  change, rollback, deletion, tag, merge, or baseline overwrite occurred.
- Branch and source identity:
  - branch `codex/experiment/baby-teacher-baseline`;
  - clean launch and formal declaration commit
    `6948baca05008a8adcb65fe0e83d65c0878e74d4`;
  - candidate implementation commit
    `64739bbfa75e9baa657f666a78162f0b25aa590b`;
  - completed candidate smoke result commit
    `179ac0fb2eb96de0b546c99833fce6f87feaa191`;
  - pre-launch `git diff 64739bb... 6948baca... -- codes docs` was empty,
    HEAD matched the declaration commit, and the tracked working tree was
    clean. The run did not modify tracked source.
- Executed command, exactly once:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Run and environment identity: run name
  `2026-07-31 18_35_00.450484_baby_light_init_pid22864`; manifest start
  `2026-07-31T18:35:00.452484+08:00`; completion
  `2026-07-31T18:51:45.182081+08:00`; manifest elapsed approximately
  `16:44.73` and command wall time approximately `1008.9` seconds; Python
  `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`.
- Resolved profile, parameters, and override gate:
  - `dataset_config_overrides={}` and `student_config_overrides={}` before and
    during the run;
  - seed `2022`, batch size `1024`, maximum `epoch=1000`, patience `7`,
    `smoke_train_batches=0`, `run_final_test=true`, and `116` full batches per
    epoch;
  - `val_test_once_v1`, validation Recall@20 selection, candidate exclusion
    `train_only`, `Ks=[10,20,40,50]`, `test_flag=part`, dataset preflight
    enabled, and duplicate-modalities policy `error`;
  - `td_distill_no_projection`, dimension `64`, random initialization with
    `td_init_from_teacher=false`, AdamW `student_lr=6e-5`, and weight decay
    `0.01`;
  - `td_distill_alpha=0.3`; component rates item-image `1.0`, item-text `0.3`,
    user-image `0.0`, and user-text `0.0`; active heads were exactly item-image
    and item-text. No projection or teacher warm start was used.
- Dataset/preflight audit: the audited Baby conversion manifest, train,
  validation, Test, image, and text hashes all matched the active identities.
  Pairwise split overlaps were `0 / 0 / 0`; both modalities were finite,
  nonzero, and non-duplicate. The sole preflight warning remained the fixed
  official-split condition: items `240`, `1212`, and `6115` are absent from
  train and cover `11` validation plus `7` Test interactions; cold users are
  `0`. No dataset or preprocessing asset changed.
- Optimization and stopping evidence: epochs `0-80` completed, each with all
  `116` batches, for `9396` optimization batches. The run then stopped
  naturally after `7/7` consecutive non-improving validation evaluations.
  All `81` recorded values in every loss and validation series were finite.
  First-to-last epoch sums were total `81.4317097067833 -> 58.99385267496109`,
  BPR `80.39764374494553 -> 58.53531700372696`, semantic
  `3.4468876123428345 -> 1.5284535139799118`, item-image
  `3.4144498836249113 -> 1.1873003570362926`, and item-text
  `3.555012971162796 -> 2.6656305082142353`; both inactive user-side losses
  remained exactly `0.0`.
- Validation selection result at best epoch `73`:
  - exact Recall@20 `0.06842596333982363`;
  - exact Recall@50 `0.12096829886492674`;
  - exact NDCG@20 `0.03004422336248142`;
  - exact NDCG@50 `0.04076260342592865`;
  - raw-log vectors rounded at K `[10,20,40,50]`: Recall
    `[0.04203, 0.06843, 0.10397, 0.12097]`, Precision
    `[0.00440, 0.00360, 0.00273, 0.00254]`, Hit Ratio
    `[0.04397, 0.07169, 0.10831, 0.12600]`, and NDCG
    `[0.02308, 0.03004, 0.03754, 0.04076]`.
- One-time final Test result after restoring validation-best epoch `73`, with
  exact vectors ordered by K `[10,20,40,50]`:
  - Precision `[0.004674723579326367, 0.0036616096682952452, 0.0028683466186679293, 0.0026320390845977456]`;
  - Recall `[0.0421596544365896, 0.06659234097521655, 0.1041240614189935, 0.11925134349664723]`;
  - NDCG `[0.024242770942206778, 0.03081209766712305, 0.03912474568059561, 0.042142503473832]`;
  - Hit Ratio `[0.046490100282849466, 0.07271792234507622, 0.11313962458215178, 0.12959629724864688]`;
  - AUC `0.0`, expected because `test_flag=part` omits AUC.
- Test-access and chronology audit: dataset preflight read Test structure and
  identity under the declared contract. The raw log contains `81` validation
  events, one early-stop event, and exactly one student final-Test event after
  restoring best epoch `73`. It contains no teacher Test-evaluation event;
  teacher metrics printed at reuse came from the already frozen checkpoint.
  No post-run command invoked ranking, validation, training, or Test again.
- Manifest and eligibility audit: manifest status is `completed`, model stage
  `td_distill`, protocol `val_test_once_v1`, selection split `validation`,
  primary metric `Recall@20`, candidate exclusion `train_only`,
  `run_final_test=true`, `final_test_performed=true`,
  `paper_ready_eligible=true`, and `paper_ready_blockers=[]`. Its best epoch,
  best validation Recall, final Test vectors, profile, arguments, checkpoint
  paths, and empty override maps agree with the raw log, convergence record,
  and checkpoint metadata.
- Run-specific artifacts, all present under the same run identity:
  - raw log `logs/2026-07-31 18_35_00.450484_baby_light_init_pid22864`, `39273`
    bytes, SHA256
    `1b544004190bf48f3364d6fae820fcf5f838205fd7ad4b1d667ba5bb9966ba28`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-07-31 18_35_00.450484_baby_light_init_pid22864.json`,
    `3399` bytes, SHA256
    `ef771f6373643947eb790179cb4434607251521221481a813138c62faadec187`;
  - manifest
    `exp/runs/baby/run_manifest__2026-07-31 18_35_00.450484_baby_light_init_pid22864.json`,
    `25398` bytes, SHA256
    `967779e828f820ece44bd2a75331ee48a1a4e45ca2cefbb7b43bbe5eb3793480`;
  - convergence record
    `exp/converge/baby/auto__2026-07-31 18_35_00.450484_baby_light_init_pid22864.pkl`,
    `12770` bytes, SHA256
    `c8eb564c3a03e5ae74fe769d5f27f865e906ac989caf64751db2d7212a8093dc`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-07-31 18_35_00.450484_baby_light_init_pid22864.pth`,
    `20354975` bytes, SHA256
    `e764fbac66172391e7ee89006665e95484954574f4b6c2df51c2f099144bf375`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-07-31 18_35_00.450484_baby_light_init_pid22864.pth`,
    `6785997` bytes, SHA256
    `f1df1389823902b08f0e4617d1deff856a021a9342c0b7191d3b00895316652a`.
- Checkpoint/deployment audit: the format-v2 full checkpoint contains finite
  user `(19445,64)` and item `(7050,64)` ID-embedding parameters plus exactly
  two optimizer-state entries. The inference checkpoint contains exactly those
  two finite embeddings plus `embedding_dim=64`, `n_users=19445`,
  `n_items=7050`, and variant `td_distill_no_projection`. Its user and item
  tensors are bit-for-bit equal to the full checkpoint tensors. No teacher,
  prompt, modality, semantic cache, projection, graph, or optimizer state is in
  the inference export.
- Frozen-teacher and output-scope audit: the shared teacher remained
  `141098540` bytes with unchanged timestamp and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  No teacher optimizer step, run-local teacher checkpoint, generic student
  checkpoint, or shared alias write occurred. A post-run timestamp inventory
  found exactly the six declared run-specific files and no additional output.
- Verification evidence: exact pre-launch profile resolution and clean Git
  gate; source-diff, environment, GPU, disk, conversion-manifest, and teacher
  checks; process exit review; raw-log event counts and chronology; structured
  manifest and preflight inspection; convergence-series finiteness, length,
  argmax, and final-result equality assertions; PyTorch checkpoint key, shape,
  finiteness, optimizer-state, metadata, and full-to-inference exact-equality
  assertions; SHA256/size inventory; forbidden-artifact and teacher-mutation
  checks; and final tracked Git status all passed.
- Experimental meaning: the candidate improved exact validation Recall@20 over
  the seed-2022 ID-only reference by `0.025512924120535754`. Its final Test
  Recall@20 `0.06659234097521655` exceeded the reference
  `0.044279857310199594` by `0.022312483665016952` absolute, approximately
  `50.39%` relative, while remaining `0.0200645727233522` below the frozen
  teacher. This supports the predeclared asymmetric directional-distillation
  hypothesis for this seed only; it does not establish variance, multi-seed
  significance, or a generally optimal semantic setting.
- Unresolved risks: only seed `2022` has formal candidate evidence; the already
  consumed final Test result must not tune future semantic weights; the fixed
  official cold-item warning remains; and ignored raw logs/checkpoints are not
  protected by Git and still need explicit physical/off-device backup at a
  future stable milestone.
- Unique next action: after this outcome commit and only with explicit user
  authorization, append and commit a declaration-only matched seed-2023
  replication stage. It must specify separate override-free seed-2023 baseline
  and candidate profiles, keep every non-seed control and the frozen teacher
  fixed, require sequential baseline-then-candidate execution with validation
  selection and one-time Test per run, and stop without implementation or
  execution. The expected artifact is one auditable `TRAINING_LOG.md`
  declaration commit defining the later profile/run gates.

### 2026-07-31 | matched Baby seed-2023 baseline/candidate replication stage (pending; declaration only)

- Purpose and hypothesis: predeclare one matched seed-2023 replication pair to
  test whether the direction of the seed-2022 asymmetric-distillation gain
  persists under a second student initialization and sampling stream. The
  comparison remains the ID-only BPR baseline versus the independently pinned
  asymmetric no-projection candidate; this stage changes the stochastic run
  seed only and does not tune or revise either method.
- Status and authorization boundary: pending replication specification only.
  This task authorizes this append-only declaration, its separate declaration
  outcome, focused Markdown/Git verification, and one coherent commit. It does
  not authorize profile implementation or resolution, project-code execution,
  dataset loading, training, validation ranking, Test evaluation, run artifact
  creation, tag, bundle, baseline overwrite, or merge to `main`.
- Parameter-selection firewall:
  - run seed `2023` is the next predeclared replication seed, selected to address
    the already recorded single-seed risk rather than to optimize any metric;
  - both future seed-2023 profiles and their complete parameter mappings must
    be implemented and committed together before the seed-2023 baseline run,
    so baseline validation/Test evidence cannot alter the later candidate;
  - the completed seed-2022 reference and candidate Test results motivate
    replication but do not select a new learning rate, semantic rate, stopping
    rule, or other method control;
  - no seed-2023 baseline result may tune, cancel, reorder, or otherwise change
    the already frozen seed-2023 candidate. Low or reversed paired results
    remain valid evidence and cannot trigger automatic rollback or deletion.
- Branch and preservation point: branch
  `codex/experiment/baby-teacher-baseline`; clean declaration predecessor and
  rollback/preservation point
  `92b64136e3ca75fa0f41a11e29f907a9403d54b3`, the completed seed-2022
  candidate formal-result record. Seed-2022 baseline result commit
  `6f806fa70de1c707dd109741b1c8fd2d3efdc29a`, candidate implementation commit
  `64739bbfa75e9baa657f666a78162f0b25aa590b`, and every existing run/result
  artifact remain immutable comparison evidence. No rollback is authorized.
- Proposed future override-free profile identities, not yet implemented:
  - baseline `baby_student_reference_seed2023_v1`;
  - candidate `baby_td_asymmetric_no_projection_seed2023_v1`.
  Current source contains neither identity. A separate authorized implementation
  stage must add both profiles and verify them without launching a runner.
- Exact seed delta: each new profile must copy its seed-2022 counterpart and
  change only the training/sampling `seed` from `2022` to `2023`. In
  particular, `hard_token_seed=2022` remains fixed because it identifies the
  frozen teacher preprocessing/cache contract rather than the student run RNG.
  No future formal command may use `--seed`; the profile must carry seed `2023`
  so `dataset_config_overrides={}` and `student_config_overrides={}` remain
  empty.
- Dataset and preprocessing identity: retain audited MMRec Baby under
  `data/baby/`, conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`,
  and the active train/validation/Test matrix plus image/text hashes recorded
  above. Keep official split assignments, cold-item policy
  `retain_official`, `pca` hard-token type, `hard_token_seed=2022`, existing
  cache provenance, dataset preflight, and duplicate-modality policy `error`.
  No dataset, split, feature, preprocessing, or cache change is allowed.
- Frozen teacher identity and controls: both future runs reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, expected size `141098540`
  bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. Teacher training, optimizer steps,
  run-local teacher checkpoints, shared-alias publication, and overwrite are
  prohibited.
- Fixed non-seed controls shared by both future profiles: batch size `1024`;
  `td_distill_no_projection`; embedding dimension `64`; random initialization
  with `td_init_from_teacher=false`; AdamW `student_lr=6e-5`; student weight
  decay `0.01`; maximum `epoch=1000`; validation every epoch; patience `7`;
  `smoke_train_batches=0`; all `116` batches per epoch; unchanged
  `data_generator.sample()` pairwise BPR sampler; no efficiency benchmark;
  and an inference export containing only user/item ID embeddings plus
  deployment metadata.
- Fixed method-specific controls:
  - seed-2023 baseline retains `td_distill_alpha=0.0` and item-image,
    item-text, user-image, and user-text rates all `0.0`;
  - seed-2023 candidate retains `td_distill_alpha=0.3`, item-image `1.0`,
    item-text `0.3`, user-image `0.0`, and user-text `0.0`;
  - the only actual behavioral parameter differences between the paired
    profiles remain alpha, item-image, and item-text; both use no projection,
    no teacher warm start, and identical deployment state.
- Evaluation contract for each future formal run: `val_test_once_v1`; train on
  `train_mat`; select and early-stop only by validation Recall@20; candidate
  exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`; restore the
  validation-best checkpoint; then execute exactly one final student Test
  evaluation for that run. Test metrics cannot influence technical acceptance
  or any profile value.
- Mandatory future sequence, with no parallel or combined launch:
  1. implement and verify both seed-2023 profiles together, commit the complete
     parameter freeze, and stop without training or evaluation;
  2. in a later task, append and commit a separate seed-2023 baseline formal
     pending declaration, then execute only that baseline once after explicit
     authorization and commit its completed/failed outcome;
  3. only after the baseline outcome commit, append and commit a separate
     seed-2023 candidate formal pending declaration, then execute only that
     candidate once after new explicit authorization and commit its outcome.
  The baseline and candidate may not be launched concurrently or in one task.
- Planned future formal commands after the two profiles exist, each requiring
  its own committed run declaration and explicit launch authorization:
  - baseline:
    `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`;
  - candidate:
    `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`.
  Neither command is executable or authorized by this declaration task.
- Future profile-implementation acceptance criteria: existing seed-2022
  profiles remain value-identical; each seed-2023 profile has the same default
  keys and differs from its counterpart only at `seed`; both new identities
  are explicitly recognized while unknown/mismatched identities remain
  blocked; formal static resolution yields empty dataset/student override maps;
  baseline/candidate non-seed equality and exact semantic delta are locked by
  focused tests; frozen-teacher, no-warm-start, no-projection, protocol, and
  inference-only deployment gates remain unchanged.
- Future run acceptance criteria for each member of the pair: launch from its
  clean committed declaration; exact profile identity and empty overrides;
  dataset and teacher validation pass; finite loss/validation series;
  validation Recall@20 alone selects and early-stops; validation-best restore
  precedes exactly one final Test; manifest status `completed`,
  `paper_ready_eligible=true`, blockers empty, and
  `final_test_performed=true`; teacher hash unchanged; six isolated same-run
  artifacts exist and are fingerprinted; full and inference embeddings are
  finite and exactly equal. There is no minimum quality threshold.
- Planned run-specific artifacts for each later baseline/candidate execution:
  raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  Every run uses a distinct timestamp/PID identity; no shared student alias is
  allowed.
- Risks and evidence limits: seed `2023` plus seed `2022` still provides only
  two paired observations and cannot by itself establish a stable variance or
  significance estimate; two future one-time Test accesses are required for
  the matched pair and must remain report-only; the official cold-item warning
  persists; long GPU runs may fail; and ignored artifacts require explicit
  physical backup at a later stable milestone. Preserve all completed or
  partial evidence without destructive recovery.
- Declaration acceptance and verification plan: this task must append only this
  pending specification plus its declaration outcome to `TRAINING_LOG.md`,
  confirm that no seed-2023 profile currently exists, run Markdown/content,
  `git diff --check`, exact staged-file, and `git diff --cached --check`
  reviews, create one local commit, and finish with a clean tree. No project
  code, dataset, training, validation, or Test command may run.
- Unique next action after this declaration commit: only with explicit user
  authorization, implement and verify both proposed seed-2023 profiles in one
  coherent code/test/documentation task, append its own pending/completed
  trace, and commit it. Stop without profile smoke, formal training,
  validation ranking, Test access, tag, bundle, or merge. Expected artifacts
  are the two pinned profile definitions, supported-identity/eligibility
  updates, focused tests, seed-2023 replication documentation, log outcome,
  and one implementation commit.

### 2026-07-31 | matched Baby seed-2023 replication declaration (completed; no profiles or runs)

- Status and scope: completed the requested declaration-recording task only;
  the matched seed-2023 replication stage above remains `pending`. No profile
  was implemented or resolved, and no project code, dataset, training,
  validation, Test, efficiency, tag, bundle, merge, or baseline action ran.
- Verification evidence: the pre-edit tree was clean at
  `92b64136e3ca75fa0f41a11e29f907a9403d54b3`; current source has no seed-2023
  profile identity; the declaration freezes both future profiles before any
  seed-2023 evidence, distinguishes run seed `2023` from frozen
  `hard_token_seed=2022`, fixes all non-seed controls, and states the mandatory
  baseline-outcome-before-candidate sequence. The final append-only diff and
  staged scope are limited to `TRAINING_LOG.md` and must pass Git checks before
  this declaration commit.
- Metrics, Test access, and generated artifacts: none. No Baby split was loaded
  or accessed; no validation/Test metric, profile-resolution output,
  checkpoint, preflight, manifest, convergence record, raw log, or efficiency
  result was generated or changed. This log declaration and its Git commit are
  the only task artifacts.
- Experimental meaning and unresolved risks: the replication contract now
  prevents seed-2023 baseline evidence from changing the candidate and makes
  training seed the sole cross-seed delta. It does not add replication evidence
  yet; two seeds will still be insufficient for a strong variance/significance
  claim, and future ignored artifacts will remain outside Git protection.
- Next action: with separate explicit authorization, implement and verify both
  override-free seed-2023 profiles together, create one scoped source/test/doc
  commit, and stop without running either member of the pair.

### 2026-07-31 | matched Baby seed-2023 profile implementation (pending)

- Purpose and rationale: implement the predeclared matched replication
  identities `baby_student_reference_seed2023_v1` and
  `baby_td_asymmetric_no_projection_seed2023_v1` together, before either arm
  produces evidence. Each profile must copy its seed-2022 counterpart and
  change only the training/sampling `seed` from `2022` to `2023`, preserving
  the parameter-selection firewall and baseline-before-candidate run order.
- Status and authorization: pending implementation and static/unit verification
  only. The user authorized both profile definitions, supported identity and
  eligibility wiring, focused tests, seed-2023 replication documentation,
  this append-only trace, and one coherent commit. Training, validation
  ranking, Test evaluation, dataset loading, profile smoke, run artifacts,
  tags, bundles, baseline overwrite, and merge to `main` are prohibited.
- Branch and rollback/preservation point: branch
  `codex/experiment/baby-teacher-baseline` at clean declaration commit
  `df114dcd67c68209e8f15cc7371a633f81a362d4`. Preserve both completed
  seed-2022 profiles and all prior experiment evidence; no rollback or deletion
  is authorized.
- Declared implementation scope:
  - add both independent seed-2023 identities, metadata, copied defaults, and
    registry entries in `codes/utility/dataset_profiles.py`;
  - narrowly make dataset-profile override resolution defer overlapping fields
    owned by an active registered student profile to the existing student
    override audit. This is required because the student run seed becomes
    `2023` while the teacher/dataset profile seed remains `2022`;
  - extend `codes/tests/test_dataset_profiles.py` to lock exact cross-seed and
    matched-arm deltas, empty formal override maps, eligibility, rejection of
    overrides/unknown identities, and all frozen controls;
  - add `docs/BABY_SEED2023_REPLICATION_PROFILES_V1.md`, then append a separate
    completed or failed outcome to this active log.
- Explicitly excluded scope: no change to parser defaults or unrelated CLI
  options, `codes/main_mmlight.py`, data loading or sampling, model/loss or
  optimizer implementation, evaluation/finalization protocol, checkpoint or
  inference-export format, dataset/teacher assets, existing profile values, or
  generated experiment artifacts. Existing parser choices and runtime
  eligibility consume the central registered-profile collections and should
  need no source edit; any material need beyond the declared files requires a
  scope amendment before continuing.
- Exact profile contract:
  - seed-2023 baseline defaults equal `baby_student_reference_v1` at every key
    except `seed=2023`;
  - seed-2023 candidate defaults equal
    `baby_td_asymmetric_no_projection_v1` at every key except `seed=2023`;
  - matched seed-2023 non-semantic controls are identical; baseline semantic
    values remain all zero, while candidate values remain alpha `0.3`,
    item-image `1.0`, item-text `0.3`, and both user rates `0.0`;
  - `hard_token_seed=2022` remains dataset/teacher-owned and is not added to a
    student profile. Frozen teacher mode, no projection, no warm start,
    dimension `64`, AdamW `student_lr=6e-5`, weight decay `0.01`, batch `1024`,
    `epoch=1000`, patience `7`, `smoke_train_batches=0`,
    `val_test_once_v1`, and final-Test-after-best-restore contract stay fixed.
- Metadata-resolution boundary: for an active registered student profile,
  overlapping fields in its defaults are student-owned for override reporting
  and must be audited by `resolved_student_profile_metadata`; dataset-only
  fields, including `hard_token_seed`, remain audited against the frozen
  dataset/teacher profile. An undeclared CLI seed change must still appear in
  `student_config_overrides` and block paper-ready eligibility. Runs without an
  active student profile retain the current dataset-profile audit unchanged.
- Risks: shared mutable mappings could alter seed-2022 defaults; overly broad
  ownership logic could hide a CLI override; registry expansion could admit an
  unknown identity; or the two seed-2023 arms could drift at a non-seed control.
  Static resolution must prove that declared profile fields are audited once,
  dataset-only fields remain protected, and unknown/mismatched metadata stays
  blocked.
- Acceptance criteria: both seed-2022 mappings remain value-identical; each
  seed-2023 mapping has the same key set and differs from its counterpart only
  at `seed`; the seed-2023 pair differs only at the three actually changed
  semantic values and their profile metadata; both formal CLI configurations
  resolve with `dataset_config_overrides={}` and
  `student_config_overrides={}`; `seed=2023` and
  `hard_token_seed=2022`; all four exact identities are eligible only with
  matching metadata and empty overrides; unknown identities and a CLI seed
  override remain blocked; all fixed teacher, protocol, initialization,
  architecture, optimizer, budget, and deployment controls remain unchanged.
- Planned verification: focused `test_dataset_profiles.py`; full
  `codes/tests/test_*.py` unit suite on temporary/synthetic fixtures only;
  Python compilation of changed Python files; static actual-parser resolution
  of both future formal commands without importing the training runner;
  focused documentation/registry checks; source/diff checks for excluded
  modules; `git diff --check`; exact staged-file review; and
  `git diff --cached --check`. No dataset split, runner, training, validation
  ranking, or Test command will execute.
- Metrics and run artifacts: none expected. The only task artifacts are the
  scoped profile source, tests, documentation, this pending/outcome trace, and
  the resulting Git commit.
- Unique next action after a verified implementation commit: with separate
  explicit authorization, append and commit only the seed-2023 baseline formal
  pending declaration, then stop before execution. The candidate declaration
  and launch remain gated until the baseline has run and its outcome commit
  exists.

#### Metadata-resolution acceptance clarification before implementation

- To preserve existing override evidence for capped `epoch`, batch, protocol,
  and other shared controls, student ownership applies only when an active
  registered student profile intentionally pins an overlapping field to a
  value different from the dataset/teacher profile. Identical overlapping
  defaults retain the existing dataset-profile comparison and may continue to
  appear in both override maps when changed by CLI.
- For the declared seed-2023 profiles the only such field is `seed`: formal
  resolution treats the registered student expectation as `2023`, while an
  undeclared CLI value differs from that expectation and is recorded by both
  metadata audits. `hard_token_seed` remains dataset-only at `2022`. This
  clarification changes no profile value, file scope, or authorization gate.

### 2026-07-31 | matched Baby seed-2023 profile implementation (completed; no run)

- Purpose and outcome: implemented and verified both predeclared seed-2023
  replication profiles together from declaration commit
  `df114dcd67c68209e8f15cc7371a633f81a362d4`. The baseline and candidate are
  now independently registered, accepted by the exact Baby student identity
  gate, and resolvable without overrides before either arm produces evidence.
- Status and authorization compliance: completed successfully as profile and
  reproducibility-enablement work only. No dataset split or training runner was
  loaded; no profile smoke, training, validation ranking, Test evaluation,
  efficiency benchmark, tag, bundle, baseline overwrite, or merge to `main`
  occurred.
- Branch and source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean implementation predecessor
  `df114dcd67c68209e8f15cc7371a633f81a362d4`. The resulting coherent commit
  will contain only the four scoped files below and must be cited by the later
  seed-2023 baseline formal declaration.
- Actual changes:
  - `codes/utility/dataset_profiles.py` defines independent seed-2023 baseline
    and candidate names, sources, copied defaults, and registry entries. The
    central profile-name tuple automatically extends the existing parser
    choices, and the central exact-identity set automatically extends the
    maintained runtime eligibility gate;
  - dataset metadata resolution now adopts a registered student profile's
    expected value only for overlapping defaults that intentionally differ
    from the dataset/teacher profile. This permits the declared run
    `seed=2023` without treating it as an override while preserving the prior
    audit behavior for identical shared controls and dataset-only fields;
  - `codes/tests/test_dataset_profiles.py` locks both seed-only cross-seed
    deltas, the matched-arm semantic delta, exact formal resolution, frozen
    controls, CLI seed-override evidence, dataset ownership of
    `hard_token_seed`, supported identities, and paper-ready blockers;
  - `docs/BABY_SEED2023_REPLICATION_PROFILES_V1.md` records the two profile
    identities, exact seed delta, frozen teacher/non-seed controls, mandatory
    baseline-outcome-before-candidate sequence, future commands, and the
    validation-best-restore-before-single-Test contract.
- Exact defaults and comparison audit:
  - `baby_student_reference_seed2023_v1` has the same default keys and values
    as `baby_student_reference_v1` except `seed: 2022 -> 2023`;
  - `baby_td_asymmetric_no_projection_seed2023_v1` has the same default keys
    and values as `baby_td_asymmetric_no_projection_v1` except
    `seed: 2022 -> 2023`;
  - the matched seed-2023 profiles share seed `2023` and every non-semantic
    control. Their actual changed-value set remains exactly
    `td_distill_alpha`, `td_item_image_rate`, and `td_item_text_rate`;
  - baseline semantic values are all `0.0`; candidate values are alpha `0.3`,
    item-image `1.0`, item-text `0.3`, user-image `0.0`, and user-text `0.0`.
- Static actual-parser resolution, using each complete future formal argument
  list without importing `main_mmlight.py`:
  - baseline resolved profile/scope/source as
    `baby_student_reference_seed2023_v1` / `student_reference` /
    `predeclared_baby_id_only_bpr_reference_seed2023`, with `seed=2023`,
    `hard_token_seed=2022`, `epoch=1000`, patience `7`,
    `smoke_train_batches=0`, frozen teacher mode, `run_final_test=true`, and
    both override maps `{}`;
  - candidate resolved profile/scope/source as
    `baby_td_asymmetric_no_projection_seed2023_v1` / `student_candidate` /
    `predeclared_baby_asymmetric_no_projection_directional_v1_seed2023`, with
    the same seed/cache identities, both override maps `{}`, and semantic
    values `0.3 / 1.0 / 0.3 / 0.0 / 0.0`;
  - an explicit seed `2024` is recorded against expected `2023` by both
    metadata maps and yields the existing student-profile override blocker;
    a changed `hard_token_seed` remains a dataset override, and unknown or
    mismatched identities remain blocked.
- Verification evidence:
  - focused `codes.tests.test_dataset_profiles`: `20/20` tests passed;
  - full `codes/tests/test_*.py` unit suite: `40/40` tests passed in the
    declared Python environment, using temporary/synthetic fixtures only;
  - Python compilation passed for the changed profile and test modules;
  - actual `utility.parser` static resolution passed for both complete future
    formal commands. An initial JSON-reporting `python -c` wrapper failed with
    a PowerShell quoting `NameError` before producing parser evidence; corrected
    tuple-reporting wrappers passed, with no repository, dataset, or experiment
    state change;
  - source-diff checks against `df114dcd...` were empty for `parser.py`,
    `main_mmlight.py`, `load_data.py`, both TD-Distill model modules, and
    `experiment_protocol.py`, proving no parser default, runner, sampler,
    model/loss, optimizer, exporter, or evaluation-protocol implementation
    changed;
  - focused registry/document review and `git diff --check` passed before this
    outcome entry. Final exact staged-file and cached-whitespace checks remain
    required before the implementation commit.
- Test-split, metrics, and generated artifacts: the real Baby train,
  validation, and Test matrices were not loaded or accessed. No validation or
  Test metric, checkpoint, preflight report, manifest, convergence record, raw
  log, dataset derivative, teacher alias, or efficiency artifact was generated
  or changed. The only task artifacts are the scoped source, tests,
  documentation, this trace, and the pending Git commit.
- Acceptance decision: all declared implementation criteria passed. Existing
  seed-2022 profile defaults remain unchanged, the two seed-2023 profiles are
  frozen before evidence, formal static resolution is override-free, and the
  baseline-before-candidate firewall remains enforceable. This is not a run or
  a claim of replicated recommendation quality.
- Unresolved risks: neither seed-2023 profile has exercised the real GPU/data
  path; future long runs may fail; two paired seeds will still be insufficient
  for a strong variance or significance claim; and future ignored run artifacts
  will require explicit fingerprints and later physical backup. These risks do
  not authorize profile changes based on the future baseline result.
- Unique next action: after this implementation commit and only with explicit
  user authorization, append and commit a standalone uncapped formal pending
  declaration for `baby_student_reference_seed2023_v1`, citing the exact
  implementation commit and full override-free command, then stop without
  training, validation, or Test. The candidate declaration and execution stay
  blocked until the baseline outcome commit exists.

### 2026-07-31 | baby_student_reference_seed2023_v1 formal uncapped baseline (pending)

- Purpose and hypothesis: predeclare the baseline arm of the matched Baby
  seed-2023 replication as one uncapped formal ID-only BPR run. The purpose is
  to measure the already frozen reference method under the new student
  initialization and sampling stream before any seed-2023 candidate evidence
  exists. This declaration changes no method control and makes no quality
  claim.
- Status and authorization boundary: pending; formal, uncapped, single run seed
  `2023`, and eligible for one execution only after this declaration is in a
  clean committed tree and the user gives new explicit launch authorization.
  This task authorizes only this append-only baseline declaration, its
  declaration outcome, focused Markdown/Git verification, and one local
  commit. It does not authorize profile resolution by project code, dataset
  access, training, validation ranking, Test evaluation, run-artifact creation,
  retry, candidate declaration or execution, tag, bundle, baseline overwrite,
  or merge to `main`.
- Branch and exact source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean profile implementation and
  declaration predecessor
  `03260ee2db0e9e90332588031eacf3152d260dbf`. That commit implemented and
  verified both seed-2023 profiles together before evidence. The commit
  containing this pending declaration must be the later clean launch HEAD, and
  `git diff 03260ee2... <launch-commit> -- codes docs` must be empty. No source,
  profile, test, or documentation change is permitted between the frozen
  implementation and launch.
- Dataset and preprocessing identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/validation/Test matrices respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text features respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve official split assignments, cold-item policy `retain_official`,
  duplicate-modalities policy `error`, dataset preflight, `pca` hard-token
  type, `hard_token_seed=2022`, and the existing cache provenance. No dataset,
  split, feature, preprocessing, or cache change is allowed.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, expected size `141098540`
  bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, run-local
  teacher checkpoint, shared-alias publication, overwrite, or new teacher Test
  evaluation is allowed.
- Student input and deployment identity: no student checkpoint is loaded.
  `td_init_from_teacher=false` requires random initialization at run seed
  `2023`; the model is `td_distill_no_projection` with user/item ID embedding
  dimension `64`. The inference export must contain only finite user/item ID
  embeddings plus `embedding_dim`, user/item counts, and the no-projection
  variant identity; it must contain no teacher, modality, prompt, semantic
  cache, graph, projection, or optimizer state.
- Full override-free formal command, frozen exactly and containing no `--seed`:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Resolved profile and override contract, already statically verified by
  implementation commit `03260ee2...`: profile
  `baby_student_reference_seed2023_v1`, scope `student_reference`, source
  `predeclared_baby_id_only_bpr_reference_seed2023`;
  `dataset_config_overrides={}` and `student_config_overrides={}`; training and
  sampling `seed=2023`; teacher preprocessing/cache
  `hard_token_seed=2022`. The command may not add CLI seed, semantic,
  optimizer, epoch, patience, batch, sampler, protocol, teacher, or final-Test
  overrides.
- Fixed baseline parameters: batch size `1024`; maximum `epoch=1000`;
  validation every epoch; early-stopping patience `7` consecutive
  non-improving validation evaluations; `smoke_train_batches=0`, using all
  `116` independently sampled batches per completed epoch; unchanged
  `data_generator.sample()` pairwise BPR sampling; AdamW
  `student_lr=6e-5`; student weight decay `0.01`; no efficiency benchmark;
  `td_distill_alpha=0.0`; and item-image, item-text, user-image, and user-text
  rates all `0.0`. The run is BPR-only with no semantic head, projection, or
  teacher warm start.
- Evaluation and Test contract: `val_test_once_v1`; train only on `train_mat`;
  select and early-stop only by validation Recall@20; candidate exclusion
  `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`. Save and restore the
  validation-best student checkpoint, then execute exactly one final student
  Test evaluation. Test metrics are report-only and cannot influence selection,
  acceptance, any profile value, or the already frozen later arm.
- Intended environment: `D:\miniconda\envs\run_5060\python.exe`; Python
  `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`. Environment, GPU,
  free space, source HEAD, profile resolution, dataset identity, and teacher
  fingerprint must be rechecked immediately before any separately authorized
  launch.
- Planned run-specific artifacts, all keyed by one new timestamp/PID run name:
  raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  No shared student alias is allowed; all six artifacts must share the same run
  identity and remain outside Git.
- Hard acceptance criteria for the later run: launch from the exact clean
  declaration commit; source diff gate and both empty override maps pass;
  profile seed is `2023` while `hard_token_seed` is `2022`; dataset preflight
  and frozen-teacher validation pass; every loss and validation series value is
  finite; every completed epoch uses all `116` batches; validation Recall@20
  alone selects and early-stops; the validation-best checkpoint is restored
  before exactly one student final Test; manifest status is `completed`,
  `paper_ready_eligible=true`, blockers are empty, and
  `final_test_performed=true`; teacher hash and aliases are unchanged; all six
  same-run artifacts exist and are fingerprinted; full/inference embeddings
  are finite and bit-for-bit equal; the inference export contains only the
  declared deployment state.
- Quality acceptance and preservation rule: there is no minimum metric
  threshold. Any finite result produced by the exact declared code and protocol
  is valid completed baseline evidence even if low, reversed, or worse than
  seed `2022`. Quality alone cannot cancel the matched replication, change the
  frozen later profile, or trigger rollback, retry, reset, deletion, or a
  `failed` label.
- Failure boundary: mark a later run `failed` only for process/code failure,
  non-finite values, source/profile/data/teacher mismatch, override drift,
  selection/Test chronology violation, teacher mutation, missing or cross-run
  artifacts, or another hard acceptance failure. Preserve all partial
  run-specific files and record the root cause without automatic retry or
  destructive recovery.
- Risks and evidence limits: the uncapped GPU run may fail or take substantial
  time; the final Test access is consumable once and cannot tune anything; the
  fixed official cold-item warning remains; one additional seed may reverse the
  seed-2022 direction; and two paired seeds will still be insufficient for a
  strong variance or significance claim. Ignored artifacts require explicit
  fingerprints and later physical backup.
- Planned later verification after separate launch authorization: verify clean
  HEAD, exact source diff, actual environment, empty profile overrides, teacher
  SHA256, data/preflight identities, and pre-run artifact inventory; after the
  single process exits, audit raw chronology, finite convergence series,
  validation selection and restoration, exactly-one Test access, manifest
  eligibility, checkpoint structure and full-to-inference equality, six
  artifact hashes, teacher/alias preservation, and tracked Git status; then
  append and commit one completed or failed outcome before any later stage.
- Declaration-only verification and next action: this task must change only
  `TRAINING_LOG.md`, run focused content/diff and Git index checks, create one
  declaration commit, and stop. After that commit, wait for explicit user
  authorization before executing the exact baseline command once. No later-arm
  declaration or execution is permitted before this baseline outcome is
  committed.

### 2026-07-31 | baby_student_reference_seed2023_v1 formal baseline declaration (completed; run not started)

- Status and scope: completed the requested declaration-recording task only;
  the formal uncapped baseline above remains `pending`. No project code,
  profile resolution, dataset, training, validation, Test, efficiency, run
  artifact, retry, later-arm declaration, tag, bundle, baseline overwrite, or
  merge action ran.
- Source and verification evidence: the pre-edit branch was clean at
  `03260ee2db0e9e90332588031eacf3152d260dbf`; focused source/document review
  confirmed the implemented baseline identity, seed-only delta, absence of
  `--seed` in the frozen command, empty-override contract, and unchanged
  `hard_token_seed=2022`. The final append-only diff and staged scope must be
  limited to `TRAINING_LOG.md` and pass `git diff --check` plus
  `git diff --cached --check` before this declaration commit.
- Metrics, Test access, and generated artifacts: none. No Baby split was loaded
  or accessed; no validation/Test metric, checkpoint, preflight, manifest,
  convergence record, raw log, dataset derivative, asset hash, or efficiency
  result was generated or changed. This log declaration and its Git commit are
  the only task artifacts.
- Experimental meaning and unresolved risks: the baseline run contract is now
  frozen before seed-2023 evidence and preserves every non-seed control from
  the completed seed-2022 reference. It provides no new replication evidence,
  runtime evidence, or quality result; the real GPU/data path for this profile
  remains unexecuted, and the future one-time Test result cannot tune or cancel
  the already frozen matched comparison.
- Unique next action: after this declaration commit and only with new explicit
  user authorization, execute the exact declared
  `baby_student_reference_seed2023_v1` command once from the clean declaration
  HEAD, audit its protocol and artifacts, append and commit its completed or
  failed outcome, and stop. Do not begin any later arm in that task.

### 2026-07-31 | baby_student_reference_seed2023_v1 formal uncapped baseline (failed)

- Purpose and outcome: executed the exact declared seed-2023 ID-only BPR
  baseline command once from clean declaration commit
  `acc0b59f2bc1b41a76c8f8bda486e655a9b09d10`. The run passed source,
  profile, environment, dataset-preflight, cache, and frozen-teacher gates but
  failed before completing any optimization batch because the external command
  execution session timed out and closed its output channel. The runner caught
  the resulting `OSError: [Errno 22] Invalid argument`, wrote a failed manifest,
  and exited; no retry was launched.
- Status and acceptance: failed under the declared process/code failure
  boundary. This is not a low-quality result: no validation-selected model or
  recommendation metric exists. The failed attempt and all valid partial
  artifacts are preserved; no reset, deletion, rollback, or automatic retry
  occurred.
- Branch and exact source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean launch HEAD
  `acc0b59f2bc1b41a76c8f8bda486e655a9b09d10`; pre-launch
  `git diff 03260ee2db0e9e90332588031eacf3152d260dbf HEAD -- codes docs`
  was empty. The tracked tree was clean before launch, and the failed process
  did not modify tracked source.
- Executed command, exactly once:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Process and environment: run name
  `2026-07-31 23_22_19.126721_baby_light_init_pid17888`; runner start
  `2026-07-31T23:22:19.129277+08:00`; failed manifest time
  `2026-07-31T23:22:31.886772+08:00`; outer execution wrapper returned exit
  code `124` after approximately `14` seconds. Environment checks immediately
  before launch reported Python `3.10.20`, PyTorch `2.11.0+cu128`, CUDA runtime
  `12.8`, NVIDIA driver `595.97`, NVIDIA GeForce RTX 5060 8 GB, GPU selector
  `0`, and approximately `405.9 GB` free on drive D.
- Resolved profile and protocol evidence: profile/scope/source resolved as
  `baby_student_reference_seed2023_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2023`; both
  `dataset_config_overrides` and `student_config_overrides` were `{}`. The
  runner recorded seed `2023`, `hard_token_seed=2022`, batch size `1024`,
  maximum `epoch=1000`, patience `7`, `smoke_train_batches=0`,
  `run_final_test=true`, AdamW `student_lr=6e-5`, weight decay `0.01`, random
  64-dimensional no-projection student initialization, and all semantic rates
  plus `td_distill_alpha` at `0.0`. The manifest recorded
  `paper_ready_eligible=true` and no blockers before the execution failure.
- Dataset, preprocessing, and teacher audit: preflight passed the declared
  conversion manifest, train/validation/Test matrix, and image/text feature
  identities; split overlaps were `0 / 0 / 0`, modalities were finite and
  non-duplicate, and the only warning was the frozen three-cold-item condition
  covering `11` validation and `7` Test interactions. Both PCA hard-token
  caches were hits with random state `2022`. The frozen teacher was loaded and
  validated, remained `141098540` bytes, and retained SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  no teacher optimizer step, run-local teacher checkpoint, alias publication,
  or overwrite occurred.
- Training, validation selection, and Test result: none. The log reaches
  student construction, disabled warm start, cache loading, and confirmation
  of BPR-only mode, but contains no completed optimization batch, epoch,
  validation event, best-checkpoint save/restore, early-stop event, or student
  final-Test event. Therefore there is no loss series, best epoch, validation
  Recall@20, checkpoint selection, or final Test metric. No validation-best
  checkpoint could be restored and the protocol-authorized final Test ranking
  was not executed.
- Test-split access and protocol compliance: the dataset preflight read Test
  matrix structure and identity as declared, and frozen teacher Test metrics
  were read from checkpoint metadata. Neither the teacher nor the seed-2023
  student performed a new Test ranking evaluation. Execution complied with the
  declared identity and Test chronology up to the process failure, but the
  formal protocol did not complete.
- Preserved partial artifacts:
  - raw log
    `logs/2026-07-31 23_22_19.126721_baby_light_init_pid17888`, `4927`
    bytes, SHA256
    `2d512a4ef40cf02c933f6eb3391c02bdf1ddddec60ec24e2343e36d1c41521c0`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-07-31 23_22_19.126721_baby_light_init_pid17888.json`,
    `3399` bytes, SHA256
    `6ccaec45bf975bc001ffd07f4fc3e4104c86052f85556c7737677f39a62f6f16`;
  - failed manifest
    `exp/runs/baby/run_manifest__2026-07-31 23_22_19.126721_baby_light_init_pid17888.json`,
    `24366` bytes, SHA256
    `6f6b6e49e8ee3495bab306693848c3cf0f960ca250dd3470335602ac8bb325bd`.
  No convergence record, full student checkpoint, inference-only checkpoint,
  run-local teacher checkpoint, or additional same-run artifact exists.
- Root cause and recovery point: the repository runner and data path reached
  normal BPR-only initialization, but the outer command transport was launched
  with a short timeout inappropriate for the uncapped formal run. Closing that
  transport caused the runner's next output operation to raise the recorded
  Windows `EINVAL`/`OSError`. Recovery must preserve this failed run and start
  from the same frozen implementation and parameters, using a separately
  declared clean commit and a long-lived execution transport without a short
  timeout. A rerun requires new explicit authorization and is not implied by
  this outcome.
- Experimental meaning and unresolved risks: this attempt contributes no
  seed-2023 quality or replication evidence. The seed-2023 baseline remains
  unresolved, so the mandatory baseline-outcome-before-candidate firewall
  continues to block candidate declaration and execution. The next long run
  could still encounter independent GPU, dependency, or code failures, and
  ignored artifacts remain outside Git protection.
- Unique next action: with explicit user authorization, append and commit a
  recovery formal-run declaration for the same
  `baby_student_reference_seed2023_v1` command, parameters, data, protocol, and
  teacher identity, adding only the operational requirement for a long-lived
  execution transport; then stop before rerunning. Do not declare or execute
  the candidate. Expected artifacts for that next step are the append-only
  recovery declaration and its clean Git commit only.

### 2026-08-01 | baby_student_reference_seed2023_v1 recovery formal run (pending)

- Purpose and rationale: predeclare one recovery execution of the unresolved
  seed-2023 ID-only BPR baseline after the first formal attempt failed solely
  because its short-lived outer command transport timed out and closed the
  output channel before any optimization batch completed. This recovery keeps
  the original formal command, method, parameters, data, preprocessing,
  `val_test_once_v1` protocol, and frozen teacher identity unchanged. It makes
  no quality claim and does not reinterpret or replace the preserved failed
  attempt.
- Status, scope, and authorization boundary: pending; formal, uncapped, single
  recovery run at seed `2023`. This task authorizes only this append-only
  declaration, focused Markdown/Git verification, and one local commit. It
  does not authorize project-code execution, profile resolution, dataset
  access, training, validation ranking, Test evaluation, run-artifact
  creation, recovery launch, automatic retry, candidate declaration or
  execution, tag, bundle, baseline overwrite, or merge to `main`. The recovery
  may run exactly once only after this declaration is committed in a clean
  tree and the user gives new explicit launch authorization.
- Branch and exact source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean declaration predecessor
  `d2d91ee73cbd2000b1ad207e7c113fbcb1c60fd6`; frozen profile implementation
  `03260ee2db0e9e90332588031eacf3152d260dbf`. The commit containing this
  pending declaration must be the later clean recovery launch HEAD, and
  `git diff 03260ee2db0e9e90332588031eacf3152d260dbf <launch-commit> -- codes docs`
  must be empty. No source, profile, test, or documentation change is
  permitted between the frozen implementation and recovery launch.
- Full override-free formal command, frozen exactly and containing no
  `--seed`:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Dataset and preprocessing identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/validation/Test matrix SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve official split assignments, cold-item policy `retain_official`,
  duplicate-modalities policy `error`, dataset preflight, `pca` hard-token
  type, `hard_token_seed=2022`, and existing cache provenance. No dataset,
  split, feature, preprocessing, or cache change is allowed.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, expected size `141098540`
  bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, run-local
  teacher checkpoint, shared-alias publication, overwrite, or new teacher Test
  evaluation is allowed.
- Frozen student and parameter contract: profile
  `baby_student_reference_seed2023_v1`, scope `student_reference`, source
  `predeclared_baby_id_only_bpr_reference_seed2023`, with
  `dataset_config_overrides={}` and `student_config_overrides={}`; random
  initialization at training/sampling seed `2023`; no loaded student
  checkpoint; `td_init_from_teacher=false`; no-projection user/item ID
  embeddings of dimension `64`; batch size `1024`; maximum `epoch=1000`;
  validation every epoch; early-stopping patience `7`; all `116` independently
  sampled BPR batches per completed epoch; `smoke_train_batches=0`; unchanged
  `data_generator.sample()` sampling; AdamW `student_lr=6e-5`; student weight
  decay `0.01`; `td_distill_alpha=0.0`; all four semantic rates `0.0`; and no
  efficiency benchmark, semantic head, projection, or teacher warm start. No
  CLI seed, semantic, optimizer, epoch, patience, batch, sampler, protocol,
  teacher, or final-Test override may be added.
- Evaluation and Test contract: unchanged `val_test_once_v1`; train only on
  `train_mat`; select and early-stop only by validation Recall@20; candidate
  exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`; save and
  restore the validation-best student checkpoint, then perform exactly one
  final student Test evaluation. Test metrics are report-only and cannot
  influence selection, acceptance, profile values, or the frozen later arm.
- Intended environment: `D:\miniconda\envs\run_5060\python.exe`; Python
  `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`. Environment, GPU,
  free space, clean source HEAD, empty override maps, dataset identity, and
  teacher fingerprint must be checked immediately before a separately
  authorized launch.
- Sole new operational launch requirement: execute the frozen command in a
  continuously attached foreground execution session for the entire process,
  with the invoking shell/tool timeout explicitly set to at least three hours
  (`10800000` ms). A default or otherwise short timeout is forbidden. Detached
  launch, output-channel closure while the process is active, and automatic
  retry are forbidden. If the foreground session or process ends for any
  reason, preserve its state and artifacts, record one outcome, and stop.
- Planned run-specific artifacts: one new timestamp/PID identity shared by raw
  log `logs/<run_name>`, preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`, manifest
  `exp/runs/baby/run_manifest__<run_name>.json`, convergence record
  `exp/converge/baby/auto__<run_name>.pkl`, full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`,
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  No shared student alias is allowed; these artifacts remain outside Git.
- Hard acceptance criteria for the later recovery run: launch from the exact
  clean declaration commit through the required foreground session and
  timeout; pass the source-diff, empty-override, environment, dataset,
  preprocessing, profile, seed, and frozen-teacher gates; keep every loss and
  validation value finite; use all `116` batches for each completed epoch;
  select and early-stop only by validation Recall@20; restore the
  validation-best checkpoint before exactly one final student Test; finish
  with manifest status `completed`, `paper_ready_eligible=true`, no blockers,
  and `final_test_performed=true`; preserve the teacher hash and aliases;
  produce and fingerprint all six same-run artifacts; and verify finite,
  bit-for-bit equal full/inference embeddings with the inference export limited
  to the declared deployment state. There is no minimum metric threshold; any
  finite protocol-compliant result is valid evidence even when low or worse
  than seed `2022`.
- Risks, failure boundary, and rollback point: the uncapped foreground GPU run
  may exceed three hours or fail independently; final Test access is consumable
  once; one additional seed may reverse the seed-2022 direction; and ignored
  artifacts remain outside Git protection. Mark the later recovery `failed`
  only for process/code failure or another hard acceptance violation, preserve
  all partial artifacts, and do not retry, reset, delete, or roll back based on
  runtime or metric quality. The traceable recovery point is this clean
  declaration commit plus the unchanged frozen implementation and preserved
  failed-attempt artifacts.
- Planned verification and unique next action: for this declaration-only task,
  verify that only `TRAINING_LOG.md` changed, run unstaged and staged diff
  checks, commit this pending declaration, and stop. Afterward, wait for new
  explicit user authorization to execute the frozen baseline command once
  using the required continuous foreground session and at-least-three-hour
  timeout, then audit and commit one completed or failed outcome. Candidate
  declaration and execution remain blocked until that recovery outcome commit
  exists.

### 2026-08-01 | baby_student_reference_seed2023_v1 recovery formal run (completed)

- Status and outcome: completed successfully and accepted as valid formal
  seed-2023 ID-only BPR baseline evidence. The one authorized recovery process
  exited naturally with code `0`; no second launch, retry, rollback, reset,
  deletion, candidate declaration, or candidate execution occurred. Its lower
  metric relative to seed `2022` is retained under the predeclared
  quality-preservation rule and is not a failure.
- Source and execution identity: branch
  `codex/experiment/baby-teacher-baseline`; exact clean launch/declaration
  commit `eaaa68d851af2256b7b5c9957eab4cc426fecc8b`; frozen profile
  implementation `03260ee2db0e9e90332588031eacf3152d260dbf`; pre-launch and
  post-run `git diff 03260ee2db0e9e90332588031eacf3152d260dbf HEAD -- codes docs`
  were empty. The tracked tree was clean before launch and after run audit.
- Executed command, exactly once and without `--seed`:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Long-lived transport compliance: the command ran continuously in one
  attached foreground shell/tool session with its timeout explicitly set to
  `10800000` ms. Main PID `29128` and run identity
  `2026-08-01 20_20_35.101504_baby_light_init_pid29128` remained unchanged;
  the process was not detached or relaunched. The foreground tool reported
  exit code `0` after approximately `1457.1` seconds. Manifest start and
  completion were `2026-08-01T20:20:35.110046+08:00` and
  `2026-08-01T20:44:36.841092+08:00`, approximately `24:01.7` apart.
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver `595.97`;
  NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`. The launch preflight found
  approximately `406.1 GB` free on drive D and no existing training process;
  the post-run audit found no remaining `main_mmlight.py` process.
- Resolved profile and parameter contract: profile/scope/source
  `baby_student_reference_seed2023_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2023`; both dataset and student
  override maps were `{}`; training/sampling seed `2023` and
  `hard_token_seed=2022`; `td_distill_no_projection`; 64-dimensional random
  user/item ID embeddings; `td_init_from_teacher=false`; batch size `1024`;
  maximum `epoch=1000`; early-stopping patience `7`;
  `smoke_train_batches=0`; AdamW `student_lr=6e-5`; student weight decay
  `0.01`; `td_distill_alpha=0.0`; all four semantic rates `0.0`; and no
  efficiency benchmark. Raw startup and checkpoint metadata agree with the
  declaration; no semantic head was active and every semantic loss series is
  identically zero.
- Dataset and preprocessing audit: all six pre-launch asset hashes matched the
  pending declaration. The generated preflight passed split overlap,
  non-duplicate modality, shape, dtype, finiteness, conversion-manifest, and
  fingerprint gates. Its only warning was the declared retained-official
  cold-item condition: item IDs `240`, `1212`, and `6115`, covering `11`
  validation and `7` Test interactions with no cold users. Image and text PCA
  caches were hits with random state `2022`; no dataset, split, feature,
  preprocessing, or cache identity changed.
- Frozen teacher audit: teacher training was skipped and the read-only alias
  `Model/baby/teacher_model_val_test_once_v1.pt` remained `141098540` bytes
  with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`
  before and after the run. No teacher optimizer step, run-local teacher
  checkpoint, shared-alias publication, or alias overwrite occurred. The
  existing teacher Test metrics were read from checkpoint metadata; no new
  teacher Test ranking ran.
- Training and validation chronology: epochs `0-79` completed contiguously.
  The uncapped training path used `train_batch_count=116` and the foreground
  progress for every completed epoch reached `116/116`; the code executed
  `data_generator.sample()` once per batch. All 80 BPR/total loss and
  validation metric series values were finite. Validation Recall@20 alone
  selected checkpoints. The best was epoch `72`; epochs `73-79` produced the
  declared seven consecutive non-improvements and natural early stopping at
  patience `7/7`.
- Exact validation-best metrics at epoch `72`: Recall@20
  `0.04098274743170617`; Recall@50 `0.07119470055957632`; NDCG@20
  `0.017848015977795412`; NDCG@50 `0.02408387902271345`. The full checkpoint,
  convergence record, and manifest agree on epoch `72`, protocol
  `val_test_once_v1`, selection split `validation`, and primary K `20`.
- One-time final student Test result: after restoring the validation-best epoch
  `72` checkpoint, the raw chronology contains exactly one final student Test
  event. Exact vectors ordered by K `[10,20,40,50]` are Precision
  `[0.0024993571612239892, 0.002057084083311931, 0.0016469529442016179, 0.0015407559784006715]`,
  Recall
  `[0.02253427453093155, 0.03680654207653432, 0.05880417887874913, 0.06895894996692173]`,
  NDCG
  `[0.013025226314658288, 0.017025150092128044, 0.02203587686177423, 0.024038280300640525]`,
  and Hit Ratio
  `[0.024839290305990986, 0.040678837747493035, 0.06500385703265722, 0.076009256878375]`.
  AUC is `0.0` because `test_flag=part` omits AUC. Test was accessed by the
  declared dataset preflight and exactly one final student ranking only after
  validation selection; it did not affect selection, acceptance, or any
  profile value.
- Manifest and checkpoint acceptance: manifest status is `completed`,
  `paper_ready_eligible=true`, blockers are `[]`, and
  `final_test_performed=true`. The full model state contains only
  `user_id_embedding.weight` and `item_id_embedding.weight` plus optimizer and
  declared metadata. The inference export contains exactly those two
  embeddings plus `embedding_dim`, `n_users`, `n_items`, and `variant`; it has
  no teacher, modality, prompt, cache, graph, projection, or optimizer state.
  User `(19445, 64)` and item `(7050, 64)` embeddings are finite and
  bit-for-bit equal between the full and inference checkpoints.
- Run-specific artifacts and SHA256:
  - raw log `logs/2026-08-01 20_20_35.101504_baby_light_init_pid29128`,
    `39035` bytes,
    `e8f3ebaafe0e420e4816df80869393fb746de17979c3eb7faf9656a190e01596`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-08-01 20_20_35.101504_baby_light_init_pid29128.json`,
    `3399` bytes,
    `badd4f0e1da36bf9cef6067e84e779568327252c0f7a1a69265006937811b437`;
  - manifest
    `exp/runs/baby/run_manifest__2026-08-01 20_20_35.101504_baby_light_init_pid29128.json`,
    `25383` bytes,
    `edcc1b4d858eecbad5724ac0c46608abb572bb9412f78a1555156cff278a7d33`;
  - convergence
    `exp/converge/baby/auto__2026-08-01 20_20_35.101504_baby_light_init_pid29128.pkl`,
    `12631` bytes,
    `07a45c071cebbc0b7591d36064f73d27c3af286156c4e418e68c1bab26b8e2b1`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-01 20_20_35.101504_baby_light_init_pid29128.pth`,
    `20354975` bytes,
    `8101a7b522d382fdee1934b18590060651d58064b3d74fe40bf4accaa69fa80b`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-01 20_20_35.101504_baby_light_init_pid29128.pth`,
    `6785997` bytes,
    `664ff1474d5108876c700d5161c5d29ac314fcad70ea2a9fcc941f0049b6d1b6`.
  All six files exist, share the same run identity, and remain outside Git.
- Experimental meaning and unresolved risks: seed `2023` produced validation
  Recall@20 `0.04098274743170617` and Test Recall@20
  `0.03680654207653432`, below the seed-2022 reference values
  `0.04291303921928788` and `0.044279857310199594`. This is valid evidence of
  seed sensitivity, not authorization to alter the frozen matched candidate or
  retry the baseline. Two paired seeds remain insufficient for a strong
  variance or significance claim; the retained cold items remain a fixed
  protocol limitation; and ignored artifacts require explicit physical or
  off-device backup at a stable milestone.
- Unique next action: after committing this outcome and only with new explicit
  user authorization, append and commit a standalone formal pending
  declaration for the already frozen matched profile
  `baby_td_candidate_seed2023_v1`, citing this baseline outcome commit and
  preserving the baseline-before-candidate firewall; then stop without
  training, validation, or Test. The expected artifacts for that next task are
  only the append-only candidate declaration and its clean Git commit.

### 2026-08-01 | baby_td_asymmetric_no_projection_seed2023_v1 formal uncapped candidate (pending)

- Append-only identity correction: the preceding baseline outcome's unique
  next action used `baby_td_candidate_seed2023_v1` as an erroneous shorthand.
  No profile, alias, registry entry, or executable identity with that name
  exists. The canonical, implemented, and already frozen profile ID is
  `baby_td_asymmetric_no_projection_seed2023_v1`. This entry corrects the
  handoff prospectively without editing or deleting the old record; all future
  declarations, commands, manifests, and outcomes must use only the canonical
  ID.
- Purpose and hypothesis: predeclare the candidate arm of the matched Baby
  seed-2023 replication as one uncapped formal asymmetric no-projection
  directional-distillation run. The hypothesis, frozen before either seed-2023
  arm produced evidence, is that the fixed item-dominant semantic transfer can
  improve the deployable student over the matched ID-only BPR baseline while
  preserving the same initialization seed, sampler, optimizer budget,
  deployment state, data, teacher, and evaluation protocol. This declaration
  makes no quality claim and does not tune the candidate from the completed
  baseline result.
- Baseline-outcome prerequisite and parameter-selection firewall: the matched
  baseline `baby_student_reference_seed2023_v1` completed from declaration
  commit `eaaa68d851af2256b7b5c9957eab4cc426fecc8b`; its audited outcome is
  committed at `1749d46bf46087f84208a39bc3caefe92654aa50`. It selected
  validation epoch `72`, with validation Recall@20
  `0.04098274743170617` and final Test Recall@20
  `0.03680654207653432`. Those results satisfy the required
  baseline-outcome-before-candidate gate but did not select, cancel, reorder,
  or alter any candidate parameter. The candidate was implemented and frozen
  together with the baseline before this evidence existed.
- Status and authorization boundary: pending; formal, uncapped, single run at
  training/sampling seed `2023`. This task authorizes only this append-only
  correction and candidate declaration, focused Markdown/Git verification,
  and one local commit. It does not authorize profile resolution by project
  code, dataset access, training, validation ranking, Test evaluation,
  run-artifact creation, candidate launch, retry, tag, bundle, baseline or
  teacher overwrite, merge to `main`, or any later experiment stage. The run
  may execute exactly once only after this declaration is committed in a clean
  tree and the user gives new explicit launch authorization.
- Branch and exact source identity: branch
  `codex/experiment/baby-teacher-baseline`; clean declaration predecessor and
  completed baseline outcome commit
  `1749d46bf46087f84208a39bc3caefe92654aa50`; frozen matched-profile
  implementation commit `03260ee2db0e9e90332588031eacf3152d260dbf`.
  Implementation commit `03260ee2...` added and verified both seed-2023
  profiles together before either run. The commit containing this pending
  declaration must be the later clean launch HEAD, and
  `git diff 03260ee2db0e9e90332588031eacf3152d260dbf <launch-commit> -- codes docs`
  must be empty. No source, profile, test, or documentation change is permitted
  between the frozen implementation and launch.
- Full override-free formal command, frozen exactly and containing no
  `--seed`:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Canonical profile and override contract, statically verified before
  seed-2023 evidence by implementation commit `03260ee2...`: profile
  `baby_td_asymmetric_no_projection_seed2023_v1`, scope
  `student_candidate`, source
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2023`;
  `dataset_config_overrides={}` and `student_config_overrides={}`; student
  initialization and sampling `seed=2023`; frozen teacher preprocessing/cache
  `hard_token_seed=2022`. The command may not add a CLI seed, semantic,
  optimizer, epoch, patience, batch, sampler, protocol, teacher, or final-Test
  override, and the nonexistent shorthand must never be passed as a profile.
- Dataset and preprocessing identity: audited MMRec Baby under `data/baby/`;
  conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/validation/Test matrix SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve official split assignments, cold-item policy `retain_official`,
  duplicate-modalities policy `error`, dataset preflight, `pca` hard-token
  type, `hard_token_seed=2022`, and existing cache provenance. No dataset,
  split, feature, preprocessing, or cache change is allowed.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, expected size `141098540`
  bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, run-local
  teacher checkpoint, shared-alias publication, overwrite, or new teacher Test
  evaluation is allowed.
- Frozen shared student controls: no student checkpoint is loaded; random
  initialization at seed `2023`; `td_init_from_teacher=false`;
  `td_distill_no_projection`; user/item ID embedding dimension `64`; batch size
  `1024`; maximum `epoch=1000`; validation every epoch; early-stopping patience
  `7`; `smoke_train_batches=0`; all `116` independently sampled batches per
  completed epoch; unchanged `data_generator.sample()` pairwise BPR sampling;
  AdamW `student_lr=6e-5`; student weight decay `0.01`; and no efficiency
  benchmark, projection, or teacher warm start.
- Frozen candidate-only semantic controls: `td_distill_alpha=0.3`; item-image
  rate `1.0`; item-text rate `0.3`; user-image rate `0.0`; user-text rate
  `0.0`. Active supervision is therefore item-image and item-text only, with
  the no-projection 64-dimensional student optimizing
  `L_BPR + 0.3 * ((1.0 * L_item_image + 0.3 * L_item_text) / 1.3)`.
  Relative to the matched baseline, the only actual changed-value set is
  `td_distill_alpha`, `td_item_image_rate`, and `td_item_text_rate`; all other
  controls remain value-identical.
- Student deployment identity: the full validation-selected checkpoint may
  contain the two user/item ID embeddings, optimizer state, and declared
  metadata. The inference export must contain only finite user/item ID
  embeddings plus `embedding_dim`, user/item counts, and the no-projection
  variant identity; it must contain no teacher, modality, prompt, semantic
  cache, graph, projection, or optimizer state. Full and inference user/item
  embeddings must be bit-for-bit equal.
- Evaluation and Test contract: unchanged `val_test_once_v1`; train only on
  `train_mat`; select and early-stop only by validation Recall@20; candidate
  exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`; save and
  restore the validation-best student checkpoint, then perform exactly one
  final student Test evaluation. Test metrics are report-only and cannot
  influence selection, technical acceptance, profile values, retry, rollback,
  or interpretation of the already completed baseline.
- Intended environment: `D:\miniconda\envs\run_5060\python.exe`; Python
  `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`. Clean source HEAD,
  source-diff gate, environment, GPU, free space, exact canonical profile,
  empty override maps, dataset identities, teacher fingerprint, and pre-run
  artifact inventory must be checked immediately before a separately
  authorized launch.
- Execution transport requirement: a separately authorized launch must keep
  the single command in one continuously attached foreground shell/tool
  session for the entire process, with the shell/tool timeout explicitly set
  to at least three hours (`10800000` ms). Default or short timeouts, detached
  launch, output-channel closure while active, and automatic retry are
  forbidden. If the process or transport ends for any reason, preserve the
  state and artifacts, record one outcome, and stop.
- Planned run-specific artifacts, all keyed by one new timestamp/PID run name:
  raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  No shared student alias is allowed; all six artifacts must share the same run
  identity and remain outside Git.
- Hard acceptance criteria for the later run: launch once from the exact clean
  declaration commit through the required foreground session and timeout;
  pass source-diff, canonical-profile, empty-override, environment, dataset,
  preprocessing, seed/cache, and frozen-teacher gates; use exactly the frozen
  semantic controls; keep all training loss and validation metric series
  finite; use all `116` batches per completed epoch; select and early-stop only
  by validation Recall@20; restore the validation-best checkpoint before
  exactly one final student Test; finish with manifest status `completed`,
  `paper_ready_eligible=true`, no blockers, and
  `final_test_performed=true`; preserve teacher hash and aliases; produce and
  fingerprint all six same-run artifacts; and pass checkpoint structure,
  finiteness, deployment-state, and full-to-inference equality audits.
- Quality acceptance, failure boundary, and preservation: there is no minimum
  metric or improvement threshold. Any finite result produced by the exact
  declared code and protocol is valid completed candidate evidence even if
  lower than the matched baseline or seed-2022 candidate. Mark a later run
  `failed` only for process/code failure, non-finite values, identity or
  override drift, data/teacher mismatch, chronology violation, teacher
  mutation, missing or cross-run artifacts, or another hard acceptance
  failure. Preserve all completed or partial files; do not automatically
  retry, reset, revert, delete, or roll back based on metric quality.
- Risks, rollback point, and evidence limits: the uncapped GPU run may fail or
  exceed three hours; final Test access is consumable once; the fixed official
  cold-item warning remains; the seed-2023 direction may differ from seed
  `2022`; and two paired seeds remain insufficient for a strong variance or
  significance claim. Ignored artifacts require explicit physical or
  off-device backup at a stable milestone. The traceable rollback/preservation
  point is this clean declaration commit plus implementation commit
  `03260ee2...`, completed baseline outcome `1749d46...`, and all immutable
  prior evidence; no destructive rollback is authorized.
- Declaration-only verification and unique next action: this task must change
  only `TRAINING_LOG.md`, verify the append-only correction, canonical command,
  frozen identities and parameters, run unstaged and staged diff checks,
  create one declaration commit, and stop. It accesses no dataset split and
  produces no metric or run artifact. After that commit, wait for new explicit
  user authorization to execute the exact canonical candidate command once
  through the required long-lived foreground session, audit the existing run
  after it exits, append and commit one completed or failed outcome, and stop.

### 2026-08-01 | baby_td_asymmetric_no_projection_seed2023_v1 formal uncapped candidate (completed)

- Status and outcome: completed successfully and accepted as valid formal
  seed-2023 asymmetric no-projection candidate evidence. The one authorized
  process exited naturally with code `0`; no second launch, automatic retry,
  parameter change, rollback, reset, deletion, tag, bundle, merge, baseline or
  teacher overwrite, or later experiment stage occurred. Technical acceptance
  is independent of metric quality.
- Source and execution identity: branch
  `codex/experiment/baby-teacher-baseline`; exact clean launch/declaration
  commit `6c529c7446b7d95529fe24df78138451b3528e51`; frozen matched-profile
  implementation commit `03260ee2db0e9e90332588031eacf3152d260dbf`; completed
  seed-2023 baseline outcome commit
  `1749d46bf46087f84208a39bc3caefe92654aa50`. Pre-launch and post-run
  `git diff 03260ee2db0e9e90332588031eacf3152d260dbf HEAD -- codes docs`
  were empty. The tracked tree was clean before launch and after run audit.
- Executed command, exactly once and using only the canonical profile ID:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Long-lived transport compliance: the command ran continuously in one
  attached foreground shell/tool session with timeout explicitly set to
  `10800000` ms. Main PID `33928` and run identity
  `2026-08-01 21_09_59.190542_baby_light_init_pid33928` remained unchanged;
  the process was not detached or relaunched. The foreground tool reported
  exit code `0` after approximately `815.5` seconds. Manifest start and
  completion were `2026-08-01T21:09:59.192048+08:00` and
  `2026-08-01T21:23:30.068328+08:00`, approximately `13:30.9` apart.
- Environment: `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`;
  PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver `595.97`;
  NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`. Pre-launch free space on
  drive D was approximately `406.0 GB`; no pre-existing training process was
  found, and no `main_mmlight.py` process remained after the run.
- Resolved canonical profile and override gate: profile/scope/source
  `baby_td_asymmetric_no_projection_seed2023_v1` / `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2023`;
  `dataset_config_overrides={}` and `student_config_overrides={}` both before
  and during the run. Training/sampling seed was `2023` and teacher
  preprocessing/cache `hard_token_seed=2022`; no CLI seed or behavior override
  was passed. The nonexistent shorthand corrected in the pending declaration
  was not used.
- Frozen model, optimizer, and semantic controls: batch size `1024`; maximum
  `epoch=1000`; patience `7`; `smoke_train_batches=0`; all `116` batches per
  completed epoch; unchanged `data_generator.sample()` pairwise sampler;
  `td_distill_no_projection`; 64-dimensional random user/item ID embeddings;
  `td_init_from_teacher=false`; AdamW `student_lr=6e-5`; student weight decay
  `0.01`; no efficiency benchmark; `td_distill_alpha=0.3`; item-image rate
  `1.0`; item-text rate `0.3`; user-image and user-text rates `0.0`. Raw log,
  convergence, manifest, and checkpoint metadata agree. Active semantic heads
  were exactly item-image and item-text; both inactive user-side loss series
  remained identically zero.
- Dataset and preprocessing audit: all six pre-launch asset hashes matched the
  declaration. Generated preflight passed conversion-manifest, matrix/feature
  fingerprints, split overlaps `0 / 0 / 0`, shapes, dtypes, modality finiteness,
  and non-duplication. Its only warning was the frozen retained-official
  cold-item condition: item IDs `240`, `1212`, and `6115`, covering `11`
  validation and `7` Test interactions, with no cold users. Image and text PCA
  caches were hits with random state `2022`; no dataset, split, feature,
  preprocessing, or cache identity changed.
- Frozen teacher audit: teacher training was skipped and the read-only alias
  `Model/baby/teacher_model_val_test_once_v1.pt` retained its timestamp,
  `141098540` byte size, and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  No teacher optimizer step, run-local teacher checkpoint, shared-alias
  publication, overwrite, or new teacher Test ranking occurred. Teacher Test
  metrics printed at reuse came from frozen checkpoint metadata.
- Training and validation chronology: epochs `0-51` completed contiguously,
  each reaching `116/116` batches, for `6032` optimization batches. All 52
  total, BPR, directional, component, Recall, and NDCG series values were
  finite. Validation Recall@20 alone selected checkpoints. Best epoch was
  `44`; epochs `45-51` were the declared seven consecutive non-improvements,
  followed by one natural early-stop event at patience `7/7`.
- Loss-series evidence from epoch `0` to `51`: total
  `81.43306374549866 -> 71.39528173208237`; BPR
  `80.3982784152031 -> 70.98279410600662`; directional
  `3.449284801259637 -> 1.3749589445069432`; item-image
  `3.410076206550002 -> 0.9987448276951909`; item-text
  `3.5799797400832176 -> 2.6290057878941298`; user-image and user-text
  remained `0.0 -> 0.0`.
- Exact validation-best metrics at epoch `44`: Recall@20
  `0.06479423036892924`; Recall@50 `0.1177596150314039`; NDCG@20
  `0.02867000085862614`; NDCG@50 `0.0395883030679091`. The full checkpoint,
  convergence record, and manifest agree on epoch `44`, protocol
  `val_test_once_v1`, selection split `validation`, and primary K `20`.
- One-time final student Test result: after restoring the validation-best epoch
  `44` checkpoint, the raw chronology contains exactly one final student Test
  event. Exact vectors ordered by K `[10,20,40,50]` are Precision
  `[0.004587297505785609, 0.0036127539213165866, 0.002798920030856163, 0.002564155309848446]`,
  Recall
  `[0.04165917893997112, 0.06531066330886427, 0.10077244289380877, 0.1156571604733053]`,
  NDCG
  `[0.023961345086793755, 0.03052040561148235, 0.03849624048344874, 0.04140749439374869]`,
  and Hit Ratio
  `[0.04571869375160746, 0.07163795320133756, 0.11031113396759815, 0.1262535356132644]`.
  AUC is `0.0`, expected under `test_flag=part`.
- Test access and protocol compliance: dataset preflight read the declared Test
  matrix structure and identity. Exactly one student Test ranking occurred
  only after validation-best restoration; no Test metric influenced selection,
  acceptance, parameters, retry, or rollback. The execution fully complied
  with `val_test_once_v1` and candidate exclusion `train_only`.
- Manifest and checkpoint acceptance: manifest status is `completed`,
  `paper_ready_eligible=true`, blockers are `[]`, and
  `final_test_performed=true`. The format-v2 full checkpoint contains only
  finite user/item ID embedding model tensors plus exactly two optimizer-state
  entries and declared metadata. The inference export contains exactly the two
  embeddings plus `embedding_dim=64`, `n_users=19445`, `n_items=7050`, and
  variant `td_distill_no_projection`; it contains no teacher, modality, prompt,
  semantic cache, graph, projection, or optimizer state. User `(19445,64)` and
  item `(7050,64)` embeddings are finite and bit-for-bit equal between full and
  inference checkpoints.
- Run-specific artifacts and SHA256:
  - raw log `logs/2026-08-01 21_09_59.190542_baby_light_init_pid33928`,
    `27194` bytes,
    `98770421a4668bba8bdce727e045c003be087065f6c59a2487dd0e8ed51987e2`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-08-01 21_09_59.190542_baby_light_init_pid33928.json`,
    `3399` bytes,
    `1200c1a4c7b67fe949404517b56b9c8a8e9ebfaabd2006b71015dc5599407659`;
  - manifest
    `exp/runs/baby/run_manifest__2026-08-01 21_09_59.190542_baby_light_init_pid33928.json`,
    `25443` bytes,
    `357afe298cc1a771ddd30aaddd046a73298894e899e636c6eae0bc546608a42a`;
  - convergence
    `exp/converge/baby/auto__2026-08-01 21_09_59.190542_baby_light_init_pid33928.pkl`,
    `8736` bytes,
    `abf94a481cbf150dbd6cdbdc3d4035f8d2ace8ae236b360148fe455abc39dca0`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-01 21_09_59.190542_baby_light_init_pid33928.pth`,
    `20354975` bytes,
    `a8d614689aab6a400b524e808b23799a7d894f6c60e3c3633af5d1122eeb06f8`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-01 21_09_59.190542_baby_light_init_pid33928.pth`,
    `6785997` bytes,
    `720e632052ed5deb08730685f42e85ab736783d1831b64bd0c2195d99c5aac2f`.
  All six files exist, share the same run identity, and remain outside Git; no
  additional PID-33928 output exists.
- Matched seed-2023 experimental meaning: versus the completed same-seed
  baseline, candidate validation Recall@20 improved by
  `0.023811482937223072`, and final Test Recall@20 improved by
  `0.028504121232329954`, approximately `77.4431%` relative. The seed-2023
  candidate Test Recall@20 was `0.0012816776663522739` below the seed-2022
  candidate, but both seeds show the same positive candidate-over-baseline
  direction.
- Two-seed descriptive summary and evidence limit: seed-2022/2023 validation
  candidate-minus-baseline deltas are `0.025512924120535754` and
  `0.023811482937223072`, with descriptive mean `0.024662203528879413`.
  Corresponding Test deltas are `0.022312483665016952` and
  `0.028504121232329954`, with descriptive mean `0.025408302448673453`.
  Baseline/candidate Test Recall@20 descriptive means are
  `0.040543199693366956` and `0.0659515021420404`. These two paired seeds
  strengthen evidence that the frozen asymmetric transfer helps in this
  environment, but remain insufficient for a strong variance, confidence
  interval, or significance claim and do not prove optimal semantic rates.
- Unresolved risks: the fixed official cold-item condition remains; both final
  Test accesses are consumed and cannot tune future work; the candidate remains
  below the frozen teacher Test Recall@20 by `0.021346250389704474`; two seeds
  do not establish generality; and ignored raw artifacts are not protected by
  Git or the existing bundle. No further run or parameter change is authorized
  by this completed result.
- Unique next action: only with new explicit user authorization, perform one
  repository-record stabilization task for the now-completed matched
  seed-2022/2023 baseline/candidate evidence: append and commit a concise
  canonical two-seed summary, create a meaningful annotated milestone tag,
  refresh the standalone recovery bundle, fingerprint it, and physically back
  up the ignored run manifests, convergence records, raw logs, and four paired
  full/inference checkpoints. Stop without training, validation, Test,
  parameter changes, broad tuning, merge to `main`, or any new experiment
  declaration. Expected artifacts are the committed summary, outcome commit,
  annotated tag, refreshed bundle plus SHA256, and explicit backup inventory.

### 2026-08-01 | matched seed-2024 baseline/candidate replication stage (pending)

- Purpose and hypothesis: predeclare the third matched Baby student replication
  pair before either seed-2024 profile is implemented or any seed-2024 result is
  produced. The hypothesis is that the already frozen asymmetric no-projection
  transfer effect can be assessed under one additional independent student
  initialization and pairwise-sampling stream without changing any method,
  data, teacher, protocol, preprocessing, optimization, or semantic control.
  This third pair is intended to support a descriptive three-seed mean and
  standard-deviation summary; it is not authorization for tuning or a claim of
  statistical significance.
- Status and present scope: pending declaration only. This task may append and
  commit only this `TRAINING_LOG.md` entry. It does not implement either
  profile, modify code or any other documentation, run project tests, start training,
  perform validation ranking, access Test, create a run artifact, tag, bundle,
  backup, or merge. At declaration time neither seed-2024 profile is implemented
  and no seed-2024 result exists.
- Source and evidence anchors: branch
  `codex/experiment/baby-teacher-baseline`; clean declaration base
  `82eb88e7fd3e8f5110ddb14388c7117abb77d22e`; matched seed-2023 profile
  implementation `03260ee2db0e9e90332588031eacf3152d260dbf`; completed
  seed-2022 baseline/candidate outcomes
  `6f806fa70de1c707dd109741b1c8fd2d3efdc29a` /
  `92b64136e3ca75fa0f41a11e29f907a9403d54b3`; and completed seed-2023
  baseline/candidate outcomes
  `1749d46bf46087f84208a39bc3caefe92654aa50` /
  `82eb88e7fd3e8f5110ddb14388c7117abb77d22e`. The Git commit containing
  this entry is the exact declaration
  identity and must be recorded in the later implementation outcome and formal
  run declarations.
- Simultaneously predeclared future profile identities, frozen now before any
  seed-2024 result:
  - baseline name/scope/source:
    `baby_student_reference_seed2024_v1` / `student_reference` /
    `predeclared_baby_id_only_bpr_reference_seed2024`;
  - candidate name/scope/source:
    `baby_td_asymmetric_no_projection_seed2024_v1` / `student_candidate` /
    `predeclared_baby_asymmetric_no_projection_directional_v1_seed2024`.
  These canonical IDs must be implemented together in one later coherent
  change. Neither arm may be implemented, revised, or run in response to an
  observed seed-2024 metric from the other arm.
- Cross-seed identity contract: each future seed-2024 profile must have exactly
  the same default keys and values as its corresponding validated seed-2022
  and seed-2023 profile, except that student training and maintained pairwise
  sampling `seed` is `2024`. Thus the only allowed cross-seed default delta is
  `seed: 2022/2023 -> 2024`. `hard_token_seed=2022` remains dataset-owned and
  unchanged because it identifies frozen teacher preprocessing and PCA cache
  provenance. Formal commands must not pass `--seed` or repeat profile values;
  both `dataset_config_overrides` and `student_config_overrides` must resolve to
  `{}`.
- Frozen dataset and preprocessing identity: audited MMRec Baby under
  `data/baby/`; conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  `train_mat`, `val_mat`, and `test_mat` SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve official split assignments, cold-item policy `retain_official`,
  duplicate-modalities policy `error`, dataset preflight, `pca` hard-token
  type, `hard_token_seed=2022`, and the existing cache identities. No dataset,
  split, feature, preprocessing, or cache change is allowed.
- Frozen teacher identity and controls: reuse the read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, size `141098540` bytes and
  SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, run-local
  teacher checkpoint, shared-alias publication or overwrite, or new teacher
  Test ranking is allowed.
- Frozen controls shared by both future arms: random student initialization;
  no student checkpoint load; `td_init_from_teacher=false`;
  `td_distill_no_projection`; user/item ID embedding dimension `64`; AdamW
  `student_lr=0.00006`; student weight decay `0.01`; batch size `1024`;
  maximum `epoch=1000`; `smoke_train_batches=0`; validation every epoch;
  early-stopping patience `7`; unchanged `data_generator.sample()` pairwise
  BPR sampler; all `116` independently sampled batches per completed formal
  epoch; and no efficiency benchmark, projection head, teacher warm start, or
  extra inference state.
- Frozen matched semantic contract: the seed-2024 baseline retains
  `td_distill_alpha=0.0` and item-image, item-text, user-image, and user-text
  rates all `0.0`. The seed-2024 candidate retains
  `td_distill_alpha=0.3`, item-image `1.0`, item-text `0.3`, user-image `0.0`,
  and user-text `0.0`, optimizing the already validated normalized asymmetric
  item-only semantic objective. Between the matched arms, the only changed
  values remain `td_distill_alpha`, `td_item_image_rate`, and
  `td_item_text_rate`; all other keys and values, including `seed=2024`, must
  be identical.
- Frozen evaluation and deployment contract: unchanged `val_test_once_v1`;
  train only on `train_mat`; select and early-stop only with validation
  Recall@20; `Ks=[10,20,40,50]`; `test_flag=part`; candidate exclusion
  `train_only`; restore the validation-best student checkpoint and only then
  perform exactly one final student Test ranking per separately authorized
  formal run. Test metrics are report-only and cannot change either profile,
  determine retry, or cancel the later arm. Each inference export must contain
  only finite user/item ID embeddings and deployment metadata, and those
  embeddings must be bit-for-bit equal to the corresponding full checkpoint.
- Frozen future formal commands, recorded for identity but not authorized by
  this declaration. Baseline:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`.
  Candidate, only after the committed baseline outcome:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`.
  Each later launch requires its own pending declaration, explicit user
  authorization, exact clean committed source, one continuously attached
  foreground shell/tool session with timeout at least `10800000` ms, exactly
  one command execution, no automatic retry, and a separately committed
  completed or failed outcome.
- Mandatory sequential stage order:
  1. In one coherent future implementation task, implement and verify both
     override-free seed-2024 profiles together before producing any seed-2024
     result, append a separate implementation outcome, and commit it. Stop
     without any profile execution.
  2. In a later task, append and commit a baseline-only formal-run pending
     declaration; only after separate explicit authorization, run the frozen
     baseline command once and commit its completed or failed outcome.
  3. Only after the baseline outcome commit, append and commit a candidate-only
     formal-run pending declaration; only after another explicit authorization,
     run the frozen candidate command once and commit its completed or failed
     outcome. The baseline result may not tune or alter the candidate.
  4. Only after both seed-2024 outcomes are committed, append and commit a
     descriptive matched summary across seeds `2022`, `2023`, and `2024`,
     reporting per-arm and paired candidate-minus-baseline means and sample
     standard deviations (`n=3`, `ddof=1`) for the already recorded validation
     and final Test metrics. Do not rerun or newly access any split to compute
     that summary, and do not present three seeds as significance evidence.
- Future implementation acceptance criteria: both profile definitions, paper-
  ready identities, documentation, and focused tests are added together; each
  seed-2024 defaults map has the same key set as its counterpart and differs
  only at `seed`; the two seed-2024 arms differ only in the three already
  declared active semantic fields; `seed=2024` and `hard_token_seed=2022`
  resolve simultaneously; canonical invocations produce empty dataset and
  student override maps with no paper-ready blockers; CLI `--seed`, semantic,
  or dataset-owned preprocessing overrides remain recorded blockers; targeted
  profile/protocol tests pass; and the implementation outcome is committed
  before any smoke, training, validation ranking, or Test access.
- Risks, evidence limit, and rollback/preservation point: seed-2024 may be low,
  reversed, or fail technically; final Test access remains consumable once per
  arm; the retained official cold-item condition remains; three seeds still
  provide limited uncertainty evidence; and ignored run artifacts will remain
  outside Git until a separately authorized stable backup milestone. Metric
  quality is not a failure criterion and cannot authorize tuning, retry,
  rollback, deletion, or profile alteration. Preserve this declaration commit,
  clean base `82eb88e7fd3e8f5110ddb14388c7117abb77d22e`, immutable
  seed-2022/2023 evidence, and every later
  partial or completed artifact; no destructive rollback is authorized.
- Declaration-task acceptance and unique next action: verify that only this
  append-only log entry changed, run unstaged and staged diff checks, commit it,
  and stop. No metric or run artifact is expected and no dataset split may be
  accessed. The only next action, requiring new explicit authorization, is
  stage 1 above: implement and verify both seed-2024 profiles simultaneously in
  one coherent committed change, append its completed or failed outcome, and
  stop without training, validation ranking, Test, tag, bundle, backup, or
  merge.

### 2026-08-01 | matched seed-2024 profile implementation (pending)

- Purpose and rationale: implement together the two override-free profile IDs
  frozen by declaration commit
  `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c` before any seed-2024 result is
  produced: baseline `baby_student_reference_seed2024_v1` and candidate
  `baby_td_asymmetric_no_projection_seed2024_v1`. Atomic implementation keeps
  either observed arm from influencing the other and extends the already
  validated seed-2023 registry pattern without changing the experiment method.
- Status and authorization boundary: pending implementation only on branch
  `codex/experiment/baby-teacher-baseline` from exact clean parent
  `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c`. No profile execution, smoke,
  training, validation ranking, Test access, formal-run declaration, generated
  checkpoint, manifest, convergence record, raw log, tag, bundle, backup, or
  merge is authorized.
- Declared file scope: append-only `TRAINING_LOG.md`; profile definitions,
  registry, and paper-ready identities in `codes/utility/dataset_profiles.py`;
  focused static/synthetic tests in `codes/tests/test_dataset_profiles.py`; and
  one new `docs/BABY_SEED2024_REPLICATION_PROFILES_V1.md`. Modify no other file
  and do not update the intentionally stale top-of-log summary.
- Frozen implementation contract: add both names/scopes/sources and defaults
  maps together. Each seed-2024 defaults map must have exactly the same keys as
  its corresponding seed-2022 and seed-2023 maps and differ only at student
  training/sampling `seed=2024`; `hard_token_seed=2022` remains dataset-owned.
  Baseline semantic values remain alpha and all four rates `0.0`. Candidate
  values remain alpha `0.3`, item-image `1.0`, item-text `0.3`, and both user
  rates `0.0`. The two arms may differ only at alpha, item-image, and item-text.
  All data, split, feature, PCA/cache, frozen-teacher, `val_test_once_v1`,
  sampler, AdamW, `student_lr=0.00006`, weight decay `0.01`, batch size `1024`,
  maximum epoch `1000`, patience `7`, dimension `64`, random initialization,
  and inference-export controls remain unchanged.
- Planned implementation: mirror the existing seed-2023 constant/default-copy
  and registry-entry structure with canonical seed-2024 sources
  `predeclared_baby_id_only_bpr_reference_seed2024` and
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2024`.
  Registry-derived parser choices and paper-ready identity eligibility must
  include both profiles without special-case relaxation or unrelated refactor.
- Acceptance criteria: existing seed-2022/2023 profile objects remain exactly
  unchanged; both seed-2024 profiles resolve through their canonical parser
  invocations with `dataset_config_overrides={}`,
  `student_config_overrides={}`, and no paper-ready blocker; explicit CLI
  student seed and semantic overrides, dataset-owned preprocessing overrides,
  and unknown profile names remain recorded/rejected as applicable. Tests must
  prove equal key sets, seed-only cross-seed deltas, the exact three-field
  matched-arm delta, frozen `hard_token_seed=2022`, and the full shared control
  set without reading Baby data.
- Planned verification: run the focused dataset-profile test module, discover
  and run the complete existing unit-test suite, compile the relevant Python
  sources, execute canonical parser static-resolution checks for both profiles,
  confirm negative override and unknown-profile gates, run
  `git diff --check`, inspect unstaged and staged file scope, and run
  `git diff --cached --check`. All fixtures must be parser-only or temporary
  synthetic data; `codes/main_mmlight.py` must not execute.
- Risks and rollback/preservation point: copy/paste drift could expose only one
  arm, mutate a prior defaults map, weaken eligibility, or incorrectly treat
  `hard_token_seed` as a student seed. The rollback/preservation point is clean
  declaration commit `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c` plus this pending record; no
  destructive rollback is authorized. If verification fails, preserve the
  pending entry, append a separate failed outcome with evidence, and stop.
- Completion and next-stage gate: after successful implementation and all
  verification, append a separate completed outcome and create one coherent
  commit containing only the declared files. The only later action is a
  separately authorized, baseline-only formal-run pending declaration for
  `baby_student_reference_seed2024_v1`; this implementation task must stop
  before declaring or launching that run.

### 2026-08-01 | matched seed-2024 profile implementation (completed)

- Status and outcome: completed successfully. Both predeclared override-free
  profiles were implemented together and accepted in one coherent working-tree
  scope before any seed-2024 execution or result:
  `baby_student_reference_seed2024_v1` and
  `baby_td_asymmetric_no_projection_seed2024_v1`. No partial one-arm state,
  formal-run declaration, smoke, training, validation ranking, Test access,
  generated run artifact, tag, bundle, backup, or merge occurred.
- Source identity: branch `codex/experiment/baby-teacher-baseline`; exact parent
  and replication-stage declaration commit
  `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c`. The implementation commit is the
  Git commit containing this outcome and must be cited by the next baseline
  formal-run declaration. The intentionally stale top-of-log summary was not
  changed.
- Actual file scope:
  - `codes/utility/dataset_profiles.py` adds both seed-2024 name/scope/source
    identities, defaults copies, and registry entries; the registry-derived
    parser choices and paper-ready identity set therefore include both arms;
  - `codes/tests/test_dataset_profiles.py` adds focused frozen-value,
    cross-seed, matched-arm, canonical-resolution, override, eligibility, and
    registry assertions for both profiles while retaining unknown-name tests;
  - `docs/BABY_SEED2024_REPLICATION_PROFILES_V1.md` records the frozen pair,
    controls, sequential execution gate, and future commands without
    authorizing execution;
  - `TRAINING_LOG.md` contains the preserved replication-stage declaration and
    separate implementation pending/completed records. No other tracked file
    changed.
- Implemented identities: baseline scope/source are `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2024`; candidate scope/source are
  `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2024`. Both
  exact identity tuples are present in
  `BABY_PAPER_READY_STUDENT_PROFILE_IDENTITIES`; unknown identities remain
  rejected by the parser and eligibility gate.
- Frozen defaults audit: both new maps contain the same `26` keys as their
  seed-2022 and seed-2023 counterparts. For each arm, comparison with either
  prior seed changes only `seed`, resolved as `2024`. The canonical parser
  resolves `hard_token_seed=2022`, `val_test_once_v1`, random 64-dimensional
  no-projection initialization, AdamW `student_lr=0.00006`, weight decay
  `0.01`, batch size `1024`, maximum epoch `1000`, patience `7`, no smoke cap,
  the frozen teacher path, final Test enabled, and efficiency disabled.
- Matched semantic audit: baseline resolves `td_distill_alpha=0.0` and all four
  component rates `0.0`; candidate resolves alpha `0.3`, item-image `1.0`,
  item-text `0.3`, and both user rates `0.0`. The computed cross-arm changed
  field set is exactly `td_distill_alpha`, `td_item_image_rate`, and
  `td_item_text_rate`; `seed=2024` and every other value are identical.
- Defaults-map fingerprints: canonical sorted compact-JSON SHA256 is
  `0865aed6197bb2781d8625cf3cf828d981228a1b969c921b83ce9a3f6f43f27f`
  for the seed-2024 baseline and
  `f7ed2a940a5547afdfd3d242c6dc1832dd7e447b2824aaf633b5dd94c3bead6e`
  for the seed-2024 candidate. The four pre-existing 26-key map hashes before
  and after implementation are unchanged: seed-2022 baseline
  `98f16504442942d0209ead388611b8adf50b64c1dfd9b2b6764bcfa98fbdc3cb`,
  seed-2022 candidate
  `018534ec9cedfe4c1d4f5fc1d23552173d3f8a33843c3e71be38f69ea8c46113`,
  seed-2023 baseline
  `29d21d20c5437be41e9af076135a1e977b3c7dcb27a706496656a203f7452650`,
  and seed-2023 candidate
  `1c4c89f253d3adcd167211481dd599591012b8881c9c590c68a6e55d797fe61c`.
- Canonical parser verification: static imports of the real `utility.parser`
  using each documented future argument list returned
  `dataset_config_overrides={}`, `student_config_overrides={}`, and combined
  paper-ready blockers `[]`. Both returned `seed=2024` and
  `hard_token_seed=2022`; baseline returned alpha/rates
  `0.0 / [0.0,0.0,0.0,0.0]`, and candidate returned
  `0.3 / [1.0,0.3,0.0,0.0]`.
- Negative gate verification: a baseline `--seed 2025` override was recorded in
  both dataset and student override maps and blocked; candidate
  `--td_item_text_rate 0.5` was recorded and blocked; baseline
  `--hard_token_seed 2024` was recorded only in the dataset override map and
  activated the existing `Baby reference profile has resolved overrides`
  blocker; and `baby_unknown_student_v1` was rejected by canonical argparse as
  an invalid choice. These checks only parsed arguments.
- Automated verification:
  - focused
    `D:\miniconda\envs\run_5060\python.exe -m unittest codes.tests.test_dataset_profiles -v`
    passed `26/26`;
  - complete
    `D:\miniconda\envs\run_5060\python.exe -m unittest discover -s codes\tests -p "test_*.py" -v`
    passed `46/46` using only static parsing and temporary synthetic fixtures;
  - `D:\miniconda\envs\run_5060\python.exe -m py_compile` passed for
    `dataset_profiles.py`, `parser.py`, `test_dataset_profiles.py`, and
    `main_mmlight.py` without executing them;
  - focused document-contract checks, defaults/identity audits, and
    `git diff --check` passed. The first attempt to run the full suite and
    `py_compile` concurrently hit Windows `WinError 5` while both commands
    replaced the same ignored `__pycache__` file; no assertion or project-code
    failure occurred, no process remained, and sequential reruns of both
    commands passed.
- Data, teacher, protocol, and artifact audit: no Baby data file or split was
  loaded or accessed; no validation or Test ranking occurred. No dataset,
  split, modality feature, PCA/cache, sampler, protocol, optimizer, checkpoint,
  manifest, convergence record, or raw log changed or was generated. The
  read-only teacher remains `141098540` bytes with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`,
  and no `main_mmlight.py` process exists. Metrics and run-specific artifact
  paths are not applicable to this implementation-only task.
- Acceptance decision and unresolved risks: all declared implementation and
  verification criteria passed, so the pair is ready for a separately declared
  sequential baseline stage. This static implementation does not itself prove
  future runtime completion or metric quality; the official cold-item warning,
  consumable one-time final Test access, three-seed evidence limit, and lack of
  stable backup for ignored future artifacts remain. No run is authorized by
  this outcome.
- Unique next action: only with new explicit user authorization, append and
  commit one baseline-only uncapped formal-run pending declaration for
  `baby_student_reference_seed2024_v1`, citing this implementation commit and
  freezing its exact canonical command, clean-source/environment/data/teacher
  gates, foreground timeout requirement, acceptance criteria, and planned
  artifacts. Stop after that declaration without training, validation ranking,
  Test, candidate declaration, tag, bundle, backup, or merge.

### 2026-08-01 | baby_student_reference_seed2024_v1 formal uncapped baseline (pending)

- Purpose and hypothesis: declare the third matched ID-only BPR baseline before
  any seed-2024 execution or result. The purpose is to measure the frozen
  reference method under student initialization and maintained pairwise-sampling
  seed `2024`, providing the baseline half of the predeclared seed-2024 pair.
  This run does not tune any value and cannot alter or authorize the separately
  frozen candidate.
- Status and authorization boundary: formal uncapped baseline pending; the run
  has not started. This declaration task may append and commit only
  `TRAINING_LOG.md`. It must not execute `codes/main_mmlight.py`, load Baby
  data, perform training, validation ranking, or Test, declare or execute the
  candidate, generate a run artifact, update the stale top-of-log summary,
  create a tag/bundle/backup, or merge `main`.
- Source and branch identity: branch
  `codex/experiment/baby-teacher-baseline`; matched seed-2024 replication
  declaration commit `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c`; simultaneous
  profile implementation and verification commit
  `cb22e9dac1f67c8f4ae713de0da3183505f78ba6`. The exact clean formal-run
  launch source is the documentation commit containing this declaration; its
  full hash must be reported at declaration completion and used unchanged by
  the later authorized launch.
- Frozen command, to be executed exactly once only after new explicit user
  authorization:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Canonical profile and override gate: profile/scope/source are exactly
  `baby_student_reference_seed2024_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2024`. Static parsing of the
  frozen command at implementation commit `cb22e9d...` resolved
  `dataset_config_overrides={}`, `student_config_overrides={}`, and combined
  paper-ready blockers `[]`. A later launch is eligible only if all three remain
  exactly empty; no CLI seed, parameter, data, preprocessing, or behavior
  override may be added.
- Frozen seed and preprocessing separation: student initialization, training,
  and maintained `data_generator.sample()` pairwise BPR sampling use
  `seed=2024`. Frozen teacher preprocessing and PCA cache identity retain
  `hard_token_seed=2022`. The latter must not follow the student seed, and no
  PCA/cache regeneration, fallback, or provenance change is allowed.
- Frozen student identity and initialization: no student checkpoint is loaded;
  user and item ID embeddings are randomly initialized at seed `2024`;
  `student_model_type=td_distill_no_projection`; embedding dimension `64`;
  `td_init_from_teacher=false`; no teacher warm start, projection head, graph
  inference state, modality encoder, prompt state, or semantic cache enters the
  deployable student.
- Frozen baseline loss: BPR only, with `td_distill_alpha=0.0` and
  `td_item_image_rate=0.0`, `td_item_text_rate=0.0`,
  `td_user_image_rate=0.0`, and `td_user_text_rate=0.0`. All directional loss
  components must remain inactive and must not influence optimization.
- Frozen optimizer and budget: AdamW `student_lr=0.00006`; student weight decay
  `0.01`; batch size `1024`; maximum `epoch=1000`; validation every epoch;
  early-stopping patience `7`; `smoke_train_batches=0`; unchanged pairwise BPR
  sampler; all `116` independently sampled batches in every completed epoch;
  `run_efficiency_benchmark=false`. The run is uncapped except for validation-
  selected early stopping and has no metric-based retry budget.
- Frozen dataset and preprocessing identity: audited MMRec Baby under
  `data/baby/`; conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  `train_mat`, `val_mat`, and `test_mat` SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve the official split assignments, `retain_official` cold-item policy,
  duplicate-modalities policy `error`, dataset preflight, shapes/dtypes,
  modality finiteness/non-duplication, pairwise split overlaps `0 / 0 / 0`,
  `pca` hard-token type, cache identities, and sampler implementation.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, exactly `141098540` bytes and
  SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, training,
  run-local teacher checkpoint, alias publication or overwrite, checkpoint
  mutation, or new teacher Test ranking is allowed. Reused teacher metrics may
  come only from frozen checkpoint metadata.
- Frozen evaluation and Test contract: `val_test_once_v1`; train only on
  `train_mat`; selection split `validation`; primary and sole checkpoint/
  early-stop selector validation Recall@20; `Ks=[10,20,40,50]`;
  `test_flag=part`; candidate exclusion `train_only`. Dataset preflight may read
  Test structure and identity but cannot rank it. After natural training stop,
  restore the validation-best full student checkpoint and perform exactly one
  final student Test ranking. Test metrics are report-only and cannot affect
  selection, acceptance, parameters, retry, rollback, or the future candidate.
- Intended environment: `D:\miniconda\envs\run_5060\python.exe`; Python
  `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 with `8151` MiB reported memory; GPU selector
  `0`. Immediately before launch, verify this environment, available GPU/disk,
  exact clean HEAD/branch, zero pre-existing `main_mmlight.py` processes,
  canonical parser resolution, source-diff gate, teacher fingerprint, dataset
  identities, and absence of conflicting new run artifacts.
- Mandatory execution transport: the later authorized command must remain in
  one continuously attached foreground shell/tool session for its entire
  lifetime, with shell/tool timeout explicitly set to at least `10800000` ms.
  Default or short timeout, detached/background launch, output-channel closure
  while active, a second command execution, and automatic retry are forbidden.
  If the process or transport ends for any reason, preserve all state and
  artifacts, audit that single attempt, append one completed or failed outcome,
  and stop.
- Predeclared run-specific artifacts, all keyed by the one timestamp/PID run
  identity created by the later single process:
  - raw log `logs/<run_name>`;
  - preflight
    `exp/runs/baby/dataset_preflight__<run_name>.json`;
  - manifest `exp/runs/baby/run_manifest__<run_name>.json`;
  - convergence record
    `exp/converge/baby/auto__<run_name>.pkl`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  No shared student alias is allowed. All six must exist under the same run
  identity, remain outside Git, and receive exact byte sizes and SHA256 values
  in the outcome audit.
- Hard acceptance criteria: launch once from the exact clean declaration commit
  with the frozen command and required foreground transport; pass source,
  branch, environment, canonical-profile, empty-override, zero-blocker,
  seed/cache, dataset/preflight, and frozen-teacher gates; keep all optimization
  loss and validation metric series finite; complete `116/116` batches in every
  completed epoch; select and early-stop only by validation Recall@20; restore
  the validation-best checkpoint before exactly one student final Test; finish
  with manifest status `completed`, `paper_ready_eligible=true`, blockers `[]`,
  and `final_test_performed=true`; preserve the teacher and all aliases; and
  produce/fingerprint all six same-run artifacts.
- Checkpoint and deployment acceptance: the full checkpoint must contain only
  the declared finite user/item ID embedding model state, optimizer state, and
  required metadata. The inference-only checkpoint must contain only finite
  user/item ID embeddings plus deployment metadata including dimension/counts
  and the no-projection variant; it must contain no teacher, modality, prompt,
  semantic cache, graph, projection, or optimizer state. Full and inference
  user/item embeddings must be bit-for-bit equal.
- Quality acceptance and failure boundary: there is no minimum metric or
  improvement threshold. Any finite result produced by the exact declared code,
  identities, command, and protocol is valid completed seed-2024 baseline
  evidence even if low or worse than prior seeds. Mark the later run `failed`
  only for process/code failure, non-finite values, source/profile/override/data/
  teacher drift, chronology or Test violation, teacher mutation, missing or
  cross-run artifacts, checkpoint/deployment mismatch, or another hard
  acceptance failure. Metric quality must never trigger reset, revert, retry,
  deletion, rollback, or suppression of evidence.
- Risks and preservation point: the uncapped GPU run may fail or exceed three
  hours; final student Test access is consumable once; the retained official
  cold-item warning remains; seed `2024` may differ materially from prior seeds;
  and ignored artifacts will not be protected by Git. Preserve the exact clean
  declaration commit, implementation commit `cb22e9d...`, frozen teacher, all
  earlier evidence, and every partial or completed same-run artifact. No
  destructive rollback or candidate action is authorized.
- Declaration verification and unique next action: this documentation-only task
  must confirm the canonical static resolution, teacher fingerprint, exact
  command and frozen contracts; verify a pure append-only one-file diff with
  unstaged/staged whitespace and scope checks; append a separate declaration-
  completed outcome; commit; and stop. Only after that clean commit and new
  explicit user authorization may the frozen baseline command be executed once
  through the required long-lived foreground session, followed by audit and a
  committed completed/failed outcome. Do not declare or execute the candidate.

### 2026-08-01 | baby_student_reference_seed2024_v1 formal declaration record (completed; run pending)

- Status and outcome: the independent uncapped baseline formal-run declaration
  is recorded completely; the formal run remains pending and has not started.
  This is a documentation outcome only, not a training or experiment result.
- Source and scope: branch `codex/experiment/baby-teacher-baseline`; clean parent
  `cb22e9dac1f67c8f4ae713de0da3183505f78ba6`; replication declaration
  `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c`. Only append-only
  `TRAINING_LOG.md` changed. No historical entry or stale top-of-log summary was
  edited, and no code, profile, test, documentation file, parameter, or protocol
  implementation changed.
- Frozen command verification: the declaration contains exactly
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`.
  It is recorded for a later separately authorized single execution and was not
  executed in this task.
- Static contract evidence: real `utility.parser` argument resolution, without
  importing `main_mmlight.py`, returned canonical profile/scope/source
  `baby_student_reference_seed2024_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2024`, empty dataset/student
  override maps, blockers `[]`, `seed=2024`, and `hard_token_seed=2022`. It also
  confirmed the frozen 64-dimensional random no-projection BPR-only student,
  alpha/rates all zero, AdamW `student_lr=0.00006`, weight decay `0.01`, batch
  size `1024`, maximum epoch `1000`, patience `7`, no smoke cap, frozen teacher
  reuse, final Test enabled, and efficiency disabled.
- Teacher and environment evidence: the read-only teacher is still `141098540`
  bytes with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  The intended environment check returned Python `3.10.20`, PyTorch
  `2.11.0+cu128`, CUDA runtime `12.8`, driver `595.97`, and NVIDIA GeForce RTX
  5060 on selector `0`. A launch must repeat all preflight gates from the clean
  declaration commit rather than rely only on this declaration-time evidence.
- Declaration verification: the exact commit references, profile ID, command,
  override requirements, seeds, model/loss/optimizer budget, data/teacher/
  protocol identities, foreground timeout, one-execution/no-retry rule, six
  artifact classes, hard acceptance criteria, low-metric preservation rule,
  and baseline-only sequencing were checked. The pre-staging diff contains only
  an append-only `TRAINING_LOG.md` change and `git diff --check` passed. Final
  staged scope and `git diff --cached --check` must pass before the declaration
  commit is created.
- Data, Test, process, metrics, and artifacts: no Baby data file or split was
  loaded or accessed; no validation or Test ranking occurred; no
  `main_mmlight.py` process exists; and no checkpoint, manifest, convergence
  record, raw log, or other run artifact was generated. Validation/Test metrics
  and artifact hashes are therefore not applicable.
- Acceptance and unresolved risks: the declaration record satisfies the
  requested formal-run gate and is ready to be committed. Runtime success,
  metric quality, exact duration, and run artifact identities remain unknown;
  the official cold-item warning, consumable one-time student Test access, GPU/
  transport failure risk, and ignored-artifact preservation risk remain. The
  formal baseline is not completed until its later single process exits and its
  outcome is audited and committed.
- Unique next action: only with new explicit user authorization, start from the
  new clean declaration commit and execute the frozen seed-2024 baseline command
  exactly once in one continuously attached foreground session with shell/tool
  timeout at least `10800000` ms. After natural exit, audit that single run and
  append/commit one completed or failed outcome. Do not declare or execute the
  candidate.

### 2026-08-02 | baby_student_reference_seed2024_v1 formal uncapped baseline (completed)

- Status and decision: completed valid formal seed-2024 baseline evidence. The
  single authorized process exited naturally with code `0`; every declared
  identity, execution-flow, numerical, artifact, and deployment hard gate
  passed. Metric magnitude was not used as an acceptance threshold, and no
  retry, reset, revert, deletion, rollback, or candidate action occurred.
- Source and launch gates: branch `codex/experiment/baby-teacher-baseline` at
  exact clean declaration commit
  `b1b5835a22060dcfe4a76d240d2be1cded157257`, whose parent/implementation and
  replication provenance remain `cb22e9dac1f67c8f4ae713de0da3183505f78ba6`
  and `f6dd2e4f1cd52e36e3a4fe6c87a8f5fb24454b8c`. Before launch, HEAD, branch,
  clean worktree, zero pre-existing Python/`main_mmlight.py` processes, source
  identity, environment, GPU, disk, canonical resolution, data/preprocessing
  identities, hard-token caches, and frozen teacher fingerprint all passed.
  Available disk was approximately `378.1 GiB`.
- Command and execution count: the command was executed exactly once, with no
  added argument or override and no automatic or manual retry:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`.
  It remained in one continuously attached foreground shell/tool call with
  explicit timeout `10800000` ms until natural exit. Foreground wall time was
  `2380.7` seconds (`39:40.7`); manifest time from
  `2026-08-01T23:31:40.282800+08:00` through
  `2026-08-02T00:11:15.857494+08:00` was `2375.574694` seconds
  (`39:35.574694`).
- Run identity and canonical contract: run
  `2026-08-01 23_31_40.280800_baby_light_init_pid15908`, main PID `15908`.
  Manifest/parser identity is
  `baby_student_reference_seed2024_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2024`, with
  `dataset_config_overrides={}`, `student_config_overrides={}`, and
  `paper_ready_blockers=[]`. Resolved student/sampling `seed=2024`, frozen
  preprocessing `hard_token_seed=2022`, random 64-dimensional
  `td_distill_no_projection` initialization, no student checkpoint, and
  `td_init_from_teacher=false` all remained exact.
- Frozen training contract: BPR only with `td_distill_alpha=0.0` and all four
  directional rates `0.0`; AdamW `student_lr=0.00006`, weight decay `0.01`,
  batch size `1024`, maximum epoch `1000`, patience `7`, and
  `smoke_train_batches=0`. The attached foreground progress completed
  `116/116` batches for every epoch. Independently, the manifest records
  `118551` train interactions and batch size `1024`, yielding
  `ceil(118551/1024)=116`; the raw logger records one completed Validation
  result for each contiguous epoch `0` through `170`. The logger file does not
  persist tqdm `116/116` text, so batch completion is supported jointly by the
  attached foreground transcript, uncapped resolved arguments, derived batch
  count, and all 171 post-epoch Validation records.
- Numerical and convergence audit: all 171 entries in total/BPR loss, five
  distillation/directional loss, and eight selection metric series are finite
  and aligned. Total and BPR loss are identical, ranging from
  `80.39728689193726` to `36.642337173223495`; distillation and every
  directional loss are exactly zero for all epochs. The raw log likewise has
  171 finite epoch records and prints all four semantic components as zero in
  every record. No NaN or infinity was found.
- Validation selection: validation Recall@20 was the sole checkpoint and
  early-stop selector. Recomputing the maximum over the convergence record
  selected epoch `163` with exact Recall@20
  `0.04086654667009528`, matching manifest, convergence, and full-checkpoint
  metadata. At that epoch the stored exact Validation values include
  Recall@50 `0.0763436554873942`, NDCG@20
  `0.017870566178556143`, and NDCG@50 `0.02531633188996191`.
  The raw rounded K=`[10,20,40,50]` vectors at epoch 163 are recall
  `[0.02539, 0.04087, 0.06618, 0.07634]`, precision
  `[0.00270, 0.00217, 0.00177, 0.00163]`, hit ratio
  `[0.02695, 0.04315, 0.07025, 0.08089]`, and NDCG
  `[0.01380, 0.01787, 0.02338, 0.02532]`.
- Early stop and Test chronology: epochs `164` through `170` produced seven
  consecutive non-improvements, followed by the sole early-stop record. The
  raw log then records exactly one student final Test line after restoring
  validation-best epoch `163`; it records no new teacher Test ranking. Dataset
  preflight accessed only Test structure/identity before training, and the
  frozen checkpoint's teacher Test metrics were reused from metadata rather
  than recomputed. Thus Test ranking did not select a checkpoint or affect
  stopping, and the only new ranking access to Test was the one report-only
  student final Test after selection was complete.
- Exact final Test result for K=`[10,20,40,50]`: precision
  `[0.0029107739778863767, 0.002409359732579101, 0.0019066598097197505, 0.0017567498071484426]`;
  recall
  `[0.025622757717126184, 0.042307913651446934, 0.06718924728310247, 0.07738434061010546]`;
  NDCG
  `[0.01465014288963667, 0.019323063741455922, 0.024947511959222856, 0.027025817147583384]`;
  hit ratio
  `[0.028747750064283543, 0.04736436101825707, 0.0744664438158912, 0.08588326047827131]`;
  AUC `0.0`. All values are finite and convergence and manifest copies are
  identical.
- Manifest and profile acceptance: manifest `status=completed`,
  `evaluation_protocol=val_test_once_v1`, `selection_split=validation`,
  `primary_selection_metric=Recall@20`, `candidate_exclusion_policy=train_only`,
  `paper_ready_eligible=true`, blockers `[]`, and
  `final_test_performed=true`. The preflight retained the declared official
  cold-item warning, finite/non-duplicate modality features, disjoint splits,
  and unchanged data hashes. Image/text PCA hard-token caches were hits with
  random state `2022`; no preprocessing fallback or regeneration occurred.
- Same-run artifacts and fingerprints, all outside Git and all keyed by the
  exact run identity:
  - raw log
    `D:\Download\PromptMM\logs\2026-08-01 23_31_40.280800_baby_light_init_pid15908`,
    `76963` bytes, SHA256
    `4458c705f2eaabf604165ab06db9e56f9efb1ae8443316022d8a37dce5593e8d`;
  - preflight
    `D:\Download\PromptMM\exp\runs\baby\dataset_preflight__2026-08-01 23_31_40.280800_baby_light_init_pid15908.json`,
    `3399` bytes, SHA256
    `aac3f8446e47067488b5d9d37c8d737a7f75c766ee67dab919741d8139565b4d`;
  - manifest
    `D:\Download\PromptMM\exp\runs\baby\run_manifest__2026-08-01 23_31_40.280800_baby_light_init_pid15908.json`,
    `25386` bytes, SHA256
    `d0f3e2bb49dd641a1b88d5a16e0fb73c0fa5e45c1b5e65d2d3418d1d1ae04b7c`;
  - convergence
    `D:\Download\PromptMM\exp\converge\baby\auto__2026-08-01 23_31_40.280800_baby_light_init_pid15908.pkl`,
    `25280` bytes, SHA256
    `28b3e9f096781a6a647ba5fabcdf10e919d0bd6b6b0960dee71013a5ea8bff79`;
  - full checkpoint
    `D:\Download\PromptMM\Model\baby\td_distill\td_distill_full__val_test_once_v1__2026-08-01 23_31_40.280800_baby_light_init_pid15908.pth`,
    `20354975` bytes, SHA256
    `4b1e4c5a227171a66fd9941eb8819676cce2d56e51005ae625dd4e78e3141245`;
  - inference-only checkpoint
    `D:\Download\PromptMM\Model\baby\td_distill\td_distill_infer_only__val_test_once_v1__2026-08-01 23_31_40.280800_baby_light_init_pid15908.pth`,
    `6785997` bytes, SHA256
    `aad455f49fe570c90dc16229e9a18511cf1188da75d137de151c7c03222c8179`.
- Checkpoint/deployment audit: the full checkpoint has exact run/profile,
  protocol, best-epoch, optimizer, rate, learning-rate, weight-decay, and
  no-warm-start metadata; its model state contains only finite
  `user_id_embedding.weight` shape `[19445,64]` and
  `item_id_embedding.weight` shape `[7050,64]`, and all optimizer tensors are
  finite. The inference-only checkpoint has exactly the two finite embeddings
  plus `embedding_dim`, `n_users`, `n_items`, and variant
  `td_distill_no_projection`; it contains no optimizer, teacher, modality,
  prompt, semantic cache, graph, projection, or other state. Both embedding
  tensors have equal dtype and are bit-for-bit equal to the full checkpoint.
- Frozen teacher and process audit: before and after the run,
  `Model/baby/teacher_model_val_test_once_v1.pt` remained exactly `141098540`
  bytes with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  Teacher training was skipped, the alias was not overwritten, no run-local
  teacher checkpoint was generated, and no new teacher Test ranking occurred.
  PID `15908` exited, and the final process audit found no remaining
  `main_mmlight.py` process.
- Experimental meaning and unresolved risks: this supplies the seed-2024
  baseline half of the predeclared matched replication under the same frozen
  non-seed controls as seeds 2022/2023. It is not a candidate comparison or a
  three-seed aggregate. The official retained cold-item condition, consumed
  one-time student Test access, seed variance, and lack of Git/off-device
  protection for ignored run artifacts remain; all six artifacts are preserved
  in place. No code, configuration, teacher, data, historical log entry, or
  stale top-of-log summary changed in this outcome task.
- Unique next action: only with new explicit user authorization, append and
  commit a separate formal-run pending declaration for
  `baby_td_asymmetric_no_projection_seed2024_v1`, citing this completed
  baseline outcome and freezing the already implemented seed-2024 candidate
  command and contracts. Stop after that declaration without launching the
  candidate.

### 2026-08-02 | baby_td_asymmetric_no_projection_seed2024_v1 formal uncapped candidate (pending)

- Purpose and hypothesis: declare the candidate half of the predeclared matched
  seed-2024 Baby replication as one uncapped formal asymmetric no-projection
  directional-distillation run. The hypothesis, frozen before either
  seed-2024 arm produced evidence, is that the fixed item-dominant semantic
  transfer can improve the deployable student over the matched ID-only BPR
  baseline while preserving initialization seed, sampler, optimizer budget,
  deployment state, data, teacher, preprocessing, and evaluation protocol.
  This declaration makes no quality claim and does not tune any value from the
  completed baseline result.
- Status and authorization boundary: formal uncapped candidate pending; the run
  has not started. This task authorizes only this append-only declaration,
  focused content/diff/index verification, and one local commit containing
  only `TRAINING_LOG.md`. It does not authorize `codes/main_mmlight.py`
  execution, dataset loading, training, validation ranking, Test access,
  candidate parameter modification, run-artifact generation, retry, tag,
  bundle, backup, merge to `main`, or any later experiment stage.
- Baseline-outcome prerequisite and parameter-selection firewall: the matched
  `baby_student_reference_seed2024_v1` baseline completed successfully, and its
  audited outcome is committed at exact clean predecessor
  `1a8104b75aab9bf8ba95d6de5beb5ca18d24c0b1`. It selected validation-best
  epoch `163`, with validation Recall@20 `0.04086654667009528` and final Test
  Recall@20 `0.042307913651446934`. These results satisfy the required
  baseline-outcome-before-candidate gate but did not select, cancel, reorder,
  or alter any candidate parameter. The matched candidate was implemented and
  frozen together with the baseline before this evidence existed.
- Branch, implementation identity, and preservation point: branch
  `codex/experiment/baby-teacher-baseline`; exact clean declaration parent and
  completed baseline outcome
  `1a8104b75aab9bf8ba95d6de5beb5ca18d24c0b1`; simultaneous matched-profile
  implementation commit `cb22e9dac1f67c8f4ae713de0da3183505f78ba6`.
  The commit containing this pending declaration is the only authorized source
  for the later single launch.
  `git diff cb22e9d... <launch-commit> -- codes docs` must remain empty.
  Preserve both commits, the frozen teacher, all prior evidence, and every
  later partial or completed same-run artifact; no destructive rollback is
  authorized.
- Frozen command, to execute exactly once only after this declaration is
  committed in a clean tree and the user gives new explicit authorization:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
  No argument may be added, removed, reordered, or overridden.
- Canonical profile and empty-override gate: profile/scope/source are exactly
  `baby_td_asymmetric_no_projection_seed2024_v1` / `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2024`.
  Canonical parsing at implementation commit `cb22e9d...` verified
  `dataset_config_overrides={}`, `student_config_overrides={}`, and combined
  paper-ready blockers `[]`. All three must remain exactly empty at launch and
  in the run manifest. Dataset and student overrides are required to be empty;
  no CLI seed, semantic, optimizer, epoch, patience, batch, sampler, data,
  preprocessing, teacher, protocol, final-Test, or other behavior override is
  allowed.
- Frozen seed and preprocessing separation: student initialization, training,
  and maintained pairwise BPR sampling use `seed=2024`. Frozen teacher
  preprocessing and PCA cache identity retain `hard_token_seed=2022`.
  `hard_token_seed` must not follow the student seed, and no PCA/cache
  regeneration, fallback, or provenance change is allowed.
- Frozen student identity and initialization: load no student checkpoint;
  randomly initialize user and item ID embeddings at seed `2024`; use
  `student_model_type=td_distill_no_projection`, embedding dimension `64`, and
  `td_init_from_teacher=false`. No teacher warm start or projection head is
  allowed. Deployable state remains only user/item ID embeddings plus required
  deployment metadata, with no graph, modality encoder, prompt, semantic
  cache, teacher, projection, or optimizer state in the inference export.
- Frozen asymmetric semantic parameters: `td_distill_alpha=0.3`,
  `td_item_image_rate=1.0`, `td_item_text_rate=0.3`,
  `td_user_image_rate=0.0`, and `td_user_text_rate=0.0`. Active supervision is
  exactly item-image and item-text, optimizing
  `L_BPR + 0.3 * ((1.0 * L_item_image + 0.3 * L_item_text) / 1.3)`.
  Relative to the matched baseline, the only changed values are
  `td_distill_alpha`, `td_item_image_rate`, and `td_item_text_rate`; user-side
  rates and every non-semantic control remain identical.
- Frozen optimizer, sampling, and budget: AdamW `student_lr=0.00006`; student
  weight decay `0.01`; batch size `1024`; maximum `epoch=1000`; validation
  every epoch; early-stopping patience `7`; `smoke_train_batches=0`; unchanged
  `data_generator.sample()` pairwise BPR sampler; all `116` independently
  sampled batches in every completed epoch; `run_efficiency_benchmark=false`.
  The run is uncapped except for validation-selected natural early stopping and
  has no metric-triggered retry budget.
- Frozen dataset and preprocessing identity: audited MMRec Baby under
  `data/baby/`; conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  `train_mat`, `val_mat`, and `test_mat` SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve official splits, cold-item policy `retain_official`, duplicate-
  modalities policy `error`, dataset preflight, shapes/dtypes, modality
  finiteness/non-duplication, split overlaps `0 / 0 / 0`, `pca` hard-token
  type, cache identities, and sampler implementation.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, exactly `141098540` bytes and
  SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, training,
  run-local teacher checkpoint, alias publication or overwrite, checkpoint
  mutation, or new teacher Test ranking is allowed. Reused teacher metrics may
  come only from frozen checkpoint metadata.
- Frozen evaluation and Test contract: `val_test_once_v1`; train only on
  `train_mat`; select and early-stop only with validation Recall@20;
  `Ks=[10,20,40,50]`; `test_flag=part`; candidate exclusion `train_only`.
  Dataset preflight may read Test structure and identity but cannot rank it.
  After natural training stop, restore the validation-best full student
  checkpoint and perform exactly one final student Test ranking. Test metrics
  are report-only and cannot affect selection, acceptance, parameters, retry,
  rollback, or interpretation of the completed baseline.
- Intended environment: `D:\miniconda\envs\run_5060\python.exe`; Python
  `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 with `8151` MiB reported memory; GPU
  selector `0`. Immediately before launch, verify the exact clean HEAD/branch,
  source-diff gate, zero pre-existing `main_mmlight.py` processes, environment,
  available GPU/disk, canonical profile, both empty override maps, zero
  blockers, data/cache identities, frozen teacher fingerprint, and absence of
  conflicting new run artifacts.
- Mandatory execution transport: the later authorized command must remain in
  one continuously attached foreground shell/tool session for its entire
  lifetime, with shell/tool timeout explicitly set to at least `10800000` ms.
  Default or short timeout, detached/background launch, output-channel closure
  while active, a second command execution, and automatic retry are forbidden.
  If the process or transport ends for any reason, preserve its state and
  artifacts, audit that single attempt, append and commit one completed or
  failed outcome, and stop.
- Predeclared run-specific artifacts, all keyed by the one timestamp/PID run
  identity created by the later single process:
  - raw log `logs/<run_name>`;
  - preflight
    `exp/runs/baby/dataset_preflight__<run_name>.json`;
  - manifest `exp/runs/baby/run_manifest__<run_name>.json`;
  - convergence record
    `exp/converge/baby/auto__<run_name>.pkl`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  No shared student alias is allowed. All six must exist under the same run
  identity, remain outside Git, and receive exact byte sizes and SHA256 values
  in the later outcome audit.
- Hard acceptance criteria: launch once from the exact clean declaration commit
  with the frozen command and required foreground transport; pass source,
  branch, environment, canonical-profile, empty-override, zero-blocker,
  seed/cache, dataset/preflight, and frozen-teacher gates; keep every total,
  BPR, semantic-component, and validation metric series finite; complete
  `116/116` batches in every completed epoch; keep active heads exactly
  item-image and item-text; select and early-stop only by validation Recall@20;
  restore the validation-best checkpoint before exactly one student final
  Test; finish with manifest status `completed`,
  `paper_ready_eligible=true`, blockers `[]`, and
  `final_test_performed=true`; preserve the teacher and all aliases; and
  produce/fingerprint all six same-run artifacts.
- Checkpoint and deployment acceptance: the full checkpoint must contain only
  finite user/item ID embedding model state, optimizer state, and declared
  metadata. The inference-only checkpoint must contain only finite user/item
  ID embeddings plus dimension/count/variant deployment metadata; it must
  contain no teacher, modality, prompt, semantic cache, graph, projection, or
  optimizer state. Full and inference user/item embeddings must be bit-for-bit
  equal.
- Quality acceptance and failure boundary: there is no minimum metric or
  improvement threshold. Any finite result produced by the exact declared
  code, identities, command, and protocol is valid completed seed-2024
  candidate evidence even if lower than its matched baseline or prior seeds.
  Mark the later run `failed` only for process/code failure, non-finite values,
  source/profile/override/data/teacher drift, chronology or Test violation,
  teacher mutation, missing or cross-run artifacts, checkpoint/deployment
  mismatch, or another hard acceptance failure. Metric quality must never
  trigger reset, revert, retry, deletion, rollback, or suppression of evidence.
- Risks and unresolved evidence limits: the uncapped GPU run may fail or exceed
  three hours; final student Test access is consumable once; the retained
  official cold-item warning remains; seed `2024` may differ materially from
  prior seeds; and ignored artifacts will not be protected by Git. This
  declaration provides no new metric, runtime, or artifact evidence and does
  not establish the three-seed comparison until the candidate outcome is
  audited and committed.
- Declaration verification and unique next action: verify that this is a pure
  append-only one-file change, check the exact command, commit references,
  empty-override requirement, frozen contracts, and unstaged/staged whitespace
  and file scope, create one local declaration commit, and stop. Only after
  that clean commit and new explicit user authorization, execute the frozen
  candidate command exactly once in one continuously attached foreground
  session with shell/tool timeout at least `10800000` ms; after it exits, audit
  the single run and append/commit one completed or failed outcome.

### 2026-08-02 | baby_td_asymmetric_no_projection_seed2024_v1 formal uncapped candidate (completed)

- Status and decision: completed valid formal seed-2024 candidate evidence.
  The single authorized process exited naturally with code `0`; every declared
  source, execution-flow, numerical, artifact, protocol, and deployment hard
  gate passed. Metric magnitude was not an acceptance threshold, and no retry,
  second launch, parameter change, reset, revert, deletion, rollback, tag,
  bundle, backup, merge, or later experiment occurred.
- Source and launch gates: branch `codex/experiment/baby-teacher-baseline` at
  exact clean declaration commit
  `e2a730748a7527fe284358e0c19cccd5744b5906`; completed matched seed-2024
  baseline outcome commit
  `1a8104b75aab9bf8ba95d6de5beb5ca18d24c0b1`; simultaneous matched-profile
  implementation commit `cb22e9dac1f67c8f4ae713de0da3183505f78ba6`.
  Before launch, the exact HEAD/branch, clean worktree, zero pre-existing
  `main_mmlight.py` processes, empty `codes`/`docs` implementation diff,
  environment/GPU/disk, canonical resolution, empty override maps, zero
  blockers, data/cache identities, frozen teacher fingerprint, and absence of
  conflicting seed-2024 candidate artifacts all passed. Post-run
  `git diff cb22e9d... e2a7307... -- codes docs` is also empty.
- Command and execution count: the frozen command was executed exactly once,
  with no added, removed, reordered, or overridden argument and no automatic
  or manual retry:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`.
  It remained in one continuously attached foreground shell/tool session with
  explicit timeout `10800000` ms until natural exit. Foreground wall time was
  approximately `884.1` seconds (`14:44.1`); manifest time from
  `2026-08-02T00:42:54.031643+08:00` through
  `2026-08-02T00:57:33.513708+08:00` was `879.482065` seconds
  (`14:39.482065`).
- Run and environment identity: run
  `2026-08-02 00_42_54.029141_baby_light_init_pid29752`, main PID `29752`;
  `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`; PyTorch
  `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver `595.97`; NVIDIA GeForce
  RTX 5060 with `8151` MiB reported memory; GPU selector `0`. PID `29752`
  exited and the final process check found no remaining `main_mmlight.py`
  process.
- Canonical profile and frozen controls: profile/scope/source are exactly
  `baby_td_asymmetric_no_projection_seed2024_v1` / `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2024`;
  `dataset_config_overrides={}`, `student_config_overrides={}`, and
  `paper_ready_blockers=[]`. Student initialization, training, and pairwise
  sampling used `seed=2024`, while frozen PCA preprocessing retained
  `hard_token_seed=2022`. The student was a randomly initialized
  64-dimensional `td_distill_no_projection` model with no checkpoint load or
  teacher warm start; `td_init_from_teacher=false`.
- Frozen optimization and semantic contract: AdamW `student_lr=0.00006`,
  student weight decay `0.01`, batch size `1024`, maximum `epoch=1000`,
  patience `7`, `smoke_train_batches=0`, and no efficiency benchmark.
  `td_distill_alpha=0.3`; item-image, item-text, user-image, and user-text rates
  were exactly `1.0 / 0.3 / 0.0 / 0.0`; active heads were exactly item-image
  and item-text. Resolved arguments, raw log, convergence record, manifest, and
  full-checkpoint metadata agree.
- Dataset and preprocessing audit: conversion manifest, train, validation,
  Test, image, and text SHA256 values all matched the pending declaration.
  Preflight confirmed shapes/dtypes, finite non-duplicate modalities, split
  overlaps `0 / 0 / 0`, and duplicate policy `error`. Its only warning was the
  frozen retained-official condition: item IDs `240`, `1212`, and `6115` are
  absent from train and cover `11` validation and `7` Test interactions, with
  no cold users. Image/text PCA caches were hits with random state `2022`; no
  fallback, regeneration, or data/cache identity change occurred.
- Frozen teacher audit: teacher training was skipped and read-only
  `Model/baby/teacher_model_val_test_once_v1.pt` remained exactly `141098540`
  bytes with pre/post SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  Its last-write time remained `2026-07-30 11:57:18`; no teacher optimizer
  step, run-local teacher checkpoint, alias publication/overwrite, or new
  teacher Test ranking occurred. Printed teacher metrics were reused from
  frozen checkpoint metadata.
- Training and numerical audit: epochs `0-67` completed contiguously, each
  reaching all `116/116` batches, for `7888` optimization batches. All `68`
  entries in all `15` convergence loss/metric series were finite and aligned.
  Epoch `0 -> 67` endpoints were total loss
  `81.4356541633606 -> 64.64407312870026`, BPR
  `80.3982520699501 -> 64.20018488168716`, normalized directional loss
  `3.458005305379629 -> 1.4796270811930299`, item-image
  `3.4248215835541487 -> 1.1291100652888417`, and item-text
  `3.5686173364520073 -> 2.648016979917884`; both inactive user-side series
  remained exactly zero. Recomputed objective residuals were at most
  `5.44e-7` for total versus BPR plus scaled directional loss and `9.02e-8`
  for the declared normalized item objective; no NaN or infinity occurred.
- Validation selection: validation Recall@20 alone selected best epoch `60`.
  Exact stored metrics there were Recall@20 `0.0671997942915925`, Recall@50
  `0.1195895627471833`, NDCG@20 `0.029492535104592942`, and NDCG@50
  `0.040256057809183156`. The raw-log K=`[10,20,40,50]` rounded vectors were
  Recall `[0.04224, 0.06720, 0.10361, 0.11959]`, Precision
  `[0.00442, 0.00352, 0.00272, 0.00251]`, Hit Ratio
  `[0.04407, 0.07010, 0.10810, 0.12466]`, and NDCG
  `[0.02298, 0.02949, 0.03724, 0.04026]`. Epochs `61-67` were seven
  consecutive non-improvements followed by one natural patience `7/7` stop.
- Test chronology and exact final result: dataset preflight accessed Test
  structure/identity before training, without ranking. After early stopping,
  the run restored validation-best epoch `60` and performed exactly one new,
  report-only student Test ranking; the log contains one final student Test
  event after the one early-stop event and no new teacher Test event. Exact
  vectors ordered by K=`[10,20,40,50]` are Precision
  `[0.004787863203908524, 0.0036950372846490643, 0.0028413473900744646, 0.002626896374389465]`,
  Recall
  `[0.043436936338710826, 0.06682881901961434, 0.10255924898506319, 0.11873135835336573]`,
  NDCG
  `[0.025207338729919423, 0.031615081004297794, 0.039582682792750394, 0.04274962109657593]`,
  and Hit Ratio
  `[0.04762149652867107, 0.07328362046798695, 0.11175109282591636, 0.1289791720236532]`;
  AUC `0.0` as expected under `test_flag=part`. Test did not influence
  selection, acceptance, parameters, retry, or rollback; execution complied
  with `val_test_once_v1` and candidate exclusion `train_only`.
- Manifest acceptance: `status=completed`, model stage `td_distill`, protocol
  `val_test_once_v1`, selection split `validation`, primary metric
  `Recall@20`, candidate exclusion `train_only`, `run_final_test=true`,
  `final_test_performed=true`, `paper_ready_eligible=true`, and blockers `[]`.
  Its run/profile/arguments, best epoch/Recall, final Test vectors, teacher/data
  identities, cache hits, and checkpoint paths agree with the other evidence.
- Same-run artifacts and fingerprints, all outside Git and all keyed only by
  the exact run identity:
  - raw log
    `logs/2026-08-02 00_42_54.029141_baby_light_init_pid29752`, `33827` bytes,
    SHA256 `4fc9661bf13decf360894ba28171e2ac754be395099ee1a93f4ca1e5306f13db`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-08-02 00_42_54.029141_baby_light_init_pid29752.json`,
    `3399` bytes, SHA256
    `6ebd2223e855f0a25df5fc9ca92a1b8e479ca87c076bd401f2ed1397b7c5ae0c`;
  - manifest
    `exp/runs/baby/run_manifest__2026-08-02 00_42_54.029141_baby_light_init_pid29752.json`,
    `25447` bytes, SHA256
    `96ad86aaae2ba745bb5ec78ad9091a691d9aa26b64eb77a909af2df294823212`;
  - convergence record
    `exp/converge/baby/auto__2026-08-02 00_42_54.029141_baby_light_init_pid29752.pkl`,
    `10960` bytes, SHA256
    `88e9cd2292e9a81e9feb003aa58b5a214a0e3a577d28d111232a07c289a8ba93`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-02 00_42_54.029141_baby_light_init_pid29752.pth`,
    `20354975` bytes, SHA256
    `5a1a10c63af169cbb9d211aa82a8fd8e6ca109190e81ab44a8a43949ad072f24`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-02 00_42_54.029141_baby_light_init_pid29752.pth`,
    `6785997` bytes, SHA256
    `4e9300358221f33beb1acb51c950bf73d774717172a44869ab3aeb28fe538fde`.
  Exactly these six files contain the run identity; no generic/shared student
  checkpoint or other file was written in the run window.
- Checkpoint/deployment audit: the format-v2 full checkpoint contains only
  finite `user_id_embedding.weight` shape `[19445,64]` and
  `item_id_embedding.weight` shape `[7050,64]` in model state, two optimizer
  state entries with all six optimizer tensors finite, and the declared
  run/profile/protocol/epoch/optimizer/semantic metadata. The inference export
  contains exactly the two finite embeddings plus `embedding_dim=64`,
  `n_users=19445`, `n_items=7050`, and variant
  `td_distill_no_projection`; it contains no teacher, modality, prompt,
  semantic cache, graph, projection, or optimizer state. User and item tensors
  have matching dtype/shape and are bit-for-bit equal between checkpoints.
- Matched seed-2024 result: versus its committed same-seed baseline, candidate
  Validation Recall@20 improved from `0.04086654667009528` to
  `0.0671997942915925`, delta `+0.026333247621497212`; final Test Recall@20
  improved from `0.042307913651446934` to `0.06682881901961434`, delta
  `+0.024520905368167402` (`+57.9582%` relative). This positive magnitude is
  descriptive evidence only and did not affect technical acceptance.
- Outcome-local three-seed calculation from already recorded manifests, with
  no new split access: seed `2022` baseline/candidate/delta Recall@20 were
  Validation `0.04291303921928788 / 0.06842596333982363 /
  +0.025512924120535754` and Test
  `0.044279857310199594 / 0.06659234097521655 / +0.022312483665016952`;
  seed `2023` were Validation
  `0.04098274743170617 / 0.06479423036892924 / +0.023811482937223072`
  and Test
  `0.03680654207653432 / 0.06531066330886427 / +0.028504121232329954`;
  seed `2024` were Validation
  `0.04086654667009528 / 0.0671997942915925 / +0.026333247621497212`
  and Test
  `0.042307913651446934 / 0.06682881901961434 / +0.024520905368167402`.
  All three paired deltas are positive.
- Descriptive `n=3`, `ddof=1` calculation from those recorded values:
  Validation baseline mean/sample SD `0.04158744444036311 /
  0.0011494680477004105`, candidate `0.06680666266678179 /
  0.0018475079022266937`, and paired delta `0.02521921822641868 /
  0.0012862821015983998`; Test baseline `0.04113143767939362 /
  0.0038730713819974285`, candidate `0.06624394110123172 /
  0.0008168451779041365`, and paired delta `0.025112503421838104 /
  0.0031379268847553854`. This is not a significance claim and does not prove
  rate optimality; the separately predeclared canonical three-seed summary
  remains the next experiment-record stage after this outcome commit.
- Experimental meaning, unresolved risks, and scope: the predeclared
  asymmetric no-projection transfer improved Recall@20 over its matched BPR
  baseline for all three recorded seeds under frozen non-semantic controls.
  Evidence remains limited to three seeds; all one-time final student Test
  accesses are consumed; retained official cold items remain; and ignored
  artifacts lack Git/off-device protection. Only this append-only
  `TRAINING_LOG.md` outcome changes tracked state; no code, profile, parameter,
  data, teacher, protocol, historical entry, or top-of-log summary changed.
- Unique next action: only with new explicit user authorization, start from the
  clean commit containing this outcome and append/commit one standalone
  canonical descriptive matched summary across seeds `2022/2023/2024`, using
  only the already recorded outcome/manifests and reporting the predeclared
  per-arm and paired mean/sample-SD statistics. Do not load a split, rerun
  training/Validation/Test, alter parameters, launch another experiment,
  create a tag/bundle/backup, or merge `main` in that summary task.

### 2026-08-02 | canonical matched Baby seeds 2022/2023/2024 descriptive summary (pending)

- Purpose and rationale: consolidate the three completed, predeclared matched
  Baby baseline/candidate pairs into one canonical descriptive Recall@20
  record. The summary will report each seed separately and then compute
  per-arm and paired candidate-minus-baseline statistics, without creating new
  experimental evidence or revisiting any ranking split.
- Status and authorization boundary: pending summary-only repository record.
  This task may read only the already recorded outcome entries in this active
  log and their six corresponding completed run manifests, perform arithmetic
  verification, append a separate completed or failed summary outcome, and
  create one local commit containing only `TRAINING_LOG.md`. It must not load
  any data split, import or execute a project entry point, run training,
  Validation, Test, or efficiency evaluation, modify code/profile/parameters,
  edit the intentionally stale top-of-log summary, create a tag/bundle/backup,
  or merge `main`.
- Branch, source, and preservation point: branch
  `codex/experiment/baby-teacher-baseline` at exact clean candidate-outcome
  commit `efd974dbdd0920a44132d7fcaefb6a489671b713`. Preserve every historical
  entry and all six formal outcomes/manifests verbatim; this task is pure EOF
  append and authorizes no rollback, deletion, replacement, or reformatting.
- Frozen outcome provenance, ordered baseline/candidate by seed:
  - seed `2022`: outcome commits
    `6f806fa70de1c707dd109741b1c8fd2d3efdc29a` /
    `92b64136e3ca75fa0f41a11e29f907a9403d54b3` and manifests
    `exp/runs/baby/run_manifest__2026-07-31 12_33_07.292506_baby_light_init_pid7872.json` /
    `exp/runs/baby/run_manifest__2026-07-31 18_35_00.450484_baby_light_init_pid22864.json`;
  - seed `2023`: outcome commits
    `1749d46bf46087f84208a39bc3caefe92654aa50` /
    `82eb88e7fd3e8f5110ddb14388c7117abb77d22e` and manifests
    `exp/runs/baby/run_manifest__2026-08-01 20_20_35.101504_baby_light_init_pid29128.json` /
    `exp/runs/baby/run_manifest__2026-08-01 21_09_59.190542_baby_light_init_pid33928.json`;
  - seed `2024`: outcome commits
    `1a8104b75aab9bf8ba95d6de5beb5ca18d24c0b1` /
    `efd974dbdd0920a44132d7fcaefb6a489671b713` and manifests
    `exp/runs/baby/run_manifest__2026-08-01 23_31_40.280800_baby_light_init_pid15908.json` /
    `exp/runs/baby/run_manifest__2026-08-02 00_42_54.029141_baby_light_init_pid29752.json`.
- Frozen comparison fields: for every manifest use only
  `best_selection_recall` as Validation Recall@20 and
  `final_test_result.recall[1]` as Test Recall@20. The candidate-minus-baseline
  paired delta is computed independently within each seed. Summaries use
  arithmetic mean and sample standard deviation
  `sqrt(sum((x - mean)^2) / (n - 1))` with `n=3`, `ddof=1`; population SD is
  not permitted.
- Required summary content and interpretation gate: record all six per-seed
  baseline/candidate values and all six paired deltas; record Validation and
  Test baseline, candidate, and paired-delta mean/sample-SD values; verify all
  three paired deltas are positive on both splits; and reproduce the exact
  user-specified Test statistics. The outcome must state that three seeds are
  descriptive evidence only, not statistical significance, parameter
  optimality, or cross-dataset generalization.
- Research-priority gate: do not recommend seed `2025`/`2026` expansion now.
  After this summary, prioritize complementary matched ablation and efficiency
  evidence. The next task may only predeclare the first highest-information
  matched ablation stage; implementation and execution require later separate
  authorization.
- Risks and failure boundary: risks are transcription error, mixing protocol or
  profile identities, selecting the wrong K index, using population rather
  than sample SD, missing/mutated ignored manifests, or disagreement between a
  manifest and its immutable outcome. Mark this summary `failed` only if those
  hard consistency checks fail; preserve the pending entry and report the
  mismatch. Metric magnitude and seed variance are not failure criteria.
- Planned verification and acceptance criteria: confirm all six manifest files
  exist and match the recorded SHA256 fingerprints; require completed status,
  `val_test_once_v1`, Validation/Recall@20 selection, `train_only`, final Test
  performed, paper-ready eligibility, empty blockers, canonical profile/seed,
  and empty dataset/student override maps; compare every extracted value with
  the corresponding outcome; independently recompute deltas, means, and
  `ddof=1` sample SD; verify the required exact Test statistics and positive
  directions; then review pure append scope, run `git diff --check`, stage only
  `TRAINING_LOG.md`, run `git diff --cached --check`, commit, and confirm a
  clean tree. No validation/Test metric is newly generated and no experiment
  artifact is created or modified.

### 2026-08-02 | canonical matched Baby seeds 2022/2023/2024 descriptive summary (completed)

- Status and outcome: completed successfully as the standalone canonical
  matched three-seed descriptive summary. All six formal outcome/manifests
  agreed on identity and Recall@20 values, every requested statistic was
  reproduced exactly with `n=3`, `ddof=1`, and all per-seed paired directions
  were positive. This is a repository-record result only; no experiment was
  implemented, launched, repeated, or extended.
- Source and scope: branch `codex/experiment/baby-teacher-baseline`; exact clean
  summary parent `efd974dbdd0920a44132d7fcaefb6a489671b713`. Only EOF-appended
  `TRAINING_LOG.md` pending/completed records are in scope. No historical
  record, code, profile, parameter, generated artifact, or intentionally stale
  top-of-log summary was edited.
- Outcome provenance verification: the completed seed-2022 baseline/candidate
  commits `6f806fa70de1c707dd109741b1c8fd2d3efdc29a` /
  `92b64136e3ca75fa0f41a11e29f907a9403d54b3`, seed-2023 commits
  `1749d46bf46087f84208a39bc3caefe92654aa50` /
  `82eb88e7fd3e8f5110ddb14388c7117abb77d22e`, and seed-2024 commits
  `1a8104b75aab9bf8ba95d6de5beb5ca18d24c0b1` /
  `efd974dbdd0920a44132d7fcaefb6a489671b713` are all ancestors of this
  summary parent. Each commit's immutable log contains its canonical completed
  heading and the exact Validation/Test Recall@20 values used below.
- Manifest fingerprint verification, ordered baseline/candidate by seed:
  - seed `2022`: `cd2c5aa1fae79af8f8161dfee4fbb2bf9f451b591607df6c646d1dbf8b67af7d` /
    `967779e828f820ece44bd2a75331ee48a1a4e45ca2cefbb7b43bbe5eb3793480`;
  - seed `2023`: `edcc1b4d858eecbad5724ac0c46608abb572bb9412f78a1555156cff278a7d33` /
    `357afe298cc1a771ddd30aaddd046a73298894e899e636c6eae0bc546608a42a`;
  - seed `2024`: `d0f3e2bb49dd641a1b88d5a16e0fb73c0fa5e45c1b5e65d2d3418d1d1ae04b7c` /
    `96ad86aaae2ba745bb5ec78ad9091a691d9aa26b64eb77a909af2df294823212`.
  All hashes matched the files declared in the pending entry.
- Manifest contract audit: every manifest has `status=completed`, protocol
  `val_test_once_v1`, selection split `validation`, primary metric
  `Recall@20`, candidate exclusion `train_only`, final Test performed,
  `paper_ready_eligible=true`, blockers `[]`, its expected canonical profile
  and seed, and empty dataset/student override maps. For each run,
  `best_selection_recall` and `final_test_result.recall[1]` exactly equal the
  corresponding outcome values.
- Canonical per-seed Recall@20 results and within-seed candidate-minus-baseline
  deltas:

  | Seed | Validation baseline | Validation candidate | Validation paired delta | Test baseline | Test candidate | Test paired delta |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `2022` | `0.04291303921928788` | `0.06842596333982363` | `+0.025512924120535754` | `0.044279857310199594` | `0.06659234097521655` | `+0.022312483665016952` |
  | `2023` | `0.04098274743170617` | `0.06479423036892924` | `+0.023811482937223072` | `0.03680654207653432` | `0.06531066330886427` | `+0.028504121232329954` |
  | `2024` | `0.04086654667009528` | `0.0671997942915925` | `+0.026333247621497212` | `0.042307913651446934` | `0.06682881901961434` | `+0.024520905368167402` |

  Candidate-minus-baseline Recall@20 is positive for every seed on both
  Validation and Test; there is no reversed pair in these six completed runs.
- Canonical descriptive statistics, computed across the three seed-level
  values with arithmetic mean and sample SD (`n=3`, `ddof=1`):

  | Split | Series | Mean | Sample SD |
  | --- | --- | ---: | ---: |
  | Validation | baseline | `0.04158744444036311` | `0.0011494680477004105` |
  | Validation | candidate | `0.06680666266678179` | `0.0018475079022266937` |
  | Validation | paired delta | `0.02521921822641868` | `0.0012862821015983998` |
  | Test | baseline | `0.04113143767939362` | `0.0038730713819974285` |
  | Test | candidate | `0.06624394110123172` | `0.0008168451779041365` |
  | Test | paired delta | `0.025112503421838104` | `0.0031379268847553854` |

- Exact required Test-statistic acceptance: candidate mean/sample SD are
  `0.06624394110123172 / 0.0008168451779041365`; baseline are
  `0.04113143767939362 / 0.0038730713819974285`; paired delta are
  `0.025112503421838104 / 0.0031379268847553854`. All exactly match the
  requested canonical values.
- Recalculation evidence: a read-only PowerShell pass independently checked all
  six paths, hashes, protocol/profile/override fields, values, positive deltas,
  and sample-SD formula. An initial aggregate-boolean wrapper emitted a
  spurious seed-2022 field-mismatch message; immediate labeled inspection found
  every field true, and the corrected explicit assertions passed all six
  manifests. A second read-only script using only Python standard-library
  `json` and `statistics` (no repository import) then asserted every exact
  per-seed value plus all six requested mean/sample-SD pairs. Neither check
  imported or executed `codes/main_mmlight.py` or other project code.
- Test access, commands, and artifacts: no data file or train/Validation/Test
  split was loaded or accessed, and no ranking, training, Validation, Test, or
  efficiency command ran. Reading already stored JSON result scalars does not
  constitute a new Test evaluation. No checkpoint, manifest, convergence
  record, raw log, metric, dataset derivative, or other experiment artifact was
  created or modified; the summary log and its local Git commit are the only
  outputs.
- Evidence meaning: under the frozen Baby dataset, teacher, protocol,
  optimizer, architecture, initialization policy, and per-seed matched design,
  the asymmetric no-projection candidate has a consistently positive
  descriptive Recall@20 delta across these three seeds. The evidence is not a
  statistical-significance result, does not prove the semantic parameters are
  optimal, and does not establish cross-dataset generalization. Three seeds
  remain a small descriptive sample, and all recorded final Test accesses are
  consumable evidence rather than a tuning signal.
- Research priority and unresolved risks: do not expand to seed `2025` or
  `2026` now; additional same-configuration seeds have lower information value
  than isolating method components and measuring deployment cost. Prioritize a
  matched ablation next, then efficiency evidence. The retained official
  cold-item condition and lack of Git/off-device protection for ignored run
  artifacts remain; neither risk changes the descriptive calculations.
- Acceptance and commit gate: all numerical, manifest, outcome, direction, and
  interpretation criteria passed. Before committing, the full working diff
  must remain a pure EOF append to only `TRAINING_LOG.md`, `git diff --check`
  must pass, the index must contain only that file, and
  `git diff --cached --check` must pass. A failure in those final repository
  checks requires preserving this outcome and reporting the exact blocker.
- Unique next action: only with new explicit user authorization, start from the
  clean summary commit and append/commit a declaration-only first
  highest-information matched ablation stage: a three-seed item-image-only
  ablation that keeps every baseline/candidate control fixed and changes only
  the full candidate's `td_item_text_rate` from `0.3` to `0.0`. This nested arm
  uses the existing baseline and full-candidate evidence to isolate whether the
  lower-weight text component adds value beyond dominant item-image transfer.
  The next task must stop after its pending plus declaration-completed log
  records and one `TRAINING_LOG.md` commit; it must not implement a profile,
  load data, run training/Validation/Test/efficiency, create artifacts, tag,
  bundle, backup, or merge `main`.

### 2026-08-02 | matched three-seed item-image-only ablation stage (pending; declaration only)

- Purpose and hypothesis: predeclare a matched three-seed modality ablation of
  the completed asymmetric no-projection candidate before any image-only
  profile is implemented or any image-only result exists. At fixed total
  distillation coefficient `td_distill_alpha=0.3`, compare item-image-only
  supervision with the completed item-image plus lower-rate item-text mixture
  to determine whether their validation-selected outcomes differ consistently
  across the already frozen student seeds `2022`, `2023`, and `2024`. This is
  an ablation comparison, not a declaration that either direction will win.
- Status and authorization boundary: pending declaration only. This task may
  append this pending record and the separate declaration-completed outcome,
  perform content/diff/index checks, and create one local commit containing
  only `TRAINING_LOG.md`. It must not implement or resolve a profile, import or
  execute project code, load any data split, train, rank Validation or Test,
  run an efficiency benchmark, or create a checkpoint, preflight, manifest,
  convergence record, raw run log, tag, bundle, backup, or merge to `main`.
- Branch, exact source, and preservation point: branch
  `codex/experiment/baby-teacher-baseline` at exact clean parent and canonical
  three-seed summary commit
  `6b1c87a70dd1578d7512eada599111b980829327`. Preserve that commit, all
  existing profile definitions, all completed baseline/full-candidate
  outcomes, and all ignored artifacts verbatim. This task is a pure EOF append
  and authorizes no rollback, deletion, replacement, or historical edit.
- Three future override-free profiles are frozen simultaneously by this
  declaration, before any member can produce evidence. Their exact canonical
  profile / scope / source identities are:
  - seed `2022`: `baby_td_item_image_only_no_projection_seed2022_v1` /
    `student_ablation` /
    `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022`;
  - seed `2023`: `baby_td_item_image_only_no_projection_seed2023_v1` /
    `student_ablation` /
    `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2023`;
  - seed `2024`: `baby_td_item_image_only_no_projection_seed2024_v1` /
    `student_ablation` /
    `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2024`.
  The IDs are independent registry identities, not aliases or CLI override
  recipes, and none is implemented by this declaration.
- Exact per-seed source-profile identity and sole defaults delta:
  - seed `2022` must copy the complete defaults mapping of
    `baby_td_asymmetric_no_projection_v1`, whose canonical sorted compact-JSON
    defaults SHA256 is
    `018534ec9cedfe4c1d4f5fc1d23552173d3f8a33843c3e71be38f69ea8c46113`;
  - seed `2023` must copy the complete defaults mapping of
    `baby_td_asymmetric_no_projection_seed2023_v1`, defaults SHA256
    `1c4c89f253d3adcd167211481dd599591012b8881c9c590c68a6e55d797fe61c`;
  - seed `2024` must copy the complete defaults mapping of
    `baby_td_asymmetric_no_projection_seed2024_v1`, defaults SHA256
    `f7ed2a940a5547afdfd3d242c6dc1832dd7e447b2824aaf633b5dd94c3bead6e`.
  For each copy, the sole allowed defaults-value change is
  `td_item_text_rate: 0.3 -> 0.0`. Profile name/scope/source metadata changes
  only to the independent identity declared above; the key set and every other
  default value must remain identical to the corresponding full candidate.
- Frozen seeds and semantic controls: each profile keeps its own student
  initialization, training, and maintained pairwise-sampling seed exactly
  `2022`, `2023`, or `2024`, respectively. All three keep dataset/teacher PCA
  cache identity `hard_token_seed=2022`, `td_distill_alpha=0.3`,
  `td_item_image_rate=1.0`, `td_item_text_rate=0.0`,
  `td_user_image_rate=0.0`, and `td_user_text_rate=0.0`. No result from one
  seed may change either seed value, hard-token seed, alpha, or component rate
  for another profile.
- Frozen architecture, initialization, optimizer, sampling, and budget: all
  three profiles retain `td_distill_no_projection`, 64-dimensional user/item
  ID embeddings, random initialization, no loaded student checkpoint,
  `td_init_from_teacher=false`, and no teacher warm start. Retain AdamW
  `student_lr=6e-5`, student weight decay `0.01`, batch size `1024`, maximum
  `epoch=1000`, validation every epoch, early-stopping patience `7`,
  `smoke_train_batches=0`, the unchanged `data_generator.sample()` pairwise
  BPR sampler, all `116` batches per completed formal epoch, and
  `run_efficiency_benchmark=false`.
- Frozen dataset and preprocessing identity: audited MMRec Baby under
  `data/baby/`; conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  `train_mat`, `val_mat`, and `test_mat` SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve the official splits, `retain_official` cold-item policy,
  duplicate-modality policy `error`, preflight, `pca` hard-token type,
  existing image/text PCA cache identities, and all preprocessing provenance;
  no fallback or cache regeneration is allowed.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, exactly `141098540` bytes and
  SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  retain `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher training, optimizer step,
  warm start, run-local teacher checkpoint, alias publication/overwrite,
  checkpoint mutation, or new teacher Test ranking is allowed.
- Frozen evaluation and deployment contract: retain `val_test_once_v1`; train
  only on `train_mat`; select and early-stop only by Validation Recall@20;
  `Ks=[10,20,40,50]`; `test_flag=part`; candidate exclusion `train_only`;
  restore the Validation-best checkpoint and only then perform exactly one
  final student Test ranking in each separately declared formal run. Each
  inference-only export must contain only finite user/item ID embeddings plus
  dimension/count/no-projection deployment metadata, with embeddings
  bit-for-bit equal to the corresponding full checkpoint and no teacher,
  modality, prompt, semantic cache, graph, projection, or optimizer state.
- Exact objective and interpretation boundary: because the maintained semantic
  loss normalizes by the sum of active rates, each image-only profile optimizes
  `L_BPR + 0.3 * L_item_image`, whereas its corresponding completed full
  candidate optimizes
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`.
  Therefore this ablation holds the total coefficient alpha at `0.3` but does
  not hold the effective coefficient on `L_item_image` fixed: it changes from
  `0.3 / 1.3` in the full mixture to `0.3` in image-only. Any outcome may be
  described only as the matched effect of switching between these two
  normalized supervision mixtures. It must not be claimed as a pure marginal
  causal contribution of text under fixed image weight, nor as preserving the
  image term's effective coefficient.
- Frozen comparison and later summary rule: compare each future image-only run
  only with the already completed same-seed full candidate listed above. After
  all three image-only outcomes are committed, report each arm by seed and
  compute arithmetic mean and sample standard deviation with `n=3`, `ddof=1`.
  Define the paired delta prospectively as `full candidate - image-only`
  within the same seed, and report its mean and `ddof=1` sample SD for both
  Validation Recall@20 and the one-time final Test Recall@20 using stored
  outcomes/manifests only. This direction is descriptive and does not relax
  the interpretation boundary above or establish statistical significance.
- Mandatory future order and firewall:
  1. implement and verify all three independent image-only profiles together
     in one coherent change, prove their canonical formal resolutions are
     override-free, commit the implementation outcome, and stop without any
     profile execution;
  2. in later separate stages, declare, run exactly once, audit, and commit the
     image-only outcome for seed `2022`, then repeat for seed `2023`, then seed
     `2024`; each seed requires its own formal pending declaration and its own
     completed or failed outcome before advancing;
  3. only after all three outcomes are committed, create the declared
     full-candidate versus image-only per-seed and mean/sample-SD summary;
  4. no Validation or Test direction, magnitude, low metric, or reversal from
     an earlier seed may modify, cancel, retry, or reorder either later
     profile/run. Low or direction-reversed protocol-valid results remain
     preserved completed evidence rather than failure.
- Future implementation acceptance criteria: add all three profiles and exact
  paper-ready identities simultaneously; each defaults map has the same key
  set as its corresponding full candidate and differs only at
  `td_item_text_rate`; prior profiles remain value-identical; canonical formal
  parsing yields the declared profile/scope/source, its own student seed,
  `hard_token_seed=2022`, `dataset_config_overrides={}`,
  `student_config_overrides={}`, and no paper-ready blocker. Focused tests must
  lock all fixed controls, exact objective rates, unknown/override rejection,
  frozen teacher reuse, no projection/warm start, and inference-only contract.
- Risks, hard failure boundary, and rollback point: risks include copying the
  wrong seed profile, profile/default drift beyond item-text rate, confusing
  student seed with `hard_token_seed`, hidden CLI overrides, using population
  SD, consuming Test before Validation-best restoration, or overstating the
  normalized-loss comparison. Metric magnitude or reversal is not failure.
  Preserve clean parent `6b1c87a70dd1578d7512eada599111b980829327`, this
  declaration, all future partial/completed artifacts, and all low results; no
  destructive rollback is authorized.
- Declaration acceptance and verification plan: require all three identities,
  source profiles/fingerprints, the sole defaults delta, every frozen control,
  exact loss formulas, interpretation boundary, paired-delta direction, and
  mandatory sequence to be present consistently. Then verify pure EOF append,
  run `git diff --check`, stage only `TRAINING_LOG.md`, run
  `git diff --cached --check`, create one local commit, and confirm a clean
  tree. No metric or experiment artifact is expected from this task.
- Unique next action after this declaration commit: only with new explicit
  user authorization, implement and verify all three image-only profiles
  simultaneously, append their implementation pending/completed trace, and
  create one coherent source/test/documentation commit. Stop after the clean
  implementation commit without profile execution, training, Validation,
  Test, efficiency evaluation, tag, bundle, backup, or merge.

### 2026-08-02 | matched three-seed item-image-only ablation declaration (completed; no profiles or runs)

- Status and outcome: completed successfully as a declaration-recording task
  only. The matched item-image-only ablation stage above remains pending for
  future implementation and sequential execution. All three future profiles
  were frozen simultaneously before any image-only evidence exists.
- Source and scope: branch `codex/experiment/baby-teacher-baseline`; exact
  clean declaration parent
  `6b1c87a70dd1578d7512eada599111b980829327`. The only tracked change is a
  pure EOF append to `TRAINING_LOG.md` containing the pending declaration and
  this independent declaration-completed outcome. No historical entry,
  intentionally stale top-of-log summary, source, test, profile, parameter,
  protocol implementation, data, teacher, cache, or generated artifact was
  changed.
- Frozen identity outcome: the canonical future IDs are
  `baby_td_item_image_only_no_projection_seed2022_v1`,
  `baby_td_item_image_only_no_projection_seed2023_v1`, and
  `baby_td_item_image_only_no_projection_seed2024_v1`, each with exact scope
  `student_ablation` and the seed-matched source identity declared above. Each
  must copy its complete corresponding full-candidate defaults mapping and
  change only `td_item_text_rate` from `0.3` to `0.0`; no profile is currently
  implemented, aliased, resolved, or executable by this record.
- Contract outcome: student/sampling seeds `2022/2023/2024`,
  `hard_token_seed=2022`, alpha `0.3`, item-image rate `1.0`, both user rates
  `0.0`, random 64-dimensional no-projection initialization without teacher
  warm start, frozen data/splits/PCA caches/teacher, AdamW `6e-5` and `0.01`,
  batch `1024`, epoch `1000`, patience `7`, `116` batches, protocol
  `val_test_once_v1`, Validation Recall@20 selection, one final Test after
  best-checkpoint restoration, and the ID-only inference contract are all
  frozen. Future formal commands must use only the independent profile ID and
  resolve both override maps to `{}`.
- Interpretation outcome: the record explicitly fixes image-only semantic
  loss as `0.3 * L_item_image` and full-candidate semantic loss as
  `0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`. It therefore forbids both
  an effective-image-coefficient invariance claim and a pure fixed-image-weight
  marginal causal interpretation of text. The later paired delta is frozen as
  `full candidate - image-only` within seed and remains descriptive.
- Sequence and preservation outcome: future work must first implement and
  verify all three profiles together, then advance through independent formal
  seed stages in order `2022`, `2023`, `2024`, and only afterward summarize
  per-arm and paired mean/sample SD with `ddof=1`. An earlier result cannot
  change, cancel, retry, or reorder a later profile, and low or reversed valid
  metrics must be retained as completed evidence.
- Test access, metrics, commands, and artifacts: no project command was run;
  no data or train/Validation/Test split was loaded or accessed; no training,
  ranking, efficiency test, or profile resolution occurred. Validation/Test
  metrics and run artifact paths are not applicable. No checkpoint, preflight,
  manifest, convergence record, raw log, tag, bundle, backup, or merge was
  created or modified; this log record and its local Git commit are the only
  outputs.
- Acceptance and unresolved risks: declaration content is complete and
  internally consistent with the canonical three-seed summary and source
  profile fingerprints. Runtime behavior and image-only metrics remain
  unknown; future implementation drift, one-time Test consumption, retained
  official cold items, limited three-seed uncertainty, and lack of Git/off-
  device protection for ignored future artifacts remain unresolved. Final
  one-file append, whitespace, staged-scope, cached-diff, commit, and clean-tree
  gates must pass before handoff.
- Unique next action: only with new explicit user authorization, implement and
  verify the three frozen image-only profiles simultaneously in one coherent
  source/test/documentation change, append and commit its pending/completed
  trace, and stop. Do not declare or launch any formal run in that task.

### 2026-08-02 | matched three-seed item-image-only profile implementation (pending)

- Purpose and rationale: implement simultaneously the three override-free
  item-image-only ablation identities frozen by declaration commit
  `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`, before any image-only profile
  execution or result. Atomic implementation preserves the parameter-selection
  firewall across seeds and makes `td_item_text_rate=0.0` an independently
  pinned profile value rather than a CLI override.
- Status and authorization boundary: pending implementation and static/unit
  verification only on branch `codex/experiment/baby-teacher-baseline` from
  exact clean parent `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`.
  `codes/main_mmlight.py`, smoke, training, Validation ranking, Test ranking,
  efficiency evaluation, formal-run declaration, dataset loading, and run-
  artifact generation are prohibited. No tag, bundle, backup, baseline/teacher
  overwrite, merge to `main`, or stale top-of-log summary update is authorized.
- Canonical profiles to implement together, with exact scope/source identity:
  - `baby_td_item_image_only_no_projection_seed2022_v1` /
    `student_ablation` /
    `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022`;
  - `baby_td_item_image_only_no_projection_seed2023_v1` /
    `student_ablation` /
    `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2023`;
  - `baby_td_item_image_only_no_projection_seed2024_v1` /
    `student_ablation` /
    `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2024`.
- Declared file scope: extend only the profile constants/default copies,
  registry entries, and exact paper-ready identities in
  `codes/utility/dataset_profiles.py`; add focused static/synthetic assertions
  to `codes/tests/test_dataset_profiles.py`; create one unified
  `docs/BABY_ITEM_IMAGE_ONLY_ABLATION_PROFILES_V1.md`; and preserve this
  pending plus a separate completed/failed outcome in append-only
  `TRAINING_LOG.md`. Existing registry-derived parser choices and runtime
  eligibility should consume the new profiles without edits. If inspection
  proves another tracked file is materially required, append a scope amendment
  before changing it; do not perform unrelated refactoring.
- Exact defaults-copy contract: seed `2022`, `2023`, and `2024` image-only
  maps must independently copy the complete mappings of
  `baby_td_asymmetric_no_projection_v1`,
  `baby_td_asymmetric_no_projection_seed2023_v1`, and
  `baby_td_asymmetric_no_projection_seed2024_v1`, respectively. Each new map
  must have exactly the same keys as its source and the sole changed value
  must be `td_item_text_rate: 0.3 -> 0.0`. Source mappings must retain their
  declaration-time sorted compact-JSON SHA256 values
  `018534ec9cedfe4c1d4f5fc1d23552173d3f8a33843c3e71be38f69ea8c46113`,
  `1c4c89f253d3adcd167211481dd599591012b8881c9c590c68a6e55d797fe61c`,
  and `f7ed2a940a5547afdfd3d242c6dc1832dd7e447b2824aaf633b5dd94c3bead6e`.
- Frozen semantic, seed, and preprocessing contract: student initialization,
  training, and sampling seeds remain `2022`, `2023`, and `2024` in their
  corresponding profiles; dataset-owned `hard_token_seed=2022` remains fixed
  for every canonical resolution. Retain `td_distill_alpha=0.3`,
  `td_item_image_rate=1.0`, `td_item_text_rate=0.0`, and both user-side rates
  `0.0`. No existing baseline or full-candidate mapping may change.
- Frozen shared controls: retain random initialization, no loaded student
  checkpoint, `td_init_from_teacher=false`, `td_distill_no_projection`,
  dimension `64`, no teacher warm start, AdamW `student_lr=6e-5`, student
  weight decay `0.01`, batch size `1024`, maximum epoch `1000`, patience `7`,
  `smoke_train_batches=0`, unchanged pairwise sampling and all `116` formal
  batches, frozen audited data/splits/PCA caches/teacher,
  `val_test_once_v1`, Validation Recall@20 selection, `train_only` exclusion,
  and the maintained inference-only user/item ID embedding export contract.
- Identity and override gate: add all three exact name/scope/source tuples to
  the central profile registry and paper-ready identity set together. The
  registry-derived parser choices must accept the three canonical IDs while
  continuing to reject unknown names. Each canonical invocation must resolve
  its own student seed, `hard_token_seed=2022`,
  `dataset_config_overrides={}`, `student_config_overrides={}`, and combined
  paper-ready blockers `[]`. Any CLI seed, semantic, optimizer, dataset-owned
  preprocessing, or other pinned-value override must remain recorded and
  blocked; mismatched scope/source metadata must remain blocked.
- Interpretation boundary to document and test: under active-rate
  normalization, image-only optimizes `L_BPR + 0.3 * L_item_image`, whereas
  the full candidate optimizes
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`.
  The implementation and documentation must not claim the image term's
  effective coefficient remains fixed and must not interpret a later
  difference as the pure marginal causal contribution of text under fixed
  image weight.
- Regression and defaults-fingerprint acceptance: focused tests must prove all
  pre-existing seed-2022/2023/2024 baseline and full-candidate mappings retain
  their exact keys/values and declaration-time fingerprints; each new map has
  the exact one-field delta and receives a reproducible sorted compact-JSON
  fingerprint. Tests must also lock all canonical identities, frozen controls,
  canonical resolutions, negative override/unknown-profile gates, and the
  maintained inference-only deployment structure without loading Baby data.
- Planned verification: inspect the current registry/test patterns; run the
  focused dataset-profile test module; run the full existing unit suite using
  only static parsing and temporary synthetic fixtures; run `py_compile` for
  relevant Python files; statically parse all three canonical invocations
  without importing or executing the training runner; exercise negative seed,
  item-text, hard-token, mismatched-identity, and unknown-profile cases; audit
  exact defaults fingerprints/deltas and documentation interpretation; confirm
  excluded runner/sampler/model/protocol files are unchanged; run
  `git diff --check`; stage only the necessary profile code, tests, unified
  document, and `TRAINING_LOG.md`; run `git diff --cached --check`; commit once;
  and confirm a clean tree.
- Risks, failure boundary, and preservation point: risks are wrong source-map
  copying, shared-mapping mutation, seed/cache confusion, registry identity
  drift, weakened override rejection, incomplete regression fingerprints, or
  an overstated loss interpretation. Mark this implementation failed only if
  code or a hard verification criterion cannot be satisfied; preserve this
  pending record and all evidence. The rollback/preservation point is clean
  declaration commit `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`; no destructive
  rollback is authorized.
- Completion and next-stage gate: after successful verification, append an
  independent completed outcome and create one coherent implementation commit
  containing only the declared necessary files. The unique later action is a
  separately authorized seed-2022 image-only formal-run pending declaration;
  this task must stop before declaring or launching that run.

### 2026-08-02 | matched three-seed item-image-only profile implementation (completed; no runs)

- Status and objective outcome: completed successfully. All three image-only
  profiles frozen by declaration commit
  `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516` were implemented and verified
  simultaneously before any one of them was executed. This outcome closes
  only the implementation stage; no smoke or formal-run stage was declared or
  started.
- Branch, parent, and exact tracked scope: branch
  `codex/experiment/baby-teacher-baseline` from exact clean parent
  `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`. The coherent implementation
  commit contains only `codes/utility/dataset_profiles.py`,
  `codes/tests/test_dataset_profiles.py`,
  `docs/BABY_ITEM_IMAGE_ONLY_ABLATION_PROFILES_V1.md`, and this append-only
  `TRAINING_LOG.md` trace. No stale top-of-log summary, runner, parser,
  sampler, model, protocol, data, or artifact file was changed.
- Implemented canonical identities: the central registry now contains
  `baby_td_item_image_only_no_projection_seed2022_v1` /
  `student_ablation` /
  `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022`,
  plus the corresponding seed-`2023` and seed-`2024` names and source suffixes.
  Registry-derived parser choices and the exact paper-ready identity set
  consume all three entries; no unknown-profile or override gate was relaxed.
- Exact defaults outcome: each new 26-key mapping is an independent `dict`
  copy of its same-seed full asymmetric candidate. The computed changed field
  set for every pair is exactly `{'td_item_text_rate'}`, with value delta
  `0.3 -> 0.0`. The source full-candidate fingerprints remain unchanged at
  `018534ec9cedfe4c1d4f5fc1d23552173d3f8a33843c3e71be38f69ea8c46113`
  for seed `2022`,
  `1c4c89f253d3adcd167211481dd599591012b8881c9c590c68a6e55d797fe61c`
  for seed `2023`, and
  `f7ed2a940a5547afdfd3d242c6dc1832dd7e447b2824aaf633b5dd94c3bead6e`
  for seed `2024`.
- New defaults fingerprints: canonical sorted compact-JSON SHA256 is
  `0669b57e57a400b2eb1eda1ff47970eb6929894b2affab12003e465e8f3418f6`
  for image-only seed `2022`,
  `2ab921262cc34c4b35c817a200267ab1047200c184f7f8ee766d93f80b5ecf64`
  for image-only seed `2023`, and
  `67b274963044d9ca7e05bad0a87f7c4fe594a69008493844beea9d9b284ca861`
  for image-only seed `2024`. Existing baseline fingerprints also remain
  unchanged at `98f16504442942d0209ead388611b8adf50b64c1dfd9b2b6764bcfa98fbdc3cb`,
  `29d21d20c5437be41e9af076135a1e977b3c7dcb27a706496656a203f7452650`,
  and `0865aed6197bb2781d8625cf3cf828d981228a1b969c921b83ce9a3f6f43f27f`
  for seeds `2022`, `2023`, and `2024` respectively.
- Frozen-controls outcome: canonical defaults preserve matching student and
  sampling seeds `2022/2023/2024`, dataset-owned `hard_token_seed=2022`,
  alpha `0.3`, item-image rate `1.0`, both user rates `0.0`, random
  64-dimensional no-projection initialization without teacher warm start,
  AdamW `student_lr=0.00006` and student weight decay `0.01`, batch `1024`,
  epoch `1000`, patience `7`, `smoke_train_batches=0`, unchanged sampler and
  `116`-batch formal epochs, frozen data/split/PCA/cache/teacher identities,
  `val_test_once_v1`, Validation Recall@20 selection, one later final Test,
  and the inference-only ID-embedding deployment contract.
- Canonical parser evidence: three independent static imports of the real
  `utility.parser`, without importing `main_mmlight.py`, resolved the three
  documented canonical invocations to their exact name/scope/source identity,
  student seed `2022`, `2023`, or `2024`, `hard_token_seed=2022`,
  `dataset_config_overrides={}`, `student_config_overrides={}`, and combined
  profile identity blockers `[]`.
- Negative-gate evidence: real parser checks recorded and blocked a seed CLI
  override, an item-text CLI override, and a dataset-owned hard-token seed
  override; the hard-token override remained absent from the student override
  map. An unknown profile was rejected by argparse choices with exit code `2`.
  Focused tests also rejected mismatched image-only scope/source identities and
  covered these cases across the three profiles.
- Verification commands and results: `python -m unittest
  codes.tests.test_dataset_profiles -v` passed `34/34`; `python -m unittest
  discover -s codes/tests -p "test_*.py" -v` passed `54/54` using only static
  and temporary synthetic fixtures; `python -m py_compile` passed for
  `codes/utility/dataset_profiles.py`, `codes/utility/parser.py`,
  `codes/tests/test_dataset_profiles.py`, and
  `codes/td_distill_model_no_projection.py`. Independent nine-map SHA256 and
  three-pair delta audits passed; documentation formula/order/content checks
  passed; excluded tracked runner/parser/model/protocol changes were empty;
  and `git diff --check` passed. Staged-scope and cached-diff checks remain the
  final pre-commit gates below.
- Interpretation outcome: the focused regression and unified document lock
  image-only as `L_BPR + 0.3 * L_item_image` and the full candidate as
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`. Thus alpha is
  fixed but the effective image coefficient is not: it is `0.3` versus
  `0.3 / 1.3`. Future results may describe only the matched switch between
  normalized supervision mixtures, not a pure marginal causal text effect at
  fixed image weight.
- Commands not executed, data/Test access, metrics, and artifacts: did not
  execute or import `codes/main_mmlight.py`; did not run smoke, training,
  Validation, Test, efficiency, or any formal experiment. No Baby data or
  train/Validation/Test split was loaded or accessed. No quality or efficiency
  metric exists for this task. No checkpoint, preflight, manifest, convergence
  record, raw run log, tag, bundle, backup, or merge was created.
- Remaining risks and sequence gate: runtime behavior and all image-only
  metrics remain unknown by design. Retained official cold items, only three
  future seeds, one-time Test discipline, ignored-artifact protection, and the
  normalized-mixture interpretation boundary remain future operational risks.
  Runs must remain separate and ordered `2022`, `2023`, `2024`; a low or
  reversed earlier outcome cannot modify, cancel, or reorder a later profile.
- Unique next action: in a separate task, append and commit only the seed-2022
  image-only formal-run pending declaration for exact profile
  `baby_td_item_image_only_no_projection_seed2022_v1`, then stop without
  starting the run. That declaration must cite the clean implementation
  commit produced by this task and re-pin its command, environment, identities,
  override-free parser result, acceptance rule, and planned isolated artifacts.

### 2026-08-02 | baby_td_item_image_only_no_projection_seed2022_v1 formal uncapped ablation (pending)

- Purpose and comparison target: declare the first run of the frozen matched
  three-seed item-image-only ablation as one uncapped formal seed-2022 run. It
  compares item-image-only supervision with the same-seed item-image plus
  item-text full candidate while keeping total distillation coefficient
  `td_distill_alpha=0.3`. This declaration makes no metric claim and does not
  tune any value from existing full-candidate evidence.
- Status and authorization boundary: pending; exactly one formal execution may
  occur only after this declaration and its declaration-record outcome are in
  a clean local commit and the user separately authorizes launch. The current
  task authorizes only append-only changes to `TRAINING_LOG.md`, static parser
  resolution, frozen-teacher fingerprinting, Markdown/Git checks, and one
  documentation-only commit. It does not authorize execution of
  `codes/main_mmlight.py`, Baby data loading, training, Validation or Test
  ranking, efficiency evaluation, run-artifact generation, automatic retry,
  seed-2023/2024 declaration or execution, tag, bundle, backup, or merge.
- Immutable provenance anchors: the three-seed ablation was predeclared by
  commit `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`; all three independent
  profiles were implemented and statically verified together by clean commit
  `9394ac0d75649e0e58a4dd4f0dbecdcabad2e306`; and the same-seed full
  asymmetric candidate's completed formal outcome is committed at
  `92b64136e3ca75fa0f41a11e29f907a9403d54b3`. The present branch is
  `codex/experiment/baby-teacher-baseline`, and the implementation commit is
  the exact clean parent and preservation point for this declaration.
- Clean-source launch gate: the commit containing this pending declaration and
  its separate declaration outcome must be the future launch HEAD. Its tracked
  delta from implementation commit `9394ac0d...` must be only this append to
  `TRAINING_LOG.md`; `git diff 9394ac0d... <launch-commit> -- codes docs` must
  be empty. No source, profile, test, or experiment documentation change is
  allowed between the frozen implementation and launch.
- Exact override-free formal command, frozen verbatim and containing no CLI
  seed or profile-value override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_item_image_only_no_projection_seed2022_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Static canonical-resolution evidence: importing only the real
  `utility.parser` with the exact argument list above, without importing the
  training entry point, resolved name/scope/source as
  `baby_td_item_image_only_no_projection_seed2022_v1` / `student_ablation` /
  `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022`;
  student initialization/training/sampling `seed=2022`;
  dataset/teacher preprocessing `hard_token_seed=2022`;
  `dataset_config_overrides={}`; `student_config_overrides={}`; and static
  `paper_ready_blockers=[]`. The dataset profile also resolved exactly as
  `baby_teacher_reference_v1` / `teacher_only` /
  `historical_baby_core_plus_matching_promptmm_prompt_defaults`.
- Frozen semantic controls: `td_distill_alpha=0.3`, item-image rate `1.0`,
  item-text rate `0.0`, user-image rate `0.0`, and user-text rate `0.0`. No CLI
  override may change a semantic rate, alpha, seed, optimizer, budget,
  protocol, teacher, preprocessing, or other profile-owned value; any resolved
  override or profile identity mismatch is a hard blocker.
- Frozen student and optimizer controls: random user/item ID embedding
  initialization at seed `2022`; no student checkpoint load;
  `td_init_from_teacher=false`; `td_distill_no_projection`; dimension `64`;
  no projection and no teacher warm start; AdamW `student_lr=0.00006` and
  student weight decay `0.01`; batch size `1024`; maximum epoch `1000`;
  Validation every epoch; early-stopping patience `7`;
  `smoke_train_batches=0`; unchanged `data_generator.sample()` pairwise BPR
  sampling; and all `ceil(118551/1024)=116` independently sampled batches per
  completed epoch. No efficiency benchmark is allowed.
- Frozen dataset and preprocessing identity: audited MMRec Baby under
  `data/baby/`; conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test matrix SHA256 values respectively
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256 values respectively
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Preserve official splits, `retain_official` cold-item policy,
  duplicate-modalities policy `error`, dataset preflight, `pca` hard-token
  type, `hard_token_seed=2022`, and the existing image/text PCA cache
  identities and provenance. No fallback, regeneration, or data/cache change
  is allowed.
- Frozen teacher identity and controls: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, statically rechecked at
  exactly `141098540` bytes with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. No teacher optimizer step, run-local
  teacher checkpoint, shared-alias publication or overwrite, or new teacher
  Test ranking is allowed.
- Frozen evaluation contract: unchanged `val_test_once_v1`; train only on
  `train_mat`; candidate exclusion `train_only`; `Ks=[10,20,40,50]` and
  `test_flag=part`; select checkpoints and early-stop only by Validation
  Recall@20; save and restore the Validation-best student checkpoint; and only
  then perform exactly one final student Test ranking. Test is report-only and
  cannot influence selection, acceptance, retry, rollback, or any later
  profile.
- Frozen deployment contract: the full checkpoint may contain the finite
  user/item ID embeddings, optimizer state, and declared metadata. The
  inference-only checkpoint must contain only finite user/item ID embeddings
  plus `embedding_dim`, user/item counts, and no-projection variant identity;
  it must contain no teacher, modality, prompt, semantic cache, graph,
  projection, or optimizer state. Full and inference user/item embeddings
  must be bit-for-bit equal.
- Interpretation boundary: active-rate normalization makes the image-only
  objective `L_BPR + 0.3 * L_item_image`, while the same-seed full candidate
  objective is
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`. Alpha is held at
  `0.3`, but the image term's effective coefficient is not held fixed: it is
  `0.3` for image-only and `0.3 / 1.3` for the full candidate. A later paired
  difference therefore compares two normalized supervision mixtures and must
  not be described as the pure marginal causal contribution of text at fixed
  image weight.
- Intended launch environment: `D:\miniconda\envs\run_5060\python.exe`;
  Python `3.10.20`; PyTorch `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver
  `595.97`; NVIDIA GeForce RTX 5060 8 GB; GPU selector `0`. Exact clean source,
  canonical resolution, empty overrides/blockers, environment/GPU/free-space,
  frozen dataset/cache/teacher identities, and absence of another training
  process must be rechecked immediately before a separately authorized launch.
- Execution transport and retry contract: execute the frozen command once in
  one continuously attached foreground shell/tool session for the entire
  process, with timeout explicitly at least `10800000` ms. A short/default
  timeout, detached process, closed output channel while active, second launch,
  or automatic retry is forbidden. If execution or transport ends for any
  reason, preserve the resulting state and artifacts, audit it once, append
  one completed or failed outcome, and stop.
- Six planned run-specific artifacts, all isolated under one new timestamp/PID
  run identity: raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  All six must share the same run identity, be fingerprinted after exit, remain
  outside Git, and never use or overwrite a shared student alias.
- Hard acceptance for the later run: launch exactly once from the exact clean
  declaration commit through the required foreground transport; pass source,
  profile, override/blocker, environment, data/preprocessing/cache, and frozen
  teacher gates; keep all loss and Validation metric series finite; use all
  `116` batches per completed epoch; select and early-stop only by Validation
  Recall@20; restore the Validation-best checkpoint before exactly one final
  Test; finish with manifest `status=completed`,
  `paper_ready_eligible=true`, `paper_ready_blockers=[]`, and
  `final_test_performed=true`; preserve the teacher hash; and pass all six
  artifact, checkpoint-structure, finiteness, deployment-state, and
  full-to-inference equality audits.
- Quality rule, risks, and preservation: there is no minimum metric or
  improvement threshold. A finite, protocol-compliant low result or a paired
  direction reversal relative to full candidate commit `92b64136...` remains
  valid completed evidence and may not trigger reset, revert, deletion,
  profile change, or rerun. Mark a later run failed only for a process/code or
  hard-acceptance failure. The uncapped run may fail or exceed three hours;
  one-time Test access is consumable; the retained official cold-item warning,
  limited three-seed evidence, normalized-mixture interpretation, and ignored
  artifact protection remain unresolved risks. Preserve every partial or
  completed artifact without destructive recovery.
- Declaration metrics, data/Test access, and artifacts: none. This declaration
  used only Git/file inspection, static canonical parser resolution, and the
  frozen-teacher fingerprint check. It did not import the training entry point,
  load Baby data or any split, perform training/Validation/Test/efficiency
  evaluation, or create a run artifact.
- Declaration acceptance and unique next action: append a separate declaration
  outcome below, verify pure append and one-file scope, pass unstaged and staged
  whitespace/scope checks, create one documentation-only local commit, confirm
  a clean tree and no training process, and stop. With separate explicit user
  authorization, the only next action is to use that clean declaration commit
  to execute the exact frozen seed-2022 image-only command once in the required
  continuously attached foreground session with timeout at least `10800000`
  ms, then audit and commit one completed or failed outcome.

### 2026-08-02 | baby_td_item_image_only_no_projection_seed2022_v1 formal declaration (completed; run pending)

- Status and outcome: completed the declaration-recording task successfully.
  The independent uncapped seed-2022 image-only formal run above remains
  `pending`; it was not started, attempted, detached, or retried. No seed-2023
  or seed-2024 run was declared, and no later ablation stage was advanced.
- Source and provenance outcome: branch
  `codex/experiment/baby-teacher-baseline` began clean at exact implementation
  commit `9394ac0d75649e0e58a4dd4f0dbecdcabad2e306`, whose parent and ablation
  declaration commit is `5ee560dcd06d73e3f9a7cb3844144b4e92d6a516`.
  The same-seed full candidate outcome remains anchored at
  `92b64136e3ca75fa0f41a11e29f907a9403d54b3`. The commit containing this
  outcome is the documentation-only declaration commit and must be the exact
  clean source for the separately authorized future launch.
- Canonical static-resolution outcome: the exact frozen command resolved to
  profile/scope/source
  `baby_td_item_image_only_no_projection_seed2022_v1` / `student_ablation` /
  `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022`;
  student/sampling `seed=2022`; `hard_token_seed=2022`;
  `dataset_config_overrides={}`; `student_config_overrides={}`; and static
  `paper_ready_blockers=[]`. It also resolved alpha/rates as
  `0.3 / 1.0 / 0.0 / 0.0 / 0.0`, random 64-dimensional no-projection
  initialization without teacher warm start, AdamW `student_lr=0.00006`,
  weight decay `0.01`, batch `1024`, epoch `1000`, patience `7`, no smoke cap,
  frozen teacher reuse, `val_test_once_v1`, final Test enabled, and efficiency
  disabled. `main_mmlight.py` was not imported by this check.
- Teacher and process verification: the frozen teacher path exists at exactly
  `141098540` bytes and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  Startup and pre-edit process inspection found no matching
  `main_mmlight.py` training process. The exact implementation-to-current
  source diff under `codes/` and `docs/` was empty before this append.
- Declaration content acceptance: the pending record freezes the exact command,
  complete semantic/student/optimizer/budget/data/PCA-cache/teacher/protocol
  and inference-only contracts, all six isolated artifact classes, continuous
  foreground execution with timeout at least `10800000` ms, one execution and
  no automatic retry, hard acceptance, and preservation of low or reversed
  valid evidence. It also locks image-only as
  `L_BPR + 0.3 * L_item_image` versus full candidate as
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`, so neither fixed
  effective image coefficient nor pure marginal text causality may be claimed.
- Repository verification: before this outcome append, the declaration diff
  was one EOF-only hunk in `TRAINING_LOG.md`, with `163` added lines and zero
  deleted lines; required-content checks and `git diff --check` passed. Final
  pure-append, exact staged-file, cached-whitespace, commit-parent, clean-tree,
  and no-training-process checks remain mandatory before handoff.
- Commands, metrics, data/Test access, and generated artifacts: no project
  command, smoke, training, Validation, Test, or efficiency evaluation ran.
  No Baby data or split was loaded or accessed, and there are no quality or
  efficiency metrics. No checkpoint, preflight, manifest, convergence record,
  raw log, tag, bundle, backup, or merge was created or changed. This appended
  declaration and its local Git commit are the only task artifacts.
- Experimental meaning and unresolved risks: the run contract is now frozen
  before image-only seed-2022 evidence, but runtime completion and metrics are
  still unknown. The official cold-item warning, consumable one-time final
  Test, normalized-mixture interpretation boundary, limited three-seed
  uncertainty, long foreground runtime, and ignored-artifact protection remain
  unresolved. None authorizes changing, cancelling, reordering, or rerunning a
  profile in response to result direction.
- Unique next action: only with new explicit user authorization, from the exact
  clean declaration commit execute the frozen seed-2022 image-only command
  once in one continuously attached foreground session with timeout at least
  `10800000` ms; after exit, audit the single run and append and commit one
  completed or failed outcome. Stop without declaring seed-2023/2024 or
  starting another experiment.

### 2026-08-02 | baby_td_item_image_only_no_projection_seed2022_v1 formal uncapped ablation (failed)

- Status and acceptance decision: failed the declared hard protocol acceptance
  rule because the frozen-teacher reuse path performed one new teacher Test
  ranking before student training. The single student run itself exited
  naturally with code `0`, completed all optimization and Validation work,
  restored its Validation-best checkpoint, performed exactly one later student
  final Test, and produced internally consistent finite artifacts. Its metrics
  and all six artifacts remain valid diagnostic evidence and are preserved,
  but the run is not accepted as paper-ready image-only comparison evidence.
  No retry, second launch, reset, revert, deletion, or rollback occurred.
- Source and execution identity: branch
  `codex/experiment/baby-teacher-baseline`; exact clean declaration and launch
  HEAD `7108e3ec39bc66ec481faf7059cadecf8975a213`; frozen implementation
  `9394ac0d75649e0e58a4dd4f0dbecdcabad2e306`. The pre-launch source diff
  `git diff 9394ac0d... 7108e3ec... -- codes docs` was empty, the tracked tree
  was clean, and no source or profile file changed during execution or audit.
- Executed command, exactly once with no added, removed, or overwritten
  argument:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_item_image_only_no_projection_seed2022_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Transport, PID, run identity, and duration: one continuously attached
  foreground shell/tool session was used with timeout explicitly set to
  `10800000` ms. The command was not detached or relaunched. Main PID `37484`
  and run identity
  `2026-08-02 02_15_00.996496_baby_light_init_pid37484` remained unchanged.
  The tool returned exit code `0` after approximately `1058.1` seconds;
  manifest start/completion were `2026-08-02T02:15:00.999008+08:00` and
  `2026-08-02T02:32:30.194256+08:00`, an internal elapsed time of about
  `1049.2` seconds.
- Pre-launch gates and environment: HEAD/branch/clean-tree and source-diff
  gates passed; no Python training process existed; Python was `3.10.20`,
  PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, NVIDIA driver `595.97`, and GPU
  `0` was an NVIDIA GeForce RTX 5060 with `8151 MiB` total and `6076 MiB` free
  at inspection. Drive D had `405907824640` bytes (`378.031 GiB`) free. The
  command resolved without importing the training entry point during the
  static gate, and all required files and caches were present.
- Resolved profile and semantic controls: profile/scope/source were exactly
  `baby_td_item_image_only_no_projection_seed2022_v1` / `student_ablation` /
  `predeclared_baby_item_image_only_no_projection_ablation_v1_seed2022`;
  `seed=2022`; `hard_token_seed=2022`; `dataset_config_overrides={}`;
  `student_config_overrides={}`; and static plus runtime
  `paper_ready_blockers=[]`. Alpha and item-image/item-text/user-image/user-text
  rates were `0.3 / 1.0 / 0.0 / 0.0 / 0.0`. The no-projection model created
  only the item-image semantic head, with no user or item-text active head and
  no teacher warm start.
- Data, preprocessing, and PCA-cache audit: preflight reproduced the declared
  conversion, train, Validation, Test, image, and text SHA256 identities;
  shapes/interactions were unchanged; split overlaps were `0 / 0 / 0`; both
  modalities were finite, distinct, and non-duplicate. The only preflight
  warning was the frozen `retain_official` condition for three cold items with
  `11` Validation and `7` Test interactions and no cold users. Image/text PCA
  caches were hits at random state `2022`: image identity
  `7634bb6e8dcfdbdd45eee436cc3b8e3f3f8e1e54c85c2f4b9f4a17719cc77c90`,
  `3610252` bytes, SHA256
  `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`;
  text identity
  `6e7c8161aaeb01f5e4bb8744be3bf9d16df71cd7cc74ea841bc5c20ad60926ee`,
  `1805450` bytes, SHA256
  `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
  No cache fallback, regeneration, or identity change occurred.
- Student training and Validation selection: epochs `0-74` completed, each
  using the uncapped `116/116` batches shown by the attached foreground
  progress stream. The convergence record contains `75` entries for every
  total/BPR/distillation/component-loss and Validation metric series; all are
  finite, and inactive component losses remain exactly zero. Validation
  Recall@20 alone selected and early-stopped the run at patience `7/7`.
  Manifest, convergence record, and full checkpoint all agree on best epoch
  `67` and exact Validation Recall@20 `0.06542176345982047`.
- Student final Test chronology and exact result: after the epoch-74 natural
  early stop, the raw log has exactly one event stating restoration of
  Validation-best epoch `67` followed by the student final Test. Ordered by K
  `[10,20,40,50]`, Precision is
  `[0.004607868346618729, 0.003674466443815945, 0.002802777063512372, 0.0025857546927232234]`;
  Recall is
  `[0.0416851781971352, 0.06670462348143064, 0.10119136969998162, 0.11673884385378007]`;
  NDCG is
  `[0.023442955935571032, 0.03022013402080786, 0.03792645906484195, 0.0409738110351617]`;
  Hit Ratio is
  `[0.04592440215993866, 0.07307791205965578, 0.11072255078426049, 0.12717922345075494]`;
  and AUC is `0.0` as expected for `test_flag=part`. All values are finite.
- Manifest and checkpoint audit: the runner manifest records
  `status=completed`, `paper_ready_eligible=true`, `paper_ready_blockers=[]`,
  and `final_test_performed=true`, and its profile, protocol, cache, teacher,
  and artifact paths agree with the other evidence. Those fields do not detect
  the teacher Test chronology violation and therefore do not override this
  failed hard-acceptance decision. The format-v2 full checkpoint contains
  finite user `(19445,64)` and item `(7050,64)` ID embeddings plus two
  optimizer-state entries and declared metadata. The inference-only checkpoint
  has exactly the two finite embeddings plus `embedding_dim`, user/item counts,
  and the no-projection variant; it contains no optimizer, teacher, modality,
  prompt, semantic-cache, graph, projection, loss, or epoch state. Both
  embeddings are bit-for-bit equal between full and inference checkpoints.
- Frozen teacher identity and root cause: the teacher remained exactly
  `141098540` bytes with pre/post SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  teacher training was skipped, alias overwrite was disabled, and no run-local
  teacher checkpoint or teacher optimizer step occurred. However, in the
  frozen-teacher branch, `codes/main_mmlight.py` lines `1572-1578` observed
  `teacher_test_ret is None` with `run_final_test=true` and explicitly called
  `restore_checkpoint_then_evaluate(... self.test(..., is_val=False,
  is_teacher=True))`. The resulting `Teacher reuse summary` and metric vector
  in the raw log are one newly computed teacher Test ranking, not merely
  checkpoint-metadata reuse. This is the sole hard failure and contradicts the
  declaration's prohibition on a new teacher Test ranking.
- Test-split access and protocol compliance: preflight structurally read and
  fingerprinted the Test matrix as declared. The process then performed one
  prohibited new frozen-teacher Test ranking before student training and one
  authorized student final Test ranking only after restoring the student
  Validation-best checkpoint. Student selection and early stopping did not use
  Test, but the teacher reranking means the overall declared protocol did not
  comply and the run cannot be paper-ready accepted.
- Six preserved same-run artifacts, absolute paths, bytes, and SHA256:
  - raw log: `D:\Download\PromptMM\logs\2026-08-02 02_15_00.996496_baby_light_init_pid37484`,
    `36787` bytes,
    `724d5b85b189d8e158e7caed027134c94035bca3fab90ee69de19eed708f7a28`;
  - preflight: `D:\Download\PromptMM\exp\runs\baby\dataset_preflight__2026-08-02 02_15_00.996496_baby_light_init_pid37484.json`,
    `3399` bytes,
    `a5924b509830ee7e19266a93be9db1e8c464ee1f18e8813aba9d2af6aa26ce9e`;
  - manifest: `D:\Download\PromptMM\exp\runs\baby\run_manifest__2026-08-02 02_15_00.996496_baby_light_init_pid37484.json`,
    `25460` bytes,
    `9f79cdee9c0f83a8dffaed77dd5f2238c1b3b1326ce4300b233cbbea380404a7`;
  - convergence: `D:\Download\PromptMM\exp\converge\baby\auto__2026-08-02 02_15_00.996496_baby_light_init_pid37484.pkl`,
    `11936` bytes,
    `e88ed22466efb3e6726f09186709e138ad74424a7ba731c89b5003e00160d505`;
  - full checkpoint: `D:\Download\PromptMM\Model\baby\td_distill\td_distill_full__val_test_once_v1__2026-08-02 02_15_00.996496_baby_light_init_pid37484.pth`,
    `20354975` bytes,
    `8734934df4d72725e54d764e4d6fe1d7301bac61818f59fbf415a2984a94f032`;
  - inference-only checkpoint: `D:\Download\PromptMM\Model\baby\td_distill\td_distill_infer_only__val_test_once_v1__2026-08-02 02_15_00.996496_baby_light_init_pid37484.pth`,
    `6785997` bytes,
    `48bc297017b24392861c85075e934fee353484e37657dc5e63e9fc8f9fbf1280`.
- Same-seed paired diagnostic differences, defined as full candidate minus
  image-only: the full candidate is run
  `2026-07-31 18_35_00.450484_baby_light_init_pid22864`. Validation Recall@20
  difference is `+0.0030041998800031666`. At Test K=20, Precision difference
  is `-0.000012856775520699634`, Recall difference is
  `-0.00011228250621408975`, NDCG difference is
  `+0.0005919636463151898`, and Hit Ratio difference is
  `-0.0003599897145795533`. The Recall@20 direction therefore reverses by a
  small amount, but quality direction did not cause this failure and does not
  authorize a rerun or artifact deletion.
- Interpretation boundary and unresolved risks: image-only optimizes
  `L_BPR + 0.3 * L_item_image`; the full candidate optimizes
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`. Alpha is matched,
  but the effective image coefficient is `0.3` versus `0.3 / 1.3`; paired
  differences compare normalized supervision mixtures and cannot identify the
  pure marginal causal contribution of text at fixed image weight. In
  addition to the frozen cold-item and three-seed limitations, the newly
  confirmed teacher-reranking behavior remains an unresolved protocol risk for
  future formal launches. Ignored run artifacts remain outside Git protection.
- Unique next action: in a separate task, append and commit only the seed-2023
  image-only formal-run pending declaration for exact profile
  `baby_td_item_image_only_no_projection_seed2023_v1`, explicitly preserving
  this failed run and its teacher-reranking blocker, then stop without starting
  training. Do not declare seed-2024, change source, rerun seed-2022, create a
  tag/bundle/backup, merge `main`, or launch any experiment in that task.

### 2026-08-02 | Frozen-teacher Test-isolation protocol fix (pending)

- Purpose and rationale: repair the frozen-teacher reuse defect confirmed by
  failed seed-2022 image-only run
  `2026-08-02 02_15_00.996496_baby_light_init_pid37484`. When
  `if_train_teacher=false`, the runner must load and freeze the declared
  teacher checkpoint for student distillation without performing any new
  teacher Test ranking, independently of `run_final_test`. Historical teacher
  quality stored in the checkpoint may remain provenance metadata but must not
  be represented as an evaluation performed by the current run.
- Status and authorization boundary: pending implementation and static/unit
  verification only. This task authorizes the scoped source, focused tests,
  this append-only trace, and one coherent local commit. It does not authorize
  loading any Baby split, importing or executing the training entry point,
  training, Validation ranking, Test ranking, smoke/formal execution,
  seed-2022 retry, seed-2023/2024 declaration or execution, tag, bundle,
  backup, baseline overwrite, or merge to `main`.
- Branch, source, and preservation point: branch
  `codex/experiment/baby-teacher-baseline` at exact clean commit
  `3934561e53aa108ea670412ee1db1a439e128ab0`. Preserve the failed seed-2022
  run and every ignored artifact recorded there; no reset, revert, deletion,
  or reinterpretation of that evidence is permitted.
- Declared implementation scope: update the frozen-teacher reuse and manifest
  logic in `codes/main_mmlight.py`; add a narrow reusable protocol guard in
  the existing protocol utility only if needed for testability; and add or
  extend focused synthetic/static tests under `codes/tests/`. The guard must
  be independent of Baby data and callable before the Test evaluator receives
  users or accesses Test ranking state. No parser/profile/model/sampler/data,
  preprocessing/cache, optimizer, loss, checkpoint-format, student selection,
  student final-Test, or efficiency behavior may change.
- Required runtime behavior and manifest contract: with
  `if_train_teacher=false`, load the frozen checkpoint, set teacher/prompt
  modules to evaluation/frozen use, set
  `teacher_final_test_performed=false`, and never call an
  `is_teacher=true, is_val=false` ranking path even when
  `run_final_test=true`. Any historical final-Test result already embedded in
  checkpoint metadata must be stored under an explicitly historical/existing
  metadata field, not `teacher_final_test_result` for the current run. A
  frozen-teacher attempt to evaluate Test must add a deterministic
  paper-ready blocker to the run manifest and raise before the evaluator or
  Test-user materialization is invoked.
- Risks: a guard placed after `data_generator.test_set` access would be too
  late; changing the shared test method without preserving the teacher-training
  and student-finalization paths could block authorized evaluation; leaving
  stale manifest fields could falsely claim current teacher Test access; and
  tests that import `main_mmlight.py` could accidentally load Baby data. The
  implementation must therefore isolate policy from data access and use only
  static or temporary synthetic fixtures.
- Acceptance criteria: focused tests prove that frozen-teacher reuse with both
  `run_final_test=true` and `false` performs no teacher Test evaluation,
  records `teacher_final_test_performed=false`, labels checkpoint-carried
  metrics only as historical metadata, and rejects a direct frozen-teacher
  Test attempt before a sentinel evaluator or Test-user supplier is touched
  while recording the blocker. Tests must also preserve authorized trained-
  teacher Test and student final-Test behavior. Relevant unit tests, Python
  compilation, static source/protocol checks, pure-append log verification,
  `git diff --check`, staged scope, and `git diff --cached --check` must pass.
- Planned verification and outcome: inspect existing protocol/manifest/test
  seams; implement only the declared scope; run focused unit tests plus the
  relevant existing protocol suite without importing the training entry point;
  run `py_compile` and static AST/text checks; confirm no Baby data/Test access
  and no training process; append a separate completed or failed outcome; then
  create one commit containing only the scoped source/tests and
  `TRAINING_LOG.md` and stop.

### 2026-08-02 | Frozen-teacher Test-isolation protocol fix (completed)

- Status and outcome: completed successfully. Frozen-teacher reuse now loads
  and freezes the declared checkpoint for student distillation without any
  teacher Test ranking, regardless of the student run's `run_final_test`
  value. The failed seed-2022 image-only run and its six artifacts remain
  preserved and failed; this code change does not retroactively accept,
  reinterpret, rerun, or replace that evidence.
- Source and scope: work began clean on branch
  `codex/experiment/baby-teacher-baseline` at exact commit
  `3934561e53aa108ea670412ee1db1a439e128ab0`. The completed scope changes only
  `codes/main_mmlight.py`, `codes/utility/experiment_protocol.py`,
  `codes/tests/test_experiment_protocol.py`, and this append-only log. No
  parser, dataset/student profile, model, loss, sampler, optimizer, data,
  preprocessing/cache, checkpoint format, efficiency path, or experiment
  parameter changed.
- Runtime behavior: the frozen branch captures the loaded checkpoint, sets the
  teacher and prompt modules to evaluation mode with all parameters
  `requires_grad=false`, and unconditionally records frozen reuse without
  teacher Test. The previous `run_final_test`-conditioned frozen-teacher
  `restore_checkpoint_then_evaluate(... is_teacher=true, is_val=false)` block
  and its `Teacher reuse summary` were removed. Authorized trained-teacher
  final Test and student final Test still restore their selected checkpoints
  and pass through the unchanged evaluator after the new access gate permits
  them.
- Manifest contract: every run initializes
  `teacher_final_test_performed=false`. Frozen reuse retains that value and
  records `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking` plus
  `status=teacher_reused_without_test`. A checkpoint-carried
  `teacher_final_test_result` or `final_test_result`, when present, is copied
  only to `teacher_checkpoint_historical_final_test_metadata` with
  `source=teacher_checkpoint_metadata`, its original field name, and
  `performed_by_current_run=false`; no current-run
  `teacher_final_test_result` is emitted by frozen reuse.
- Test-access hard gate: all runner evaluation-user materialization now uses a
  lazy supplier. Before that supplier can read Validation/Test user keys, the
  pure protocol guard rejects exactly
  `if_train_teacher=false, is_teacher=true, is_val=false`. It first appends the
  deterministic blocker
  `frozen-teacher reuse attempted a new teacher Test ranking`, sets
  `paper_ready_eligible=false`, preserves
  `teacher_final_test_performed=false`, writes the manifest, and then raises.
  `Trainer.test` repeats the same guard before model forward/ranking so direct
  evaluator calls are also blocked. Static regression requires the sole
  `data_generator.test_set` reference to remain inside the guarded lazy
  supplier.
- Focused and full verification: the first focused protocol run passed `19/20`
  and failed only because a new static assertion incorrectly prohibited
  passing `run_final_test` into the manifest builder; the implementation had
  not violated the protocol. The assertion was narrowed to prohibit
  `run_final_test` control flow, and the focused suite then passed `20/20`.
  Final `D:\miniconda\envs\run_5060\python.exe -m unittest discover -s
  codes/tests -p "test_*.py" -v` passed `58/58`. The tests prove pre-access
  blocker ordering, manifest behavior for both `run_final_test=true/false`,
  historical-only metric labeling, direct evaluator protection, preserved
  frozen-teacher Validation, authorized trained-teacher Test, and authorized
  student Test behavior using only static or temporary synthetic fixtures.
- Compilation and independent static verification:
  `D:\miniconda\envs\run_5060\python.exe -m py_compile
  codes/utility/experiment_protocol.py codes/main_mmlight.py
  codes/tests/test_experiment_protocol.py` passed. An independent AST/pure-
  function command, without importing `main_mmlight.py`, returned
  `STATIC_PROTOCOL_OK`: one guarded lazy Test-set reference, no frozen
  `Teacher reuse summary`, identical false teacher-Test manifest semantics for
  both student final-Test settings, and blocker execution before a sentinel
  user supplier. `git diff --check` passed before this outcome append.
- Commands, data/Test access, metrics, and artifacts: no Baby dataset or split
  was loaded or accessed. No training entry point was imported or executed;
  no training, Validation, Test, smoke, formal run, or efficiency benchmark
  occurred. Therefore there is no new quality/efficiency metric, run identity,
  raw log, preflight, manifest, convergence record, checkpoint, dataset/cache
  artifact, tag, bundle, backup, or merge. Verification created only ignored
  Python bytecode caches plus temporary synthetic test fixtures; no experiment
  artifact or frozen teacher file changed.
- Acceptance and residual risk: all declared static/unit/compile acceptance
  conditions pass. The live GPU/data runner path remains intentionally
  unexecuted, so a separately declared later formal run must still verify the
  manifest and Test chronology at runtime. The prior failed seed-2022 result,
  cold-item condition, normalized-mixture interpretation boundary, limited
  seed evidence, and ignored-artifact protection remain unresolved and
  unchanged.
- Unique next action: in a separate task, append and commit only the seed-2023
  image-only formal-run pending declaration for exact profile
  `baby_td_item_image_only_no_projection_seed2023_v1`, citing the clean commit
  that contains this fix and explicitly gating future launch on the new
  frozen-teacher manifest/Test-isolation contract; then stop without loading
  data or starting any run. Do not declare seed-2024 or rerun seed-2022 in that
  task.

## 2026-08-02 frozen-teacher Test chronology correction audit pending

- Status: `pending`; this is a read-only retrospective protocol audit begun from
  clean commit `ee7b3621765b63886c341d63fdc49a20d1a21bc8` on branch
  `codex/experiment/baby-teacher-baseline`.
- Purpose and rationale: audit the frozen-teacher Test chronology of the six
  existing seed-2022/2023/2024 baseline/candidate formal student runs and
  correct the prior conclusion that frozen-teacher reuse only copied checkpoint
  metadata and performed no new teacher Test ranking. The audit hypothesis is
  that the old `Teacher reuse summary` records came from the pre-fix
  `restore_checkpoint_then_evaluate(... is_teacher=true, is_val=false)` path and
  therefore represent a teacher Test ranking recomputed by each run.
- Declared scope and sources: read only the six existing raw logs, their
  manifests/preflight records as needed for identity and chronology,
  `TRAINING_LOG.md`, and Git/source history. The affected run identities are
  `2026-07-31 12_33_07.292506_baby_light_init_pid7872` (seed-2022 baseline),
  `2026-07-31 18_35_00.450484_baby_light_init_pid22864` (seed-2022 candidate),
  `2026-08-01 20_20_35.101504_baby_light_init_pid29128` (seed-2023 baseline),
  `2026-08-01 21_09_59.190542_baby_light_init_pid33928` (seed-2023 candidate),
  `2026-08-01 23_31_40.280800_baby_light_init_pid15908` (seed-2024 baseline),
  and `2026-08-02 00_42_54.029141_baby_light_init_pid29752` (seed-2024
  candidate). The associated three-seed summaries are in scope for evidence
  classification only.
- Explicit exclusions: do not modify code or historical log entries; do not
  load Baby data or any split; do not import or execute the training entry
  point; do not train, Validation, or Test; do not declare or run any seed; do
  not create tags, bundles, backups, or merge `main`.
- Risks and acceptance criteria: preserve every original raw log, manifest,
  metric, checkpoint, and other artifact unchanged; identify each run and
  prove whether its `Teacher reuse summary` is a current-run teacher Test
  ranking by ordering it against student training/final Test and matching the
  historical source path. Acceptance requires a per-run chronology finding,
  explicit correction of the prior metadata-only/no-ranking conclusion, and
  temporary classification of all six runs plus the three-seed summary as
  protocol-noncompliant descriptive/diagnostic evidence.
- Rollback point and verification plan: rollback point is the unchanged
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8` source tree and preserved log
  history. After read-only evidence collection, append a separate
  `completed`/`failed` correction outcome, run append-only and staged diff
  checks, and create one commit containing only `TRAINING_LOG.md`. No new run
  or artifact is authorized by this declaration.

## 2026-08-02 frozen-teacher Test chronology correction audit completed

- Status and correction: `completed`. The retrospective audit confirms that
  all six seed-2022/2023/2024 baseline/candidate formal student runs executed
  one new frozen-teacher Test ranking before student training. Prior statements
  that their `Teacher reuse summary` values came only from checkpoint metadata,
  or that no new teacher Test ranking occurred, are incorrect. Historical
  entries remain verbatim; this append-only outcome supersedes only that
  chronology and paper-readiness interpretation.
- Audit source and scope: branch `codex/experiment/baby-teacher-baseline`, clean
  audit parent `ee7b3621765b63886c341d63fdc49a20d1a21bc8`. Evidence was limited
  to the six existing raw logs and manifests, the active `TRAINING_LOG.md`, and
  Git/source history. No code, profile, parameter, dataset, split, cache,
  checkpoint, convergence record, raw log, manifest, metric, or artifact was
  changed.
- Historical source proof: the exact launch commits
  `8d8858547a475fd970ae9753496b2a7c06eb472c`,
  `6948baca05008a8adcb65fe0e83d65c0878e74d4`,
  `eaaa68d851af2256b7b5c9957eab4cc426fecc8b`,
  `6c529c7446b7d95529fe24df78138451b3528e51`,
  `b1b5835a22060dcfe4a76d240d2be1cded157257`, and
  `e2a730748a7527fe284358e0c19cccd5744b5906` each contain the same
  frozen-reuse control flow: with `teacher_test_ret is None` and
  `run_final_test=true`, it reads `data_generator.test_set`, calls
  `restore_checkpoint_then_evaluate(... self.test(..., is_val=false,
  is_teacher=true))`, logs `Teacher reuse summary`, and writes
  `teacher_final_test_result`. The corresponding implementation commits
  `58fda059...`, `64739bb...`, `03260ee2...`, and `cb22e9d...`, and later
  pre-fix commit `7108e3e...`, preserve that path; source diffs from each
  implementation commit to its launch commit are empty for
  `codes/main_mmlight.py`.
- Manifest semantic proof: all six old manifests lack a
  `teacher_final_test_performed` field, contain a top-level
  `teacher_final_test_result`, and have neither `final_test_result` nor
  `teacher_final_test_result` inside `teacher_checkpoint_metadata`. Thus the
  top-level teacher vector is a current-run evaluator result, not copied
  checkpoint metadata. Every run recomputed the same teacher Test vector at K
  `[10,20,40,50]`; at K=20 it is Recall
  `0.08665691369856875`, Precision `0.004836718950887038`, and NDCG
  `0.04042213264283099`.
- Affected seed-2022 baseline: run
  `2026-07-31 12_33_07.292506_baby_light_init_pid7872`, profile
  `baby_student_reference_v1`, seed `2022`, launch commit `8d885854...`.
  Raw-log lines `8/9/10` record teacher training skipped, checkpoint loading,
  and the newly computed teacher summary; line `17` is the first student
  Validation and line `360` is the sole student final Test after restoring
  validation-best epoch `163`. Raw log is `76795` bytes / SHA256
  `2664a00666cd95920f572dca5b1fc4441fd01d8534b1a711e65a45d300f1e256`;
  manifest is `25331` bytes / SHA256
  `cd2c5aa1fae79af8f8161dfee4fbb2bf9f451b591607df6c646d1dbf8b67af7d`.
- Affected seed-2022 candidate: run
  `2026-07-31 18_35_00.450484_baby_light_init_pid22864`, profile
  `baby_td_asymmetric_no_projection_v1`, seed `2022`, launch commit
  `6948baca...`. The same teacher events are at raw-log lines `8/9/10`, before
  first student Validation at line `17`; the sole student final Test is line
  `180` after restoring epoch `73`. Raw log is `39273` bytes / SHA256
  `1b544004190bf48f3364d6fae820fcf5f838205fd7ad4b1d667ba5bb9966ba28`;
  manifest is `25398` bytes / SHA256
  `967779e828f820ece44bd2a75331ee48a1a4e45ca2cefbb7b43bbe5eb3793480`.
- Affected seed-2023 baseline: run
  `2026-08-01 20_20_35.101504_baby_light_init_pid29128`, profile
  `baby_student_reference_seed2023_v1`, seed `2023`, launch commit
  `eaaa68d8...`. Teacher skip/load/recomputed-summary lines `8/9/10` precede
  first student Validation line `17`; the sole student final Test is line
  `178` after restoring epoch `72`. Raw log is `39035` bytes / SHA256
  `e8f3ebaafe0e420e4816df80869393fb746de17979c3eb7faf9656a190e01596`;
  manifest is `25383` bytes / SHA256
  `edcc1b4d858eecbad5724ac0c46608abb572bb9412f78a1555156cff278a7d33`.
- Affected seed-2023 candidate: run
  `2026-08-01 21_09_59.190542_baby_light_init_pid33928`, profile
  `baby_td_asymmetric_no_projection_seed2023_v1`, seed `2023`, launch commit
  `6c529c74...`. Teacher skip/load/recomputed-summary lines `8/9/10` precede
  first student Validation line `17`; the sole student final Test is line
  `122` after restoring epoch `44`. Raw log is `27194` bytes / SHA256
  `98770421a4668bba8bdce727e045c003be087065f6c59a2487dd0e8ed51987e2`;
  manifest is `25443` bytes / SHA256
  `357afe298cc1a771ddd30aaddd046a73298894e899e636c6eae0bc546608a42a`.
- Affected seed-2024 baseline: run
  `2026-08-01 23_31_40.280800_baby_light_init_pid15908`, profile
  `baby_student_reference_seed2024_v1`, seed `2024`, launch commit
  `b1b5835a...`. Teacher skip/load/recomputed-summary lines `8/9/10` precede
  first student Validation line `17`; the sole student final Test is line
  `360` after restoring epoch `163`. Raw log is `76963` bytes / SHA256
  `4458c705f2eaabf604165ab06db9e56f9efb1ae8443316022d8a37dce5593e8d`;
  manifest is `25386` bytes / SHA256
  `d0f3e2bb49dd641a1b88d5a16e0fb73c0fa5e45c1b5e65d2d3418d1d1ae04b7c`.
- Affected seed-2024 candidate: run
  `2026-08-02 00_42_54.029141_baby_light_init_pid29752`, profile
  `baby_td_asymmetric_no_projection_seed2024_v1`, seed `2024`, launch commit
  `e2a73074...`. Teacher skip/load/recomputed-summary lines `8/9/10` precede
  first student Validation line `17`; the sole student final Test is line
  `154` after restoring epoch `60`. Raw log is `33827` bytes / SHA256
  `4fc9661bf13decf360894ba28171e2ac754be395099ee1a93f4ca1e5306f13db`;
  manifest is `25447` bytes / SHA256
  `96ad86aaae2ba745bb5ec78ad9091a691d9aa26b64eb77a909af2df294823212`.
- Corrected Test chronology: each affected run performed two ranking accesses
  to Test: first, one prohibited frozen-teacher ranking immediately after
  checkpoint load and before student training/Validation; second, one
  authorized student final Test only after natural early stopping and
  validation-best checkpoint restoration. Student Validation selection,
  student final-Test metrics, teacher checkpoint identity, and all previously
  recorded artifact paths/bytes/SHA256 remain unchanged, but they do not erase
  the first protocol violation.
- Evidence reclassification: the six runs retain their original process
  completion, numerical results, manifests, raw logs, convergence records, and
  checkpoints, but are now temporarily classified as protocol-noncompliant
  `descriptive/diagnostic evidence`, not paper-ready formal evidence. The
  canonical matched three-seed summary and its means, sample SDs, and paired
  deltas are likewise retained unchanged only as protocol-noncompliant
  descriptive/diagnostic calculations. Historical manifest values
  `paper_ready_eligible=true`, `paper_ready_blockers=[]`, and `status=completed`
  predate the missing teacher-Test gate and cannot establish current
  paper-readiness for these runs.
- Preservation, commands, and new evidence: this audit ran only read-only text,
  JSON, hash, and Git/source-history checks. It did not load Baby or any data
  split, import/execute project code, train, Validation, Test, rank, declare or
  run any seed, or create/modify an experiment artifact. It generated no new
  quality metric, run identity, raw log, manifest, preflight, convergence
  record, checkpoint, tag, bundle, backup, or merge. All original metrics and
  artifacts remain preserved; only this EOF log correction is tracked.
- Acceptance and residual risk: every raw log contains exactly one
  `Teacher reuse summary` and exactly one student final-Test summary in the
  stated order; all six manifests and their recorded fingerprints match; and
  all six launch sources prove the prohibited evaluator call. The corrected
  `ee7b362...` source prevents recurrence but cannot retroactively make these
  six executions compliant. Their quality directions remain descriptive only
  until replacement formal evidence is produced under the fixed protocol.
- Unique recovery next action: only with separate explicit user authorization,
  begin a staged replacement of this protocol-noncompliant frozen-teacher run
  line from the fixed source. The first separate task should append and commit
  only a seed-2022 baseline recovery formal-run pending declaration citing
  `ee7b362...` and this audit outcome, then stop without loading data or
  starting a run; later baseline/candidate and seed stages require their own
  authorization and outcomes.

## 2026-08-02 matched seed-2022 baseline/candidate recovery execution amendment (pending)

- Purpose and rationale: replace only the protocol-noncompliant seed-2022 main
  experiment baseline/candidate pair after the retrospective audit proved that
  their pre-fix frozen-teacher reuse paths performed a prohibited new teacher
  Test ranking. This recovery is authorized solely because of that confirmed
  implementation defect. It is not motivated by the historical Validation or
  Test metric magnitudes, candidate direction, or any quality threshold, and it
  does not tune, reinterpret, delete, overwrite, or supersede the preserved old
  declarations, results, raw logs, manifests, convergence records, or
  checkpoints.
- Status and amended authorization: pending execution amendment. The user
  explicitly authorizes the two matched seed-2022 main-experiment recovery arms
  to be declared and executed strictly serially within this one task. This is a
  narrow amendment to the prior one-stage handoff: first declare, commit, run,
  audit, and commit the baseline; only if its outcome is `completed` and that
  outcome commit exists may the candidate be declared, committed, run once,
  audited, and committed. Candidate declaration or launch is prohibited after
  a failed baseline, interrupted connection, blocker, or hard-gate violation.
- Exact branch and starting identity: branch
  `codex/experiment/baby-teacher-baseline` began clean at exact commit
  `1d24afea6e7aaf1653ad952179744f92899def2b`, the completed chronology-audit
  record. The frozen-teacher Test-isolation implementation is commit
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`. Pre-amendment
  `git diff --exit-code ee7b362... 1d24afe... -- codes` passed, and the sole
  successor is the audit-only `TRAINING_LOG.md` commit, proving no post-fix
  behavior drift under `codes/`.
- Recovery scope: recover only `baby_student_reference_v1` and
  `baby_td_asymmetric_no_projection_v1` at seed `2022`. Do not run or declare
  image-only ablation, seed `2023`, seed `2024`, another seed, efficiency,
  broad tuning, a smoke, or any other experiment. Do not modify code, parser or
  profile defaults, parameters, data, preprocessing, PCA caches, teacher,
  checkpoint formats, sampler, optimizer, evaluation policy, or run commands.
  Do not create a tag, bundle, backup, or merge to `main`.
- Frozen common controls: audited MMRec Baby; `val_test_once_v1`; train on
  `train_mat`; Validation Recall@20 alone selects and early-stops; candidate
  exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`; random
  64-dimensional `td_distill_no_projection` student initialization;
  `td_init_from_teacher=false`; AdamW `student_lr=6e-5`, weight decay `0.01`;
  batch size `1024`; maximum epoch `1000`; patience `7`; all `116` batches per
  completed epoch; `smoke_train_batches=0`; no efficiency benchmark; student
  and sampling `seed=2022`; PCA/cache `hard_token_seed=2022`; and canonical
  commands with empty dataset/student override maps.
- Frozen data and preprocessing identity: conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test matrix SHA256 values
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text SHA256 values
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  The frozen image/text PCA caches remain random state `2022`, identities
  `7634bb6e8dcfdbdd45eee436cc3b8e3f3f8e1e54c85c2f4b9f4a17719cc77c90`
  and `6e7c8161aaeb01f5e4bb8744be3bf9d16df71cd7cc74ea841bc5c20ad60926ee`,
  with SHA256 `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`
  and `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
- Frozen teacher identity: reuse only read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, `141098540` bytes, SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. Teacher training, optimizer steps,
  mutation, publication, overwrite, run-local teacher checkpoints, and any new
  frozen-teacher Test ranking are prohibited.
- Test-isolation recovery contract: each run may read Test structure and
  identity during preflight, but frozen-teacher reuse must perform zero Test
  ranking and record `teacher_final_test_performed=false` plus
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`. A historical
  final-Test vector carried by a checkpoint, if present, may appear only under
  `teacher_checkpoint_historical_final_test_metadata` with
  `performed_by_current_run=false`; the current frozen checkpoint contains no
  top-level carried final-Test field, so `null` is acceptable and a current-run
  `teacher_final_test_result` is forbidden. Each student must select and
  restore only its Validation-best checkpoint and then perform exactly one new
  final Test ranking. Any frozen-teacher Test attempt is a hard failure.
- Execution and failure rule: every formal command must run exactly once in a
  single continuously attached foreground shell/tool session with the explicit
  timeout at least `10800000` ms. Parallel launch, detached/background launch,
  output-channel closure while active, automatic retry, or a second process is
  forbidden. Low metrics or a direction reversal remain completed evidence;
  only process/code failure or a hard gate may produce `failed`. Preserve all
  old and new artifacts on either outcome and never reset, revert, delete, or
  modify parameters in response to quality.
- Amendment acceptance and rollback/preservation point: this record and the
  baseline declaration below must be committed together before launch from a
  clean tree. The preservation point is exact start commit `1d24afe...`, fixed
  source `ee7b362...`, the frozen teacher/data/cache assets, and all prior
  diagnostic evidence. No destructive rollback is authorized. If the baseline
  fails, append and commit its failed outcome and stop immediately without any
  candidate declaration or execution.

## 2026-08-02 baby_student_reference_v1 matched recovery formal run (pending)

- Purpose and hypothesis: execute one replacement formal seed-2022 ID-only BPR
  baseline under the corrected frozen-teacher Test-isolation implementation.
  The goal is protocol recovery, not metric improvement. The student method and
  all parameters remain exactly the original `baby_student_reference_v1`
  contract, and the preserved old baseline remains diagnostic evidence rather
  than a rollback or overwrite target.
- Status and authorization: pending formal uncapped baseline, authorized for
  exactly one launch after this declaration commit is clean. The candidate is
  not yet declared and cannot be started until this baseline has naturally
  exited, passed the complete artifact/protocol audit, and its `completed`
  outcome has been committed. A baseline failure requires a committed failed
  outcome and immediate stop without retry.
- Exact command, frozen verbatim with no CLI seed or parameter override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Source launch gate: branch
  `codex/experiment/baby-teacher-baseline`; pre-declaration clean HEAD
  `1d24afea6e7aaf1653ad952179744f92899def2b`; frozen behavior implementation
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; no `codes/` drift between them.
  The commit containing this amendment and pending declaration must be the
  exact clean launch HEAD. No tracked edit may intervene.
- Canonical static resolution: the exact command resolved profile/scope/source
  as `baby_student_reference_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference`; `dataset_config_overrides={}`;
  `student_config_overrides={}`; student-identity blockers `[]`; `seed=2022`;
  `hard_token_seed=2022`; `td_distill_no_projection`; dimension `64`; random
  initialization; no teacher warm start; AdamW `student_lr=6e-5`; weight decay
  `0.01`; batch `1024`; epoch `1000`; patience `7`; no smoke cap; final Test
  enabled; and efficiency disabled.
- Baseline loss contract: BPR only. `td_distill_alpha=0.0` and item-image,
  item-text, user-image, and user-text rates are all `0.0`; no semantic head,
  projection head, or teacher initialization may affect the student.
- Environment and startup evidence: intended interpreter
  `D:\miniconda\envs\run_5060\python.exe`; Python `3.10.20`; PyTorch
  `2.11.0+cu128`; CUDA runtime `12.8`; NVIDIA driver `595.97`; NVIDIA GeForce
  RTX 5060 with `8151` MiB; GPU selector `0`; approximately `405.7 GB` free on
  drive D during declaration checks. The teacher, all six dataset assets, and
  both PCA cache hashes matched the amendment, and the training-process count
  was `0`. These gates must be repeated immediately before launch.
- Evaluation chronology and hard acceptance: dataset/preflight and frozen
  teacher validation must pass; all loss and Validation series must be finite;
  every completed epoch must use all `116` batches; Validation Recall@20 alone
  must save/select and naturally early-stop; the Validation-best full
  checkpoint must be restored; frozen teacher Test ranking count must be zero;
  student final Test ranking count must be exactly one and only after restore.
  The manifest must report `status=completed`, protocol `val_test_once_v1`,
  selection `validation` / `Recall@20`, exclusion `train_only`, both override
  maps `{}`, `paper_ready_eligible=true`, blockers `[]`,
  `teacher_final_test_performed=false`, policy
  `frozen_checkpoint_reuse_no_test_ranking`, and
  `final_test_performed=true`, with no current-run teacher final-Test result.
- Planned isolated artifacts, all sharing one new timestamp/PID run identity:
  raw log `logs/<run_name>`; dataset preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; run manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  All six must exist, share the run identity, and receive byte counts and SHA256
  fingerprints. The teacher hash must remain unchanged.
- Checkpoint/deployment acceptance: the full checkpoint must contain finite
  user/item ID embeddings plus the declared optimizer/metadata state. The
  inference-only checkpoint must contain exactly finite user/item ID embeddings
  plus dimension/count/no-projection deployment metadata and no optimizer,
  teacher, modality, prompt, semantic cache, graph, or projection state. Full
  and inference embedding tensors must be shape/dtype identical and bit-for-bit
  equal.
- Quality, preservation, and next gate: no minimum Validation or Test metric is
  required. A finite protocol-compliant low result remains `completed` and is
  never a retry or rollback trigger. After natural exit, audit the raw log,
  preflight, manifest, convergence, full checkpoint, and inference-only
  checkpoint, append and commit one completed or failed baseline outcome, and
  then apply the amendment's candidate gate exactly.

## 2026-08-02 baby_student_reference_v1 matched recovery formal run (completed)

- Status and decision: completed successfully as protocol-compliant replacement
  seed-2022 baseline evidence. The single authorized process exited naturally
  with code `0`; every source, identity, frozen-teacher Test-isolation,
  selection/Test chronology, numerical, artifact, and deployment hard gate
  passed. No retry, second launch, parameter change, rollback, reset, deletion,
  image-only run, or other experiment occurred.
- Source and launch identity: branch
  `codex/experiment/baby-teacher-baseline`; exact clean amendment/baseline
  declaration and launch commit
  `6cac3434b41c46741151e452de9b62067969953f`; corrected frozen-teacher
  implementation `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; chronology-audit
  start commit `1d24afea6e7aaf1653ad952179744f92899def2b`. Pre-launch
  `git diff --exit-code ee7b362... HEAD -- codes` passed, the worktree was
  clean, and the run did not modify tracked source.
- Executed command, exactly once and without any added override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Transport, PID, run identity, and duration: the command remained in one
  continuously attached foreground shell/tool cell for its entire lifetime,
  with timeout explicitly `10800000` ms. It was not detached, parallelized, or
  relaunched. Main PID `17532`; run
  `2026-08-02 11_06_37.355218_baby_light_init_pid17532`; foreground wall time
  approximately `2897.4` seconds. Manifest start/completion were
  `2026-08-02T11:06:37.356218+08:00` and
  `2026-08-02T11:54:49.445552+08:00`, elapsed `2892.089334` seconds.
- Environment and canonical controls: Python `3.10.20`; PyTorch
  `2.11.0+cu128`; CUDA runtime `12.8`; driver `595.97`; NVIDIA GeForce RTX
  5060; GPU `0`. Profile/scope/source are exactly
  `baby_student_reference_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference`; dataset and student override maps
  are `{}`; seed and `hard_token_seed` are both `2022`; model is random
  64-dimensional `td_distill_no_projection` with
  `td_init_from_teacher=false`; AdamW `student_lr=6e-5`, weight decay `0.01`;
  batch `1024`; epoch budget `1000`; patience `7`; no smoke cap or efficiency
  benchmark. Alpha and all four semantic rates remained `0.0`, so total and BPR
  losses are identical and all semantic series are exactly zero.
- Dataset, preprocessing, and preflight audit: conversion manifest, all three
  split matrices, image/text features, and both PCA cache identities/hashes
  matched the amendment. Split overlaps are `0 / 0 / 0`; modalities are finite,
  distinct, and non-duplicate; both PCA caches were hits at random state
  `2022`. The only preflight warning is the preserved official condition for
  items `240`, `1212`, and `6115`, covering `11` Validation and `7` Test
  interactions with no cold users. No data, split, preprocessing, or cache
  asset changed.
- Frozen-teacher Test-isolation audit: teacher training was skipped and
  `Model/baby/teacher_model_val_test_once_v1.pt` remained `141098540` bytes,
  last-write time `2026-07-30T11:57:18.0973235+08:00`, and SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  Raw log contains zero `Teacher reuse summary` and zero teacher final-Test
  events, plus exactly one frozen-reuse/no-ranking message. Manifest records
  `teacher_final_test_performed=false` and
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`; it has no
  `teacher_final_test_result`. Because the checkpoint carries no top-level
  final-Test vector, `teacher_checkpoint_historical_final_test_metadata` is
  `null`; existing teacher selection/identity values remain checkpoint
  metadata only. Frozen-teacher Test ranking count is exactly `0`.
- Training and Validation chronology: epochs `0-170` completed contiguously,
  each through all `116` batches in the attached progress stream, for `171`
  Validation evaluations. All 15 convergence loss/selection series have 171
  finite entries. Total/BPR loss changed from `80.3966903090477` to
  `36.4888471364975`. Validation Recall@20 alone selected best epoch `163`;
  epochs `164-170` formed the final seven consecutive non-improvements and the
  raw log then records one natural patience `7/7` early stop.
- Exact Validation-best metrics at epoch `163`: Recall@20
  `0.04291303921928788`; Recall@50 `0.08005846771724855`; NDCG@20
  `0.01879878388702351`; NDCG@50 `0.026503690746942317`. Manifest,
  convergence, full checkpoint, and raw-log rounded vector agree on the best
  epoch and selector.
- Student Test chronology and exact result: only after natural early stopping,
  the run restored Validation-best epoch `163` and performed exactly one final
  student Test ranking. Exact vectors ordered by K `[10,20,40,50]` are
  Precision
  `[0.002993057341218855, 0.0024787863203908794, 0.0019940858832605012, 0.0018441758806892075]`,
  Recall
  `[0.026797499663274653, 0.044279857310199594, 0.07085836323043866, 0.08179702900453686]`,
  NDCG
  `[0.015145598433562857, 0.019996624710057805, 0.02605878370945137, 0.02827561590531854]`,
  and Hit Ratio
  `[0.029724865003856682, 0.04906145538698948, 0.0783234764721007, 0.09040884546155713]`;
  AUC is `0.0` under `test_flag=part`. Test structure/identity was read during
  preflight; ranking accesses were frozen teacher `0` and final student `1`.
  No Test value influenced selection, stopping, acceptance, or parameters.
- Manifest acceptance: `status=completed`, protocol `val_test_once_v1`,
  selection split `validation`, primary metric `Recall@20`, candidate exclusion
  `train_only`, `paper_ready_eligible=true`, blockers `[]`, final Test enabled
  and performed, exact canonical profile/arguments, both override maps `{}`,
  correct checkpoint paths, and the frozen-teacher fields above. The manifest,
  raw log, preflight, convergence record, and checkpoints share one run
  identity.
- Six preserved run-specific artifacts, bytes, and SHA256:
  - raw log `logs/2026-08-02 11_06_37.355218_baby_light_init_pid17532`,
    `76554` bytes,
    `2b9ebb1b002c5755a59638084379a3fee3fed5fcfdf9ca865041dc2ab9e4caa0`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-08-02 11_06_37.355218_baby_light_init_pid17532.json`,
    `3399` bytes,
    `68a02e979391298dcab0b4da73c07879f7085798b00887ef15ee643142d0bbf6`;
  - manifest
    `exp/runs/baby/run_manifest__2026-08-02 11_06_37.355218_baby_light_init_pid17532.json`,
    `24904` bytes,
    `56ab1754b5785280ae3f1c837b7828ff5d659e7d315be2e8b37eeb71fe98a911`;
  - convergence
    `exp/converge/baby/auto__2026-08-02 11_06_37.355218_baby_light_init_pid17532.pkl`,
    `25280` bytes,
    `3a7a4a4e933e3b7a33fd1a251ab28b81b98ccda5c878b80c8fee61d20dc86a6c`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-02 11_06_37.355218_baby_light_init_pid17532.pth`,
    `20354975` bytes,
    `a92fa9d1b070e7e69ade2b80d328b417ea0c0e0b085d8bd324b0936ba380df9d`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-02 11_06_37.355218_baby_light_init_pid17532.pth`,
    `6785997` bytes,
    `7a44b83ddb2aba3da6a91f05477febd06acc237f48ee5e886a2df875b4e311ec`.
- Checkpoint/deployment audit: format-v2 full checkpoint model state contains
  only finite `user_id_embedding.weight` shape `[19445,64]` and
  `item_id_embedding.weight` shape `[7050,64]`, plus exactly two finite
  optimizer states and declared metadata. Inference export contains exactly
  those two finite embeddings plus `embedding_dim=64`, `n_users=19445`,
  `n_items=7050`, and variant `td_distill_no_projection`; it contains no
  optimizer, teacher, modality, prompt, semantic cache, graph, projection, or
  other training state. Full/inference tensors have identical shapes/dtypes
  and are bit-for-bit equal.
- Verification evidence and repository state: foreground process exit `0`;
  raw event-count/order audit; structured manifest/preflight assertions;
  convergence length/finiteness/argmax/result equality assertions; checkpoint
  key/shape/finiteness/optimizer/deployment/exact-equality assertions; artifact
  hashes; teacher preservation; source/status checks; and final process count
  `0` all passed. The first read-only structured audit wrapper compared NumPy
  vectors directly and raised an ambiguous-truth `ValueError`; the corrected
  element-wise wrapper passed `BASELINE_AUDIT_OK` without modifying an artifact
  or re-running any training/evaluation.
- Experimental meaning and remaining risks: the corrected-source replacement
  reproduces the old seed-2022 baseline numerical trajectory and Test values
  while removing the prohibited frozen-teacher Test ranking, so it restores
  paper-ready protocol eligibility for this baseline. It does not validate the
  candidate, erase old diagnostic evidence, estimate multi-seed variance, or
  resolve the official cold-item and ignored-artifact backup risks.
- Next gate under the committed amendment: after committing this outcome and
  confirming a clean tree, append and commit the exact
  `baby_td_asymmetric_no_projection_v1` seed-2022 recovery formal pending
  declaration, then launch that candidate once using the same source,
  foreground timeout, identity, Test-isolation, and audit requirements. Do not
  run image-only or any other seed/stage.

## 2026-08-02 baby_td_asymmetric_no_projection_v1 matched recovery formal run (pending)

- Purpose and hypothesis: execute one replacement formal seed-2022 asymmetric
  no-projection candidate under the corrected frozen-teacher Test-isolation
  implementation. This recovery exists solely because the old run performed a
  prohibited frozen-teacher Test ranking. The already frozen hypothesis remains
  that item-dominant image/text semantic transfer improves the matched ID-only
  student; neither the old candidate metrics nor the newly recovered baseline
  metrics selected or changed any candidate value.
- Baseline prerequisite and authorization: the corrected-source matched
  `baby_student_reference_v1` baseline completed all hard gates from launch
  commit `6cac3434b41c46741151e452de9b62067969953f`; its completed outcome is
  committed at `3bf22b6a6a1460fc99ca69b5b497706a22e6ee64`. It selected epoch
  `163`, with Validation Recall@20 `0.04291303921928788` and final Test
  Recall@20 `0.044279857310199594`. Those values satisfy only the serial
  baseline-outcome gate and did not tune, cancel, reorder, or alter the
  candidate.
- Status and execution boundary: pending formal uncapped candidate, authorized
  for exactly one launch after this declaration is committed and the tree is
  clean. A process failure, connection interruption, or hard-gate violation
  requires preservation plus one committed failed outcome and immediate stop;
  no retry or second process is allowed. No image-only, seed-2023/2024,
  efficiency, smoke, tag, bundle, backup, or merge is authorized.
- Exact command, frozen verbatim and containing no CLI seed, semantic,
  optimizer, budget, or protocol override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Source and clean-launch gate: branch
  `codex/experiment/baby-teacher-baseline`; exact clean declaration parent and
  completed baseline outcome `3bf22b6a6a1460fc99ca69b5b497706a22e6ee64`;
  frozen Test-isolation implementation
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`. The commit containing this
  pending declaration must be the exact launch HEAD, and
  `git diff --exit-code ee7b362... <launch-commit> -- codes` must remain empty.
- Canonical profile and override contract: exact name/scope/source
  `baby_td_asymmetric_no_projection_v1` / `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1`;
  `dataset_config_overrides={}`; `student_config_overrides={}`; profile
  blockers `[]`; student/sampling seed `2022`; PCA/cache
  `hard_token_seed=2022`; `td_distill_no_projection`; dimension `64`; random
  initialization; `td_init_from_teacher=false`; no loaded student checkpoint,
  projection head, or teacher warm start.
- Frozen optimizer, sampling, and budget: AdamW `student_lr=6e-5`; student
  weight decay `0.01`; batch size `1024`; maximum epoch `1000`; Validation
  every epoch; early-stopping patience `7`; `smoke_train_batches=0`; unchanged
  pairwise BPR sampler; all `116` batches per completed epoch;
  `run_efficiency_benchmark=false`. No parameter or command differs from the
  original candidate profile.
- Frozen semantic controls and objective: `td_distill_alpha=0.3`;
  item-image `1.0`; item-text `0.3`; user-image `0.0`; user-text `0.0`.
  Active heads must be exactly item-image and item-text, optimizing
  `L_BPR + 0.3 * (L_item_image + 0.3 * L_item_text) / 1.3`. The only behavioral
  delta from the recovered baseline remains alpha, item-image, and item-text;
  every non-semantic control is matched.
- Frozen data/preprocessing identity: audited MMRec Baby conversion manifest
  SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test hashes
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text hashes
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`;
  frozen image/text PCA cache SHA256
  `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`
  and `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
  No split, feature, cache, sampler, or preprocessing change is allowed.
- Frozen teacher and Test-isolation contract: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, `141098540` bytes, SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, no alias overwrite, no teacher mutation, and zero
  frozen-teacher Test rankings. Manifest must record
  `teacher_final_test_performed=false`, policy
  `frozen_checkpoint_reuse_no_test_ranking`, no current-run
  `teacher_final_test_result`, and historical field `null` for this checkpoint
  or, if checkpoint-carried metadata exists, only
  `performed_by_current_run=false`.
- Evaluation and ranking chronology: unchanged `val_test_once_v1`; train only
  on `train_mat`; Validation Recall@20 alone selects and early-stops;
  candidate exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`.
  Preflight may access Test structure/identity. The Validation-best full
  checkpoint must be restored after natural stop, followed by exactly one
  student final Test ranking. Test must not affect selection, acceptance,
  parameter interpretation, retry, or rollback.
- Execution transport: launch the command once in one continuously attached
  foreground shell/tool session with timeout explicitly at least `10800000`
  ms. Detached/background launch, output-channel closure while active,
  parallel execution, a second launch, and automatic retry are forbidden.
- Planned six same-run artifacts: raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  All must share one run identity, remain outside Git, and be fingerprinted.
- Hard acceptance: exact clean source/profile/empty-overrides/identity gates;
  matching environment/data/cache/teacher; finite total/BPR/directional/
  component and Validation series; all `116` batches per completed epoch;
  active semantic heads exactly item-image/text; Validation Recall@20-only
  selection; best-checkpoint restore; frozen teacher Test count `0`; student
  final Test count `1`; manifest `status=completed`, paper-ready true, blockers
  `[]`, final Test performed; unchanged teacher hash; all six artifacts; finite
  full/inference checkpoints; ID-only inference structure; and bit-for-bit
  embedding equality.
- Quality and preservation rule: there is no minimum metric, improvement, or
  direction threshold. A low or reversed protocol-compliant result is
  `completed` evidence and cannot trigger parameter changes, retry, rollback,
  reset, deletion, or suppression. Preserve the prior diagnostic candidate and
  all new files on every outcome.
- Next action: commit this declaration only, repeat the clean launch/profile/
  process/environment/data/cache/teacher gates, execute the exact command once
  through the required foreground session, audit all six artifacts and Test
  chronology after natural exit, append and commit one completed or failed
  outcome, and stop.

## 2026-08-02 baby_td_asymmetric_no_projection_v1 matched recovery formal run (completed)

- Outcome and scope: completed exactly one seed-2022 replacement candidate run
  authorized by the matched recovery amendment. This recovery was required
  solely by the confirmed frozen-teacher Test-isolation defect; no historical
  metric, comparison, or direction was used to change parameters, retry, or
  roll back. The prior diagnostic candidate and all old declarations/results
  remain preserved.
- Launch identity and command: branch
  `codex/experiment/baby-teacher-baseline`, clean launch HEAD
  `df7595a230bfeecf0ad402df26a7115f0bde88d9`, frozen implementation baseline
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`, and empty
  `git diff --exit-code ee7b362... <launch-HEAD> -- codes`. The attached
  foreground command was executed once with a `10800000 ms` tool timeout and
  exit code `0`:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
  Run identity:
  `2026-08-02 12_05_26.329228_baby_light_init_pid27620`.
- Protocol and resolved identity: `val_test_once_v1`, seed `2022`,
  `hard_token_seed=2022`, `baby_td_asymmetric_no_projection_v1`,
  `td_distill_no_projection`, dimension `64`, random initialization,
  `td_init_from_teacher=false`, AdamW `student_lr=6e-5`, weight decay
  `0.01`, batch size `1024`, epoch cap `1000`, patience `7`, all `116`
  batches per completed epoch, `smoke_train_batches=0`, and
  `run_efficiency_benchmark=false`. Empty dataset/student override maps and
  the canonical `baby_teacher_reference_v1` teacher-only profile were
  preserved. Active semantic heads were exactly item-image and item-text with
  rates `1.0` and `0.3`; user rates were `0.0` and `0.0`.
- Data, cache, and teacher identity: conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test SHA256
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text feature SHA256
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`,
  `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`;
  image/text PCA cache SHA256
  `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`,
  `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
  Frozen teacher remained read-only, `141098540` bytes, SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  The official cold-start warning remains: Validation has `11` and Test `7`
  interactions involving items `240`, `1212`, and `6115` absent from train;
  no split overlap or data asset changed.
- Test isolation and chronology: teacher training was skipped; raw log has
  zero `Teacher reuse summary` events, exactly one frozen-checkpoint message
  stating teacher Test ranking is prohibited, and zero frozen-teacher Test
  rankings. Manifest records
  `teacher_final_test_performed=false`,
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`, no current
  teacher final-Test result, and null checkpoint-carried final-Test metadata.
  Validation epochs `0-80` are contiguous (`81` evaluations), all convergence
  series are finite, and natural patience reached `7/7`. Validation
  `Recall@20` alone selected epoch `73` at
  `0.06842596333982363`; the best checkpoint was restored and exactly one
  student final Test ranking followed. Student final Test Recall@20 was
  `0.06659234097521655` (Precision `[0.004674723579326367,
  0.0036616096682952452, 0.0028683466186679294,
  0.0026320390845977456]`; Recall
  `[0.0421596544365896, 0.06659234097521655, 0.1041240614189935,
  0.11925134349664723]`; NDCG `[0.024242770942206778,
  0.03081209766712305, 0.03912474568059561, 0.042142503473832]`; Hit Ratio
  `[0.046490100282849466, 0.07271792234507622, 0.11313962458215178,
  0.12959629724864688]`; AUC `0.0`). No Test value affected selection,
  stopping, acceptance, parameters, or retry policy.
- Manifest and checkpoint gates: manifest is `status=completed`,
  `paper_ready_eligible=true`, blockers `[]`, final Test enabled/performed,
  and contains the canonical profile, empty overrides, teacher policy, and
  run paths. The format-v2 full checkpoint has exactly finite
  `user_id_embedding.weight` `[19445,64]` and
  `item_id_embedding.weight` `[7050,64]` plus exactly two optimizer states.
  Inference-only checkpoint has exactly those finite embeddings plus
  `embedding_dim=64`, `n_users=19445`, `n_items=7050`, and variant
  `td_distill_no_projection`; it has no optimizer, teacher, modality, prompt,
  cache, graph, or projection state. Full/inference tensors are bit-for-bit
  identical.
- Six same-run artifacts and SHA256: raw log
  `logs/2026-08-02 12_05_26.329228_baby_light_init_pid27620`
  `cc684c58a93b9ec1811562ef459e2411bbb6feb7dd89b6a3427f1560802b5156`;
  preflight
  `exp/runs/baby/dataset_preflight__2026-08-02 12_05_26.329228_baby_light_init_pid27620.json`
  `8be646a7b71dc787c3146e8077b6163a446da6f833f2e3b0ec1c41d0285f073c`;
  manifest
  `exp/runs/baby/run_manifest__2026-08-02 12_05_26.329228_baby_light_init_pid27620.json`
  `15bbc9b33594e6b428f2d91dd8a443790d20e9b849bec199dc44e082b946b356`;
  convergence
  `exp/converge/baby/auto__2026-08-02 12_05_26.329228_baby_light_init_pid27620.pkl`
  `bf0c521c9118256b133ec9cfe813af9632ef705b1da32e9801bd2df143e587e0`;
  full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-02 12_05_26.329228_baby_light_init_pid27620.pth`
  `5b3c0da2524a3d38a823f750d62e14245056cb05247d07f56e8dbc9224cab370`;
  inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-02 12_05_26.329228_baby_light_init_pid27620.pth`
  `e7aa155b44526cd172e4c25e5550cc70d356bbee4971a15b3fcaa8cd3371db18`.
- Verification and meaning: corrected structured audit passed
  (`CANDIDATE_AUDIT_OK`), including raw event counts/order, manifest and
  preflight identity, finite 15-series convergence and argmax/result equality,
  checkpoint key/shape/optimizer/deployment/exact-equality checks, artifact
  hashes, teacher preservation, source diff, environment, and final training
  process count `0`. Test split structure/identity was read in preflight and
  the Test ranking protocol was compliant: frozen teacher `0`, student `1`.
  This is valid protocol-compliant candidate evidence regardless of the
  candidate's lower Test Recall@20 than its Validation selection value. No
  unresolved execution blocker remains; official cold-start and ignored-large-
  artifact backup risks remain documented repository risks.
- Next action: wait for explicit user authorization before any further
  experiment or stage; do not run another seed, ablation, efficiency test,
  tag, bundle, backup, or merge in this task.

## 2026-08-02 matched seed-2023 baseline/candidate recovery execution amendment (pending)

- Purpose and recovery rationale: replace only the protocol-noncompliant
  seed-2023 main-experiment baseline/candidate pair after the retrospective
  audit proved that both pre-fix frozen-teacher reuse paths performed a
  prohibited new teacher Test ranking. This recovery is authorized solely by
  that confirmed Test-isolation implementation defect. It is not motivated by
  either old run's Validation/Test values, candidate direction, or any quality
  threshold, and it does not tune, reinterpret, delete, overwrite, or replace
  the preserved old declarations, metrics, raw logs, manifests, convergence
  records, or checkpoints.
- Preserved diagnostic evidence: the old baseline run
  `2026-08-01 20_20_35.101504_baby_light_init_pid29128` selected epoch `72`
  with Validation Recall@20 `0.04098274743170617` and Test Recall@20
  `0.03680654207653432`; the old candidate run
  `2026-08-01 21_09_59.190542_baby_light_init_pid33928` selected epoch `44`
  with Validation Recall@20 `0.06479423036892924` and Test Recall@20
  `0.06531066330886427`. Both remain protocol-noncompliant diagnostic
  evidence because each ranked frozen-teacher Test before student training;
  none of these values changes this recovery's commands or parameters.
- Status and amended authorization: pending serial recovery execution. The
  user explicitly authorizes both matched seed-2023 main-experiment arms in
  this one task, strictly baseline first. Append/commit this amendment and the
  baseline pending declaration, run/audit the baseline once, and commit its
  completed or failed outcome. Only if that outcome is `completed`, all hard
  gates pass, and its outcome commit exists may the candidate pending record
  be appended/committed and the candidate launched. A baseline failure,
  connection interruption, blocker, or hard-gate violation requires a
  committed failed outcome and immediate stop with no candidate declaration,
  launch, or retry.
- Exact branch and source identity: branch
  `codex/experiment/baby-teacher-baseline` began clean at exact commit
  `7d60c61a38f54b186b3a4534763fd640ed857dac`. Frozen-teacher Test-isolation
  implementation commit is
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; startup
  `git diff --exit-code ee7b362... 7d60c61... -- codes` was empty, so the
  intervening recovery/audit history introduces no post-fix behavior drift.
  The commit containing this amendment and baseline declaration must be the
  exact clean baseline launch HEAD.
- Recovery scope: recover only `baby_student_reference_seed2023_v1` and, if
  the serial gate opens, `baby_td_asymmetric_no_projection_seed2023_v1`. Do
  not run or declare image-only, seed `2024`, another seed, efficiency, smoke,
  broad tuning, or any other experiment. Do not modify code, parser/profile
  defaults, parameters, data, split, preprocessing, PCA caches, teacher,
  checkpoint format, sampler, optimizer, evaluation policy, or frozen
  commands. Do not create a tag, bundle, backup, or merge to `main`.
- Frozen common controls: audited MMRec Baby; `val_test_once_v1`; train only
  on `train_mat`; Validation Recall@20 alone selects and early-stops;
  candidate exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`;
  random 64-dimensional `td_distill_no_projection` initialization;
  `td_init_from_teacher=false`; AdamW `student_lr=6e-5`; student weight decay
  `0.01`; batch size `1024`; maximum epoch `1000`; Validation every epoch;
  patience `7`; all `116` batches per completed epoch;
  `smoke_train_batches=0`; `run_efficiency_benchmark=false`; student and
  sampling `seed=2023`; frozen preprocessing/cache `hard_token_seed=2022`;
  no student checkpoint; and canonical commands with empty dataset/student
  override maps.
- Frozen data/preprocessing identity: conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test hashes
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text hashes
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Frozen image/text PCA caches remain random state `2022`, identities
  `7634bb6e8dcfdbdd45eee436cc3b8e3f3f8e1e54c85c2f4b9f4a17719cc77c90`
  and `6e7c8161aaeb01f5e4bb8744be3bf9d16df71cd7cc74ea841bc5c20ad60926ee`,
  with SHA256 `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`
  and `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
- Frozen teacher and environment: reuse only read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, `141098540` bytes, SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. Teacher training, optimizer steps,
  mutation, publication, alias overwrite, run-local teacher checkpoints, and
  frozen-teacher Test ranking are prohibited. Intended environment is
  `D:\miniconda\envs\run_5060\python.exe`, Python `3.10.20`, PyTorch
  `2.11.0+cu128`, CUDA runtime `12.8`, driver `595.97`, NVIDIA GeForce RTX
  5060 with `8151` MiB, GPU `0`; startup training-process count was `0`.
- Test-isolation and execution contract: preflight may read Test structure and
  identity, but frozen-teacher reuse must rank Test zero times and record
  `teacher_final_test_performed=false` plus
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`. A historical
  checkpoint-carried final-Test vector, if any, may appear only under
  `teacher_checkpoint_historical_final_test_metadata` with
  `performed_by_current_run=false`; a current-run
  `teacher_final_test_result` is forbidden. Each student must select and
  restore only its Validation-best checkpoint, then rank final Test exactly
  once. Every formal command must run once in one continuously attached
  foreground shell/tool session with timeout at least `10800000` ms. Parallel
  launch, detached/background execution, output closure while active,
  automatic retry, and a second process are forbidden.
- Acceptance, quality, and preservation: hard gates cover exact clean source,
  canonical profile, empty overrides/blockers, environment/data/cache/teacher
  identity, finite loss and Validation series, all `116` batches, Validation
  Recall@20-only selection, natural stop, best-checkpoint restore, teacher Test
  count `0`, student Test count `1`, completed/paper-ready manifest fields,
  six same-run artifacts, finite checkpoint structure, ID-only inference
  export, and bit-for-bit full/inference embedding equality. There is no
  minimum metric or improvement threshold. Low or reversed compliant results
  remain completed evidence and cannot trigger parameter changes, retry,
  rollback, reset, deletion, or suppression. Preserve all old/new artifacts.

## 2026-08-02 baby_student_reference_seed2023_v1 matched recovery formal run (pending)

- Purpose and hypothesis: execute one replacement formal seed-2023 ID-only
  BPR baseline under the corrected frozen-teacher Test-isolation source. The
  goal is protocol recovery, not metric improvement; every method and runtime
  parameter remains the previously frozen
  `baby_student_reference_seed2023_v1` contract.
- Status and authorization: pending formal uncapped baseline, authorized for
  exactly one launch after this amendment/declaration commit is clean. The
  candidate is not yet declared. It remains blocked until this run naturally
  exits, passes the full protocol/artifact audit, and its `completed` outcome
  commit exists. A failure requires a committed failed outcome and immediate
  stop without retry.
- Exact command, frozen verbatim with no CLI seed, semantic, optimizer,
  budget, or protocol override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Source launch gate: branch
  `codex/experiment/baby-teacher-baseline`; pre-declaration clean HEAD
  `7d60c61a38f54b186b3a4534763fd640ed857dac`; fixed implementation
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; no `codes/` drift between
  them. The commit containing this amendment and pending declaration must be
  the exact clean launch HEAD, with no intervening tracked edit.
- Canonical static resolution: the exact command resolves profile/scope/source
  `baby_student_reference_seed2023_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2023`;
  `dataset_config_overrides={}`; `student_config_overrides={}`; profile
  blockers `[]`; training/sampling seed `2023`;
  `hard_token_seed=2022`; `td_distill_no_projection`; dimension `64`; random
  initialization; `td_init_from_teacher=false`; AdamW `student_lr=6e-5`;
  weight decay `0.01`; batch `1024`; epoch `1000`; patience `7`; no smoke cap;
  final Test enabled; and efficiency disabled.
- Baseline objective: BPR only. `td_distill_alpha=0.0` and item-image,
  item-text, user-image, and user-text rates are all `0.0`; no semantic head,
  projection head, loaded student checkpoint, or teacher warm start may affect
  the student.
- Evaluation and hard acceptance: pass preflight and frozen-teacher validation;
  every loss/Validation series value is finite; every completed epoch uses all
  `116` batches; Validation Recall@20 alone saves/selects and naturally
  early-stops; restore the Validation-best full checkpoint; frozen teacher Test
  ranking count is `0`; student final Test ranking count is exactly `1` and
  occurs only after restore. Manifest must report `status=completed`, protocol
  `val_test_once_v1`, selection `validation` / `Recall@20`, exclusion
  `train_only`, both override maps `{}`, `paper_ready_eligible=true`, blockers
  `[]`, `teacher_final_test_performed=false`, policy
  `frozen_checkpoint_reuse_no_test_ranking`, and
  `final_test_performed=true`, with no current-run teacher final-Test result.
- Six planned isolated artifacts, all sharing one new timestamp/PID identity:
  raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  All six must exist, share the run identity, and receive byte counts and
  SHA256 fingerprints; the teacher hash must remain unchanged.
- Checkpoint/deployment acceptance: the full checkpoint must contain finite
  user/item ID embeddings plus declared optimizer/metadata state. The
  inference-only checkpoint must contain exactly finite user/item ID embeddings
  plus dimension/count/no-projection deployment metadata and no optimizer,
  teacher, modality, prompt, semantic cache, graph, or projection state. Full
  and inference embeddings must have identical shape/dtype and be bit-for-bit
  equal.
- Next gate: after the single foreground process exits, audit raw log,
  preflight, manifest, convergence, full checkpoint, inference checkpoint,
  frozen teacher, Test chronology, process state, and source/worktree status;
  append and commit one completed or failed baseline outcome. Only a committed
  completed outcome opens the amendment's candidate gate.

## 2026-08-02 baby_student_reference_seed2023_v1 matched recovery formal run (completed)

- Status and outcome: completed successfully as the protocol-compliant
  replacement seed-2023 baseline. The single authorized process exited
  naturally with code `0`; no second launch, retry, parameter change, rollback,
  reset, deletion, image-only run, seed-2024 run, or other experiment occurred.
  This recovery exists solely to remove the confirmed pre-fix frozen-teacher
  Test-isolation defect; its metric magnitude did not drive acceptance or any
  later candidate value.
- Source and launch identity: branch
  `codex/experiment/baby-teacher-baseline`; exact clean amendment/baseline
  declaration and launch commit
  `0df1797954967d2b72635f5f599d0243444f0a47`; fixed Test-isolation source
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; recovery start commit
  `7d60c61a38f54b186b3a4534763fd640ed857dac`. Pre-launch
  `git diff --exit-code ee7b362... HEAD -- codes` was empty, the worktree was
  clean, and tracked source did not change during the run.
- Executed command exactly once, with no CLI seed or parameter override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Transport, PID, and run identity: the command remained in one continuously
  attached foreground session with explicit timeout `10800000 ms`, was not
  detached or relaunched, and used main PID `36572`. Run identity:
  `2026-08-02 12_48_21.776381_baby_light_init_pid36572`. The foreground cell
  returned exit code `0` after approximately `1062.1` seconds.
- Resolved profile and controls: profile/scope/source are exactly
  `baby_student_reference_seed2023_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2023`; both override maps are
  `{}` and profile blockers are `[]`. Student/training seed is `2023`, frozen
  PCA/cache `hard_token_seed=2022`, model is random 64-dimensional
  `td_distill_no_projection`, `td_init_from_teacher=false`, no student
  checkpoint or teacher warm start, AdamW `student_lr=6e-5`, weight decay
  `0.01`, batch `1024`, epoch cap `1000`, patience `7`,
  `smoke_train_batches=0`, and efficiency benchmark disabled. BPR-only
  controls are `td_distill_alpha=0.0` and all four semantic rates `0.0`.
- Dataset and preflight identity: conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test hashes
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text hashes
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`,
  `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`;
  PCA cache hashes
  `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`,
  `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
  Split overlaps remained `0 / 0 / 0`; modalities were finite, distinct,
  and non-duplicate. The only warning was the preserved official cold-item
  condition: items `240`, `1212`, and `6115` cover `11` Validation and `7`
  Test interactions, with no cold users.
- Frozen-teacher Test-isolation audit: teacher training was skipped and the
  read-only teacher remained `141098540` bytes with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  Raw log contains zero `Teacher reuse summary`, zero teacher Test-ranking
  events, and exactly one frozen-reuse/no-ranking message. Manifest records
  `teacher_final_test_performed=false`,
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`, no current
  `teacher_final_test_result`, and null
  `teacher_checkpoint_historical_final_test_metadata`. Frozen teacher Test
  ranking count is exactly `0`.
- Training, Validation, and final Test: epochs `0-79` completed contiguously,
  each through all `116` batches, for `80` Validation evaluations. All `15`
  convergence loss/selection series entries were finite; total and BPR loss
  were identical because semantic loss was disabled. Natural patience reached
  `7/7`. Validation Recall@20 alone selected epoch `72` at
  `0.04098274743170617`; epochs `73-79` were the final non-improvements. The
  selected checkpoint was restored and exactly one student final Test ranking
  followed. Exact final Test vectors ordered by K `[10,20,40,50]`: Precision
  `[0.0024993571612239892, 0.002057084083311931, 0.0016469529442016179,
  0.0015407559784006715]`; Recall
  `[0.02253427453093155, 0.03680654207653432, 0.05880417887874913,
  0.06895894996692173]`; NDCG
  `[0.013025226314658288, 0.017025150092128044, 0.02203587686177423,
  0.024038280300640525]`; Hit Ratio
  `[0.024839290305990986, 0.040678837747493035, 0.06500385703265722,
  0.076009256878375]`; AUC `0.0`.
- Manifest and checkpoint acceptance: manifest is `status=completed`,
  `evaluation_protocol=val_test_once_v1`, selection `validation` /
  `Recall@20`, exclusion `train_only`, `paper_ready_eligible=true`, blockers
  `[]`, `run_final_test=true`, and `final_test_performed=true`. Full model
  state contains only finite user/item ID embeddings with shapes
  `[19445,64]` and `[7050,64]`, plus exactly two optimizer-state entries and
  declared metadata. Inference-only state contains exactly those two finite
  embeddings plus dimension/count/no-projection metadata, with no optimizer,
  teacher, modality, prompt, cache, graph, projection, or other training
  state. Full/inference tensors are bit-for-bit equal.
- Six same-run artifacts, bytes, and SHA256:
  - raw log `logs/2026-08-02 12_48_21.776381_baby_light_init_pid36572`,
    `38789` bytes,
    `6bdb329c10b15dd48f6976a728e70232e14362524d129e389696da9c01223d15`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-08-02 12_48_21.776381_baby_light_init_pid36572.json`,
    `3399` bytes,
    `b6bd00847ef253a92af07ea162afb9a56903803fe68299be6f65934d38b7b28b`;
  - manifest
    `exp/runs/baby/run_manifest__2026-08-02 12_48_21.776381_baby_light_init_pid36572.json`,
    `24947` bytes,
    `e5fbc3f0136825d4687af12add16f465ae6f6bbad3fe5935be9654e5ffcfdadb`;
  - convergence
    `exp/converge/baby/auto__2026-08-02 12_48_21.776381_baby_light_init_pid36572.pkl`,
    `12631` bytes,
    `6fab1e8d293eff9be9fe04705fd5238f56fdc4f6d3366e875174f2496423fab4`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-02 12_48_21.776381_baby_light_init_pid36572.pth`,
    `20354975` bytes,
    `b22b4f18c3893818cac17d70161c01a9702ad8f4c03c64c785b6958e75ae7166`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-02 12_48_21.776381_baby_light_init_pid36572.pth`,
    `6785997` bytes,
    `51bac1a42132a4823fdad78f0868032b4d95012c9b7e2ce71b8e0dc16ec4559b`.
- Verification and meaning: structured `BASELINE_AUDIT_OK` passed raw event
  counts/order, manifest/preflight identity, 80-epoch finite convergence and
  argmax/result equality, checkpoint keys/shapes/finiteness/optimizer/deploy-
  ment/exact-equality checks, teacher preservation, six hashes, source gate,
  and process exit. Test split structure/identity was read only in preflight;
  execution complied with the declared protocol using frozen teacher `0` and
  student final Test `1`. This is valid seed-2023 baseline evidence. The
  required baseline-outcome gate is now open solely because this completed
  outcome is committed; its metric does not tune or alter the candidate.
- Next gate: after this outcome commit and a clean-tree confirmation, append
  and commit the separate canonical `baby_td_asymmetric_no_projection_seed2023_v1`
  candidate pending declaration. Only then launch that candidate once under
  the same foreground timeout and Test-isolation audit rules.

## 2026-08-02 baby_td_asymmetric_no_projection_seed2023_v1 matched recovery formal run (pending)

- Purpose and hypothesis: execute one replacement formal seed-2023 asymmetric
  no-projection candidate under the corrected frozen-teacher Test-isolation
  implementation, matched to the completed recovered seed-2023 BPR baseline.
  This recovery is required solely by the confirmed historical
  frozen-teacher Test-ranking defect. The candidate's semantic settings were
  frozen before this baseline evidence and are not selected, tuned, cancelled,
  reordered, or changed by the baseline metric.
- Baseline prerequisite and authorization: recovered baseline
  `baby_student_reference_seed2023_v1` completed all hard gates; its outcome is
  committed at `9111fcf`. It selected epoch `72`, Validation Recall@20
  `0.04098274743170617`, and final Test Recall@20 `0.03680654207653432`.
  These values satisfy only the required baseline-before-candidate sequence
  gate and did not alter any candidate value. The old protocol-noncompliant
  candidate and its artifacts remain preserved diagnostic evidence.
- Status and execution boundary: pending formal uncapped candidate, authorized
  for exactly one launch after this declaration commit is clean. A process
  failure, connection interruption, or hard-gate violation requires preserving
  all state, appending and committing one failed outcome, and stopping. No
  retry or second process is allowed. No image-only, seed-2024, efficiency,
  smoke, tag, bundle, backup, merge, or other experiment is authorized.
- Exact command, frozen verbatim with no CLI seed, semantic, optimizer,
  budget, or protocol override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Source and clean-launch gate: branch
  `codex/experiment/baby-teacher-baseline`; completed baseline outcome
  `9111fcf`; fixed Test-isolation implementation
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; and no `codes/` behavior drift
  relative to that fix. The commit containing this pending declaration must be
  the exact clean launch HEAD; no tracked edit may intervene.
- Canonical profile and override contract: profile/scope/source exactly
  `baby_td_asymmetric_no_projection_seed2023_v1` / `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2023`;
  `dataset_config_overrides={}`; `student_config_overrides={}`; profile
  blockers `[]`; student/training/sampling seed `2023`; frozen PCA/cache
  `hard_token_seed=2022`; `td_distill_no_projection`; dimension `64`; random
  initialization; `td_init_from_teacher=false`; no loaded student checkpoint,
  projection, or teacher warm start.
- Frozen optimizer, sampling, and budget: AdamW `student_lr=6e-5`; student
  weight decay `0.01`; batch size `1024`; maximum epoch `1000`; Validation
  every epoch; patience `7`; `smoke_train_batches=0`; unchanged pairwise BPR
  sampler; all `116` batches per completed epoch; and
  `run_efficiency_benchmark=false`. No parameter or command differs from the
  independently frozen seed-2022 candidate method except the profile-owned
  student/sampling seed `2023`.
- Frozen semantic objective: `td_distill_alpha=0.3`; item-image rate `1.0`;
  item-text rate `0.3`; user-image and user-text rates `0.0`. Active heads must
  be exactly item-image and item-text, optimizing
  `L_BPR + 0.3 * ((1.0 * L_item_image + 0.3 * L_item_text) / 1.3)`. The only
  actual behavioral delta from the matched recovered baseline is the declared
  semantic set `td_distill_alpha`, `td_item_image_rate`, and
  `td_item_text_rate`; all non-semantic controls remain equal.
- Frozen data/preprocessing identity: audited MMRec Baby conversion manifest
  SHA256 `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test hashes
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text hashes
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`,
  `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`;
  image/text PCA cache SHA256
  `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`,
  `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
  No split, feature, preprocessing, cache, sampler, or data identity change
  is allowed.
- Frozen teacher and Test-isolation contract: reuse read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, exactly `141098540` bytes,
  SHA256 `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, no alias overwrite or mutation, and zero
  frozen-teacher Test rankings. Manifest must record
  `teacher_final_test_performed=false`, policy
  `frozen_checkpoint_reuse_no_test_ranking`, no current-run
  `teacher_final_test_result`, and historical metadata only under
  `teacher_checkpoint_historical_final_test_metadata` with
  `performed_by_current_run=false` if present. The current checkpoint carries
  no top-level final-Test vector, so `null` is expected.
- Evaluation chronology and acceptance: unchanged `val_test_once_v1`; train
  only on `train_mat`; select and early-stop only with Validation Recall@20;
  candidate exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`.
  Preflight may access Test structure/identity. After natural stop, restore
  the Validation-best full checkpoint and perform exactly one student final
  Test ranking. Test must not affect selection, acceptance, parameter
  interpretation, retry, rollback, or profile values. All losses and metric
  series must be finite, each completed epoch must reach `116/116`, and the
  manifest must finish `status=completed`, `paper_ready_eligible=true`,
  blockers `[]`, and `final_test_performed=true`.
- Planned six same-run artifacts: raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  All must share one new timestamp/PID identity, remain outside Git, and be
  fingerprinted with byte counts and SHA256.
- Checkpoint/deployment acceptance: full checkpoint must contain finite
  user/item ID embeddings plus declared optimizer/metadata state; inference
  export must contain exactly finite user/item ID embeddings plus
  dimension/count/no-projection metadata and no optimizer, teacher, modality,
  prompt, semantic cache, graph, projection, or other training state. Full and
  inference embedding tensors must have identical shape/dtype and be bit-for-
  bit equal.
- Quality and preservation rule: there is no minimum metric or improvement
  threshold. A low or direction-reversed protocol-compliant result is valid
  completed evidence and cannot trigger parameter changes, retry, rollback,
  reset, deletion, or suppression. Mark failed only for process/code failure
  or a hard acceptance violation; preserve every old and new artifact.
- Next action: commit this declaration only, confirm clean HEAD, process count,
  source diff, environment, data/cache identity, and teacher hash, then execute
  this exact command once in the required continuously attached foreground
  session with timeout at least `10800000 ms`. After natural exit, audit the
  six artifacts and Test chronology, append and commit one completed or failed
  candidate outcome, and stop.

## 2026-08-02 baby_td_asymmetric_no_projection_seed2023_v1 matched recovery formal run (completed)

- Status and outcome: completed successfully as the protocol-compliant
  replacement seed-2023 asymmetric no-projection candidate. The single
  authorized process exited naturally with code `0`; no second launch,
  automatic retry, parameter change, rollback, reset, deletion, image-only,
  seed-2024, efficiency, tag, bundle, backup, merge, or later experiment
  occurred. Acceptance is independent of metric direction or magnitude.
- Source and launch identity: branch
  `codex/experiment/baby-teacher-baseline`; exact clean candidate declaration
  and launch commit `80c8c732980cd920845ed82d0dfa3171ba2e9c49`;
  completed recovered baseline outcome `9111fcf`; fixed Test-isolation source
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`. Pre-launch source diff under
  `codes/` was empty, the worktree was clean, and tracked source did not change
  during execution or audit.
- Executed command exactly once, with no added seed or parameter override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_td_asymmetric_no_projection_seed2023_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Transport, PID, run identity, and duration: one continuously attached
  foreground shell/tool session with timeout `10800000 ms`; no detach,
  connection interruption, or relaunch. Main PID `4516`; run identity
  `2026-08-02 13_11_41.705929_baby_light_init_pid4516`; foreground wall time
  approximately `744.5` seconds; process exit code `0`.
- Canonical profile and frozen controls: profile/scope/source exactly
  `baby_td_asymmetric_no_projection_seed2023_v1` / `student_candidate` /
  `predeclared_baby_asymmetric_no_projection_directional_v1_seed2023`;
  dataset/student overrides `{}` / `{}`; blockers `[]`; student/training seed
  `2023`; `hard_token_seed=2022`; random 64-dimensional
  `td_distill_no_projection`; no student checkpoint, projection, teacher warm
  start, or efficiency benchmark; `td_init_from_teacher=false`; AdamW
  `student_lr=6e-5`; weight decay `0.01`; batch `1024`; epoch cap `1000`;
  patience `7`; `smoke_train_batches=0`; all `116` batches per epoch.
  Semantic controls remained alpha `0.3`, item-image `1.0`, item-text `0.3`,
  user-image `0.0`, user-text `0.0`; active heads were exactly item-image and
  item-text.
- Dataset and preflight identity: conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test hashes
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text hashes
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`,
  `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`;
  PCA cache hashes
  `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`,
  `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
  Split overlaps remained zero and the sole warning was the fixed official
  cold-item condition for items `240`, `1212`, and `6115`, covering `11`
  Validation and `7` Test interactions with no cold users.
- Frozen-teacher Test-isolation: teacher training was skipped and the read-only
  teacher remained `141098540` bytes with SHA256
  `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`.
  Raw log contains zero `Teacher reuse summary`, zero teacher Test-ranking
  events, and exactly one frozen-reuse/no-ranking message. Manifest records
  `teacher_final_test_performed=false`,
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`, no current
  `teacher_final_test_result`, and null historical final-Test metadata.
  Frozen teacher Test ranking count is exactly `0`.
- Training and numerical audit: epochs `0-51` completed contiguously, each
  through all `116` batches, for `52` Validation evaluations. All entries in
  all `15` total/BPR/directional/component and selection series were finite;
  inactive user-side losses remained exactly zero. Epoch `0 -> 51` endpoints:
  total `81.4330638051033 -> 71.39528197050095`; BPR
  `80.39827847480774 -> 70.98279428482056`; directional
  `3.449284791946411 -> 1.374958942644298`; item-image
  `3.4100762009620667 -> 0.9987448286265135`; item-text
  `3.579979732632637 -> 2.629005778580904`. Recomputed objective residuals
  were at most `3.52e-7` and normalized semantic residuals at most `9.35e-8`.
- Validation selection and Test chronology: Validation Recall@20 alone selected
  epoch `44` at exact `0.06479423036892924`; epochs `45-51` formed seven
  consecutive non-improvements, followed by one natural patience `7/7` stop.
  The run restored epoch `44` and then performed exactly one student final
  Test ranking. Exact vectors ordered by K `[10,20,40,50]`: Precision
  `[0.004587297505785609, 0.0036127539213165866, 0.002798920030856163,
  0.002564155309848446]`; Recall
  `[0.04165917893997112, 0.06531066330886427, 0.10077244289380877,
  0.1156571604733053]`; NDCG
  `[0.023961345086793755, 0.03052040561148235, 0.03849624048344874,
  0.04140749439374869]`; Hit Ratio
  `[0.04571869375160746, 0.07163795320133756, 0.11031113396759815,
  0.1262535356132644]`; AUC `0.0`. No Test value affected selection,
  stopping, acceptance, parameters, retry, or rollback.
- Manifest and checkpoint acceptance: manifest is `status=completed`, protocol
  `val_test_once_v1`, selection `validation` / `Recall@20`, exclusion
  `train_only`, `paper_ready_eligible=true`, blockers `[]`, and final Test
  enabled/performed. Full model state contains only finite user/item ID
  embeddings `[19445,64]` and `[7050,64]`, exactly two optimizer-state entries,
  and declared metadata. Inference-only state contains exactly the two finite
  embeddings plus `embedding_dim=64`, `n_users=19445`, `n_items=7050`, and
  variant `td_distill_no_projection`; it contains no optimizer, teacher,
  modality, prompt, cache, graph, projection, or other training state. Full
  and inference tensors have identical shape/dtype and are bit-for-bit equal.
- Six same-run artifacts, bytes, and SHA256:
  - raw log `logs/2026-08-02 13_11_41.705929_baby_light_init_pid4516`,
    `26943` bytes,
    `06dce735b4f7d81fd5e6029dbd78bb07c0c80a31b98f59dc95f9fa485a9dffee`;
  - preflight
    `exp/runs/baby/dataset_preflight__2026-08-02 13_11_41.705929_baby_light_init_pid4516.json`,
    `3399` bytes,
    `12e7ee5f4f20e5335228b2134d86fccca2b618a1e0bd510a1e5f2acbcd98d6db`;
  - manifest
    `exp/runs/baby/run_manifest__2026-08-02 13_11_41.705929_baby_light_init_pid4516.json`,
    `24999` bytes,
    `de0baf59683546168535d9a365a901838842b2b7153f16d82cdcbd02fc87e5e3`;
  - convergence
    `exp/converge/baby/auto__2026-08-02 13_11_41.705929_baby_light_init_pid4516.pkl`,
    `8734` bytes,
    `10fd9cf553bb1796ec7738ea79cb6b53e33b1765ac25a96a8ec46a0914dcb712`;
  - full checkpoint
    `Model/baby/td_distill/td_distill_full__val_test_once_v1__2026-08-02 13_11_41.705929_baby_light_init_pid4516.pth`,
    `20354897` bytes,
    `65166779894ea4634026d760f56c09d140e75c7067c7253d66a2b59f9478e36b`;
  - inference-only checkpoint
    `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__2026-08-02 13_11_41.705929_baby_light_init_pid4516.pth`,
    `6785989` bytes,
    `af29acbc4ad09f07b734fc020aaf34dd4d96a92b7b66d01ad23fc52eb7cdb6ba`.
- Matched recovered seed-2023 meaning: versus the recovered baseline, candidate
  Validation Recall@20 delta is `+0.023811482937223072` and final Test
  Recall@20 delta is `+0.028504121232329954`. The values reproduce the old
  seed-2023 numerical trajectories while removing the prohibited teacher Test
  ranking; therefore the pair is now protocol-compliant evidence. This one
  recovered seed does not by itself establish variance, significance, rate
  optimality, or cross-dataset generalization.
- Verification, Test access, and unresolved risks: structured
  `CANDIDATE_AUDIT_OK` passed event/order, finite-series, objective, manifest,
  preflight, checkpoint structure/equality, teacher, artifact, source, and
  process-exit gates. Test structure/identity was read in preflight; ranking
  access was frozen teacher `0`, student final Test `1`, fully compliant with
  `val_test_once_v1`. Remaining risks are the official cold-item condition,
  only seeds `2022/2023` currently recovered under the fixed protocol, and
  ignored artifact protection. No further stage is authorized in this task.
- Next action: stop and wait for explicit user authorization before any other
  seed, ablation, efficiency test, tag, bundle, backup, or merge.

## 2026-08-02 matched seed-2024 baseline/candidate recovery execution amendment (pending)

- Purpose and recovery rationale: replace only the protocol-noncompliant
  seed-2024 main-experiment baseline/candidate pair after the retrospective
  audit proved that both historical frozen-teacher reuse paths performed a
  prohibited new teacher Test ranking before student training. This recovery
  is authorized solely by that confirmed Test-isolation implementation defect.
  It is not motivated by either old run's Validation/Test values, candidate
  direction, or any quality threshold, and it does not tune, reinterpret,
  delete, overwrite, or replace the preserved old declarations, metrics, raw
  logs, manifests, convergence records, or checkpoints.
- Preserved diagnostic evidence: the old baseline run
  `2026-08-01 23_31_40.280800_baby_light_init_pid15908` selected epoch `163`
  with Validation Recall@20 `0.04086654667009528` and final Test Recall@20
  `0.042307913651446934`; the old candidate run
  `2026-08-02 00_42_54.029141_baby_light_init_pid29752` selected epoch `60`
  with Validation Recall@20 `0.0671997942915925` and final Test Recall@20
  `0.06682881901961434`. Both runs, their metrics, and all six artifacts per
  run remain protocol-noncompliant diagnostic evidence because each performed
  one frozen-teacher Test ranking. None of those values changes this recovery's
  profiles, commands, parameters, acceptance, or execution order.
- Status and amended authorization: pending serial recovery execution. The
  user explicitly authorizes both matched seed-2024 main-experiment arms in
  this one task, strictly baseline first. Append and commit this amendment and
  the baseline pending declaration, launch and audit the baseline once, then
  commit one completed or failed outcome. Only if that outcome is `completed`,
  all hard gates pass, and the outcome commit exists may the candidate pending
  record be appended and committed and the candidate launched. A baseline
  failure, connection interruption, or hard-gate violation requires a
  committed failed outcome and immediate stop without candidate declaration,
  launch, retry, or second process.
- Exact branch and source identity: branch
  `codex/experiment/baby-teacher-baseline` began clean at exact commit
  `e1cd465623452c41d95582e252112575be0a3f8d`, the completed seed-2023
  recovery candidate outcome. Frozen-teacher Test-isolation implementation
  commit is `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; startup
  `git diff --exit-code ee7b362... e1cd465... -- codes` was empty, proving no
  post-fix behavior drift. The commit containing this amendment and baseline
  declaration must be the exact clean baseline launch HEAD.
- Recovery scope: recover only `baby_student_reference_seed2024_v1` and, if
  the serial gate opens, `baby_td_asymmetric_no_projection_seed2024_v1`. Do
  not declare or run image-only ablation, another seed, efficiency, smoke,
  broad tuning, or any other experiment. Do not modify code, parser/profile
  defaults, parameters, data, splits, preprocessing, PCA caches, teacher,
  checkpoint format, sampler, optimizer, evaluation policy, or frozen commands.
  Do not create a tag, bundle, backup, or merge to `main`.
- Frozen common controls: audited MMRec Baby; `val_test_once_v1`; train only
  on `train_mat`; Validation Recall@20 alone selects and early-stops;
  candidate exclusion `train_only`; `Ks=[10,20,40,50]`; `test_flag=part`;
  random 64-dimensional `td_distill_no_projection` initialization;
  `td_init_from_teacher=false`; AdamW `student_lr=6e-5`; student weight decay
  `0.01`; batch size `1024`; maximum epoch `1000`; Validation every epoch;
  patience `7`; all `116` batches per completed epoch;
  `smoke_train_batches=0`; `run_efficiency_benchmark=false`; student and
  sampling `seed=2024`; frozen preprocessing/cache `hard_token_seed=2022`;
  no loaded student checkpoint; and canonical override-free profiles.
- Frozen data and preprocessing identity: conversion manifest SHA256
  `cf2d0d8c8aff9b321aad0b11d48d078794d12a4920afaa4c8efedfe3beda9df2`;
  train/Validation/Test SHA256 values
  `3cead4c601ccef4c2424cd692951935840ccc8f22df15577ced4e5f3fec37fac`,
  `f1458ff1dc28c5371699780270e3a3e270c8f9ce19def3d5e67d4ba9644ce9d2`,
  and `773de4f57f2c1bcb6695ea57995280e7cbe114b2b579263a599bb097bb6555ac`;
  image/text SHA256 values
  `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70`
  and `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4`.
  Frozen image/text PCA caches remain random state `2022`, identities
  `7634bb6e8dcfdbdd45eee436cc3b8e3f3f8e1e54c85c2f4b9f4a17719cc77c90`
  and `6e7c8161aaeb01f5e4bb8744be3bf9d16df71cd7cc74ea841bc5c20ad60926ee`,
  with SHA256 `187852dca1f62554e2d29714ca363e4a92036e948dddfb6c090a5577b6175c8c`
  and `5e9df184aca47a3c5d96d4db72977878981eca1cf67abba85b06e43528e1e562`.
- Frozen teacher and environment: reuse only read-only
  `Model/baby/teacher_model_val_test_once_v1.pt`, exactly `141098540` bytes,
  SHA256 `b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4`;
  `if_train_teacher=false`, `teacher_only=false`, and
  `allow_teacher_alias_overwrite=false`. Teacher training, optimizer steps,
  mutation, publication, alias overwrite, run-local teacher checkpoints, and
  frozen-teacher Test ranking are prohibited. Verified launch environment is
  `D:\miniconda\envs\run_5060\python.exe`, Python `3.10.20`, PyTorch
  `2.11.0+cu128`, CUDA runtime `12.8`, driver `595.97`, NVIDIA GeForce RTX
  5060 with `8151` MiB, GPU `0`; startup training-process count was `0` and
  drive D free space was `405601091584` bytes.
- Test-isolation and execution contract: preflight may read Test structure and
  identity, but frozen-teacher reuse must rank Test zero times and record
  `teacher_final_test_performed=false` plus
  `teacher_test_policy=frozen_checkpoint_reuse_no_test_ranking`. Historical
  checkpoint-carried final-Test metadata, if any, may appear only under
  `teacher_checkpoint_historical_final_test_metadata` with
  `performed_by_current_run=false`; a current-run
  `teacher_final_test_result` is forbidden. Each student must select and
  restore only its Validation-best checkpoint, then rank final Test exactly
  once. Every formal command must execute once in one continuously attached
  foreground shell/tool session with timeout at least `10800000` ms. Parallel
  or background launch, output closure while active, automatic retry, and a
  second process are forbidden.
- Acceptance, quality, and preservation: hard gates cover exact clean source,
  canonical profile, empty overrides/blockers, environment/data/cache/teacher
  identity, finite loss and Validation series, all `116` batches, Validation
  Recall@20-only selection, natural stop, best-checkpoint restore, teacher Test
  count `0`, student final Test count `1`, completed/paper-ready manifest
  fields, six same-run artifacts, finite checkpoint structure, ID-only
  inference export, and bit-for-bit full/inference embedding equality. There
  is no minimum metric or improvement threshold. Low or reversed compliant
  results remain completed evidence and cannot trigger parameter changes,
  retry, rollback, reset, deletion, or suppression. Preserve all old and new
  artifacts.

## 2026-08-02 baby_student_reference_seed2024_v1 matched recovery formal run (pending)

- Purpose and hypothesis: execute one replacement formal seed-2024 ID-only
  BPR baseline under the corrected frozen-teacher Test-isolation source. The
  goal is protocol recovery, not metric improvement; every method and runtime
  parameter remains the previously frozen
  `baby_student_reference_seed2024_v1` contract.
- Status and authorization: pending formal uncapped baseline, authorized for
  exactly one launch after this amendment/declaration commit is clean. The
  candidate is not yet declared and remains blocked until this run naturally
  exits, passes the full protocol/artifact audit, and its `completed` outcome
  commit exists. A failure requires a committed failed outcome and immediate
  stop without retry.
- Exact command, frozen verbatim with no CLI seed, semantic, optimizer,
  budget, or protocol override:
  `D:\miniconda\envs\run_5060\python.exe codes\main_mmlight.py --dataset baby --student_profile baby_student_reference_seed2024_v1 --gpu_id 0 --if_train_teacher false --teacher_checkpoint Model/baby/teacher_model_val_test_once_v1.pt --run_final_test true`
- Source launch gate: branch
  `codex/experiment/baby-teacher-baseline`; pre-declaration clean HEAD
  `e1cd465623452c41d95582e252112575be0a3f8d`; fixed implementation
  `ee7b3621765b63886c341d63fdc49a20d1a21bc8`; no `codes/` drift between
  them. The commit containing this amendment and pending declaration must be
  the exact clean launch HEAD, with no intervening tracked edit.
- Canonical profile and override contract: exact profile/scope/source
  `baby_student_reference_seed2024_v1` / `student_reference` /
  `predeclared_baby_id_only_bpr_reference_seed2024`;
  `dataset_config_overrides={}`; `student_config_overrides={}`; profile
  blockers `[]`; training/sampling seed `2024`; `hard_token_seed=2022`;
  `td_distill_no_projection`; dimension `64`; random initialization;
  `td_init_from_teacher=false`; AdamW `student_lr=6e-5`; weight decay `0.01`;
  batch `1024`; epoch `1000`; patience `7`; no smoke cap; final Test enabled;
  and efficiency disabled.
- Baseline objective: BPR only. `td_distill_alpha=0.0` and item-image,
  item-text, user-image, and user-text rates are all `0.0`; no semantic head,
  projection head, loaded student checkpoint, or teacher warm start may affect
  the student.
- Evaluation chronology and hard acceptance: dataset preflight and frozen-
  teacher validation must pass; every loss/Validation series value must be
  finite; every completed epoch must use all `116` batches; Validation
  Recall@20 alone must save/select and naturally early-stop; restore the
  Validation-best full checkpoint; frozen teacher Test ranking count must be
  `0`; student final Test ranking count must be exactly `1` and occur only
  after restore. Manifest must report `status=completed`, protocol
  `val_test_once_v1`, selection `validation` / `Recall@20`, exclusion
  `train_only`, both override maps `{}`, `paper_ready_eligible=true`, blockers
  `[]`, `teacher_final_test_performed=false`, policy
  `frozen_checkpoint_reuse_no_test_ranking`, and
  `final_test_performed=true`, with no current-run teacher final-Test result.
- Planned six isolated artifacts, all sharing one new timestamp/PID identity:
  raw log `logs/<run_name>`; preflight
  `exp/runs/baby/dataset_preflight__<run_name>.json`; manifest
  `exp/runs/baby/run_manifest__<run_name>.json`; convergence record
  `exp/converge/baby/auto__<run_name>.pkl`; full checkpoint
  `Model/baby/td_distill/td_distill_full__val_test_once_v1__<run_name>.pth`;
  and inference-only checkpoint
  `Model/baby/td_distill/td_distill_infer_only__val_test_once_v1__<run_name>.pth`.
  All six must exist, share the run identity, and receive byte counts and
  SHA256 fingerprints; the teacher hash must remain unchanged.
- Checkpoint/deployment acceptance: the full checkpoint must contain finite
  user/item ID embeddings plus declared optimizer/metadata state. The
  inference-only checkpoint must contain exactly finite user/item ID embeddings
  plus dimension/count/no-projection deployment metadata and no optimizer,
  teacher, modality, prompt, semantic cache, graph, projection, or other
  training state. Full and inference embeddings must have identical
  shape/dtype and be bit-for-bit equal.
- Next gate: commit this amendment and declaration only; repeat the exact clean
  HEAD, process, source, canonical-profile, environment, data/cache, and
  teacher gates; then launch the frozen baseline command exactly once through
  the required continuously attached foreground session. After natural exit,
  audit raw log, preflight, manifest, convergence, both checkpoints, teacher,
  Test chronology, process state, and source/worktree status; append and commit
  one completed or failed baseline outcome. Only a committed completed outcome
  opens the amendment's candidate gate.
