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
