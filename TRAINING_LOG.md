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
