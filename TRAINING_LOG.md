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
- Current completed stage: frozen Baby teacher reference, seed `2022`
- Current incomplete stage: Baby student baseline and distillation experiments
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
tables. As of this reset, the only completed paper-ready Baby result is the
seed-2022 teacher reference.

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

No formal Baby student, distillation, ablation, multi-seed, or efficiency result
exists yet. No Baby student checkpoint is an accepted reference.

Known configuration boundary before the first Baby student run:

- Baby teacher semantic dimension is `64`.
- The parser still defaults `student_embed_size` to `32`.
- `td_distill_no_projection` requires student and teacher semantic dimensions
  to match exactly.
- The parser defaults to `student_model_type=lightgcn`, four component rates of
  `1.0`, `td_distill_alpha=0.1`, and teacher warm start enabled.
- Those parser defaults are not a validated Baby student profile and do not
  represent the intended asymmetric no-projection method.
- Old Amazon values such as `student_lr=6e-5`, `td_distill_alpha=0.3`, and
  component rates `1/0.3/0/0` are hypotheses for Baby, not confirmed defaults.

## Next Experiment Gate

Before launching a Baby student run:

1. Define and commit a Baby student protocol/profile without observing Baby
   test metrics.
2. Reuse `Model/baby/teacher_model_val_test_once_v1.pt` read-only with
   `--if_train_teacher false`; never overwrite it.
3. Make the baseline/proposed comparison fair on student embedding dimension,
   optimizer budget, sampling, evaluation protocol, and seed.
4. Set `student_embed_size=64` for any no-projection run unless a separately
   justified teacher target projection changes the method identity.
5. Predeclare initialization policy. Treat teacher warm start as a separate
   control because it was a major confounder historically.
6. Run a short execution smoke before any uncapped student training.

Recommended evidence sequence after the profile is declared:

1. ID-only BPR student baseline.
2. Asymmetric no-projection directional-distillation candidate.
3. Directional-loss-off control under the same initialization.
4. Symmetric four-component and projection controls.
5. Single-seed gate, then declared multi-seed confirmation.
6. Efficiency benchmark using an explicitly pinned accepted student checkpoint.

Do not begin broad hyperparameter sweeps before the baseline/proposed sanity
pair establishes that the method is viable on Baby.

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
