# Repository Guidance

## Mandatory Startup For Every AI

Before making experiment recommendations, editing repository files, or running
project code in this workspace:

- Read this `AGENTS.md` completely.
- Read the complete active `TRAINING_LOG.md` at the repository root. Follow its
  archive links only when historical detail is needed; never substitute an old
  snapshot for the active log.
- Inspect `git status --short --branch` before editing. Preserve and work around
  unrelated user changes; never reset, discard, overwrite, stage, or commit them.
- Identify whether the task can affect model behavior, experiment protocol,
  data identity, preprocessing, parameters, environment, commands, metrics,
  checkpoints, or reproducibility. If it can, the mandatory before/after trace
  below applies before the first edit.

For any training, tuning, or result-comparison task in this repository:

- Treat `TRAINING_LOG.md` as the canonical experiment memory for this repo.
- Append every completed run and every confirmed parameter change to `TRAINING_LOG.md`.
- The active experiment path is `codes/main_mmlight.py`.
- The default argument source for that path is `codes/utility/parser.py`.
- `codes/run_patent.py` is a separate standalone script and should not be treated as part of the active training line unless the user explicitly asks for it.
- When checking or changing `student_lr`, remember that a CLI flag such as `--student_lr` overrides the parser default.

## Mandatory Before/After Experiment Trace

For every change that can affect experiment behavior, result interpretation, or
reproducibility:

1. Before editing, append a `pending` entry to `TRAINING_LOG.md`. Record the
   purpose, hypothesis or rationale, scope, affected files/parameters, risks,
   acceptance criteria, rollback point, and planned verification or next run.
2. Do not make the experiment-affecting edit until that pending entry exists.
3. Implement only the declared scope. If the scope changes materially, append
   an amendment before continuing.
4. After implementation and verification, append a separate `completed` or
   `failed` entry. Preserve the pending entry; never rewrite it into an outcome.
   Record actual changes, verification evidence, unresolved risks, artifact or
   source identity available at that time, and the next action.
5. Create one coherent local Git commit after the requested change is verified.
   Report the resulting commit hash at task completion and record it in the next
   formal-run declaration when the change will be used for training.

Ordinary prose, spelling, or formatting changes that cannot affect experiments
do not require pending/completed entries in `TRAINING_LOG.md`; their Git commit
is the trace. Repository-policy changes and training-log reorganizations are
material and must record their own preservation and verification evidence.

## Formal Run Gate

- Never start a formal or long training run from dirty or uncommitted source.
- Before launch, ensure the exact source is committed and record the commit
  hash, branch, full command, environment, dataset and preprocessing identity,
  teacher/student checkpoint identity, protocol, parameters, seeds, acceptance
  rule, and planned artifact paths in `TRAINING_LOG.md` or the run manifest.
- A smoke run may be used before the final commit only when it is explicitly
  labeled non-formal, capped, isolated in run-specific artifacts, and recorded.
- After every completed or failed run, append status, resolved parameters,
  validation selection metrics, one-time final test metrics when applicable,
  and artifact paths/hashes before starting the next formal run.

## Version Control and Traceability

For any repository modification in this workspace:

- Inspect `git status --short --branch` before editing. Never reset, discard, overwrite, stage, or commit unrelated user changes.
- Treat `main` and existing baseline tags as protected stable history. If a material code, experiment, or repository-policy change starts while on `main`, create an appropriately scoped branch such as `experiment/<name>`, `fix/<name>`, `maintenance/<name>`, or `docs/<name>` before editing.
- Group one coherent, reviewable change into one commit. Do not create a commit for every keystroke, and do not mix unrelated refactors or formatting into the same commit.
- After the requested change is complete, verify it in proportion to risk and create a local Git commit on the task branch. Documentation-only changes need focused content/diff checks rather than the full training test suite.
- Promote a verified task branch to `main` only when the change is intended to become the new stable repository state. Use a non-destructive fast-forward when possible; never move or recreate an existing baseline tag.
- Append to `TRAINING_LOG.md` when a confirmed change affects model behavior, experiment protocol, data identity, preprocessing, parameters, environment, run commands, metrics, checkpoints, or reproducibility. Git history alone is sufficient for ordinary prose/formatting changes that cannot affect experiments.
- Before any formal or long training run, ensure the exact code is committed and record its commit hash, branch, full command, environment, and data/checkpoint identity in `TRAINING_LOG.md` or the run manifest.
- After every completed run, record status, parameters, selected validation metrics, final test metrics, and artifact paths in `TRAINING_LOG.md` as already required above. Do not commit generated checkpoints or raw run outputs.
- Keep `data/`, `Model/`, `logs/`, `exp/`, `zhuanli/`, and other ignored large artifacts out of normal Git history. Update asset hashes only when those assets change or at a deliberate milestone; do not rehash the full asset set after an unrelated small source edit.
- Create Git tags and refresh the standalone bundle only for meaningful stable milestones, not for routine commits. Tags should identify states such as a passed protocol smoke, a pinned formal teacher, a completed multi-seed result set, or a paper-result freeze.
- Every material repository modification must at least be preserved by its
  coherent branch commit. Meaningful stable milestones additionally require an
  annotated tag and refreshed standalone bundle. Routine commits do not need a
  tag or bundle.
- Git-ignored datasets, checkpoints, manifests, caches, and raw logs are not
  protected by commits or bundles. Fingerprint them when they define a run, and
  make an explicit physical/off-device backup at stable milestones.
- Never slim, replace, or reorganize a canonical log until a verbatim immutable
  snapshot has been created and its byte length and SHA256 have been verified
  and recorded in the new log and archive index.
- At task completion, report the branch, commit hash, verification performed, and whether the working tree is clean. If a safe commit is impossible because unrelated user changes overlap the task, preserve those changes and explain the blocker instead of forcing a commit.
