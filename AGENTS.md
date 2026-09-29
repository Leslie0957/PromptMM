# Repository Guidance

## Mandatory Startup For Every AI

Before making experiment recommendations, editing repository files, or running
project code in this workspace:

- Read this `AGENTS.md` completely.
- Read the root `README.md` overview, then the short active `TRAINING_LOG.md`.
  These are the default core context for a new task. Use
  `docs/experiments/README.md` to select only the relevant evidence family.
  Do not read full archived logs or every research note at startup. Historical
  plans and pending declarations are not current task queues or launch authority.
- Read the active `TRAINING_LOG.md` header/current-state material and the newest
  entries relevant to the task. Use targeted search and bounded sections first;
  read the full historical log only when an older dependency is unresolved or
  a cited experiment anchor cannot otherwise be verified. Never substitute an
  old snapshot for the active log.
- Read root `THESIS_ROADMAP.md` as the stable thesis objective before research decisions. Do not silently change its core goals; follow its versioned change rule. Experiment status belongs in the active log, not in repeated charter edits.
- Inspect `git status --short --branch` before editing. Preserve and work around
  unrelated user changes; never reset, discard, overwrite, stage, or commit them.
- Identify whether the task can affect model behavior, experiment protocol,
  data identity, preprocessing, parameters, environment, commands, metrics,
  checkpoints, or reproducibility. If it can, the mandatory before/after trace
  below applies before the first edit.

For any training, tuning, or result-comparison task in this repository:

- Treat `TRAINING_LOG.md` as the canonical experiment memory for this repo.
- The active log carries current state and new append-only entries; verified
  immutable snapshots under `archive/training/` carry prior detailed records.
  Search `archive/training/ENTRY_INDEX_2026-09-26.md` by topic/profile first,
  then read only the indicated source section when historical detail is needed.
  `docs/experiments/FILES.md` indexes source/documents; `docs/experiments/RUN_RECORDS.md` indexes
  saved JSON records; `archive/catalog/` contains on-demand raw file inventories.
  Catalogs are navigation snapshots, not replacements for audited outcomes.
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

A declaration-only task that changes no implementation, profile, parameter,
data, environment, or protocol is the exception to the duplicate outcome rule:
one `pending` formal-run record plus its coherent Git commit completes the
declaration task. Do not append a second declaration-completed record that merely
repeats it. The later execution must still receive a separate `completed` or
`failed` run outcome.

Ordinary prose, spelling, or formatting changes that cannot affect experiments
do not require pending/completed entries in `TRAINING_LOG.md`; their Git commit
is the trace. Repository-policy changes and training-log reorganizations are
material and must record their own preservation and verification evidence.

## Deterministic Run Record Routing

- When declaring a run/cohort or auditing its outcome, read
  `docs/experiments/RUN_CLOSEOUT.md`. It defines mandatory versus conditional
  file updates, the active result-matrix pointers, generated catalogs and the
  closeout checklist. Do not infer update targets from chat memory.
- Each new run declaration must include explicit record-update targets:
  outcome audit path, experiment-family row, active matrix cells/seeds and
  conditional paper/gap consumers. An old declaration without targets must
  have them identified in its audit before closeout; this does not authorize
  any additional execution.
- Preserve historical records. Update current consumers only when triggered;
  record unchanged/not-applicable targets with reasons instead of rewriting
  every past report. Refresh generated navigation after manual edits, then
  verify and commit the coherent scope as required below.

## Mandatory Completion Handoff

1. At the start of every task, read this complete `AGENTS.md`, the active log
   sections required by the startup rule above, and inspect
   `git status --short --branch` before recommending experiments, editing files,
   or running project code.
2. Use the active log to identify the current experiment stage, completed work,
   unresolved risks, and the single recommended next step. Repository records,
   not chat memory, are the source of continuity.
3. For every experiment-affecting change, append a `pending` entry before the
   change and a separate `completed` or `failed` entry afterward. Preserve old
   entries verbatim; never rewrite a declaration into its outcome. Apply the
   declaration-only exception above when no experiment-affecting field changes.
4. After a change, run verification proportional to its risk and automatically
   create one coherent local Git commit for the completed scope. Never stage or
   commit unrelated files.
5. Make the completion handoff proportional to the task. A declaration-only
   handoff needs only status, branch/commit/cleanliness, exact launch command,
   confirmation that no run or Test access occurred, and one next step. A
   completed or failed formal run still reports commands and parameters,
   verification and metrics, artifact paths/fingerprints, Test chronology and
   protocol compliance, branch/commits/cleanliness, experimental meaning,
   unresolved risks, and exactly one copy-ready next step.
6. If a task fails, report the root cause, valid artifacts that remain, and a
   concrete recovery plan before recommending the next action. Never describe a
   failed or partial result as completed.
7. A clear next step is not authorization to start it. Formal or long training,
   broad tuning, test-split evaluation, merge to `main`, tag creation, and
   baseline-asset overwrite still require explicit user authorization. When the
   user explicitly authorizes one specific run, or unambiguously approves the
   sole recorded next run, that authorization may cover the run's pending
   declaration, necessary in-scope preparation, verification, clean Git commit,
   immediate launch from that exact commit, post-run audit, and outcome commit
   in one continuous task. Do not request a redundant second `continue` after
   the declaration commit. The authorization does not extend to another seed,
   arm, retry, tuning sweep, Test access beyond the declared protocol, or any
   later experiment.
8. Complete in-scope checks, fixes, verification, logging, and Git commits
   autonomously when they require no user choice. Do not repeatedly return
   discoverable or self-verifiable questions to the user.
9. Advance only one explicit experiment stage per task. For an explicitly
   authorized specific run, its declaration, clean launch commit, one launch,
   audit, and outcome commit together count as one stage. Do not launch parallel
   experiment stages or advance to another seed/arm/run in the same task.
10. Never automatically roll back, reset, revert, or delete an experiment
    result solely because its metrics are low, worse than expected, or degraded
    relative to another method. Preserve and record low metrics as valid
    completed evidence when the code, execution flow, and hard acceptance
    conditions succeeded. Mark a task `failed` only when code, execution flow,
    or hard acceptance conditions fail; preserve the failure state and report
    its cause and recommended recovery point. Without explicit user
    confirmation, never revert, reset, restore a branch, delete changes, or
    delete experiment results.

## Formal Run Gate

- Never start a formal or long training run from dirty or uncommitted source.
- Freeze stable dataset, preprocessing, teacher, protocol, environment, and
  common-parameter identities once as a named shared anchor in
  `TRAINING_LOG.md` or a committed/run manifest. A later run may cite that exact
  anchor instead of repeating its hashes and parameters. Repeat or rehash a
  common identity only when it changed, is stale or untrusted, or no valid
  anchor exists.
- Before launch, ensure the exact source is committed. A minimal declaration
  must record the run/profile and seed, exact shared anchor, explicit delta from
  it, full command, branch and launch-source rule, comparison anchor, Test
  policy, expected artifact family, and one-launch/no-retry/no-next-stage guard.
  The commit containing the declaration may be the launch HEAD; it need not
  embed its own not-yet-known hash in that same commit.
- Keep prelaunch verification proportional: verify the changed fields and the
  referenced anchor, and let the runtime preflight/manifest capture resolved
  common identities. Do not repeatedly inventory or hash unchanged assets just
  to restate an already verified anchor.
- A run-specific authorization remains valid while the agent records and
  commits that exact run declaration. If the declaration, command, parameters,
  identities, acceptance rule, artifact paths, or intended Test access changes
  materially, append an amendment and obtain new authorization before launch.
  Otherwise, after the tree is clean and every declared preflight gate passes,
  launch the exact run immediately without asking for the same approval again.
- Launch an authorized formal command at most once. A crash, timeout, transport
  loss, hard-acceptance failure, or partial artifact is not permission to retry,
  resume, roll back, or start the next run. Preserve and audit the state, append
  a failed or partial outcome as appropriate, commit the record, and request
  new authority for any recovery execution.
- A smoke run may be used before the final commit only when it is explicitly
  labeled non-formal, capped, isolated in run-specific artifacts, and recorded.
- After every completed or failed run, append status, resolved parameters,
  validation selection metrics, one-time final test metrics when applicable,
  artifact paths/hashes, Test chronology (zero teacher Test and exactly one
  student Test under `val_test_once_v1`), finite objective/metric evidence,
  selected-versus-tested checkpoint equality, and clean-source evidence before
  starting the next formal run.

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
- A clean Git commit identifies and makes tracked launch source recoverable; it
  does not preserve ignored run artifacts or guarantee that interrupted
  training can resume. Never describe source rollback as automatic experiment
  recovery. Preserve failure artifacts and treat any source reset, revert, or
  rerun as a separate authorized action.
- Never slim, replace, or reorganize a canonical log until a verbatim immutable
  snapshot has been created and its byte length and SHA256 have been verified
  and recorded in the new log and archive index.
- Keep current-state summaries short. Archive only at an explicitly authorized
  organization task, preserve snapshot bytes and SHA256 (including Git EOL
  handling), and keep new entries in the active log. Update experiment indexes
  after completed stages; never move/delete referenced runtime assets merely
  to make the directory look cleaner. The metadata-only catalog builder is
  `tools/build_experiment_navigation.py`; it never launches experiments.
- At task completion, report the branch, commit hash, verification performed, and whether the working tree is clean. If a safe commit is impossible because unrelated user changes overlap the task, preserve those changes and explain the blocker instead of forcing a commit.
