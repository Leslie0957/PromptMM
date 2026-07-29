# Repository Guidance

For any training, tuning, or result-comparison task in this repository:

- Read `TRAINING_LOG.md` before making parameter changes or giving experiment recommendations.
- Treat `TRAINING_LOG.md` as the canonical experiment memory for this repo.
- Append every completed run and every confirmed parameter change to `TRAINING_LOG.md`.
- The active experiment path is `codes/main_mmlight.py`.
- The default argument source for that path is `codes/utility/parser.py`.
- `codes/run_patent.py` is a separate standalone script and should not be treated as part of the active training line unless the user explicitly asks for it.
- When checking or changing `student_lr`, remember that a CLI flag such as `--student_lr` overrides the parser default.

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
- At task completion, report the branch, commit hash, verification performed, and whether the working tree is clean. If a safe commit is impossible because unrelated user changes overlap the task, preserve those changes and explain the blocker instead of forcing a commit.
