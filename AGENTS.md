# Repository Guidance

For any training, tuning, or result-comparison task in this repository:

- Read `TRAINING_LOG.md` before making parameter changes or giving experiment recommendations.
- Treat `TRAINING_LOG.md` as the canonical experiment memory for this repo.
- Append every completed run and every confirmed parameter change to `TRAINING_LOG.md`.
- The active experiment path is `codes/main_mmlight.py`.
- The default argument source for that path is `codes/utility/parser.py`.
- `codes/run_patent.py` is a separate standalone script and should not be treated as part of the active training line unless the user explicitly asks for it.
- When checking or changing `student_lr`, remember that a CLI flag such as `--student_lr` overrides the parser default.
