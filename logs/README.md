# Raw Training Logs

This directory is the runtime output location used by
`codes/utility/logging.py`. The existing files are raw per-run records from
2026-04-09 through 2026-04-17, including completed, interrupted and launcher
diagnostic runs.

Related records:

- Canonical current experiment memory: `../TRAINING_LOG.md`
- Full historical summary and handoffs: `../archive/training/`
- Convergence artifacts: `../exp/converge/`
- Efficiency artifacts: `../exp/efficiency/`
- Checkpoints: `../Model/`

Do not delete small, empty or failed-run logs solely based on file size. They
may explain interrupted experiments or command-line failures.

