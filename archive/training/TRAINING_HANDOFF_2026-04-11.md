# Training Handoff - 2026-04-11

Use `TRAINING_LOG.md` as the canonical experiment record.
This file is only a short next-session handoff.

## Current Situation
- Active training path: `codes/main_mmlight.py`
- Default argument source: `codes/utility/parser.py`
- Active dataset: `amazon`
- Current default `student_lr`: `5e-5`
- KD-related weights: unchanged from the 2026-04-10 baseline
- Main goal of this run: check whether the higher student learning rate speeds up convergence without hurting Recall@20 and NDCG@20

## Current Baseline To Beat
- Baseline date: 2026-04-10
- Best Recall@20: 0.08522735
- Best NDCG@20: 0.09222543
- Best Precision@20: 0.01581643
- Baseline interpretation: training was stable, but convergence was slow

## Current Run In Progress
- Start time seen in console: 2026-04-11 00:27
- PID: 17924
- Command:
  - `python .\main_mmlight.py --data_path d:/Download/PromptMM/data/ --dataset amazon`
- Effective run notes:
  - `student_lr=5e-05`
  - `point=''`, so no manual point-name override is being used
  - `if_train_teacher=True`
  - `student_model_type='lightgcn'`

## Where Artifacts Go
- Log file:
  - `logs/2026-04-11 00_27_22_amazon_light_init_pid17924`
- Latest teacher alias:
  - `Model/amazon/teacher_model_great.pt`
- Archived teacher checkpoint for this run:
  - `Model/amazon/runs/teacher_model_great__2026-04-11 00_27_22_amazon_light_init_pid17924.pt`
- Archived converge result for this run:
  - `exp/converge/amazon/auto__2026-04-11 00_27_22_amazon_light_init_pid17924.pkl`

## Important Notes
- Result saving has been updated so archived run artifacts are unique and should not overwrite prior runs.
- The current running process was started before the latest warning-cleanup patch, so you may still see two known PyTorch warnings in this specific run.
- Those warnings were cleaned in code for the next launch and are not currently believed to invalidate this run.
- `codes/run_patent.py` is separate and should not be mixed into this training line.

## What To Do Tomorrow
1. Check whether the current run finished normally or stopped early.
2. Open the latest log file and extract the best Recall@20, NDCG@20, Precision@20, and any convergence/early-stopping signal.
3. Append the completed run to `TRAINING_LOG.md`.
4. Compare the finished run against the 2026-04-10 baseline.
5. If training is still stable but still too slow, the next candidate change is `student_lr = 1e-4`.

## Reminder For A New Chat Window
- In this repo, read `AGENTS.md` and `TRAINING_LOG.md` first.
- If needed, also read this file for the short current-session snapshot.
