# Sports zero-update gradient diagnostic

Manual command from repository root:

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/diagnose_sports_gradients.py
```

Seed2022, eight fixed batches of1024 at the unchanged pinned shared initialization. Reuses Data.sample and the active student/loss functions. Diagnostic RNG reset after construction means these are not claimed to replay historical training batches. Reads train_mat only; bypasses Data constructor and does not read Validation/Test splits. No teacher forward, ranking, optimizer construction or parameter update.

Objective: BPR + (0.3/1.3) D_image + (0.3*0.3/1.3) D_text, with D=MSE(normalized vectors), mean over batch and dimension. Report raw and weighted losses, user/item/combined embedding gradient norms, relative norms and cosines; zero-vector cosine is null. Repeated positive IDs accumulate into actual embedding-table gradients. Verifies total-gradient linearity, parameter and input fingerprints unchanged. No AdamW step prediction is claimed: raw gradient size is not an adaptive-optimizer update ratio, and this initial-state diagnostic cannot establish effects later in training.

Output: exp/gradient_checks/sports_sharedinit_seed2022_v2/report.json plus exact batches.json and SHA256. Requires clean committed source; exclusive output directory, no retry or overwrite. On failure preserve artifacts and report error. After completion request audit of this report before deciding further experiments. No formal training profile or evaluator changes.

Revision v2 preserves the failed v1 report. Sampled NumPy integer IDs are converted to Python int before strict JSON serialization, preserving values, order and sampling. The v1 attempt failed before any gradient measurement; no old training results need rerunning.
