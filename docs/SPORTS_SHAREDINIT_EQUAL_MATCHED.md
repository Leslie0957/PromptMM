# Sports shared-initialization equal-target matched-scale arm

Manual one-time launch after the preparation commit, from a clean checkout:

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sharedinit_equal_matched.py
```

This single seed-2022 arm uses the pinned shared six-tensor initial state and targets, 300 epochs, patience 300, student learning rate 6e-5, batch size 1024, AdamW weight decay 0.01, and dimension 64. The objective is BPR + 0.4111064309069839 × (Dimage + Dtext)/2, with zero user distillation. Only the item text rate (0.3 to 1.0) and global distillation coefficient (0.3 to 0.4111064309069839) differ from the existing shared Full arm. The coefficient matches the aggregate initial raw item-gradient RMS over eight fixed train-only batches, as recorded in `exp/gradient_checks/sports_matched_target_direction_seed2022_v1/report.json` (SHA256 `b990d361fa56ae94a3b20480d0db9c1dac3a672bf20a36c2d409d66453972cc2`). It does not match each batch, AdamW update, or later gradient.

Compare with the existing shared Full Validation Recall@20 0.095438076853 and BPR-only 0.095477873096. Initial Validation is checked for state equality but not selected. Select the best of 300 epochs by Validation Recall@20. No teacher or student Test ranking; existing loader structural Test reads remain. This is a single-seed direction comparison, with no expected gain claim or paper-ready eligibility.

The launcher requires clean committed source, the exact completed shared-pair anchor and calibration report, and an exclusive output path. It launches one child once, with no retry or additional arm. Completion verifies finite curves, selected checkpoint, source/asset identity, and exact initial state and initial Validation equality to the prior Full arm. The report is `exp/initialization_checks/sports_sharedteacherinit_equal_matched_seed2022_val300_v1/run.json`; the timestamped manifest and checkpoint are referenced there. Preserve a failed or low-metric run for audit. After completion, give the report path for a read-only outcome audit.
