# Sports shared-initialization alpha3 sensitivity arm

```powershell
Set-Location 'D:\Download\PromptMM'
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sharedinit_alpha3.py
```

Single seed2022 arm,300epochs,patience300,lr6e-5,batch1024,AdamW decay0.01,dimension64. Only behavior delta from completed shared-init full is alpha0.3 to3.0. Image:text rates remain1:0.3; userdistillation0. Objective BPR+3*(Dimage+0.3Dtext)/1.3. Same pinned six tensors, teacher/data and legacy TD evaluator. Initial Validation excluded from selection; best epoch chosen by Validation Recall20. No Test ranking; loader Test structural reads remain.

Compare existing alpha0 BPR Recall20=0.095477873096 and alpha0.3 full=0.095438076853. This is a single-seed sensitivity probe, not guaranteed gain or an optimal coefficient. Low valid metrics remain completed evidence. Do not change batch/evaluator or extend epochs during this run.

Launcher requires clean committed source and exact completed pair anchor; exclusive new output, one child invocation, no retry/resume/additional arm. Runtime completion verifies300finite curves, selected checkpoint/source/asset and exact initialstate/metrics against prior pair. New output exp/initialization_checks/sports_sharedteacherinit_alpha3_seed2022_val300_v1/run.json; timestamped run manifest and checkpoint/curve referenced therein. Old artifacts retained. Prior full run took about5h04m and BPR3h13m under differing load; allow several hours, no precise runtime guarantee.

After completion ask for local report audit. Assistant preparation does not launch training.
