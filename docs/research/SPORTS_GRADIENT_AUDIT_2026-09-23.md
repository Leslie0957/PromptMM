# Shared-initialization gradient diagnostic audit — 2026-09-23

V2 completed from clean 82ced9edec7fc89898c1993f35b62872d5ee9d61, 22:57:58–22:58:01 +08:00. Command: `& 'D:\miniconda\envs\run_5060\python.exe' -B tools/diagnose_sports_gradients.py`. Seed2022,8 fixed batches of1024 at pinned initialstate; CUDA0/run_5060/RTX5060. Objective BPR+0.3*(Dimage+0.3Dtext)/1.3. Shared asset and training SHA anchors unchanged.

| Quantity | Eight-batch mean |
|---|---:|
| Weighted distillation / BPR gradient norm, all parameters | 0.456150% |
| Weighted distillation / BPR gradient norm, item parameters | 1.045023% |
| Weighted image / BPR gradient norm, item parameters | 1.087236% |
| Weighted text / BPR gradient norm, item parameters | 0.252189% |
| Distillation versus BPR cosine, item parameters | 0.010416 |

Overall norm ratio ranges0.302896%–0.648397%; item ratio0.894858%–1.298162%. User distillation gradient zero as configured; zero-vector cosine null. Image/text gradient cosine ranges -0.285789 to -0.277350, indicating partial local opposition, not evidence that text is harmful to ranking. Distillation/BPR directions are nearly orthogonal at this initialstate, not strongly antagonistic. Combined norm is only slightly smaller than image-only norm: partial cancellation is not sufficient to explain the overall small gradient.

Mean raw BPR loss0.0161237260; weighted distillation0.00567510375. Comparable loss values do not imply comparable gradients. Directional MSE averages over batch and64 dimensions; its 1/d scaling is part of the implemented objective, not a proven implementation bug.

This supports the hypothesis that current distillation is a weak raw-gradient perturbation at warm start, consistent with the near-equal full/BPR results. It does not prove causality for300epochs, statistical equivalence, no later effect, or the fraction of actual AdamW parameter updates. No optimal coefficient or gate design has been established.

Single recommended next stage: prepare ONE seed2022 shared-init full-method alpha3.0 sensitivity arm, retaining300epochs/lr6e-5/batch1024/teacher/data/selection protocol and all other weights. Compare existing alpha0 and0.3 anchors; no new BPR rerun. Alpha3 is a conservative decade sensitivity probe, not an optimum: at identical initialstate the raw distill norm scales10x (mean item ratio about10.45%), with no guarantee of improved metrics. Use only Validation; no automatic further sweep. Preparation/launch not performed by this audit.

Verification: report completion/8rows and finite values; exact initial/final tensor metadata; current source fingerprints and shared asset file SHA; saved batches SHA/count/shape; norm-ratio algebra and cosine bounds. Runtime reports gradient linearity and unchangedparameters. This audit recomputed report arithmetic only, not gradients. Zero optimizersteps, zero Validation/Test evaluations and Test split not loaded; no selected/tested checkpoint, no recommendation accuracy measured. V1failed report preserved. Babyseed2023 auditexception/originalmanifestfalse preserved; Innovation2undecided.

Artifacts:

- exp\gradient_checks\sports_sharedinit_seed2022_v2\report.json; SHA256 5219ef3e3ccb7399bb484f42dd7edb7bc0dfe0281f7e3a555accbb68c48b74ff
- exp\gradient_checks\sports_sharedinit_seed2022_v2\batches.json; SHA256 d0489b97de4733f14b9dbd10bbea82dffdd1cbbc4420181a91681cf0ccfa75f0
