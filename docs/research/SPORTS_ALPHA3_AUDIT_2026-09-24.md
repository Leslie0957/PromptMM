# Sports alpha3 outcome audit — 2026-09-24

Valid completed run from clean a4017bab544ed7c78dfd9be2ce8e02b1b2e09449 on codex/experiment/baby-teacher-baseline. Manual command: `& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sharedinit_alpha3.py`. Full child command/common identities in prior declaration and run.json. Started09-23 23:05:25,finished09-24 02:14:49 +08:00 (3h09m).

Seed2022,300epochs,patience300,lr6e-5,batch1024,dim64,AdamWdecay0.01. Only behavioral argument deltaalpha0.3->3.0; profile name/config profile/config source metadata also change. Same shared initialization, semantic targets and initialValidation. InitialValidation excluded from selection. All300 losses/selection curves finite; earliest maximum Recall20 and checkpointmetadata/finiteweights/hash verified; current source fingerprints and shared asset match. No new training/forward/ranking by assistant. Teacher/student Test flagsfalse; existing Teststructuralreads retained,selected-versus-tested N/A,eligibilityfalse.

| alpha | Best epoch (1-based) | Recall20 | NDCG20 at selected epoch | Last20 mean Recall20 |
|---|---:|---:|---:|---:|
|0 BPR|286|0.095477873096|0.043805670602|0.095232970136|
|0.3|283|0.095438076853|0.043780603134|0.095234881917|
|3.0|3|0.094367089719|0.043365329516|0.092193144649|

Alpha3 finalRecall0.092054380611 below commoninitial0.094184495191; minimum0.091965190207 at epoch270. Bestprefix30/120/200/300 all0.094367089719. Distillation loss epoch sum4.034550214->2.860641601, versusalpha0.3 4.038396500->3.830487356. BPR epoch sum3.213209464->1.283759787. These are214-batch epoch sums. Stronger semantic fitting did not improve validation ranking. Valid negative evidence, not executionfailure; no rollback or automatic extension.

Supports an objective/target tradeoff hypothesis in this warm-start configuration, not proof of its exact cause, universal distillation failure, or text harm. Small initial rawgradient did not imply that10xalpha would help. Single seed and Validation sensitivity cannot establish statistical significance or generalization. Cold-init evidence retains separate scope.

Single nextstage: read-only mechanism review of teacher-derived initial ID vectors versus image/text targets and directional constraints, incorporating cold-init evidence. Identify a concrete inconsistency before proposing changes. No automatic alpha sweep, longer budget, gate or multiseed expansion; no new experiment prepared here.

Babyseed2023 auditexception accepted while originalmanifestfalse remains; Innovation2undecided. All oldresults preserved. Ignored artifacts are not backed up byGit.

Artifacts:

- exp\initialization_checks\sports_sharedteacherinit_alpha3_seed2022_val300_v1\run.json; SHA256 8b924097c9e7471c906c292d39ddd3c730d7a4e2379b4eca05fa8e8388c7e657
- D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-23 23_05_29.825139_sports_light_init_pid44284.json; SHA256 71293fa6d249b5a40a636cea60c4b21bcb626b68b66623a82f58d3bac78db9ff
- D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-23 23_05_29.825139_sports_light_init_pid44284.pth; SHA256 4ea04e859703f54c49f28568b2d0b99fe9df08bf21a7d5199a9b16ceca6275b3
- D:\Download\PromptMM\exp\converge\sports\auto__2026-09-23 23_05_29.825139_sports_light_init_pid44284.pkl; SHA256 689dd4126aafabe4d8391f09dfcdbf7531eefb78b8b6d5455507ce7df641a557

Relative Recall alpha3 versus BPR: -1.1633935081762692%.
