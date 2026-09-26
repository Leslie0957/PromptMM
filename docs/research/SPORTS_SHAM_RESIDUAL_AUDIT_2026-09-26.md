# Sports real-text versus geometry-matched sham: outcome audit (2026-09-26)

Status: **completed; six valid Validation-only runs**. User executed the declared serial cohort; this audit only read saved reports, curves and checkpoints. No model forward, gradient calculation, optimizer step, new Validation/Test ranking or retry was performed during the audit.

## Contract and provenance

- Protocol: [predeclared contract](SPORTS_SHAM_RESIDUAL_PROTOCOL_2026-09-25.md). Launch source `0f6d3ac62f55fec7e825237cf7b64eed0c4c1dbe`, branch `codex/experiment/baby-teacher-baseline`, clean source recorded and checked by the serial launcher before/after each child. Current tracked implementation fingerprints match all six manifests.
- Execution interval: `2026-09-25T17:46:19.635720+08:00` through `2026-09-26T11:26:47.082661+08:00`. Exact Python: `D:\miniconda\envs\run_5060\python.exe`; GPU0. The full command and ordered six child commands are preserved below.
- Common settings: Sports, frozen epoch37 teacher, random ID64 initialization, seeds 2022/2023/2024, 300 epochs/patience300, 214 batches/epoch, batch1024, student_lr6e-5, AdamW decay0.01. Objective BPR + 0.3*(Dimage + 0.3 Dtext)/1.3; user distillation rates zero. Earliest maximum Validation Recall@20 selects the checkpoint; NDCG is read at that same epoch.
- Within each seed both arms replay exactly the same saved initial user/item tensors and all 64,200 triplet batches, shape [300,214,3,1024]. All six arms use the same pinned six-tensor teacher semantic asset. Sham RNGs are 9022022/9022023/9022024; only item-text residual direction is randomized, preserving each item's image/text cosine, raw text norm and combined target norm.
- Shared anchors: SPORTS_CONVERTED_20260916 and SHARED_TD_TENSORS_PROCESS0_V1; prior paired cohort is context and replay source. Dataset/teacher identities in every manifest match the declaration; actual replay/semantic/target assets and manifest source fingerprints were rehashed by the existing read-only verifier.

```powershell
& 'D:\miniconda\envs\run_5060\python.exe' -B tools/run_sports_sham_residual_three_seed.py
```

Exact saved child argv:

```json
[
  [
    "D:\\miniconda\\envs\\run_5060\\python.exe",
    "-B",
    "codes/main_mmlight.py",
    "--dataset",
    "sports",
    "--student_profile",
    "sports_student_real_textresidual_seed2022_val300_v1",
    "--gpu_id",
    "0"
  ],
  [
    "D:\\miniconda\\envs\\run_5060\\python.exe",
    "-B",
    "codes/main_mmlight.py",
    "--dataset",
    "sports",
    "--student_profile",
    "sports_student_sham_textresidual_seed2022_val300_v1",
    "--gpu_id",
    "0"
  ],
  [
    "D:\\miniconda\\envs\\run_5060\\python.exe",
    "-B",
    "codes/main_mmlight.py",
    "--dataset",
    "sports",
    "--student_profile",
    "sports_student_real_textresidual_seed2023_val300_v1",
    "--gpu_id",
    "0"
  ],
  [
    "D:\\miniconda\\envs\\run_5060\\python.exe",
    "-B",
    "codes/main_mmlight.py",
    "--dataset",
    "sports",
    "--student_profile",
    "sports_student_sham_textresidual_seed2023_val300_v1",
    "--gpu_id",
    "0"
  ],
  [
    "D:\\miniconda\\envs\\run_5060\\python.exe",
    "-B",
    "codes/main_mmlight.py",
    "--dataset",
    "sports",
    "--student_profile",
    "sports_student_real_textresidual_seed2024_val300_v1",
    "--gpu_id",
    "0"
  ],
  [
    "D:\\miniconda\\envs\\run_5060\\python.exe",
    "-B",
    "codes/main_mmlight.py",
    "--dataset",
    "sports",
    "--student_profile",
    "sports_student_sham_textresidual_seed2024_val300_v1",
    "--gpu_id",
    "0"
  ]
]
```

## Preflight evidence (read from saved report)

All three saved preflight gates passed before training. Each seed has 18,352 valid rows and five unchanged degenerate rows. Largest raw-text norm error is 3.814697265625e-6, image/text cosine error 1.4901161193847656e-7, combined norm error 2.384185791015625e-7; all below 2e-5. The launcher required train-positive target validity. There was no Validation-based sham reselection or alpha adjustment.

| Seed | Real item-gradient RMS | Sham item-gradient RMS | Sham / real | Real / BPR RMS |
|---|---:|---:|---:|---:|
| 2022 | 0.003087916679 | 0.003088510979 | 1.000192460 | 2.331078 |
| 2023 | 0.003114434205 | 0.003113803336 | 0.999797437 | 2.355102 |
| 2024 | 0.003095763229 | 0.003095996112 | 1.000075226 | 2.337557 |

RMS is the square root of the mean squared full item-parameter gradient norm over eight fixed epoch0 train batches. Repeated positives contribute through the actual saved interactions. These are initial raw gradients, not AdamW update magnitudes. The cold-start RMS ratios above cannot be equated with the prior warm-start mean per-batch norm ratio of 1.045023%; the initialization and averaging estimand differ.

## Selected Validation outcomes

Epochs in this table are one-based; raw manifests use zero-based best_epoch.

| Seed | Real epoch | Real Recall@20 | Sham epoch | Sham Recall@20 | Real minus sham | Real NDCG@20 | Sham NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 294 | 0.082328259836 | 300 | 0.077468502122 | +0.004859757715 | 0.037207407800 | 0.035192384421 |
| 2023 | 292 | 0.082050198914 | 300 | 0.076926022677 | +0.005124176237 | 0.037094821672 | 0.035119391452 |
| 2024 | 293 | 0.082542735948 | 300 | 0.076222432799 | +0.006320303149 | 0.037306458146 | 0.034536445192 |

- Mean Recall@20: real **0.082307064899**, sham **0.076872319199**; mean paired difference **+0.005434745700** (+7.0698% relative to sham mean). All three differences are positive and exceed the predeclared mean threshold of 0.00209.
- Mean same-selected-epoch NDCG@20: real **0.037202895873**, sham **0.034949407022**; all three paired differences are positive.
- All six curves contain 300 finite values for total/BPR/distillation loss and selected Recall/NDCG. Earliest max-Recall epoch and saved selected Recall match manifests and full checkpoints. Selected model tensors are finite; inference user/item tensors equal the selected full checkpoint exactly. All six saved result records were independently reproduced with the existing read-only verifier. Exactly one manifest exists per declared profile.

## Experimental meaning and limits

Under the fixed Sports teacher, cold initialization, exact batches, objective and 300-epoch budget, the correct item-text residual directions outperform the predeclared random residual controls despite matching per-item target geometry and nearly identical initial total distillation gradient RMS. This meets the declared support criterion and strengthens the claim that the actual residual directions carry useful training information beyond this random second-direction control.

The intervention also changes cross-item residual structure. It does not separately identify natural-language semantics, item correspondence and cross-item geometry, establish that every possible second constraint is inferior, or prove a unique mechanism. Initial scale matching does not match later gradients or adaptive optimizer trajectories. Each training seed has one fixed sham seed; three pairs are not a significance test. All sham best epochs are at the final epoch300, so this is a fixed-budget result, not proof of fully converged superiority.

The current real-arm selected metrics reproduce the prior strict Full metrics; this does not imply byte-identical checkpoints. Prior Full minus image-matched mean Recall advantage +0.004177997837 remains a separate comparison. Earlier F-versus-1:0.3 geometry and warm-start near-ties remain valid: closeness to teacher final F alone does not establish recommendation quality, and this cold-start result does not demonstrate a warm-start gain. It neither measures inference speed nor removes the potential value of ID-only deployment.

No teacher/student Test ranking occurred in these six runs: run_final_test, final_test_performed and teacher_final_test_performed are false throughout. Existing trainer structural Test reads remain part of preflight. Audit itself opened no Validation/Test split matrix. Final Test metrics and selected-versus-tested checkpoint equality are **N/A**; selected-versus-exported equality passed. paper_ready_eligible=false. No Baby/Test generalization claim; Baby2023 accepted audit exception/originalmanifestfalse and Innovation2 undecided remain unchanged.

## Artifact identity ledger

All paths below are workspace-relative under D:/Download/PromptMM. Ignored run assets were not modified or committed; no physical/off-device backup is evidenced. Source commits preserve tracked source and this audit, not ignored artifacts.

| Artifact | SHA256 |
|---|---|
| exp/sham_residual/sports_three_seed_v1/batch.json | `32cd2b6af6b1c851b358d0b58af3c6dc26ff873954eb09cf8cc410677c10244e` |
| exp/sham_residual/sports_three_seed_v1/preflight.json | `058589544332696aabfae90c9b8eafe45a9d3d0f6b196f099ac5ab71ecd80102` |
| tools/run_sports_sham_residual_three_seed.py | `f718015eabdeb6976d8081f139e5695b23ecd57d329b729e3ffdb9b656019b11` |
| Prior paired batch | `b254d9e969edd3e001a4a85c2b14b69f8eb68b3d279d667100973c62fe7797b8` |
| Shared teacher semantics | `e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20` |
| train_mat | `5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8` |
| Frozen teacher checkpoint | `57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea` |

### Seed 2022 shared assets

- Initial tensor SHA256 `2357868cefeb086dba14c3808415eac97936f67cf5e6bef8382d95ff43bcffff`; tape SHA256 `e8d982c6a2d1be20a934502d4a1ad6ba7e2b699b21098668343341f10db3d77b` under `exp/paired_coldinit/sports_three_seed_v1/seed2022/initial.pt` and `triplets.npy`.
- Sham target `D:\Download\PromptMM\exp\sham_residual\sports_three_seed_v1\seed2022\sham_text.pt`, SHA256 `d9d12790af5294355aebe65ddd2ee801ca9e2e48ef7c8c5d27f98ece4f577d17`.

### Seed 2023 shared assets

- Initial tensor SHA256 `bb5429aa8a1edfe53b36fed977146460d1101ae79539d2badf856d5514ef9d85`; tape SHA256 `3763128ba003228d60c09924b9370b4cab0b8c16679366742835fc4559d89085` under `exp/paired_coldinit/sports_three_seed_v1/seed2023/initial.pt` and `triplets.npy`.
- Sham target `D:\Download\PromptMM\exp\sham_residual\sports_three_seed_v1\seed2023\sham_text.pt`, SHA256 `5c388e253ff59eb3f9bbbad1a2a7d87448601edc8c16066a907288deb0540f85`.

### Seed 2024 shared assets

- Initial tensor SHA256 `e2c6885debdec09a8ff8401fe4f4f96ea5f46dcc1fd90f24dcc4ecdbd2188e37`; tape SHA256 `131cc14ebe6853e79216183f06923f3689d47d26c69eac2521d7f93b1b314bc7` under `exp/paired_coldinit/sports_three_seed_v1/seed2024/initial.pt` and `triplets.npy`.
- Sham target `D:\Download\PromptMM\exp\sham_residual\sports_three_seed_v1\seed2024\sham_text.pt`, SHA256 `1874806335056e3d8bbf7ffccb71d4240e520e9161e3bf8b419b05eeafba99e3`.

### sports_student_real_textresidual_seed2022_val300_v1 (completed)

- Manifest: `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-25 17_46_36.818082_sports_light_init_pid56688.json`; SHA256 `da4d884a3a025b406aa31a0ad1c5eadf0446076af89025159f6b9b78396a5a86`.
- Selected full checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-25 17_46_36.818082_sports_light_init_pid56688.pth`; SHA256 `81e802609e1f0836e9816d2528763c656805a1940ae15fab244698b744dd8819`.
- Inference checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-25 17_46_36.818082_sports_light_init_pid56688.pth`; SHA256 `6658c6878b06aa48764cdadaf5fe74dc178df15ca2d105dac325e7dd0ae9cf25`.
- Curve: `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-25 17_46_36.818082_sports_light_init_pid56688.pkl`; SHA256 `4d5255d667e235070ffab6e10cc8270a0465c50e40cb9637a1f9c54ee67158e1`.

### sports_student_sham_textresidual_seed2022_val300_v1 (completed)

- Manifest: `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-25 20_41_43.842353_sports_light_init_pid9624.json`; SHA256 `2850a9957ff86e45d2e664456eb8375f5341b298eaf7c28fd0073bcf1e99ab51`.
- Selected full checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-25 20_41_43.842353_sports_light_init_pid9624.pth`; SHA256 `17998b0465741cc934ddf29762406eccea73690aa302536388568efcfad8f3aa`.
- Inference checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-25 20_41_43.842353_sports_light_init_pid9624.pth`; SHA256 `df7c8f7bc360ac4813a4f328920de6a2d86f6e70bda2d002e3c912eeb255faf6`.
- Curve: `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-25 20_41_43.842353_sports_light_init_pid9624.pkl`; SHA256 `c6d7956bf705cbc83efea63c6fc705368c6cb01d6a7d5cd86eff8a5032502c65`.

### sports_student_real_textresidual_seed2023_val300_v1 (completed)

- Manifest: `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-25 23_42_04.894528_sports_light_init_pid15332.json`; SHA256 `f533fd24d899aee50c1915b26c848ace3f96d138efdee915e596596f566562ef`.
- Selected full checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-25 23_42_04.894528_sports_light_init_pid15332.pth`; SHA256 `8fac9e37fb5199e1474d1a974ab33bb3be1b1141dad1443f34ea038674e4d877`.
- Inference checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-25 23_42_04.894528_sports_light_init_pid15332.pth`; SHA256 `e19b2ae1d4c0fc273e6d8b2d367d443e6b09c5e2f25ccea3fb18a18216902315`.
- Curve: `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-25 23_42_04.894528_sports_light_init_pid15332.pkl`; SHA256 `60fa7daf32581850becf0dd3a2286e55ccc9a7326800d4a9b707b2bf6049cc0b`.

### sports_student_sham_textresidual_seed2023_val300_v1 (completed)

- Manifest: `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-26 02_38_22.563718_sports_light_init_pid596.json`; SHA256 `389e96d21242eaa0844cdac993ff327d56bc3abad15bc0c2d3ea73b49044438c`.
- Selected full checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-26 02_38_22.563718_sports_light_init_pid596.pth`; SHA256 `d58c766700099d928d18c0dbcd73d1f7108a3294a9b912f423ea52f1cf10e9a2`.
- Inference checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-26 02_38_22.563718_sports_light_init_pid596.pth`; SHA256 `fbb8d44cddf551cbad3749a0546ec7aaaddad6a5fd181a4b447ccfec3dd8f307`.
- Curve: `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-26 02_38_22.563718_sports_light_init_pid596.pkl`; SHA256 `0a8d455a7c1292d195c21ce8a9c1481cf1571c206ffe200c7df267891d93adff`.

### sports_student_real_textresidual_seed2024_val300_v1 (completed)

- Manifest: `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-26 05_34_16.930265_sports_light_init_pid42156.json`; SHA256 `2ed3c0cc86f7388d70434f8407c7dd47adf209115c8ea5c7be06470f2c053908`.
- Selected full checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-26 05_34_16.930265_sports_light_init_pid42156.pth`; SHA256 `3eecf0e85df00018485fde36ebd6a87a72ac11eaa66004c7a7a95a99f037edd0`.
- Inference checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-26 05_34_16.930265_sports_light_init_pid42156.pth`; SHA256 `c83981118198b14b3b7c4d17ecc8c2a7f6f60c3da03fa4336df4a17ced7bfe29`.
- Curve: `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-26 05_34_16.930265_sports_light_init_pid42156.pkl`; SHA256 `416f2c3d9d7acdd4f48b6ebb9d53bb363b4d03f922134af381bbb07bd4a94809`.

### sports_student_sham_textresidual_seed2024_val300_v1 (completed)

- Manifest: `D:\Download\PromptMM\exp\runs\sports\run_manifest__2026-09-26 08_30_07.418852_sports_light_init_pid30132.json`; SHA256 `b728fdfad2da6edc9784dea4c035e777210f0196afaf417f8e9e7e35f534270b`.
- Selected full checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_full__val_test_once_v1__2026-09-26 08_30_07.418852_sports_light_init_pid30132.pth`; SHA256 `5500922727544ed6d35d43b647ec99223af1d7b08a52135ec08c91adb71dffee`.
- Inference checkpoint: `D:\Download\PromptMM\Model\sports\td_distill\td_distill_infer_only__val_test_once_v1__2026-09-26 08_30_07.418852_sports_light_init_pid30132.pth`; SHA256 `205156875817f14e1d77b529fe083b93208fdcd96c02087eea357637cd8610c5`.
- Curve: `D:\Download\PromptMM\exp\converge\sports\auto__2026-09-26 08_30_07.418852_sports_light_init_pid30132.pkl`; SHA256 `6d3840080f23746ceb5f0b27d950611ff5134281ad60185a9fb2b387e83b28f7`.

## Single next step

Read-only synthesis of this sham-control result with the strict Full/image-matched and warm-start evidence to freeze the supported mechanism claim and its remaining limits before choosing any separately authorized experiment. No extension, new seed, Test run, tag or merge is authorized by this audit.
