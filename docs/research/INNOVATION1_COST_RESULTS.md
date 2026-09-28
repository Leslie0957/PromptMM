# Innovation1 cost stage status

2026-09-28; source plan: [bounded cost roadmap](INNOVATION1_COST_ROADMAP_2026-09-28.md). This matrix tracks measurement status, not the existing 18-cell quality results.

| Stage | Scope | Status | Evidence |
|---|---|---|---|
| M0 | Sports nine selected students plus one shared teacher; export/request fidelity and bounded resources | Completed; one real non-formal smoke audited | [M0 audit](INNOVATION1_COST_M0_AUDIT.md): 9 student + 1 teacher exports, 12 Train-only requests, zero score delta, 13.953 s parent wall, limits passed, Test0/Validation0/updates0 |
| M1 | Same-checkpoint deployment preparation and cached service | `serial_v1` failed 0/30; separately authorized `serial_v2` completed 30/30 and accepted | [M1 audit](INNOVATION1_DEPLOYMENT_COST_AUDIT.md) and [v2 declaration](INNOVATION1_M1_DEPLOYMENT_RECOVERY_V2_2026-09-29.md); bounded direct timings, no new quality evaluation |
| M2 | Native early update windows, three methods × three seeds × three rounds | [One serial run prepared](INNOVATION1_M2_UPDATE_LAUNCH_2026-09-29.md); not launched | No new update measurements; 27-condition protocol and command ready for user review |

### M1 accepted cells

Each cell is the median across the three round-level measurements for the fixed selected checkpoint. Online numbers are cached-service batch1/128/1024 latency in ms on identical Train-only requests; offline is the runner's bounded representation preparation in seconds. One shared teacher is listed separately. Source and exact per-round raw samples: [v2 report](../../exp/efficiency/innovation1_cost_v1/m1/serial_v2/report.json).

| Method / seed | Offline s | Online batch1 / 128 / 1024 ms |
|---|---:|---:|
| BPR / 2022 | 0.100625 | 0.32800 / 0.49790 / 2.09450 |
| BPR / 2023 | 0.100598 | 0.33780 / 0.49535 / 2.09005 |
| BPR / 2024 | 0.101238 | 0.33160 / 0.49865 / 2.09565 |
| Full / 2022 | 0.099610 | 0.32930 / 0.49545 / 2.08840 |
| Full / 2023 | 0.100113 | 0.32810 / 0.49570 / 2.09885 |
| Full / 2024 | 0.103354 | 0.33715 / 0.49805 / 2.09505 |
| Release / 2022 | 0.157107 | 0.34745 / 0.49820 / 2.08220 |
| Release / 2023 | 0.171081 | 0.33595 / 0.49785 / 2.08715 |
| Release / 2024 | 0.172312 | 0.33585 / 0.50020 / 2.09350 |
| Shared teacher | 2.940174 | 0.33295 / 0.49960 / 2.09830 |

M1 does not establish 300-epoch total training cost, time to equal quality, dynamic refresh cost, or lossless cost reduction. M2 update cost remains unmeasured. No M1/M2 estimates are filled in from historical wall clocks.
