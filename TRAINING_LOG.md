# Training Log — current state and new records

Canonical experiment memory: this short active log plus verified immutable history.
Read README.md and AGENTS.md first; expand only the evidence relevant to the task.
Updated: 2026-09-26. Latest audited source: f9d0eb504cb0fb17eddcb58cba8aef2517e85979.

## Current state

- 写作阶段澄清（2026-09-26）：已有证据足以开始第一创新点的阶段性写作，不代表整篇论文实验完成或可定稿。第二创新点尚未确定并完成实验；当前主要证据集中在Baby与Sports，前者有正式Test，后者相关机制结果主要为Validation，不能计作两套同等完整的正式验证。历史其他数据集记录不能自动补足覆盖。整篇论文是否达到学位要求仍需核对第二创新点、正式评价、数据集覆盖及实际要求，当前不能保证。

- Sports mechanism and cached-deployment efficiency stage is complete. No run is pending for automatic execution; old pending declarations are historical, not a work queue.
- Baby already has three-seed BPR/Full/original Image-0.3 formal results and seven fixed120 Validation diagnostic runs documented in the Baby evidence book. Broader thesis claims and final publication freeze remain unfinished; do not describe Baby as only a seed2022 baseline.
- Baby seed2023 Image-0.3 is accepted by user-audited exception; original paper_ready_eligible=false remains unchanged. Seed2024 low outcome is valid and preserved.
- Sports random-init strict Full/image mean Val Recall delta +0.004177997837; real/sham residual +0.005434745700, all three pairs positive. Correct text residual structure helps under the tested teacher/budget; not a unique semantic mechanism, significance or convergence proof.
- Warm-start seed2022 BPR0.095477873096 versus Full0.095438076853: no demonstrated incremental distillation benefit. Equal matched target0.095383454558; alpha3 best0.094367089719. No global alpha/target optimality claim.
- PromptMM release adapter lr6e-5 mean0.08757300 exceeds random-init Full0.08230706; no blanket claim of superiority over PromptMM.
- Efficiency v2 completed36/36: cached online costs/table size are similar. Offline teacher generation15.889450ms versus Full table copy0.109900ms is a phase-specific contrast, not end-to-end/online/training speedup. Failed v1 preserved.
- Sports quality remains Validation-only/paper_ready_eligible=false. Historical TD loaders structurally read Test but did no Test ranking; deployment benchmark used no held-out split files at all. No new Test evaluation is authorized.
- Innovation2 remains undecided. No new model, sweep, training, benchmark or Test run requested by the current organization task. All real experiment execution remains user-manual.

## Core navigation (read selectively)

| Need | Entry |
|---|---|
| All experiment families and latest audited result | [Experiment map](docs/experiments/README.md) |
| Thesis completion conditions and gaps | [Gap table](docs/paper/THESIS_COMPLETION_GAPS_2026-09-26.md) |
| Paper results and limitations | [Reviewed paper wording](docs/paper/SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md) |
| Baby formal/Validation evidence and exceptions | [Baby evidence](docs/research/INNOVATION1_THESIS_EVIDENCE_2026-09-15.md) |
| Latest Sports mechanism | [Mechanism conclusion](docs/research/SPORTS_INTEGRATED_MECHANISM_CONCLUSION_2026-09-26.md) |
| Latest Sports efficiency | [Efficiency audit](docs/research/SPORTS_CACHED_DEPLOYMENT_AUDIT_2026-09-26.md) |
| Every source/document path | [File navigation](docs/experiments/FILES.md) |
| Saved run/report metadata, including failures | [Run records](docs/experiments/RUN_RECORDS.md) |
| Exact historical declarations/outcomes/commands | [History section index](archive/training/ENTRY_INDEX_2026-09-26.md) |
| Raw local file inventory | [Archive catalog](archive/catalog/README.md) |

## Active implementation and protocol

- Training: codes/main_mmlight.py; defaults: codes/utility/parser.py; CLI overrides defaults.
- Standalone codes/run_patent.py and legacy Photo/LATTICE material are separate historical lines.
- Formal protocol val_test_once_v1: Validation selection then declared one-time student Test where authorized; Validation-only diagnostics retain their explicit no-final-Test policy.
- Preserve dataset/teacher/initial/tape hashes via named anchors in the original declarations and focused audits. No need to rehash all unchanged assets for ordinary documentation tasks.
- New behavior/protocol/interpretation changes need pending then completed/failed entries and a coherent local commit. Full operational rules are in AGENTS.md.
- Never treat a historical command as permission to rerun. Failure/low results remain valid history. Do not overwrite ignored assets; Git commits do not back them up.

## Verified immutable history

The complete previous active log, including this reorganization's pending entry, was copied byte-for-byte BEFORE slimming:

- [2026-09-26 full snapshot](archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-09-26.md): **1,004,172 bytes; 11,527 lines**.
- SHA256: `9def8d80aeba27e4482290444a7e514c568ad7d831f75c07b43a5da0eb95409b`.
- The archive index records the same identity; Git -text preserves exact original EOL bytes; local read-only attribute protects against accidental edits.
- [Earlier archive index](archive/training/README.md) retains April/July snapshots and handoffs. No historical entry was deleted or rewritten. Archive sections can contain superseded plans or failed attempts.
- Check the section index first, then read a bounded historical section. Do not load all snapshots at task startup. The catalogs are navigation, not new audit acceptance.

## Current authorized task / handoff

The thesis completion/gap table is now prepared at docs/paper/THESIS_COMPLETION_GAPS_2026-09-26.md. Existing Baby/Sports evidence is separated from missing formal coverage, novelty and Innovation2 evidence; degree-specific requirements remain unconfirmed. Partial writing may proceed; whole-thesis readiness is not established. Single next step: a read-only Innovation1 prerequisite review and Innovation2 candidate decision memo, without selecting RGCS by default or declaring the implementation gate passed. No new experiment is authorized.

## 2026-09-26 Paper results prose and experiment navigation reorganization (pending)

- User authorizes writing paper results/limitations and organizing experiment navigation/history so new agents read only core state first. Scope: Chinese paper draft; root overview/current-state log; experiment/document/source/raw-artifact catalogs; indexed verbatim log snapshot; original upstream README archive; startup guidance aligned with targeted reading; small stdlib catalog builder for future maintenance. Historical source/scripts, datasets, checkpoints, manifests, raw outputs and failure/low results retain their paths and bytes. No model behavior, metric, parameter, execution environment or training protocol changes; no training/forward/gradient/Validation/Test access.
- Before slimming: copy the complete active TRAINING_LOG.md including this pending entry byte-for-byte to archive/training/TRAINING_LOG_ARCHIVE_FULL_2026-09-26.md; verify bytes and SHA256, record in archive index and new active log; use Git -text for this snapshot to preserve exact bytes and mark its local file read-only. Never rewrite that archive. Prior April/July archives remain untouched. Read-only metadata inventories index all reachable experiment records/files without interpreting repeated reports as independent runs; private tool caches are excluded and exclusions documented.
- Risks: losing old commands/identities, breaking hard-coded paths, stale header/plan mistaken for authority, false completeness or duplicate experiment counts, archive EOL conversion, unsupported paper claims. Acceptance: original log preserved exactly including pending entry; all old headings indexed with source/line; shorter current log retains Baby formal evidence/seed2023 exception and latest Sports boundaries; full tracked-file navigation and experiment artifact paths discoverable; source links and paper numbers checked; raw assets unchanged; no old plans treated as runnable authorization. Basef9d0eb504cb0fb17eddcb58cba8aef2517e85979; no reset/revert/delete/tag/merge. Verification: snapshot byte/hash and Git-blob identity, generated catalog coverage, local links, focused diff and one coherent commit. Next action after completion: hand off the new overview/paper draft; no further experimental stage authorized.


## 2026-09-26 Paper results prose and experiment navigation reorganization (completed)

- Completed the authorized documentation/catalog stage on `codex/experiment/baby-teacher-baseline`, based on source `f9d0eb504cb0fb17eddcb58cba8aef2517e85979`. This outcome and its coherent commit identify the organization change; no formal launch source is introduced.
- Added Chinese paper prose separating text-direction gains, warm-start limits and offline/online efficiency, with linked audits and explicit Validation-only, statistical, initialization, teacher-semantic, PromptMM and resource boundaries. Checked the quoted metrics against the existing focused audit tables; no new result or eligibility decision.
- Replaced stale root/docs entry pages with layered navigation; updated AGENTS targeted-startup guidance. The active log now holds core state and new append-only entries. Earlier Baby evidence, 2023 exception, low/failed runs and undecided Innovation2 remain discoverable.
- Before replacement, the complete working log plus pending declaration was copied and verified: 1,004,172 bytes / 11,527 lines, SHA256 `9def8d80aeba27e4482290444a7e514c568ad7d831f75c07b43a5da0eb95409b`. Its hash is recorded here and in the archive index; `-text` preserves checkout bytes in Git. April/July snapshots remain unchanged. Original upstream README preserved at 5,000 bytes, SHA256 `f70c58347906f8da904ae3070aee4218387475d7ea1227c09296fda25c7651c1`. Original HEAD contents match the preserved prefix/original after accounting for preexisting checkout EOL normalization.
- Metadata-only catalog generation completed: 202 source/document paths, 1,365 local asset paths, 217 JSON records (including 71 run manifests), 279 historical section locators. All 217 JSON records parsed; source navigation and existing exp JSON coverage checked. 517 local Markdown links resolved; immutable original documents retain their historical relative links. New mutable content passes focused whitespace checks; archived whitespace is deliberately preserved.
- Verification used only stdlib file/path/JSON operations and Git. No training, model forward, gradients, optimizer steps, matrix/weight loading, Validation/Test split reads, ranking evaluation or experiment rerun. Source runtime files and runtime asset paths were not changed. Final staged/committed archive byte identities and clean tree are checked at handoff.
- Limitations: catalogs describe current reachable local files and are refreshable, not new audit results or independent experiment counts. Private tool/cache directories are excluded explicitly. This is routine organization, not a publication freeze; no tag, main merge, asset deletion or new off-device backup. Ignored assets still require their own backup.
- Single next step: review `docs/paper/SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md` for incorporation into the thesis. No further experiment is authorized by this handoff.

## 2026-09-26 Paper wording review (pending)

- Authorized scope: review the Sports paper draft against existing audits, choose body/discussion/appendix wording and revise the same paper file plus its navigation/current-state references. Base `a178d981a06630a2b370d02b0010edcbb008e502`; clean branch `codex/experiment/baby-teacher-baseline`. No experiment, metric, protocol or eligibility change.
- Rationale/risks: distinguish image-coefficient control from total-gradient control, initial matching from full training, fixed-budget Validation evidence from Test/general claims, and offline table generation from cached online cost. Avoid implying full baseline tuning, unique semantics or causal initialization interaction; clearly attribute alpha3 below-initial performance to its final value only.
- Acceptance/verification: every quantitative claim linked to existing focused audits; separate quality/efficiency tables and paste-ready prose from editorial notes; preserve negative outcomes and limitations; local links and focused diff pass. Refresh only metadata navigation if file sizes change. No model/data loading, training, new evaluation or Test access. Previous draft recoverable from the base commit; no automatic rollback. One coherent local commit, then hand off reviewed wording; no new run authorized.

## 2026-09-26 Paper wording review (completed)

- Revised the existing paper file into reviewed body sections (random-init text benefit; warm-start boundary; offline/cached online cost; limitations), separate quality/efficiency tables, and discussion/appendix/editorial notes. Updated the existing overview links; no new experiment or file family.
- Retained all limiting/negative evidence: Validation-only status, three pairs without significance, finite budget, single-seed warm start, stronger PromptMM configuration, initial versus trajectory gradient matching and shared student deployment structure. Explicitly distinguished image-coefficient control from total-gradient matching and alpha3 final-below-initial from best-above-initial. The 144.58 phase ratio and F/equal/alpha3 diagnostics are appendix/discussion material, not headline inference claims.
- Verification: checked the existing strict-pair, sham, warm-pair, equal-target, weighted-direction, alpha3, release-baseline and latest efficiency audits; recomputed quoted relative gains/difference/phase ratio with scalar arithmetic; all edited-document local links and focused whitespace checks passed. Navigation asset-size metadata refreshed only; immutable history and source/runtime assets unchanged.
- Based on `a178d981a06630a2b370d02b0010edcbb008e502`, branch `codex/experiment/baby-teacher-baseline`; preserved as one local documentation commit. No training/model forward/gradient computation, dataset/checkpoint load, Validation/Test split access or evaluation. No change to original metrics, eligibility, protocol or Baby exceptions. This is wording review, not publication freeze or full-thesis integration.
- Single next step: incorporate sections 1-5 of `docs/paper/SPORTS_RESULTS_AND_LIMITATIONS_2026-09-26.md` into the thesis with its actual chapter/table numbering. No new experiment authorized.

## 2026-09-26 Thesis readiness clarification (pending)

- User asks to preserve the latest conversation correction. Scope: current-state handoff, root overview and paper editorial note; refresh affected file-size catalog metadata. Base `650bb42`; clean existing branch. Clarify partial writing readiness versus whole-thesis evidence sufficiency without altering any result/protocol or drafting new experiment plans.
- Acceptance: explicitly preserve Innovation2 unfinished, Baby formal Test versus Sports Validation distinction, limited dataset coverage and unknown thesis acceptance requirements; replace the current next-step recommendation with a thesis completion/gap table. Prior outcome entries remain verbatim. Verify focused diff/links; no training, data/model load or Test access. Prior source is recoverable from base commit; no rollback performed.

## 2026-09-26 Thesis readiness clarification (completed)

- Recorded the latest correction in the root overview, current log header/handoff and paper editorial note: partial writing is supported; whole-thesis sufficiency is not established. Innovation2 is unfinished; Baby formal Test and Sports mechanism Validation are not equally complete dataset validation. Existing controlled findings have bounded value, without a guarantee of meeting unknown degree requirements.
- The current recommendation is a thesis completion/gap table before selecting Innovation2; the table itself has not been prepared. Earlier outcome entries remain historical and unchanged. No experiment was authorized or executed, and no data/model/Test access occurred.
- Verified focused content/diff and local links; refreshed only the three changed documents' byte counts in the existing navigation catalog. Base `650bb42`, branch `codex/experiment/baby-teacher-baseline`; one local documentation commit. Metrics, protocols, original audits and immutable archives unchanged. Next step: prepare the completion/gap table when requested.

## 2026-09-26 Thesis completion conditions and gap table (pending)

- User authorizes a documentation-only completion/gap map for the whole thesis. Base `e3129ab`, clean branch `codex/experiment/baby-teacher-baseline`. Scope: new paper planning document, current navigation/handoff links and generated metadata catalogs; no experiment preparation, implementation, formal declaration or execution.
- Rationale: distinguish audited Baby/Sports evidence, contribution novelty, second-point prerequisites, formal evaluation, dataset breadth, writing and milestone preservation. Historical 09-14/09-15 plans must be reconciled with completed Sports controls/efficiency rather than copied as outstanding runs. Degree-specific contribution/dataset/baseline requirements are unknown; request them optionally and label any working targets as provisional.
- Risks/acceptance: do not turn Validation into Test, exception into clean automatic acceptance, negative evidence into failure, or old RGCS hypothesis into verified contribution. Each gap needs evidence/status, closure criteria, priority and whether new execution would require separate authorization. One recommended next stage only; preserve historical entries and all artifacts. Verify source links, focused diff and catalog coverage. Source rollback point is base commit, not asset recovery; no rollback planned. No data/checkpoint loading, training, gradient/forward computation or Test access.

## 2026-09-26 Thesis completion conditions and gap table (completed)

- Added `docs/paper/THESIS_COMPLETION_GAPS_2026-09-26.md`: 11 evidence/closure gaps, dataset/split coverage, updated Innovation1 prerequisite status, conditional Innovation2 milestones, formal-evaluation boundaries and one next planning step. Linked it from the root/docs/family/paper/current-log entry points; refreshed navigation metadata.
- Reconciled historical plans with completed Sports controls and efficiency: do not repeat completed stages by default. Baby formal Test/2023 exception/2024 low result retained; Sports Validation not promoted to Test; old Amazon not counted as qualified coverage. No blanket baseline superiority, global novelty, significance, completed second point or graduation guarantee asserted. Dataset/degree requirements requested optionally and remain unconfirmed; no fixed third-dataset or run-count mandate invented.
- Verification: source-checked Baby evidence book, Sports three-seed audit, current mechanism/efficiency/baseline evidence and the historical second-point prerequisites; local links and focused whitespace checks passed. Catalog covers the new document. Base `e3129ab`, existing `codex/experiment/baby-teacher-baseline`; one documentation commit preserves this stage. Original audit/log archive files and runtime assets unchanged. No training, implementation, formal declaration, checkpoint/data loading, new metrics or Validation/Test access.
- Single recommended next stage: read-only Innovation1 prerequisite review and Innovation2 candidate decision memo, resolving retained claims and unmet gates before preparing any diagnostics. This table does not pass the historical implementation gate or authorize an experiment. All actual experiment execution remains user-manual.
