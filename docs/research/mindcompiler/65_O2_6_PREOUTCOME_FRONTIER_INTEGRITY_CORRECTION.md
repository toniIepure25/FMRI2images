# O2.6 — Pre-Outcome Frontier Integrity Correction

**Status label:** `O2_6_PREOUTCOME_FRONTIER_INTEGRITY_CORRECTION_RESTART`
(not a scientific failure).

## What was frozen / sealed (unchanged)

- O2.6 methodology frozen at commit `8d5849b626cc184bd45cb661782ff32dc57f5f19`
  (config `config_sha256 f4e94517e0032844547d7b9ccebeb166ed55686ebd4e979b6b9a267bd668f160`).
- Native-residual state extension sealed at commit `45a1813e6ff143e3f44d816eaa977c9920b3c761`
  — `O2_6_STATE_EXTENSION_OK`, max replay discrepancy `0.0`, 96/96 cells certified.

## What happened

- The frontier Job had begun compute. **No real frontier result or status was inspected**
  (no `R_AUG` / `E_AUG` / `TOTAL_RECOVERY` / terminal p-value / `D50` / `M50` / ROI or
  program status). The only partially-written file was the 53-byte sealed-state hash check
  (no scientific metric); it was quarantined/deleted.
- A static code review before outcome identified provenance/evaluability-handling gaps.
- The running Job was stopped before any outcome inspection.
- Preserved intact: frozen config, sealed RD/O2.4R state, the certified native-residual
  extension files, raw NSD data. The extension was **not** regenerated (no hash mismatch found).

## Fixes (implementation-integrity only; scientific methodology unchanged)

1. **Extension hash verification** (`--ext-manifest`): before any compute, every
   `deltanat_<s>_<roi>_fold<f>.npz` is checked against the committed extension manifest —
   existence, exact filename, SHA256, byte size, stored `delta_native.shape`, exactly 10
   training residuals, and voxel dimension == the sealed ROI voxel dimension. Require
   96 checked / 96 matched / 0 mismatch, else `O2_6_NATIVE_RESIDUAL_EXTENSION_HASH_FAILURE`.
2. **Identity-order verification** (not only file hash): `extension.train_ids` must equal the
   sealed RD cell's `train_ids` element-for-element and in the same order; extension
   `voxel_hash` must equal the sealed ROI voxel hash; family labels must align. Else
   `O2_6_NATIVE_RESIDUAL_IDENTITY_ALIGNMENT_FAILURE`. (Calibration subset indices index both
   X/Z rows and native-residual rows — they must refer to the same identities.)
3. **rd_results provenance** (`--rd-results-sha256`): the `rd_results.json` used for
   `R_NATIVE_ORACLE` is verified against the committed/sealed SHA256; mismatch stops the gate.
   The verification artifact records path, SHA256, expected SHA256, match boolean.
4. **No silent drop of rank-insufficient subsets**: a rank-insufficient balanced subset is
   marked `RANK_INSUFFICIENT`; a fold's `(M,d)` cell is evaluable **only if every** frozen
   balanced subset (25/100/100/25/1) is evaluable, else `FOLD_MD_RANK_INSUFFICIENT` (no
   averaging over the survivors).
5. **Participant/group evaluability**: a participant `(M,d)` requires all 6/6 outer folds
   evaluable, else `PARTICIPANT_MD_RANK_INSUFFICIENT`. Group median/status rules apply only
   when all 8 participants are evaluable; `n_evaluable_participants` is recorded for every cell.
6. **Terminal completeness**: the confirmatory cell `M=10,d=10` is valid only with 1/1 subset,
   6/6 folds, 8/8 participants, both primary ROIs evaluable. If rank-insufficient anywhere, the
   affected ROI is `TARGET_BASIS_AUGMENTATION_INCONCLUSIVE` — no sign-flip on fewer
   folds/participants, no substitution of `d=8` or the maximum observed rank (d=10 was frozen).
7. **D50/M50 completeness**: a candidate D50/M50 cell qualifies only with
   `n_evaluable_participants == 8` (every participant used all six folds and every frozen subset).
8. **Outside-oracle denominator diagnostic**: the frozen `OUT_BASIS_RECOVERY` formula is
   unchanged; `R_COND − R_BASE` is recorded per participant×M×d, and cells with
   `R_COND ≤ R_BASE + 1e-12` are flagged `DENOMINATOR_DEGENERATE` (diagnostic only; does not
   affect terminal feasibility / D50 / M50, which depend on `TOTAL_RECOVERY` and `E_AUG`).
9. **Tests** (kept all 15 prior O2.6 tests): extension SHA mismatch, swapped file, wrong row
   order, wrong voxel_hash, and changed rd_results all block execution; one rank-insufficient
   subset makes the fold cell non-evaluable (no selective averaging); one non-evaluable fold
   makes the participant cell non-evaluable; terminal cannot run without 8 complete effects;
   D50 cannot be chosen from an incomplete cell.

## Unchanged (explicitly)

M grid, D grid, numerical-rank threshold, outside-support definition, SVD basis, Procrustes
`P_IN`, random-outside null (frozen seed strings, `N_NULL=100`), `TOTAL_RECOVERY`, terminal
Holm family of exactly 2, D50/M50 thresholds, and program-status rules are all unchanged.
On rerun, seed strings remain exactly frozen (not re-seeded on the new commit), and the
interrupted run's partial output is not reused. `O3` remains `O3_NOT_READY`.
