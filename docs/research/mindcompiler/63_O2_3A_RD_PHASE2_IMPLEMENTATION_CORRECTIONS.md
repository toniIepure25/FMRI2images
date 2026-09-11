# O2.3A-RD / O2.4R Phase-2 — Pre-Outcome Implementation Corrections

**Status label for the interrupted run:**
`O2_3A_RD_PHASE2_PREOUTCOME_IMPLEMENTATION_CORRECTION_RESTART`

This is **not** a failure, **not** an inconclusive scientific result, and **not** a
scientific outcome. A static code review detected implementation-contract mismatches
and the run was stopped **before any scientific outcome was produced or inspected**.

## Frozen methodology is unchanged

- O2.3A-RD frozen config: `c1b2ddb0414ad541e68ab8435e51306d9490d9b287735070964045ea9fbd33f0`
- O2.4R frozen config: `ad9594461bf6c3d3b3fa208a32d19b6dc30037cf0a1c6ade1a60466015127989`

No K candidates, block assignments, rank folds, null count, inference rule, Holm
procedure, M\_STAR thresholds, or status definitions were changed. These corrections
**only align the implementation to the already-frozen contracts**.

## What was (and was not) done on the interrupted run

- Discovered by **static code review**, before any scientific outcome.
- The stopped Job (`rd-fit`, code `c61ab82`) had reached only **input / anchor
  assembly** — all 8 subjects' anchor centroids were assembled (anchor reps 3/3/3;
  ventral/lateral voxel counts logged) as input validation.
- **No real K-selection result was inspected.** (The `K_selected: 8` line in that log
  is from the data-free synthetic **selftest** that runs before the fit — not a real
  outcome.)
- **No orientation / retention outcome was inspected.**
- **No O2.4R M>0 result was computed.**
- No `rd_results.json` / `o2_4r_results.json` was written.
- Public NSD inputs on the PVC were **preserved** (103.6 GB core betas + 10.9 GB
  imagery + support). Only the derived anchor-centroid caches were invalidated for
  rebuild (see FIX 6). No raw NSD data was re-downloaded.

## The seven corrections

1. **Target-specific K.** K is now selected separately for every
   `target participant × ROI` (`select_K_for_target`): for each candidate K and anchor
   validation block `b`, SRM is fit on the seven training subjects (≠ target) on
   `block_id != b`, the target is aligned on its own `block_id != b` rows, and the
   target's held-out block `block_id == b` is reconstructed and scored by cross-subject
   native-pattern reconstruction Pearson r (mean over 5 blocks; max K; tie ≤ 1e-12 →
   smaller K). Produces `K[target, ROI]`, not `K[ROI]`. Per-target scores persisted.

2. **CV-clean `TRAIN_ONLY_VOXEL_ZSCORE`.** During K-block validation each participant's
   voxel scaler is fit on `block_id != b` **only** and applied to both its train and
   validation rows; no validation-block statistics enter K selection. The final
   all-anchor perception fit (no remaining anchor-block validation) uses the full
   512-anchor scaler, which is persisted. CV-scaler provenance recorded.

3. **RD oracle recovery.** The frozen `≥ 0.50` oracle criterion now uses the historical
   O2.3A convention preserved prospectively:
   `RD_ORACLE_RECOVERY = R_ZERO_RD / max(R_ORACLE, 1e-12)` — per target × ROI × fold,
   then median over the six folds (participant), then median over the eight
   participants (group). Raw `R_ORACLE` is never labelled recovery. All of
   `R_ZERO_RD`, `R_ORACLE`, fold/participant/group recovery are persisted separately.
   (Regression-anchored to the historical value 0.1444999174837383 / 0.2308295784225333
   = 0.6260026053473587.)

4. **O2.4R oracle recovery.** `median_oracle_recovery` is now the calibration recovery
   fraction, not raw oracle retention:
   `(R_M − R_0) / max(R_ORACLE − R_0, 1e-12)`, per participant, using fold-averaged
   quantities; both unclipped and `clip(·, 0, 1)` values are persisted, and the clipped
   median across the 8 participants feeds the `≥ 0.50` M\_STAR condition.

5. **Derangement null.** The O2.4R identity-correspondence null now draws a deterministic
   **derangement** (frozen seed → PCG64; redraw until zero fixed points), so no
   calibration identity remains paired to its own target identity. `M = 2` gives the
   unique swap. Seed text, 100 iterations, PCG64, and subset enumeration are unchanged.

6. **Cache provenance.** The union-anchor cache identity now binds the full provenance —
   canonical 512-anchor manifest SHA256, ordered 512 anchor-ID hash, `nsd_expdesign.mat`
   hash, B0/b2 lineage, exact session list, union voxel-coordinate hash, `/300` scaling,
   and code/config version — stored in an adjacent immutable manifest. A cache is reused
   only if every field matches. Pre-correction caches were invalidated and rebuilt once.

7. **Execution wrapper committed.** The canonical bootstrap
   (`scripts/mindcompiler/o2_3a_rd_phase2.sh`) and Job manifest
   (`infra/k8s/o2_3a_rd_phase2_job.yaml`) are committed, recording image digest, PVC,
   mount path, `--data`/output/state paths, CPU/RAM, r770 node selector, BLAS thread
   settings, and source/config hashes.

## Mandatory two-stage access order

The wrapper runs exactly one stage per Job. **Stage A** = RD fit → `RD_SEAL.json` →
certify `O2_4R_M0_EQUALS_O2_3A_RD` → `M0_CERT.json`, then **stops**. **Stage B**
(`--o2-4r`) refuses to run (`O2_4R_ACCESS_ORDER_VIOLATION`, exit 3) unless
`RD_SEAL.json` and a valid `M0_CERT.json` are both present. Stage B is launched only
after Stage A is sealed, committed, pushed, and server-verified.

## New tests (all green, data-free)

`tests/mindcompiler/operator_o2_3a_rd/test_rd_phase2_corrections.py`: target-specific K
can differ; scaler cannot see validation-block rows; RD oracle recovery matches the exact
historical value; O2.4R recovery fraction; derangement null has zero fixed points and is
deterministic (M ∈ {2,4,6,8,10}); provenance hash binds every field; and Stage B / M0
certification refuse to run without the required upstream artifacts.

## Unchanged program state

`O3` remains `O3_NOT_READY`. Historical O2.3A / O2.4 / O2.4-PROVREC verdicts are
untouched and are never relabelled as reproduced.
