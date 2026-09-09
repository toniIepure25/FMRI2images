# O2.4-PROVREC — Forensic O2.3A Re-Derivation & Historical-Replay Certification

**Class:** `FORENSIC_PROVENANCE_RECONSTRUCTION_AND_EXACT_REPLAY_GATE` · **Source HEAD:** `2a80bbc`
**Reconstruction config SHA256:** `9e3060af…` · **Status:** `O2_4_PROVREC_HISTORICAL_DETAIL_MISSING`

Purpose: reconstruct the historical O2.3A dense-perception machinery from its *original frozen contracts*,
prove it replays the historical fold-level results (categorical exact + scalar 1e-6), persist the reusable
state, and — only on replay PASS — resume the already-frozen O2.4 calibration frontier. Reconstruction, not
model selection: an implementation is valid only if it replays history sufficiently exactly.

## P0 forensic inventory (executed)

Historical O2.3A commits all present: methodology freeze `d2b2bc7`, acquire-131GB-b2-core `7a0f653`, anchor
manifest 512 `66d48fa`, **execution `5f13e4a`**, docs `c73f304`. All 21 historical O2.3A artifacts are
present and hash-recorded (`historical_o2_3a_inventory.json`), including `anchor_manifest.csv` (512
leakage-clean anchor images), `anchor_K_selection.csv` (selected K per subject×ROI), and
`residual_template_selection.csv` (selected rank per subject×ROI×fold).

**Decisive finding — the generator source was never committed.** Commit `5f13e4a` (execution) added *only*
artifacts + test files — **no generator `.py`**; across *all* git history the only O2.3A `.py` files are the
two test files. The dense-perception generator (anchor centroids, DetSRM, K-selection, residual template,
orientation, null, oracle) ran from ephemeral scratch and is gone.

## Why exact replay cannot be certified (P5/P6 determination)

The frozen contracts specify the **method family**, not the numerically-determining implementation. The
required replay (categorical K/rank EXACT + deterministic scalar `max_abs_err ≤ 1e-6` ∧ Pearson ≥ 0.999999
+ null 1e-6) cannot be certified because these **historical implementation details are genuinely absent**:

1. **The O2.3A generator source** — never committed; the authoritative implementation.
2. **DetSRM specifics** — init method, `n_iter`, convergence. The contract says only `DETERMINISTIC_SRM`;
   these fix the common-space gauge and thus every recon-r / retention / orientation scalar to > 1e-6.
3. **K-selection nested-CV block partition** — `n_blocks=5` is recorded, but the exact assignment of the
   512 ordered anchor images to the 5 blocks (contiguous / by-session / interleaved / seeded) is not
   persisted, and it determines the selected K per cell.
4. **Null random-subspace generator** — the seed *rule* is known
   (`SHA256("O2.3A"|subject|ROI|identity_fold|null_iteration)`), but the seed→subspace RNG/draw/dimension
   mapping is not in the contracts, and it determines `median_null` to > 1e-6.
5. **Exact anchor voxel selection** for subj02–08 — `ncsnr` absent locally (7/8).

Multiple contract-compatible implementations would diverge well beyond 1e-6. Per the gate this is a **STOP**:
do **not** loosen tolerance, do **not** pick the closest scorer. Compounded by a data blocker — the **131 GB
b2 NSD-core anchor betas are absent** (local: only subj01/40 sessions; the `7a0f653` cache is gone; pod
`nsddata_betas` = 3.5 G) and NSD-Imagery betas are absent for 7/8 (both re-acquirable from public S3, which
the gate authorizes, but the missing *implementation details* remain the binding blocker regardless of data).

## Determination

`O2_4_PROVREC_HISTORICAL_DETAIL_MISSING`. Replay not run; no reusable state produced; `M=0` **not** certified;
**no target-imagery calibration opened**; nothing fabricated. Consequently the O2.4 scientific status remains
`TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE`, now with the stronger reason
`HISTORICAL_O2_3A_STATE_NOT_RECONSTRUCTABLY_REPRODUCIBLE`. **This is a reproducibility limitation — explicitly
NOT evidence against the calibration hypothesis.** The historical O2.3A execution itself remains valid, and
`O3` remains `O3_NOT_READY`.

## Exact resolution (user/program decision — not taken unilaterally)

The binding blocker is the absent O2.3A **generator source + the numerically-determining settings**
(DetSRM init/iter, K-block partition, null RNG). To unblock O2.4 exactly, one of: (a) recover the original
O2.3A generator code (if it exists outside git, e.g., a prior session/scratch backup) and commit it; or
(b) authorize a **new prospective gate** that re-derives an O2.3A-equivalent common space under a *freshly
frozen* contract (accepting it supersedes the historical numbers rather than bit-replaying them), then runs
O2.4 on that. Either is a program decision. O2.4's frozen design (`321b42f9`) and method certification
(gauge-invariant to 6.7e-16) stand ready to run once a reusable, hash-certified common-space state exists.
