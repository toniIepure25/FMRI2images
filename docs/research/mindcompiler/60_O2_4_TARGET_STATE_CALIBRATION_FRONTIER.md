# O2.4 — Target-State Calibration Frontier (identity-diversity sample complexity)

**Analysis class:** `PROSPECTIVE_IDENTIFIABILITY_SAMPLE_COMPLEXITY_ON_EXISTING_B0_O1_O2_DATA`
**Source HEAD:** `773a893` · **Frozen config SHA:** `321b42f9…`
**Status:** `TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE` (provenance)

**Question.** What is the minimum number of *distinct target-imagery identities* required to identify the
subject-specific orientation of the otherwise-shared imagery component, such that the calibrated
orientation generalizes to completely unseen stimulus identities? (Budgets `M ∈ {0,2,4,6,8,10}`,
orthogonal-Procrustes-only, participant `N=8` sign-flip + Holm, M_STAR rule.)

## Design frozen before any outcome (Part A)

Budgets (identity-diversity, balanced M/2 simple + M/2 naturalistic; all balanced subsets: 25/100/100/25/1),
6 reused outer folds (hold out 1 simple + 1 naturalistic, never used in fit/rank/selection), reuse of the
frozen O2.3A common space + `O2_2_IMAGERY_RESIDUAL` (small-visual-span), shared template from training
subjects only, **orthogonal-Procrustes-only** estimator (`Q*=argmin‖X_CQ−Z_C‖`, deterministic SVD; no
affine/ridge/CCA/RRR/nonlinear), orientation application `U_target=Q^T U_res → B_target=W_target U_target →
P_CAL`, identity-correspondence null (100 perms), M_STAR = smallest M with median E_M>0 ∧ ≥6/8 positive ∧
Holm p<.05 ∧ median ΔZERO>0 ∧ median oracle-recovery ≥0.50. Frozen + committed + pushed before evaluation.

## Method certification (data-free, executed)

The estimator was certified on synthetic data with **no NSD data**: the **gauge-invariance test passes to
machine precision** (`|ΔR_CAL| = 6.7e-16` under an arbitrary common-space orthogonal gauge applied
consistently to W / shared coords / U_res / calibration coords); the estimator recovers orientation
monotonically to the oracle in the identifiable case (R_CAL 0.65→1.0 as M grows); the leakage detector
fires on injected test-identity overlap; low-M discrimination is present (M=2 true 0.85 vs shuffled 0.68).
(In a pure noiseless low-rank toy, R_CAL is a subspace metric that saturates at high M regardless of exact
correspondence — a property of the toy, not the estimator.) **The design and estimator are sound.**

## Why the real frontier could not be run — provenance

O2.4 must **reuse** the exact frozen O2.3A dense-perception common space (`W_target`, `K`, training-subject
common space, vision alignment) + residual-orientation template, and **M=0 must reproduce O2.3A to
numerical/hash-level agreement** (`P_ZERO`). Those products are **not present in reusable form**:

- **No in-repo O2.3A generating code** — the repository contains only the O2 small-visual-span phase code
  (`operator_o2_phases.py`, `operator_o2/`); the O2.3A dense-core-anchor + residual-orientation code was
  never committed.
- **No persisted O2.3A intermediate matrices** (W_target / common space / residual coordinates / imagery
  centroids) — only **summary result CSVs**.
- Therefore `M=0 = P_ZERO` cannot be certified against O2.3A, and `W_target/K/common-space` cannot be reused.
- The gate **forbids rebuilding** them (`No new NSD-core preprocessing`; `Do NOT refit`) — a reconstruction
  would be a forbidden refit and could not guarantee reproducing O2.3A's exact numbers.

(O2.3A itself was **validly executed** — `O2_3A_CORE_ANCHOR_FEASIBILITY_PASS` via public NSD unsigned S3 —
and NSD-Imagery betas remain obtainable from public S3; the block is purely the **reusability/provenance**
of O2.3A's frozen products, not a raw-data impossibility.) **No calibration outcome was computed or
fabricated; no target-imagery calibration was opened.**

## Determination & resolution

`TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE` (provenance). `ventral_M_STAR` / `lateral_M_STAR` =
`NOT_REACHED` (not computed). Zero-target closeout recorded:
`ZERO_TARGET_IMAGERY_ORIENTATION_NOT_IDENTIFIED_UNDER_TESTED_PUBLIC_ANCHORS` — meaning only that
vision-derived alignment did not identify orientation and the public rest route could not be validly
completed at fine scale; **not** "impossible" or "theoretically non-identifiable".

**Resolution (user/program decision, not taken unilaterally):** either (1) persist + commit the O2.3A
generating code and its frozen intermediate matrices so O2.4 can reuse them and certify `M=0 = P_ZERO`,
then re-run the frozen frontier; or (2) authorize a fresh prospective re-derivation gate (outside this
gate's reuse constraint). **O3 remains `O3_NOT_READY`** through all O2.4 outcomes. Prior-art boundary:
MindEye2 / MindAligner acknowledged; **no** "first few-shot cross-subject fMRI adaptation" claim; any
narrow novelty (a prospectively quantified target-*imagery* calibration frontier on unseen identities)
stays `NOT_CLAIMED` pending a full audit.
