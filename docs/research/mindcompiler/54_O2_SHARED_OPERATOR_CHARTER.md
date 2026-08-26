# O2 — Shared + subject-specific perception→imagery operators (charter)

**Gate:** O2 · **Input HEAD:** `299df3c` · **Class:** `FROZEN_CROSS_SUBJECT_AND_CROSS_STIMULUS_OPERATOR_GENERALIZATION`
(preregistered new generalization analysis on an existing dataset — not independent-dataset confirmation).

## Immutable
O1 `STIMULUS_INVARIANT_OPERATOR_PARTIAL`; O1.1 `OPERATOR_HETEROGENEITY_REGION_STRUCTURE_DOMINANT`;
Track R `INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`; H-A unchanged. Primary ROIs frozen from
O1/O1.1: **ventral, lateral, parietal**; secondary V1/V3; **V2/hV4 prospectively excluded** (no
cohort Holm evidence in O1; multifactorial failure in O1.1) — not silently reintroduced.

## Question & claim class
Does a perception→imagery transformation component exist that is **shared across people**
(`T_s = T_shared + Δ_s`)? Primary claim class:
`IMAGERY_ZERO_SHOT_SUBJECT_TRANSFER_WITH_VISION_ONLY_CALIBRATION` — **not** zero-calibration, **not**
completely-unseen-brain. Estimating the held-out participant's functional mapping requires only their
**vision** responses for the **training** identities.

## Measurement contract
**B0 only** (`PRIMARY_O2_MEASUREMENT_BRANCH`; b2-compatible, stronger reliability, O1 primary).
`B0_MEASUREMENT_DEPENDENCE_IS_A_LIMITATION = true`. No B1 primary; B1 never sets common-space/operator
parameters.

## Common space (vision-only)
`DETERMINISTIC_SHARED_RESPONSE_MODEL` (X_s ≈ W_s S, W_s orthonormal). Fit on **VISION ONLY**; the same
W_s maps both vision and imagery into shared space (never align subjects with imagery). New-subject W
estimated by orthogonal Procrustes of the target's **vision** on the training identities to the trained
S. Hyperalignment/GPA retained only as `COMMON_SPACE_METHOD_SENSITIVITY_FUTURE`, never an O2 selection
branch.

## Design
8 LOSO subject folds × 6 identity folds (1 simple + 1 naturalistic each; global holdout across ALL
subjects) × 3 primary ROIs = 144 primary cells. Per cell: 7 train subjects × 10 train identities;
1 target subject × 2 held-out identities. Subject-specific vision scaling (μ_vis/σ_vis on train-identity
vision, applied to both states). K ∈ {2..7} selected by **vision-only** nested CV (inner LOO subject ×
inner identity fold; max mean VISION_IDENTITY_MARGIN; tie smaller K). λ selected on **training subjects
only** (LOO train subject × identity holdout; mean held-out identity pattern r; tie larger λ).

## Operator & baselines (gauge-aware)
`FULL_SHARED_SCALAR_RIDGE` (one scalar λ; **no per-target λ, no diagonal** — those depend on arbitrary
SRM axis orientation). Runtime gauge-equivariance check: X'=XQ, Y'=YQ ⇒ T'=QᵀTQ and predictions map
back identically. Baselines: **S0 GROUP_IMAGERY_MEAN**, **S1 GLOBAL_SHARED_GAIN** (no diagonal common-
space baseline). Primary metric `NATIVE_HELD_OUT_PATTERN_R` (target native voxels, after inverse vision
scaler). Effects: `G_shared = r(T_shared) − r(S0)`; `G_beyond_gain = r(T_shared) − r(S1)`.

## Inference & hypotheses
Participant (N=8) is the unit; exact 2⁸ sign-flip; Holm across the **3 primary ROIs**; α=0.05. H-S1
T_shared > S0; H-S2 T_shared > S1; H-S3 regionally coherent in ventral/lateral/parietal. `SUPPORTED`
requires: common space validated; G_shared Holm-rejects ≥2/3 ROIs; all 3 medians >0; ≥6/8 participants
positive per rejecting ROI; G_beyond_gain positive median ≥2/3 & Holm ≥1/3; no leave-one-out sign flip.
`PARTIAL` / `NOT_SUPPORTED` / `INCONCLUSIVE` per the frozen status contract.

## Phases (freeze chronology)
- **Phase A** — vision-only common-space K-selection + validation → go/no-go. `COMMON_SPACE_VALIDATED`
  only if all 3 ROIs median inner margin >0, ≥75% outer cells positive, no subject fails all 3 ROIs;
  else STOP with `O2_COMMON_SPACE_VALIDATION_FAILURE` (no alternative method inside this gate).
- **Phase B** — primary imagery-zero-shot (ventral/lateral/parietal); seal status.
- **Phase C** — secondary V1/V3 (cannot change primary), Δ_s decomposition (training subjects only),
  gauge-invariant geometry. Shared-action fraction → SHARED/MIXED/SUBJECT-SPECIFIC dominant.

## Leakage guard (sacrosanct)
Per cell, hard counters must be zero: target-subject imagery in SRM/λ/T; test-identity rows in
SRM/λ/T. Any nonzero → `O2_SUBJECT_OR_IDENTITY_LEAKAGE_FAILURE`.

## Gauge-invariant geometry only
Singular values, spectral/Frobenius norm, effective rank, condition number, polar stretch, and
`NONTRIVIAL_TRANSFORMATION_INDEX = ||T − a*I||_F/||T||_F` (a*=trace/K). **No off-diagonal fraction** (no
invariant meaning). No raw-W cross-subject comparison.

## Forbidden interpretations
Not zero-calibration / universal brain operator / brain-to-brain transfer / causal neural transition /
neural compiler / thought reading; raw SRM axes have no biological meaning.

## Next
If SUPPORTED → O3 multi-state operator composition (separate gate). If PARTIAL → O2.1 shared-operator
heterogeneity/calibration audit. If NOT_SUPPORTED → preserve O1's within-subject result.
