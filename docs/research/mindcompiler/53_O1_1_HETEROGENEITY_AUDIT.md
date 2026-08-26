# O1.1 — Operator heterogeneity, measurement dependence & failure-mode audit (charter)

**Gate:** O1.1 (post-hoc descriptive audit — NO refit, NO new models) · **Input HEAD:** `64da8cf`
**Class:** `POST_HOC_DESCRIPTIVE_OPERATOR_HETEROGENEITY_AUDIT`.

## Immutable
O1 scientific status `STIMULUS_INVARIANT_OPERATOR_PARTIAL` is **not changed** (neither upgraded nor
downgraded). O1 O0–O4 predictions immutable; Track R `INDEPENDENT_METHOD_REPRODUCTION_PARTIAL` closed;
`H_A_CROSS_PARTICIPANT_PARTIAL` unchanged. O1.1 is explanatory only.

## Question
Why does stimulus-held-out perception→imagery operator evidence emerge strongly in some
participant×ROI cells but weakly/absent in others? Distinguish (without assuming one cause) among:
baseline/ceiling, imagery/vision measurement quality, weak vision-dependent info, operator-estimation
instability, geometry mismatch, participant heterogeneity, ROI structure, beta-preparation dependence,
stimulus-family dependence. Multiple failure modes may coexist.

## Authoritative effect
`G(s,r) = held_out_r_O4 − held_out_r_O0` = **OPERATOR_INCREMENTAL_GAIN** (raw O4 alone is NOT operator
evidence, since O0 ≈ 0.7–0.8). Secondary: G_gain=O4−O1, G_diag=O4−O2, G_lowrank=O4−O3. Diagnostic
`normalized_incremental_gain = G/max(1−O0, 0.05)` (headroom-normalized, diagnostic only).

## Source boundary
Committed O1 artifacts + Track-R measurement context (voxel counts, B0/B1 sensitivity, and repeat
**reliability** computed **once** as a split-half measurement summary — even-vs-odd repeats, pattern
correlation across voxels per identity, averaged, Spearman-Brown corrected — NOT an operator-predictor
sweep, no refit). Absent variables → `NOT_AVAILABLE_UNDER_O1_1_SOURCE_BOUNDARY`.

## Method boundary
No operator refit; no new/nonlinear models; no alternate ridge/rank/scaler/centroid/fold; no D1; no
participant/ROI exclusion; no stepwise/feature-selection/RF/SHAP; no multivariable model over 56 cells;
no p-values on N=8; **56 rows are not 56 independent subjects.** Two levels kept separate: L1 across
participants within ROI (N=8), L2 across ROIs within participant (N=7). Descriptive Spearman only.

## Frozen failure-mode axes & status rules
FM1 baseline/ceiling · FM2 imagery measurement · FM3 vision measurement · FM4 beta sensitivity (B0 vs
B1) · FM5 stability · FM6 geometry regime · FM7 model-class necessity · FM8 stimulus residuals · FM9
voxel count. Per-axis status ∈ {SUPPORTED_PATTERN, PARTIAL_PATTERN, NOT_SUPPORTED, INCONCLUSIVE}.
Thresholds frozen in `failure_mode_contract.json` (e.g. FM4 tracks-reliability if Spearman(ΔR,ΔG_beta)≥0.5;
FM6 notable if |median diff / pooled MAD| ≥ 0.8). Per-ROI case rules and participant regimes frozen in
`case_classification_contract.json`. Descriptive ROI groups: EVIDENCE {ventral, lateral, parietal},
TRANSITIONAL {V1, V3}, LOW/NO {V2, hV4}.

## Overall heterogeneity status (Part U)
One of: MEASUREMENT_DOMINANT / REGION_STRUCTURE_DOMINANT / STABILITY_DOMINANT / MULTIFACTORIAL /
UNRESOLVED — from the multifactorial decision table; does not modify the O1 PARTIAL status.

## O2 readiness (frozen, `o2_readiness_contract.json`)
`O2_READY` requires: (A) ≥3 ROIs with reproducible B0 O4>O0 Holm evidence; (B) not baseline-ceiling
artifact; (C) stability ≥ MODERATE for the evidence set; (D) a freezable cross-subject measurement
contract; (E) B0 measurement dependence acknowledged. Else `O2_NOT_READY`. Candidate primary O2 ROIs
{ventral, lateral, parietal} only if confirmed; V2/hV4 exclusion must be prospectively justified.
O1.1 does NOT select the common-space method; O2's critical test = leave-one-subject-out + held-out
stimulus identities.

## Forbidden interpretations
Not: "B1 proves the operator doesn't exist" / "B0 is the true signal" / "higher cortex computes
imagination" / "the brain uses this matrix" / "V2/hV4 have no perception→imagery transformation."
Failure to detect a linear mapping under one measurement contract is not proof of biological absence.
