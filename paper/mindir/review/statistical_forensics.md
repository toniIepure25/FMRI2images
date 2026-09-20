# Phase 6 — Statistical forensics

Audit of every major inferential claim against `evidence_table.csv` and the sealed `*inference*.json`. No new
computation; verification of internal consistency and inferential validity only.

## Inferential unit
- **Participant N = 8 throughout.** Every sign-flip test is over 2^8 = 256 participant sign assignments.
- **Folds, schedules, identity subsets, and repeats are NEVER inferential units.** O2.15 associations are
  computed at the schedule level *within participant* and then aggregated to the participant (Fisher-z) before
  the N=8 test. O2.9/O2.14 aggregate to participant TOTAL/FCF/coverage before the group criterion. **No
  pseudoreplication found.** *(Manuscript action: state this explicitly in every ρ claim + figure caption.)*

## Minimum attainable p
- The exact one-sided sign-flip minimum at N=8 is **1/256 = 0.00390625**. It appears in the evidence for:
  X1 (both ROIs, B_angles), X3-ANGLE (both, but triviality-blocked → not counted as support), X4-H1 ANISO
  (both ROIs). Each of these has all 8 participants on the same side, which is the only way to reach 1/256.
  **Every occurrence is valid.** It does **not** appear on any claim that lacks 8/8 concordance.

## Test direction / null expectation
- O2.15 (Q_OUT→margin) and X4-H1 (anisotropy) are one-sided (positive predicted); reported ρ/effects are
  positive and Holm-significant. Direction matches the pre-registered hypothesis.
- X4-H1 uses the **centred** statistic (real − isotropic-null median), null expectation 0 — this is the
  disclosed correction of the frozen-config ratio inconsistency; the ratio is descriptive only. Correct.
- Negatives (X2 K2−K1, X3-SCALE, X4-H2) have effects at/below 0 or opposite sign with non-significant sign-flip
  p; they are reported as "not supported," never as proof of absence. Correct.

## Holm families (each exactly the pre-declared size)
- O2.15: IN/OUT per ROI within the gate family. X1: 4 invariant blocks. X2: 2 tests. X3: 4 tests. X4: 4 tests.
  O2.16-SEC: 6 tests (preregistered, not evaluated). **No family is resized post hoc; no cross-gate pooling.**

## Sufficiency rules vs hypothesis tests
- O2.9 sufficiency (median ≥ 0.50, ≥6/8) and O2.14 schedule-robustness (coverage ≥ 0.75, ≥6/8) are **frozen
  decision rules, not permutation tests** — no p-value is attached, correctly. *(Manuscript action: label them
  as pre-registered decision rules so readers do not expect a null.)*

## Effect aggregation / denominators
- TOTAL = (R−R0)/(R_native−R0); FCF = (R−R0)/(R111−R0). Denominators are the native oracle and the
  full-resource ceiling respectively; both defined per participant×ROI. Consistent across O2.9/O2.14/O2.15.
- Replay integrity: O2.14/O2.15/X4 reproduce O2.9 M4T2 margins to ≤ 5.6e-17 (sealed `*_replay*.json`),
  confirming no divergence in the reused estimator.

## Verdict
**No CRITICAL statistical flaw identified.** Residual actions are transparency/wording:
(1) add a "Statistical scope and multiplicity" paragraph (participant N; within-gate families; no cross-gate
correction; exploratory vs prospective; sufficiency rules vs tests);
(2) attach the inferential unit to every ρ/effect claim and figure caption;
(3) state the 1/256 floor once.
