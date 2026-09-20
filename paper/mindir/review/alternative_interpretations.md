# Phase 3 — Central story test: falsification attempt and competing interpretations

## The narrative under test (A→I)
A perception predicts substantial imagery structure → B perception is insufficient for target-native geometry →
C missing structure is low-complexity → D orientation is participant-specific under the tested representation →
E small direct calibration recovers substantial structure → F 8 observations is the smallest prospectively
tested common burden closing ≥half the native-oracle gap under the frozen estimator → G that average frontier is
not schedule-robust → H repeat geometric agreement predicts calibration success → I exploratory analyses
indicate structured-but-distributed repeat variability.

## Transition-by-transition logic check
- **A→B:** supported; the outside-support residual is a measured gap between the perception-support ceiling and
  the native oracle (O2.5/O2.6). *Caveat:* the size of the gap is representation-dependent.
- **B→C:** supported under the frozen D=2 outside representation (O2.6). *Caveat:* "low-complexity" is
  conditional on that representation.
- **C→D:** supported as failure of cross-subject transfer + failure of prediction from tested covariates
  (O2.10/O2.11/O2.13). *Caveat:* this is "subject-specific under the tested representation/transfer," not
  proven biological uniqueness.
- **D→E→F:** supported; the M×T grid recovers ≥half the gap at M4T2 (O2.9). *Caveat:* estimator/ROI/cohort
  specific; not a demonstrated lower bound.
- **F→G:** supported; schedule coverage 0.656/0.579, no robust N* (O2.14).
- **G→H:** supported; Q_OUT predicts robustness margin (O2.15). *Caveat:* predictive, not causal.
- **H→I:** exploratory; X1–X4 constrain but do not identify the structure.

**Verdict:** every transition is logically supported *with its caveat attached*. The chain does not require any
causal claim, and does not require independent replication to be internally valid as a development-cohort result.

## Strongest competing interpretations

### SUPPORTED CONCERN (must be acknowledged in the manuscript)
1. **Estimator/representation dependence.** "Outside-support residual", "low-dimensional", and "8-observation
   minimum" are all conditional on the frozen direct-SVD + D=2 estimator. A different estimator could shrink the
   residual or change the minimum. → Acknowledged in Limitations; wording qualifiers added.
2. **Cumulative same-cohort testing.** Repeated frozen gates on one N=8 cohort inflate program-wide
   multiplicity even though each gate is clean. → This is exactly why replication is preregistered; stated.

### PLAUSIBLE BUT UNTESTED (name, do not claim resolved)
3. **Limited-N apparent subject-specificity.** With N=8, failure of cross-subject transfer could partly reflect
   sampling. Not resolvable here; the independent cohort (O2.16 primary) is the test.
4. **Task specificity.** Effects may be specific to the NSD imagery task and identity set rather than general
   imagery. Untested here.
5. **Voxel/ncsnr selection consequences.** Reliability-based selection could shape the residual geometry.
   Disclosed as a frozen rule; its influence on the residual is not separately quantified.
6. **Regression-to-mean in schedule scoring.** Schedule-level margins may show mean-reversion; the
   participant-first aggregation mitigates but does not eliminate the concern.

### ALREADY RULED OUT (by a sealed control, not by new work)
7. **"Q_OUT is just measurement reliability."** Ruled out by X3: a scalar Student-t reliability variable does
   NOT predict the calibration margin (ρ 0.004 ventral, 0.064 lateral), whereas outside-support geometric
   agreement does (ρ 0.423/0.477, 8/8). The predictor is geometric/directional, not a scalar noise level.
8. **"Anisotropy is a projector-geometry artifact."** Addressed by X4's Frobenius-norm/rank-matched isotropic
   null + subset-centering + cross-half reproducibility; the centred statistic (real−null, never the ratio)
   exceeds the matched null in 8/8 participants.
9. **"The dominant drift mode explains calibration failure."** Explicitly falsified in-program by X4-H2
   (rho_FAILURE −0.128/−0.099, opposite sign) — which is why the manuscript claims *distributed*, not
   single-mode.
10. **"Cross-subject 'no transfer' is a native-space comparability artifact."** Cross-subject analyses use
    coordinate-invariant / donor-mean quantities, not raw voxel spaces (O2.10 design).

## Net
No competing interpretation falsifies the development-cohort narrative *as bounded*. Two supported concerns
(estimator dependence; cumulative same-cohort testing) and four plausible-but-untested concerns must be stated
plainly; four candidate confounds are already ruled out by sealed controls.
