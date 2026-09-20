# Reviewer risk register

Each anticipated objection with the strongest scientifically honest answer supported by the sealed program.
These are defensive framings, not new claims.

### 1. N = 8 is small.
Acknowledged and foregrounded. The participant is the inferential unit; we use exact sign-flip tests (minimum
one-sided p = 1/256 ≈ 0.0039) and never count schedules/folds/trials as N. We do not claim population
generality — the cohort is explicitly a discovery/development cohort, and an independent replication is
preregistered and frozen (O2.16 / O2.16-SEC).

### 2. Same discovery cohort used repeatedly.
True and disclosed as a primary limitation. Each gate was prospectively frozen (config SHA + server-verify)
before outcomes, which controls *within-gate* garden-of-forking-paths but not cumulative cross-gate reuse. That
residual risk is exactly why three surviving findings are preregistered for an independent cohort rather than
claimed as established.

### 3. The eight-trial "minimum" may be estimator-specific.
Correct — we state this explicitly. Eight observations (M4T2) is the smallest common-sufficient burden under the
frozen direct-SVD + two-dimensional outside estimator, in these two ROIs, in this cohort. We do not claim a
universal minimum; a different estimator or ROI could differ.

### 4. Calibration schedule is fragile.
This is one of our findings, not a flaw (Result 4). O2.14 shows the eight-observation frontier is an average
result with median schedule coverage 0.656 (ventral) / 0.579 (lateral) and no schedule-robust minimal set. We
report the fragility rather than hide it, and identify the geometric predictor of failure (O2.15).

### 5. Is Q_OUT just measurement reliability?
Directly tested and argued against. X3 shows a scalar Student-t reliability variable does **not** predict the
calibration margin (ρ 0.004 ventral, 0.064 lateral; not supported), whereas the outside-support geometric
agreement Q_OUT does (ρ 0.423 / 0.477; 8/8). The failure-predictive signal is geometric and directional, not a
scalar noise level.

### 6. Does anisotropy simply arise from projector geometry?
Controlled in X4. Anisotropy is tested against a Frobenius-norm/rank-matched isotropic null with subset-
centering, and required cross-half reproducibility across disjoint identity subsets. A rank-2-projector artefact
would appear in the matched null; the centred statistic (real − null, never the ratio) still exceeds it in 8/8
participants. We also state the honest nuance that the drift cloud is high-rank with a reproducible dominant
direction on top (effective rank ≈ 19–28), not globally rank-1.

### 7. Why should subject specificity be biologically meaningful?
We make no biological-mechanism claim. We show only that the target-state orientation does not transfer across
participants and is not predicted by the tested perception phenotype, anatomy, or behaviour (O2.10/O2.11/O2.13).
Whether this reflects biology, measurement, or task idiosyncrasy is left open and flagged for replication.

### 8. Are ROI effects robust?
Primary conclusions are reported for both ventral and lateral streams and agree in sign/direction across the
core results (O2.9, O2.14, O2.15, X4-H1). Where ROIs differ we label it (e.g. X1 is multiregime). Parietal was
reserved and is not a primary claim.

### 9. How does this differ from Roy et al.?
The project began as an **independent method reproduction and extension** of Roy et al. ("A transformation from
vision to imagery in the human brain"). We preserve `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE` and
`BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS` and do not claim an exact replication. Our
contribution is the target-state residual decomposition, the minimal-calibration frontier, its schedule
fragility, and the repeat-geometry predictor — not a re-run of the original.

### 10. Why no independent replication yet?
Because we refuse to relabel discovery as replication. The independent cohort requires new human acquisition,
which is prepared and frozen (O2.16 primary + O2.16-DATA acquisition protocol + readiness package) but awaits
human authorization and data collection. We preregistered the secondary endpoints (O2.16-SEC) so the future
test is confirmatory, not exploratory.
