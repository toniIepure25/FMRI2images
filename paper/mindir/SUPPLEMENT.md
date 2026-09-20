# Supplement

Detailed material kept out of the main text. Nothing here is a new analysis; every item points to a sealed
artifact. This is an index + narrative, not a re-computation.

## S1. Complete gate ledger (discovery cohort, N = 8)
| Gate | Question | Sealed status | Seal commit |
|---|---|---|---|
| O1/O2/O2.1/O2.2 | perception->imagery operator + shared transport | operator established; residual exists | (program lineage) |
| O2.5 | perception support ceiling vs native oracle | support ceiling < native oracle | (program lineage) |
| O2.6 | dimensionality of the missing basis | low-dimensional outside-support residual | (program lineage) |
| O2.9 | minimal M x T calibration frontier | MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED (N*=8) | 380c9be |
| O2.10 | cross-subject transfer of target geometry | COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT | 7078b71 |
| O2.11 | perception-only prediction of orientation | NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE | 5f1a1fd |
| O2.12 | perception-prior hybrid | PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED | 1af0087 |
| O2.13 | anatomy/behaviour covariate prediction | NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE | 291061f |
| O2.14 | schedule robustness of the 8-obs frontier | EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST | 5cad808 |
| O2.15 | repeat-geometry predictor of failure | REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE | ba763a6 |
| O2.16 | independent replication | O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE | 0aa6f2d |
| X1 | coordinate-invariant shared transformation invariant | INVARIANT_STRUCTURE_MULTIREGIME (B_angles only) | 6726b98 |
| X2 | discrete latent imagery states | NO_REPRODUCIBLE_LATENT_STATE_STRUCTURE | bc0f5a2 |
| X3 | conserved angular scaffold / scalar reliability | NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE | 7940f1c |
| X4 | low-dimensional geometric drift | GEOMETRIC_DRIFT_EXISTS_BUT_FAILURE_LINK_PARTIAL | 27d4918 |
| O2.16-SEC | secondary endpoint preregistration | O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT | 1302ea5 |

## S2. Negative controls and nulls
- Participant-level exact sign-flip (2^8 = 256) one-sided tests; Holm within each frozen family.
- O2.15: pair-vs-complement construction; Fisher-z aggregation; within- vs outside-support comparison.
- X1: constrained-random-operator pipeline-triviality baseline (three of four blocks pipeline-trivial).
- X2: Gaussian / two-state / Student-t held-out likelihood; degeneracy diagnostics.
- X3: outside-projected vs raw-residual angle handling (raw residuals used, matching X1); constrained null.
- X4: Frobenius-norm/rank-matched isotropic null (1000 iters); cross-half mode reproducibility; repeat-label
  permutation null (1000 iters); centred statistic (real - null), never the unstable ratio.

## S3. Preprocessing and estimator freezes
- ROI atlas: ventral = streams label 5, lateral = label 6.
- Composite estimator: direct-SVD within-support (q = min(r_best, M)), fixed D = 2 outside; orthogonal ranges;
  no Procrustes. All ranks/dimensions frozen before outcomes.
- BLAS threads = 1; committed source injected to the compute node; no ad-hoc code.

## S4. Complete statistical tests
Every gate's exact family is in its `*_frozen_config.json` and `*inference*.json`. Key families: O2.15 IN/OUT
per ROI; X1 four invariant blocks; X2 two tests; X3 four tests; X4 four tests; O2.16-SEC six tests. Minimum
one-sided p at N = 8 is 1/256 ≈ 0.00390625.

## S5. Exploratory X analyses (full)
X1 invariants (spectral / angles / support / compression), X2 latent-state modelling, X3 angular reliability,
X4 geometric drift — see each gate's `results/`. The X4 disclosed frozen-config integrity resolution (inference
on the centred statistic, not the ratio) is documented in `x4_geometric_drift/results/scientific_status.json`.

## S6. Provenance and reproducibility
Each gate: freeze config (SHA) + driver + tests -> commit -> push -> server-verify (remote HEAD == local HEAD)
-> compute on fixed node -> pull results -> seal. Reused estimators replay upstream to <= 1e-10 (O2.14/O2.15/X4
replay O2.9 M4T2 margins to <= 5.6e-17). Artifacts carry `hashes.json`, `execution_provenance.json`,
`test_report.json`. No raw beta files committed; no GPU used.

## S7. Historical Roy reproduction limitations
Independent method reproduction and extension of Roy et al. Preserve `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE`
and `BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS`. No claim of exact reproduction.

## S8. Discovery vs replication (exact distinction)
Historical N = 8 = discovery/development cohort (definitions, frozen formulas, descriptive references). O2.16 =
preregistered/frozen independent replication, NOT executed. O2.16-SEC = three preregistered secondary geometric
endpoints for that cohort. Historical participants never enter replication inference.
