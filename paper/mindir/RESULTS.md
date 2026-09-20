# Results

All values are exact sealed results; each traces to an artifact and commit in `evidence_table.csv`. The
inferential unit is the **participant (N = 8)**; schedules, folds, identities, repeats, and dimensions are
never inferential units. Participant-level exact sign-flip tests have a minimum one-sided p of 1/256 ≈
0.00390625. Two regions of interest (ROIs) are analysed throughout: the **ventral** and **lateral** visual
streams.

> **Cohort status.** All results below come from a **discovery / development cohort** (N = 8). They are not an
> independent replication. The independent replication (O2.16) is **preregistered and frozen but not yet
> executed** (see the Replication box).

---

## Result 1 — Perception predicts imagery structure but does not fully determine target-state geometry
*Gates O1, O2, O2.1, O2.2, O2.5 — prospectively frozen (primary)*

**Question.** After accounting for perception, does a real target-imagery component remain? **Status.**
Prospectively frozen primary analyses. **Result & bounded interpretation below.**

The perception→imagery operator carried significant, reproducible predictive structure: a substantial fraction
of the target-imagery state lies within the perception-support subspace and is recovered by the perception-
derived operator. However, a **substantial target-native component lies outside perception support** — the
perception-support ceiling sits well below the native oracle. Imagery is therefore not fully determined by the
perceptual representation in these streams; a real residual target-state geometry remains after perception is
accounted for.

**Bounded message:** perception is informative but incomplete; a genuine outside-support residual exists.

---

## Result 2 — The missing imagery geometry is low-complexity but subject-specific (under the tested representation)
*Gates O2.6, O2.10, O2.11, O2.13 — prospectively frozen (primary)*

**Question.** Is the outside-support residual shared across people or individual, and is it recoverable from
perception/covariates? **Status.** Prospectively frozen primary analyses.

The outside-support residual is **low-dimensional under the frozen D = 2 outside representation** (a compact
missing basis; O2.6). Its **orientation did not transfer between participants**: cross-participant donor
transfer of the composite target-state geometry was subject-specific dominant (O2.10,
`COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT`). It was **not predictable from the tested
perception phenotype** (O2.11, `SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE`),
nor from the tested anatomy/behaviour covariates (O2.13, `NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE`).

**Bounded message:** the missing basis is low-complexity in dimension (under the frozen representation) but
**subject-specific under the tested representation and transfer procedure** — it was not recovered from the
tested perception phenotype, anatomy, or behaviour. This is a failure of cross-subject transfer and of the
tested predictors, not a demonstration that the geometry is biologically unique to each person.

---

## Result 3 — Minimal direct target-state calibration recovers substantial geometry
*Gates O2.9, O2.12 — prospectively frozen (primary)*

**Question.** How little direct imagery data recovers the missing geometry, under a fixed estimator? **Status.**
Prospectively frozen; the sufficiency criterion (median ≥ 0.50 with ≥ 6/8 participants ≥ 0.50 on both metrics)
is a **pre-registered decision rule**, not a permutation test.

Under the fixed direct-SVD within-support estimator plus a two-dimensional outside-support basis (no
Procrustes; frozen), we swept an M-identities × T-repeats calibration grid. The **smallest prospectively tested
common calibration burden, under the frozen estimator, closing ≥ 50% of the native-oracle gap in both primary
streams on this development cohort** was **8 observations at M = 4 identities × T = 2 repeats (M4T2)**:

| Stream | median TOTAL recovery | participants ≥ 0.5 | median FCF | participants ≥ 0.5 |
|---|---|---|---|---|
| ventral | **0.569** | **7/8** | 0.596 | 8/8 |
| lateral | **0.530** | **6/8** | 0.591 | 8/8 |

`N_TRIALS_STAR = 8`; status `MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED`. Adding a fixed
perception-only prior (O2.12) **improved** recovery but did **not** reduce the eight-observation minimum
(`PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED`).

**This is not a universal minimum.** It is specific to the tested estimator, ROIs, and cohort.

---

## Result 4 — The eight-observation frontier is an average result, not a robust acquisition guarantee
*Gates O2.14, O2.15 — prospectively frozen (primary)*

**Question.** Is the eight-observation frontier robust to acquisition schedule, and what predicts success?
**Status.** Prospectively frozen; O2.14 schedule-robustness uses the pre-registered coverage rule (median ≥
0.75, ≥ 6/8), O2.15 uses participant-level Spearman with Fisher-z aggregation (participant is the inferential
unit; schedules are within-participant repeated measures, not independent N).

Holding the M4T2 burden fixed, **schedule composition strongly affected success**. Median schedule coverage
(fraction of schedules meeting the joint criterion) was **0.656 (ventral)** and **0.579 (lateral)** — i.e. a
large minority of eight-observation schedules failed the operational criterion — with no schedule-robust
minimal resource set (`N_TRIALS_ROBUST_STAR = None`; status
`EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST`; the M4T2 result was leave-one-fold-out stable but not
schedule-robust).

The **repeat-to-repeat geometric agreement**, especially **outside** perception support, predicted whether a
calibration generalised: schedule-level Spearman(Q_OUT, robustness margin) had median ρ **0.423 (ventral)** and
**0.477 (lateral)**, **8/8 participants positive** in both streams, Holm-significant (status
`REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE`). The within-support agreement Q_IN was weaker
(median ρ 0.131 ventral, 0.119 lateral; also 8/8), confirming the outside component carries the stronger
failure-predictive signal.

**Bounded message:** eight observations is an average frontier; acquisition success is governed by outside-
support repeat geometry, not merely trial count.

---

## Result 5 — The failure-predictive repeat geometry is structured but distributed
*Exploratory mechanistic analyses — gates X1, X2, X3, X4*

> **These analyses are exploratory mechanistic follow-up on the same discovery cohort.** They refine — they do
> not establish — the primary story, and several are well-controlled negatives.

- **X1 (invariants).** Of four candidate coordinate-invariant blocks, **only the perception↔imagery principal-
  angle invariant `B_angles`** survived the pipeline-triviality control, and it did so in **both** ROIs
  (status `INVARIANT_STRUCTURE_MULTIREGIME`, one surviving invariant per ROI). The spectral, support, and
  compression blocks were pipeline-trivial.
- **X2 (latent states).** There was **no reproducible discrete two-state structure**: held-out two-state minus
  one-state log-likelihood had median **−0.113 (ventral)** and **−0.100 (lateral)**, with **0/8** participants
  positive in both streams; a two-state model never beat one state, and a continuous heavy-tailed (Student-t)
  description performed at least as well (status `NO_REPRODUCIBLE_LATENT_STATE_STRUCTURE`).
- **X3 (angular reliability).** There was **no repeat-stable angular scaffold** (the shared angle relation was
  not stable across single repeats; pipeline-constrained), and **scalar Student-t reliability did not predict**
  the calibration margin (median ρ **0.004 ventral**, **0.064 lateral**; not supported; status
  `NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE`).
- **X4 (geometric drift).** Repeat projector perturbations were **reproducibly anisotropic**: the centred
  dominant-direction statistic was **+0.093 (ventral)** and **+0.101 (lateral)**, **8/8** participants
  positive, both Holm-significant, with cross-half reproducibility in all participants. **But that dominant
  mode did not predict calibration failure** (median rho_FAILURE **−0.128 ventral**, **−0.099 lateral** —
  opposite the hypothesised sign; not supported), giving status
  `GEOMETRIC_DRIFT_EXISTS_BUT_FAILURE_LINK_PARTIAL`.

**Synthesis (bounded):** the repeat variability that predicts calibration failure (Result 4) is genuinely
structured, but it is **not** reducible to a discrete latent state, a scalar reliability variable, a repeat-
stable angular scaffold, or a single dominant geometric drift mode. It appears **distributed / higher-rank** —
consistent with the outside-support agreement Q_OUT being the strongest single predictor while no low-
dimensional latent summary captures it.
