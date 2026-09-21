# Research brief — Subject-specific geometry and calibration limits in human visual imagery
*(~2 pages · discovery cohort N = 8 · independent replication frozen but not yet executed)*

## Question
How much of the human perception→imagery transformation is shared across people, how much is subject-specific,
and how little direct imagery data are needed to calibrate a decoder to a person's imagined ("target") neural
state?

## Why it matters
Decoding mental imagery is bottlenecked by how much imagery data must be collected per person. If the
imagery-specific geometry perception cannot supply is low-complexity, it might be calibrated cheaply — but only
if we know how much data, and which data, are actually needed. This reframes imagery decoding as a data-cost and
acquisition-design problem, and clarifies where perception-based and shared-across-people approaches stop
working.

## Data
The Natural Scenes Dataset (NSD), including the NSD-Imagery task; eight participants, treated as a
**discovery/development cohort**. The participant is the sole inferential unit.

## Approach
A prospectively frozen pipeline that (i) separates the imagined state into a perception-supported component and
an outside-of-perception residual, (ii) tests whether that residual transfers across people, (iii) measures how
little direct imagery calibration recovers it, and (iv) tests whether the acquired repeats' geometry predicts
whether a calibration will generalise. Group inference is exact participant-level sign-flip testing (N = 8).

## Four main findings
1. **Perception is informative but insufficient** to determine the full imagery target-state geometry — a real,
   low-dimensional component lies outside the perceptual representation.
2. **The missing component is low-complexity but participant-specific** under the tested representation and
   transfer procedure: it does not transfer across people and is not predicted by the tested perception,
   anatomy, or behaviour.
3. **A small amount of direct imagery calibration substantially recovers it** — eight observations (four
   imagined identities, each measured twice) was the smallest prospectively tested common burden under the
   frozen estimator, recovering ≥ half of the native-oracle gap in both streams. **This is not a universal
   minimum.**
4. **That frontier is acquisition-schedule fragile**, and repeat-to-repeat agreement of the
   outside-of-perception geometry predicts whether a calibration generalises.

## Numerical snapshot (development cohort, N = 8)
| Quantity | Ventral | Lateral |
|---|---|---|
| Minimal-calibration total recovery (8 obs; 4 identities × 2 repeats) | **0.569** | **0.530** |
| Full-capacity fraction (8 obs) | 0.596 | 0.591 |
| Eight-observation schedule coverage | 0.656 | 0.579 |
| Repeat-geometry → calibration robustness (median ρ) | ~0.423 | ~0.477 |
| Exploratory anisotropy (centred) | ~+0.093 | ~+0.101 |

> **Exploratory mechanistic follow-up** *(exploratory, same N = 8 cohort)*
> One aggregate perception↔imagery principal-angle invariant is shared across participants; there are **no**
> reproducible discrete latent states; a scalar reliability variable does **not** explain calibration success;
> repeat perturbations are anisotropic, but the dominant anisotropic direction does **not** explain failure.
> **Conclusion:** repeat-level variability is structured but *distributed*.

## Biggest limitation
N = 8, a single discovery/development cohort analysed repeatedly; the eight-observation and subject-specific
results are conditional on the tested estimator and representation. There is no independent replication yet.

## Independent replication plan
A frozen, preregistered replication in a genuinely independent cohort (planned N = 12; minimum confirmatory
valid N ≥ 8) tests the minimal-calibration/generalisation pipeline prospectively, with three preregistered
secondary geometric endpoints. It is **frozen but not yet executed** (see `REPLICATION_PLAN_1PAGE.md`).

## What we need from a collaborator
Scientific feedback on the manuscript; a PI willing to evaluate/host the independent replication; MRI facility
access; guidance on ethics submission; and coauthorship only where scientifically warranted (see
`COLLABORATION_REQUEST.md`).
