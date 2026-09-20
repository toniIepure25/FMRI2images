# Reviewer B — Statistics / Methods

Scope: N=8, repeated cohort use, multiple testing, prospective vs exploratory, sign-flip inference,
pseudoreplication, model selection, null construction, pipeline-constrained results, calibration-minimum
interpretation, circularity/leakage, robustness. Severity: CRITICAL / MAJOR / MODERATE / MINOR.

## Summary judgement
Methodologically this is stronger than most N=8 imaging papers: participant is the inferential unit,
prospective config freezes with server-verify, estimator replay to <=1e-10, exact sign-flip, and Holm within
declared families. I found **no CRITICAL statistical flaw**. My concerns are about interpretation and
transparency, all addressable by wording.

## Major concerns
- **[MAJOR] Cumulative exploratory reuse of one cohort.** Each gate is individually frozen, but the whole
  program tests the same eight participants many times. Within-gate Holm does not control the program-wide
  family. The manuscript must state that cross-gate multiplicity is *not* corrected, that this is why the three
  surviving findings are preregistered for an independent cohort, and that the exploratory X-series is not
  confirmatory.
- **[MAJOR] "Minimum" language.** N_TRIALS_STAR=8 is the smallest *prospectively tested common-sufficient*
  burden under a *frozen estimator*, on the *development cohort*. It is not a demonstrated lower bound (finer
  grids or other estimators were not the test). Every occurrence must carry these qualifiers.
- **[MODERATE] Pipeline-constrained positives.** X1 (3/4 blocks) and X3-ANGLE are pipeline-constrained; the
  manuscript correctly excludes them, but the Results must make explicit that a "significant" sign-flip alone
  did not qualify — the triviality control gated inclusion.
- **[MODERATE] Sufficiency-criterion nulls.** O2.9/O2.14 use frozen thresholds (0.50; coverage 0.75), not
  permutation nulls. That is legitimate but must be labeled as a *pre-registered decision rule*, not a
  hypothesis test, so readers do not expect a p-value.

## Minor concerns
- **[MINOR]** State the minimum attainable one-sided p (1/256 ≈ 0.0039) once, and note it bounds achievable
  significance at N=8.
- **[MINOR]** Fisher-z aggregation and participant-first medians should be named in each ρ claim.
- **[MINOR]** Clarify that O2.15 associations use schedules within participant but aggregate to the participant
  before inference (no schedule-level pseudoreplication).

## What would cause rejection
- Counting schedules/folds/identity-subsets/repeats as independent N anywhere (I did not find this — verify it
  stays true in every figure and caption).
- Any Holm family that is not the exact pre-declared size.
- Presenting a negative as proof of absence.

## What would change my confidence
- A "Statistical scope and multiplicity" paragraph in Methods stating: participant N=8; within-gate families;
  no cross-gate correction; exploratory vs prospective; sufficiency rules vs tests.
- Explicit inferential-unit statements in every ρ/effect claim and figure caption.
