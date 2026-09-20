# Phase 5 — Conservative novelty statement

> **Important:** the bibliographic details in `literature_positioning_matrix.csv` are drawn from domain
> knowledge and **must be verified against primary sources by the human authors before submission.** The
> novelty claim below is deliberately scoped so that it does not depend on any single citation being the
> literal "first."

## One-sentence novelty (survives all three reviewers)

*Using a public imagery dataset and a prospectively frozen pipeline, we decompose the human perception→imagery
transformation into a shared, perception-recoverable component and a low-dimensional but subject-specific
target-state residual, and we characterise how little direct imagery data is needed to calibrate to that
residual, how fragile that calibration is to acquisition schedule, and that repeat-to-repeat outside-support
geometric agreement — not a scalar reliability variable or a single dominant drift mode — predicts whether a
calibration generalises.*

## What this adds beyond the closest work (bounded)
- Beyond the **perception→imagery transformation** literature (incl. the Roy et al. work we extend): an explicit
  **within-/outside-support decomposition** of the target-state, quantifying the residual that perception does
  not recover.
- Beyond **shared-response / hyperalignment**: evidence that the residual **orientation does not transfer**
  across subjects under the tested representation, i.e. a limit on shared-space approaches for this component.
- Beyond **few-shot decoding** (e.g. MindEye2): a **minimal-calibration frontier** framed as a data-cost
  question with an explicit **schedule-robustness** audit, rather than a reconstruction-accuracy result.
- Beyond **reliability / repeat-variability** accounts: a demonstration (via the exploratory X-series) that the
  failure-predictive repeat geometry is **not** captured by a scalar reliability variable or a single dominant
  drift mode.

## Honesty clause
We do **not** claim to be "first" or "novel" in absolute terms. The transformation concept, imagery decoding,
subspace geometry, and few-shot calibration all have substantial prior literature. Our contribution is the
**specific decomposition + minimal-calibration + fragility + geometric-predictor package on a development
cohort, with a preregistered independent replication.** If the human literature check finds a paper that already
reports this exact package, the novelty sentence must be downgraded accordingly.
