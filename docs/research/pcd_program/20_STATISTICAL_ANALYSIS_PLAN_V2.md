# 20 — Statistical Analysis Plan v2

**Date:** 2026-07-16 · **Status:** DRAFT — **not preregistered.** Freezes at Gate 4, before E-05.
Incorporates reviewer objections O-4 and O-6 (`11_REVIEWER_REPORTS/phase2_thesis_review.md`).

---

## 1. Unit of inference

**The subject.** n = 8 (fewer if NSD-Imagery overlaps partially — a blocking unknown).
Trials are **not** independent evidence about humans (rule 9). Stimuli are a second crossed
random factor.

## 2. Primary estimand (O-6 — not a ratio)

Per `18` §6: paired difference on the variance-stabilised scale `φ = arcsin(√·)`, with a
`γ`-weighted perception-matching term; `γ` fixed before unblinding.

**Preferred equivalent:** hierarchical (Bayesian or mixed-effects) model
```
φ(A) ~ condition * lambda_neural + (1 | subject) + (1 | stimulus)
```
The **`condition × λ_neural` interaction** is the primary coefficient. Report the posterior /
CI for that term. **No ratios of noisy estimates anywhere in the paper.**

## 3. Uncertainty

Hierarchical bootstrap resampling **subjects and stimuli** (≥10 000 draws); subject-level
paired contrasts reported individually — **all 8 subjects shown, including outliers**;
permutation tests for null calibration; 95% intervals; heterogeneity reported, never averaged away.

With n = 8, asymptotic p-values are not definitive and are reported only alongside intervals
and per-subject effects.

## 4. Multiplicity

| Family | Members | Correction |
|---|---|---|
| Primary | the interaction term | none (single preregistered endpoint) |
| Controls | O-1 matched-reg, O-3 shuffled | none (each is a distinct preregistered question) |
| Ablations (`18` §7) | ~8 arms | **FDR (BH), q = 0.05** |
| Per-ROI analyses | 17 ROIs | **FDR (BH), q = 0.05** |
| Exploratory | everything else | reported as exploratory, **never as confirmatory** |

## 5. Smallest effect size of interest and equivalence (O-4)

**SESOI: not yet set — and it cannot be honestly set from our own data.** It must be derived
from **NSD-Imagery's reported perception-vs-imagery effect sizes**, which requires reading the
paper in full. **This is a blocking item and the panel's top objection.**

Provisional placeholder pending that read: SESOI = **3 pp** on the arcsin-transformed
image-level R@1 interaction. **This number is a placeholder and must not be cited.**

**Equivalence testing is mandatory for any null claim** (TOST or ROPE against the SESOI). A
null without an equivalence bound at n = 8 and imagery SNR is uninterpretable — it would be a
power failure reported as a finding, which is exactly the error F-001 already made once
(reading a gap before convergence).

## 6. Power

**Not yet computed.** Requires NSD-Imagery's effect sizes and subject count. **Doing the power
analysis after seeing the data is not acceptable** (R3). If the analysis shows the design
cannot detect the SESOI, the honest actions are: narrow the claim, invoke the `17` §5
fallback, or report the program as underpowered — **not** to run it and interpret whatever
emerges.

## 7. Pipeline validation

**Before touching real results**, every analysis (bootstrap, hierarchical model, variance
partitioning, noise-ceiling normalisation) is validated on **synthetic data with known ground
truth** plus permutation nulls. Ships as `tests/test_analysis_pipeline_synthetic.py`.
Rationale: `pcd_neuroscience_analysis.py` produced plausible figures from untrained weights
(F-002). We do not get to make that mistake twice.

## 8. Discipline

- Primary metric and selection criterion declared **before** E-05 and never changed
  post-hoc; a change after seeing results is labelled **exploratory** (rule 14).
- SHARED1000 touched **once**, at the end, never for selection (rule 3).
- All seeds reported, including failures (rules 13, 15).
- Both image- and trial-level R@1 reported permanently (D-005).
- Negative results published (rule 13; Outcome B is an acceptable destination).
