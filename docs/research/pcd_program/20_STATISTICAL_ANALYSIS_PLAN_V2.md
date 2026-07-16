# 20 — Statistical Analysis Plan v2

**Date:** 2026-07-16 · **Revised Phase 2.5.** **Status:** DRAFT — **not preregistered.**
Freezes after the E-P1 pilot supplies measured variance (`27` §3).
Incorporates reviewer objections O-4 and O-6.

---

## 0. Phase 2.5 revision — what changed and why

The thesis is now **multi-shift generalization** (`24`), not imagery. Consequences:

- **n = 8 for shifts 1, 2, 4, 5, 6.** Only shift 3 (imagery) is n = 4. **The power crisis is
  confined to one secondary test** — the most important statistical consequence of the
  reframing.
- **NSD-Synthetic is the primary OOD test**: 8 subjects, 284 stimuli, CC-BY 4.0 (`26`).
- **The primary NSD-Synthetic endpoint is a slope, not a mean.** Contrast (5 levels) and
  phase-coherence (4 levels) give a **parametric OOD axis**. The thesis predicts ARM-B's
  margin **grows with OOD degree**; a constant offset is a capacity effect, not
  generalization. Model the slope, not 4 independent cell comparisons (`26` §5).
- **Power numbers below cannot be written until the E-P1 pilot reports measured variance.**

## 1. Units of inference

**Two crossed random factors: SUBJECT and UNIQUE STIMULUS IDENTITY.**

- Trials are **not** independent evidence (rule 9). Repeated presentations of one image are
  repeated measures of one stimulus.
- **Reconstruction seeds are not independent stimulus evidence either** — a model sampled
  twice on one stimulus contributes one stimulus observation, not two.
- Hierarchical model: `(1 | subject) + (1 | stimulus)`. Never pool trials as if independent.

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

## 8. Per-dataset protocols (Phase 2.5)

### NSD-Imagery (shift 3, n = 4 — secondary, sealed)

- **Report simple / complex / conceptual conditions separately.** Stimulus type is a
  **preregistered fixed factor**, never an exploratory slice — with 18 stimuli and 3
  conditions, slicing until something is significant is the obvious failure mode (R-16).
- Hierarchical uncertainty over subjects **and** stimuli.
- **Do not rely on asymptotic subject-level tests with 4 subjects.** Use **randomization /
  exact tests** where feasible — with n = 4 the exact permutation distribution over sign
  flips has only 2⁴ = 16 points, which bounds the achievable p-value at 1/16 = 0.0625.
  **State that ceiling explicitly rather than reporting an asymptotic p that pretends
  otherwise.**
- **Leave-one-stimulus-out** analyses, and per-stimulus sensitivity: report whether any
  single stimulus drives the effect.
- **Zero-shot invariant is a hard rule, test-enforced**
  (`test_no_arm_touches_imagery_or_the_sealed_set`): no imagery sample may touch training,
  model selection, λ selection, checkpoint selection, early stopping, or representation
  design. A leak collapses our setting into Spera et al. (2026) and forfeits the
  contribution (`23` §5).

### NSD-Synthetic (shift 2, n = 8 — primary OOD)

- **Confirmatory families (scene-derived, 52 stimuli):** `natural` (8), `manipulated` (12),
  `contrast` (16), `phase` (16). **FDR (BH) across the 4 families, q = 0.05.**
- **Primary endpoint:** slope of decoding performance against **OOD degree** along the
  contrast and phase continua.
- **Cross-family generalization is required.** An effect present in one family only is **not
  generalization** and is reported as such — this is the specific way a positive result here
  could be an artefact.
- **Exploratory (232 non-semantic stimuli: spiral, chromatic noise, words, noise):** reported
  separately, never pooled with the confirmatory set, never used for confirmation.
  **CLIP-decoding near-chance on pink noise is the expected null and proves nothing** — it is
  a measurement artefact of using a semantic target, not evidence about generalization.
- **Neural predictivity is valid on all 284** and is the one endpoint the non-semantic stimuli
  can support. Use noise-ceiling-normalised explained variance, comparable to Gifford et al.
- NSD-Synthetic is **one session** — lower SNR than NSD-core's 40. Never compare raw
  accuracies across datasets; normalise by noise ceiling (R-19).

## 9. Confirmatory vs exploratory

| Confirmatory (preregistered, corrected) | Exploratory (reported, never confirmatory) |
|---|---|
| ARM-B vs A/C/D/E/F/G/H on the §2 estimand | anything on the 232 non-semantic stimuli |
| NSD-Synthetic OOD slope, 4 scene families | per-ROI analyses beyond the preregistered set |
| Imagery 2AFC by condition | post-hoc subgroup or stimulus slices |
| Held-out neural predictivity | any endpoint chosen after unblinding |

## 10. Discipline

- Primary metric and selection criterion declared **before** E-05 and never changed
  post-hoc; a change after seeing results is labelled **exploratory** (rule 14).
- SHARED1000 touched **once**, at the end, never for selection (rule 3).
- All seeds reported, including failures (rules 13, 15).
- Both image- and trial-level R@1 reported permanently (D-005).
- Negative results published (rule 13; Outcome B is an acceptable destination).
