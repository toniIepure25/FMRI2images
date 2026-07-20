# 22 — MINDIR-Atlas Retrospective Experiment Plan (Track R)

**Date:** 2026-07-17 · **Nothing has been run. No GPU.**

**Track R** asks: do neural information reorganizations show reproducible structure across
different state transitions? **Track R is not a substitute for Track P** — it cannot support
causal path dependence, prospective counterfactual prediction, or a universal operator.

---

## Revised theory — forced by the data, not chosen

**Discarded:** "mental-state transitions implement information *contraction*."

**Refuted by already-published measurements:**
- Roy et al.: perception→imagery **contracts** dimensionality in early visual cortex (and is
  near-identity in ventral/lateral/parietal).
- Li, Yang & Bao: perception→VWM **expands spatially** — robust *ipsilateral* representation,
  **70–90% of ipsilateral LOC**, far exceeding unilateral perception.

> **Two internally-generated states reorganize information in opposite geometric directions.**
> A pure contraction thesis is dead on arrival. **Do not force "contraction" onto expansion.**

**Adopted:** transitions **reorganize** information through **contraction, expansion, rotation,
redistribution and noise injection**. The discovery target is whether there are **reproducible
regularities in *which information* is retained and *how* its neural implementation moves.**

**Transition signature** (computed identically for every transition): effective rank ·
singular spectrum · subspace alignment · retained feature information · removed feature
information · spatial redistribution · hemispheric redistribution · ROI flow · uncertainty.

---

## E-M1 — Neural specificity of the Roy transformation ★ RUN FIRST

**Data:** NSD-Imagery (class 1, public). **Blocked by nothing.**

**Question:** is the vision→imagery transformation *neural*, or does it reduce to observable
stimulus features? **Roy et al. ran no semantic control.**

**Seven predictors of measured imagery activity:**

| # | Predictor |
|---|---|
| 1 | stimulus identity only (one-hot) |
| 2 | low-level visual features (Gabor / SF / colour) |
| 3 | image embedding (CLIP-image) |
| 4 | semantic / text embedding (CLIP-text) |
| 5 | **source-state perception neural activity** (Roy's `vis2img`) |
| 6 | source neural **residualized against** stimulus features |
| 7 | source neural **+** stimulus features |

**Primary estimand:** *incremental target-state neural variance explained by source-state
neural activity beyond observable stimulus features* — i.e. **(7) − (1–4 best)**, with (6) as
the direct test of the unique neural component.

**Method:** content-held-out CV · nested hyperparameter selection · matched capacity ·
variance partitioning (unique-neural / unique-semantic / shared) · conditional permutation ·
noise-ceiling normalisation · subject-level effects (n = 8).

**Kill:** increment ≈ 0 → *"the imagery transformation" is not demonstrably neural.* A
publishable negative about an actively-cited preprint.

> **A neural transformation may NOT be claimed merely because predictor 5 predicts imagery.**
> With **12 conditions**, predictor 5 can succeed by implicitly identifying *which of 12
> stimuli* it was. This is the whole point of the experiment.

## E-M2 — Perception → working-memory reorganization

**Data:** Li/Yang/Bao Dryad (class 1, public, 7.37 GB). **Blocked by nothing** — but inspect
the file manifest first; **whether trial-level responses exist is UNVERIFIED**, and
`wm_avged.zip` may be the minimum proof-of-concept subset.

Estimate direct perception→WM mappings; compare 1-item vs 2-item load; quantify dimensionality,
retained subspace, **spatial and hemispheric redistribution**, object-identity retention,
position retention, cross-decoding, and **incremental neural value beyond object and position
labels** (the E-M1 control, transplanted).

**Classify the transition** as contraction / expansion / rotation / redistribution / mixed.
**The published result predicts expansion.** If our analysis says "contraction", we have a bug.

## E-M3 — Encoding → immediate → delayed retrieval ★ the scientific core

**Data:** Oedekoven (class 3). **BLOCKED on an access request that has not been sent.**

For held-out videos, compare: direct encoding→delayed · encoding→immediate ·
immediate→delayed · **composed** encoding→immediate→delayed · independent direct translators ·
identity/semantic-only · shuffled-video controls.

**Questions:** does the composed mapping predict target-state neural structure? Does immediate
retrieval **mediate** what survives a week? Which event features are lost vs stabilized? Is the
transformation shared across the 21 participants?

> **Terminology, binding:** this is **predictive factorization of multi-stage memory
> transformation**. It is **NOT causal path dependence** — the route was not randomized.
> Participants did not receive different paths; everyone went encoding→immediate→delayed.

## Cross-transition feature assay

Common hierarchy: low-level structure · spatial position · orientation · shape · object
identity · category · scene structure · relations · semantic content · event identity.

| Dataset | Measurable levels |
|---|---|
| NSD-Imagery (12 stimuli) | low-level, category, semantic — **identity underpowered at n=12** |
| Li/Yang/Bao | **position, hemifield, object identity** — strongest spatial assay |
| Oedekoven (24 videos) | **event identity, narrative, semantic, relations** — no low-level control |
| NSD-Synthetic | **orientation, SF, contrast, phase, colour** — best low-level reference, but perception-only |

> **Do not force oriented gratings and naturalistic videos into one feature space.** Use a
> partially shared assay plus dataset-specific components. Cross-dataset convergence is
> claimable only on the **ordering** of feature retention, never on absolute values.

## Claim boundaries

**Track R may support:** neural specificity beyond stimulus features · reproducible transition
signatures · predictive multi-stage factorization · cross-dataset convergence of
feature-retention ordering.

**Track R may NOT support:** causal path dependence · universal mental-state algebra ·
prospective counterfactual prediction · closed-loop control · universal cross-person operator ·
the complete MindStates theory. **All require Track P.**
