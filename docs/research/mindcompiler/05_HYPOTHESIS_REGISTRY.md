# 05 — MINDIR Hypothesis Registry (Gate M0.1)

**Date:** 2026-07-17 · Revised against the full Roy audit (`01`).
**Nothing here has been tested. No data has been run.**

Terminology contract: **`law`, `universal`, `operator`, `algebra`, `counterfactual` remain
provisional.** None may appear as a claim until its row below passes.

---

## Cross-cutting definitions (these four are NOT equivalent — mission §7)

| Concept | Meaning | Established by Roy? |
|---|---|---|
| **Common content information** | both states carry decodable info about the same stimulus | yes (long established) |
| **Aligned representational geometry** | the subspaces coincide | **measured: near-100% in ventral/lateral/parietal, 25–50% in V1–V4** |
| **Predictive state transformation** | source-state activity predicts target-state activity | **yes, r ≈ 0.3–0.5** |
| **Composable state operator** | transformations chain lawfully across ≥3 states | **untested by anyone** |

---

## H7 — Semantic shortcut ★ **RUN THIS FIRST**

**Statement:** the vision→imagery transformation is explained by stimulus content, not by
source-state *neural* activity. Roy's `vis2img` may be recovering "what the stimulus was",
not "how the brain transforms state".

- **Estimand:** ΔR = r(vis2img, neural source) − r(semantic-only predictor), per ROI, per subject.
- **Semantic-only baselines:** content identity (one-hot over the 12 / 512 conditions), CLIP
  image embedding, CLIP text embedding, category label, low-level Gabor/SF/colour features.
  Each fit to predict **imagery** activity directly, no neural source.
- **Identifying assumption:** with content identity one-hot, the semantic predictor has access
  to *everything* stimulus-derived; any residual advantage of `vis2img` is neural.
- **Null:** ΔR ≤ 0 — the transformation carries no information beyond stimulus identity.
- **Method:** variance partitioning (unique neural / unique semantic / shared).
- **Success:** ΔR > 0 with subject-level CI excluding 0, in ≥2 ROI families.
- **Kill:** ΔR ≈ 0 → **"the imagery transformation" is not demonstrably a neural
  transformation.** That is a publishable negative about a published result, and it would
  terminate MINDIR's neural framing.
- **Data:** NSD-Imagery (8 subjects). **Available now. No new data. No GPU.**

> **Why first:** it is the cheapest, it validates or invalidates the entire substrate, and it
> is the one test whose *negative* result is as valuable as its positive. This is the B5-first
> discipline inherited from NCD, applied to someone else's finding rather than our own.

## H1 — Universal contraction spectrum

**Statement:** subjects share the relative ordering/strength of information loss per ROI.
- **Estimand:** inter-subject correlation of the per-ROI singular-value spectrum of `W_im`.
- **Null:** ISC ≤ that of degree-matched permuted spectra.
- **CRITICAL identifying assumption:** requires **subject-specific functional alignment**.
  arXiv 2409.06843 shows anatomical alignment detects only ~10 shared dimensions — a null
  under anatomical alignment would be **methodological, not biological**.
- **Baseline:** scale-free spectrum prediction (2409.06843) — a shared spectrum may be a
  generic cortical property, not a *transition* property. **This baseline is the real threat.**
- **Success:** ISC exceeds both permuted null *and* the scale-free prediction.
- **Kill:** spectra match the generic scale-free form → nothing state-specific.
- **Data:** NSD-Imagery, 8 subjects. Available now. Underpowered but testable.

## H2 — Shared retained/removed subspaces

**Statement:** after functional alignment, retained (and removed) dimensions are more similar
across subjects than matched nulls.
- **Estimand:** principal angles between subjects' retained subspaces vs permutation null.
- **Confound:** subjects share *stimuli*; shared subspaces may reflect shared stimulus
  statistics. **Requires the H7 control to be passed first.**
- **Kill:** angles indistinguishable from null, or fully explained by stimulus features.

## H3 — Hierarchical internalization

**Statement:** low-level sensory information is lost as cognition becomes internally
generated; semantic/relational information is preferentially retained.
- **STATUS: LARGELY ESTABLISHED, NOT NOVEL.** Roy shows early-visual contraction with
  higher-visual parity; the memory-compression literature shows perception→memory loss.
- **Only novel form:** a *quantitative, feature-resolved* retention profile (orientation, SF,
  colour, texture, position, shape, identity, category, relations, layout) across ≥3 states.
- **Data:** feature-resolved assays need Dataset 2's factorial structure (64 objects × 8
  locations) or new data. **Not answerable on NSD-Imagery's 12 stimuli.**

## H4 — Path dependence

**Statement:** the final representation depends on the route through intermediate states.
- **Estimand:** distance between `perception→WM→recall` and `perception→recall` endpoints.
- **Data: NONE PUBLIC.** Requires ≥3 states on the same content, trial level.
- **Status: PROSPECTIVE-ONLY.**

## H5 — Predictable lossy composition

**Statement:** composition predicts *cumulative information loss* even when it does not beat
a direct mapper on raw accuracy.
- **This is the reframing that survives Roy.** The original algebra thesis required composition
  to *win*; on a lossy map it will not. But a model that **predicts how much is lost along a
  path** is falsifiable and does not require composition to be accuracy-improving.
- **Estimand:** predicted vs measured rank/variance retention along composed paths.
- **Kill:** composed loss unpredictable from constituent transitions.
- **Data: NONE PUBLIC. PROSPECTIVE-ONLY.**

## H6 — Individual compression fingerprint

**Statement:** individual transition spectra predict stable differences in vividness/recall.
- **PARTIALLY PRE-EMPTED:** Roy reports VVIQ ↔ V1 imagery dimensionality **r = 0.71, n = 8,
  bootstrap CI [0.15, 0.98]**.
- **That CI is nearly the entire positive range.** With n = 8 this is a *hypothesis*, not an
  established effect. **Replication with adequate n is a legitimate contribution** — and would
  be a genuine service, since the finding is currently being cited from a preprint.
- **Kill:** fails to replicate at adequate n → report the negative.

## H0 — Independent transitions (the null model)

Every state pair has an unrelated mapping; no shared structure, no transferable spectrum.
**H0 is the default and must be rejected before any MINDIR claim.**

---

## The first falsification experiment (E-M1)

> **Test H7 on NSD-Imagery: does `vis2img` beat every stimulus-only predictor at predicting
> measured imagery activity?**

- Public data, 8 subjects, no GPU, no new acquisition.
- Reuses the inherited verified-split and leakage tooling.
- **Both outcomes are publishable.** Positive: the transformation is neural, and MINDIR's
  substrate is validated. Negative: a published transformation is a stimulus-identity effect —
  a substantive correction to an active literature.
- **This is the only MINDIR experiment currently executable, and it must precede everything.**
