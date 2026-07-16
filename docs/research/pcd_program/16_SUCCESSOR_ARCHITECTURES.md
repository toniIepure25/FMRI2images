# 16 — Successor Architectures

**Date:** 2026-07-16 · Four candidates. Novelty assessed against `13`/`14`, not against ambition.

Shared constraints on all candidates (from `15`): per-ROI nodes (no pooling of
category-selective ROIs); no unrestricted `nsdgeneral_other` bypass; every interpreted
quantity has an identifying objective + gradient test; standardised protocol; splits reused.

---

## Candidate A — Identifiable Predictive Cortical Inference

**Question:** does top-down prediction of lower-level ROI activity, with recurrent inference,
learn representations that predict held-out neural activity better than feed-forward?

**Architecture:** per-ROI nodes → shared latent state per level; top-down heads
`g_l(z_{l+1}) → ŷ_l`; residual `e_l = y_l − ŷ_l`; precision weighting `Π_l`; 2 shared
inference iterations; retrieval head on the fused state.
**Losses:** retrieval + held-out ROI prediction + precision NLL.
**Identifiable:** `ŷ_l` (supervised by held-out ROI activity); `e_l` (defined by `ŷ_l`).

- **Novelty: DEAD.** ESANN 2025 already shows PC dynamics improve fMRI predictivity across
  ROIs over feedforward; deep PC nets vs. temporal adaptation likewise. Hi-DREAM owns the
  ROI hierarchy. We would arrive second into a literature that uses **laminar fMRI and
  encoding models**, where retrieval accuracy is not the currency.
- **Fatal confound:** NSD's temporal resolution cannot resolve recurrent dynamics. Any
  recurrence result tests a computational prior, never measured recurrence.
- **Reviewer objection:** *"ESANN 2025 did this with a simpler model and a proper encoding
  evaluation. What is new?"* — no answer.
- **Compute:** high (recurrence × 8 subjects). **Verdict: REJECT.**

## Candidate B — Cross-Subject Cortical World Model

**Question:** does a shared cortical latent + constrained subject adapters + masked-ROI
pretraining improve zero/few-shot transfer to unseen subjects?

**Architecture:** shared ROI encoder `E`; subject adapter `A_s = W_shared + U_s V_sᵀ`
(rank r ≪ d); masked-ROI reconstruction pretraining; retrieval head.

- **Novelty: DEAD, comprehensively.** MindEye2 (shared latent + subject ridge, ICML 2024);
  MindTuner (LoRA + Skip-LoRA subject adapters, AAAI 2025); ZEBRA (shared/subject
  disentanglement, **claims first zero-shot**, NeurIPS 2025, code public); MindAligner
  (explicit functional alignment, ICML 2025); Region-Aware fMRI FM (ROI-guided masking).
  Plus three 2026 preprints (Pictorial Cortex, StableMind, MindAdapter).
- **Reviewer objection:** *"This is MindEye2 with MindTuner's adapters. ZEBRA already does
  zero-shot without adaptation and released weights. Why is this a paper?"*
- **Verdict: REJECT.** A saturated axis moving monthly, against better-resourced groups.
  Our own model is at 16.4% R@1. We would lose.

## Candidate C — Identifiability-Constrained Perception→Imagery Decoder ★ **SELECTED**

**Question:** *Is the perception→imagery generalization gap caused by **unconstrained
intermediate representations** rather than by model capacity per se? Does forcing
representations to predict held-out neural activity improve imagery transfer at matched
perception performance?*

**Why this is the one.** NSD-Imagery (CVPR 2025) published the benchmarked finding that
**complex architectures overfit to vision and transfer worse to imagery than linear models**.
That is a *phenomenon without a mechanism*. "Complexity" is a correlate. Our hypothesis names
a mechanism, and it is testable, cheap to falsify, and useful when negative.

**Architecture (deliberately minimal — see `18` for the formal spec):**
- per-ROI nodes, shared encoder + **low-rank subject adapter** (hygiene, not the hypothesis);
- `nsdgeneral_other` as a **constrained context node** (strict parameter budget, no bypass);
- **masked-ROI neural-prediction head** — mask a subset of ROI nodes, predict their activity
  from the rest (this is the identifiability constraint and the independent variable);
- existing vMF retrieval head, unchanged, for comparability.

**Independent variable:** `λ_neural` — the weight on masked-ROI prediction. **Everything
else held fixed.** The experiment is one knob.

**Identifiable quantities:** masked-ROI predictions (supervised by held-out real activity —
out-of-sample by construction, leakage-safe by masking).

- **Novelty: TENTATIVE-OPEN.** Joint encode/decode exists (LEA) as a *performance* device;
  no work found evaluating a neural-prediction objective as an **identifiability constraint
  measured on imagery transfer**. Must confirm by full-text reading of NSD-Imagery + LEA
  before any novelty wording (`14` §3).
- **Fatal confounds:** (i) imagery SNR is low — a null may be a power failure, not a
  mechanism failure; **an equivalence test is mandatory, not optional**; (ii) NSD-Imagery
  subject overlap with our 8 must be verified; (iii) masked-ROI prediction may simply act as
  a regularizer — hence the **parameter- and regularization-matched control arm** (noise/
  dropout tuned to match training loss) is load-bearing, not decorative.
- **Reviewer objections:** *"Isn't this just multi-task regularization?"* → the matched-
  regularization arm answers it. *"n=8 and low SNR"* → subject-level hierarchical inference +
  equivalence bounds, declared in advance. *"NSD-Imagery already showed linear is better"* →
  we explain *why*, and test whether the mechanism transfers to complex models.
- **Pilot:** 1 subject, λ_neural ∈ {0, 0.5}, perception-only training, imagery eval.
  **Kill:** if λ_neural>0 does not improve the transfer ratio beyond the matched-regularization
  arm's CI on 4 subjects → identifiability hypothesis falsified → report negative.
- **Compute:** LOW. No diffusion, no recurrence, no reconstruction. Retrieval only.
- **Venue:** CCN / NeuroImage / a CVPR-or-NeurIPS benchmark-and-analysis track. Positive →
  mechanism for a documented open problem. Negative → narrows it, publishably.
- **Hard dependency:** NSD-Imagery is **not on the pod** (`14` §5).

## Candidate D (proposed here) — Identifiability Audit of Brain-Inspired Decoders

**Question:** across published brain-inspired fMRI decoders, are the biologically-labelled
components identifiable, or arbitrary task-useful features that survive ablation for
non-biological reasons?

**Motivation — this is our own case study generalised.** We shipped a "per-level uncertainty"
head that never received a gradient and would have produced stable, plausible, publishable
figures (T13/F-002). Hi-DREAM's random/reversed control degrading performance is consistent
with *"the anatomy is right"* **and** with *"any coherent structure helps"* — the two are
distinguished only by parameter-matching, which is unverified.

**Deliverable:** a gradient-reachability + identifiability test suite; re-analysis of open
models (ZEBRA has public code/weights); a taxonomy of unidentifiable-interpretation failures.

- **Novelty: OPEN but meta-scientific.** No such audit surfaced.
- **Why not primary:** it is a critique, and critiques of others' work from a lab whose own
  model scores 16.4% invite an obvious rejoinder. It also depends on reproducing several
  external codebases — high engineering risk, low scientific ceiling.
- **Verdict: RETAIN AS SECONDARY.** The test suite is a real artifact we already partly have
  (`tests/test_pcd_gradient_flow.py`) and it strengthens Candidate C's methods section at
  near-zero marginal cost. **Do not staff it as a separate paper.**

## Summary

| | A | B | **C** | D |
|---|---|---|---|---|
| Novelty | dead | dead | **tentative-open** | open |
| Identifiability | good | weak | **good** | n/a |
| Feasibility | med | high | **high** | low |
| Compute | high | med | **low** | low |
| Data risk | low | low | **HIGH (NSD-Imagery absent)** | med |
| Beats better-resourced rivals? | no | no | **not required** | n/a |
| Useful if negative | partly | no | **yes** | yes |
| **Verdict** | REJECT | REJECT | **PRIMARY** | SECONDARY |
