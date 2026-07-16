# 22 — B-LIT: NSD-Imagery Full-Text Audit

**Date:** 2026-07-16 · **Source:** arXiv 2506.06898 (CVPR 2025), full HTML text read
**Status:** discharges the top blocker in `00`. **Thesis survives — statistical plan does not.**

---

## 1. The novelty question — ANSWERED, FAVOURABLY

> **They tested no auxiliary objective.** No neural-prediction loss, no encoding objective, no
> masked-ROI prediction, no multitask learning. They attribute the imagery gap to
> **architectural choices**, not training modifications, and explicitly recommend future work
> on *"optimal decoding architectures."*

**The gap in `14` §3 is real and confirmed at full-text level for this paper.**
`06_CLAIM_EVIDENCE_REGISTRY` may move the novelty claim from FORBIDDEN to **TENTATIVE**.
Still outstanding before any novelty wording: full-text LEA (arXiv 2303.14730) and the
forward citation graph from this paper (`13` §7).

## 2. Their mechanism is a COMPETING hypothesis, not ours

**Their proposal:** *"the high-dimensional ViT-bigG image embeddings used to drive the SDXL
unCLIP generator in MindEye2 tend to break down when decoded from mental images, which have
brain activity responses of much lower SNR."*

This is a **target-dimensionality / SNR** hypothesis. Ours (`17` §1) is an
**unconstrained-intermediate-representation** hypothesis. They are distinct and both testable.

> **Design consequence (new, load-bearing).** Target dimensionality becomes a **confound that
> must be held fixed**, not a free parameter. All NCD arms use one target embedding
> (768-d CLIP, per the existing `embedding_column: fused`), so `λ_neural` is manipulated at
> **constant target dimensionality**. Their hypothesis predicts no effect of `λ_neural`;
> ours predicts an effect. **The experiment now discriminates two named mechanisms rather
> than testing one against nothing — a materially stronger design.**

## 3. The data — much smaller and stranger than assumed

| Property | Value | Consequence |
|---|---|---|
| Subjects | **4** (NSD subj **1, 2, 5, 7**) — only these completed 40 sessions | **n = 4, not 8.** Guts the `20_SAP` power assumptions |
| Unique stimuli | **18** — 6 geometric (4 bars, 2 crosses), 6 "complex" (5 scenes + 1 artwork), 6 word concepts | **Not NSD-like natural scenes.** Severe distribution shift |
| Repetitions | 8 (vision) / 16 (imagery) | helps trial-level SNR, not subject-level n |
| Trials/subject | 576 | |
| Metrics used | PixCorr, and **2AFC human identification** | |
| Access | naturalscenesdataset.org, institutional procedure | CC-BY-NC-ND 4.0 (paper) |

**Fortunate alignment:** the repo's default `SUBJECTS = subj01 subj02 subj05 subj07`
(`CLAUDE.md`) is **exactly** the four NSD-Imagery subjects. No subject-selection work needed.

## 4. Published baselines — O-5 discharged for free

Table 2, 2AFC identification accuracy (vision → imagery):

| Model | Vision | Imagery | **Drop** |
|---|---|---|---|
| MindEye1 | 84.3% | 73.0% | **11.3 pp** |
| Brain Diffuser | 80.1% | 74.0% | **6.1 pp** |
| iCNN | 74.5% | 66.2% | **8.3 pp** |
| **MindEye2** | 83.1% | **57.0%** | **26.1 pp** ← near 50% chance |

> **This discharges reviewer objection O-5** ("the effect may be specific to your weak base
> model"). Four strong published models are benchmarked on this exact dataset and metric. We
> compare against them directly rather than pleading capacity-regime scoping.

> **And it rescues the endpoint.** Our model does retrieval, not reconstruction — we descoped
> diffusion (`18` §10) and cannot compute PixCorr. But **2AFC identification needs no
> reconstruction**: given fMRI, score CLIP similarity of μ against 2 candidate stimuli. This
> is directly computable from our existing retrieval head, and directly comparable to Table 2.

## 5. What this breaks

**The `18` §6 / `20` §2 estimand is dead as written.** It assumed image-level R@1 on a
6 956-image gallery and n = 8.

| Broken | Replacement |
|---|---|
| Image-level R@1 on 6 956 gallery | **2AFC identification accuracy** (gallery of 18 makes R@1 meaningless; 2AFC is their metric and has published comparators) |
| n = 8 subjects | **n = 4** (subj01, 02, 05, 07) |
| SESOI = 3 pp (placeholder) | **SESOI = 6 pp on 2AFC**, justified below |
| "Detect small effects" | **Only large effects are detectable.** Say so, in the paper |

**SESOI justification (now data-grounded, per O-4).** The published between-architecture
spread in imagery 2AFC is 57.0–74.0% (17 pp), and the vision→imagery drop ranges 6.1 pp
(Brain Diffuser) to 26.1 pp (MindEye2). An intervention worth reporting should move the drop
by at least as much as the *smallest* architectural difference that paper treats as
meaningful, ≈ **6 pp**. **SESOI = 6 pp reduction in the vision→imagery 2AFC drop.**

**Power reality (blunt).** With n = 4, subject-level inference has almost no power for
anything smaller. This program can detect a MindEye2-sized collapse or a Brain-Diffuser-sized
robustness, and **nothing subtle**. Any null must be reported with a TOST/ROPE equivalence
bound against the 6 pp SESOI and an explicit statement that effects < 6 pp are undetectable
here. **A null at n = 4 is weak evidence of absence — this must be stated in the abstract, not
buried in limitations.**

## 6. New risks

- **R-15 — Distribution shift is enormous.** Training: 73k natural scenes. Evaluation: 18
  stimuli, two-thirds of which are geometric shapes and word concepts. A model trained on
  natural scenes has no reason to represent "4 oriented bars". Our 768-d CLIP target may be
  a poor basis for geometric/conceptual stimuli. **This threatens the whole endpoint** and
  must be checked on the 6 "complex" (scene) stimuli separately — likely the only
  NSD-comparable subset, and **6 stimuli is a vanishingly thin endpoint.**
- **R-16 — n = 4 with a 3-way stimulus-type split (simple/complex/conceptual)** invites
  slicing until something is significant. **Stimulus type is preregistered as a fixed factor,
  not an exploratory slice.**
- **R-17 — Institutional data access** may take time or be refused; not yet attempted.

## 7. Verdict

**Proceed — with a materially better experiment and a materially worse statistical position.**

Better: novelty confirmed at full text; a *named competing mechanism* to discriminate against;
four strong published baselines on the exact metric; an endpoint computable from our existing
retrieval head; the repo's default subjects already match.

Worse: n = 4; 18 stimuli; only large effects detectable; a severe distribution shift that
could sink the endpoint before `λ_neural` is ever manipulated.

**Revised kill criterion (supersedes `17` §6 row 1):** if NCD's *vision* 2AFC on NSD-Imagery
stimuli is at or near chance — i.e. the model cannot identify these 18 out-of-distribution
stimuli even from *perception* trials — then imagery transfer is unmeasurable and the thesis
is dead on data grounds, independent of `λ_neural`. **This is now the first thing to test,
and it is cheap: an eval-only pass, no training.**
