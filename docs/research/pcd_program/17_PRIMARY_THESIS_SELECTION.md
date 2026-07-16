# 17 — Primary Thesis Selection

**Date:** 2026-07-16 · **Decision:** D-006 (see `04_DECISION_LOG.md`)

---

## 1. Selected thesis

> **The perception→imagery generalization gap in fMRI decoding is a failure of
> identifiability, not of capacity. Constraining a decoder's intermediate representations to
> predict held-out neural activity (masked-ROI prediction) improves perception→imagery
> transfer at matched perception performance — and if it does not, capacity-independent
> identifiability is not the mechanism, which is itself worth reporting.**

**Name:** drop "NeuroPC". It reasserts the "predictive cortical" label Gate 0 disproved
(T6/D-002) and Hi-DREAM owns that architecture. Working name: **INI — Identifiability-
constrained Neural Inference**. Rename freely; do not resurrect "predictive cortical".

**The mission's tentative thesis is retained but demoted and specialised.** Its wording —
*"a cortical inference model trained to predict both held-out neural activity and visual
representations can learn more transferable and scientifically interpretable
representations"* — survives in mechanism but not in framing: "cortical inference" becomes
an identifiability constraint (no biological claim), and "transferable" is specialised to
**perception→imagery**, because cross-subject transfer is a dead axis (`14` §1).

## 2. Scoring (1–5; adversarial panel, `11_REVIEWER_REPORTS/phase2_thesis_review.md`)

| Criterion | A (PC inference) | B (world model) | **C (INI)** | D (audit) |
|---|---|---|---|---|
| Genuine novelty | 1 | 1 | **3** | 4 |
| Identifiability | 4 | 2 | **4** | — |
| Feasibility | 3 | 4 | **4** | 2 |
| Expected impact | 2 | 2 | **4** | 2 |
| Broad relevance | 3 | 3 | **4** | 2 |
| Data sufficiency | 4 | 4 | **2** ← risk | 3 |
| Reproducibility | 3 | 3 | **4** | 3 |
| Risk (5 = low) | 2 | 2 | **3** | 2 |
| Compute (5 = cheap) | 1 | 2 | **5** | 4 |
| Publication potential | 2 | 1 | **4** | 3 |
| **Total** | 25 | 24 | **37** | 25 |

**C wins on every axis except data sufficiency, which is its single serious risk (§5).**

## 3. Why the simplest sufficient program

Per the mission's instruction not to select the most complex candidate: C is the *least*
complex candidate. No diffusion, no recurrence, no reconstruction, no precision weighting, no
adversarial disentanglement. **One independent variable — `λ_neural` — and everything else
held fixed.** That is not a compromise; it is why the result will be interpretable.

Our own history is the argument: v1→v4 changed seven regularization knobs simultaneously and
learned nothing from four runs (F-001). A one-knob experiment is the corrective.

## 4. Contributions (at most three, ordered)

1. **A mechanistic account of the imagery-generalization gap** — identifiability, not
   capacity — tested with a matched-regularization control that separates "constraint" from
   "regularizer". *(Or its rigorous refutation with an equivalence bound.)*
2. **An identifiability protocol for brain-inspired decoders**: every interpreted quantity
   requires an identifying objective and a gradient-reachability test. Motivated by a
   documented real failure of our own (T13/F-002), shipped as a reusable test suite.
3. **A leakage-audited, protocol-standardised NSD→NSD-Imagery transfer evaluation** with
   image-level splits, sealed SHARED1000, and both image- and trial-level retrieval reported.

**Explicit non-contributions:** SOTA reconstruction; cross-subject transfer; uncertainty
calibration; any claim about cortical implementation.

## 5. The hard dependency and the fallback

**NSD-Imagery is not on the pod** (verified 2026-07-16; 181 TB free, so this is an access
task, not a capacity one). Before any INI training:

**Compatibility audit (blocking):** subject overlap with our 8; trial counts and imagery
conditions; ROI availability matching our 17; preprocessing parity; beta version; licensing;
expected distribution shift.

**Fallback if NSD-Imagery is unusable (declared now, before seeing results):** the
identifiability hypothesis is testable on **any** distribution shift the sealed data already
supports. Substitute endpoint: transfer from perception to **low-reliability / reduced-data
regimes** (`19` §2 regimes 7–8) — same one-knob design, same mechanism, weaker claim and
weaker venue. **Declared in advance to prevent post-hoc endpoint substitution** (rule 14).

## 6. Kill criteria (predefined)

| Condition | Action |
|---|---|
| `λ_neural>0` does not beat the matched-regularization arm on transfer ratio, 4 subjects, subject-level CI | Identifiability hypothesis **falsified** → report negative with equivalence bound |
| Masked-ROI prediction fails to beat a per-ROI mean baseline out-of-sample | The constraint is not identifying anything → **fix or abandon before** interpreting |
| NSD-Imagery subjects do not overlap ours, or ROIs are incompatible | Invoke §5 fallback |
| Improvement appears only at degraded perception performance | Not a mechanism — a capacity trade. Report as such |
| Full-text reading shows NSD-Imagery or LEA already tested an auxiliary neural-prediction objective | **Novelty dead** → demote to replication + mechanism analysis, or invoke Outcome C |

## 7. Venue routing (evidence-determined, not chosen)

- **Positive, clean, with mechanism:** CVPR/NeurIPS benchmark-and-analysis track, or CCN.
- **Positive but modest:** CCN, or NeuroImage as a methods contribution.
- **Negative with a tight equivalence bound:** CCN or a NeurIPS workshop — a genuine
  narrowing of a documented open problem.
- **Never:** a decoding-SOTA venue. We are at 16.4% R@1 (T11) and will not be competitive,
  and this thesis does not require us to be.
