# 14 — Novelty Gap Analysis

**Date:** 2026-07-16 · **Basis:** `13_SYSTEMATIC_LITERATURE_REVIEW.md` (abstract-level, one index)
**Verdict:** the PCD program as conceived has **no surviving novelty claim**. One genuine gap
remains open, and it is not the one the project was pursuing.

---

## 1. Novelty ledger — what the literature already owns

| Proposed PCD / spec contribution | Status | Owned by |
|---|---|---|
| Group visual ROIs into an anatomical hierarchy (V1/V2 → V3/V4 → category-selective) | **DEAD** | **Hi-DREAM** (2511.11437) — identical early/mid/late grouping, same ROI assignment |
| Map cortical levels to different visual-representation levels | **DEAD** | **MindHier** (2510.22335) hierarchy-to-hierarchy CLIP-layer alignment; **BrainMCLIP** (2510.19332) |
| Reverse/parallel visual-pathway decoding | **DEAD** | **DREAM** (WACV 2024) |
| Test anatomical grouping vs **random / reversed** ROI assignment | **DEAD — and already answered positively** | **Hi-DREAM** ran exactly this control; **BrainMCLIP** ran a 1 000-permutation ROI-label null |
| Low-rank / parameter-efficient subject adapters | **DEAD** | **MindTuner** (AAAI 2025) LoRA + Skip-LoRA |
| Shared latent space + subject-specific projections | **DEAD** | **MindEye2** (ICML 2024) |
| Explicit shared vs subject-specific factor separation | **DEAD** | **ZEBRA** (NeurIPS 2025) adversarial disentanglement, claims *first* zero-shot |
| Explicit cross-subject functional alignment | **DEAD** | **MindAligner** (ICML 2025) Brain Transfer Matrix |
| Joint encoding + decoding objective | **DEAD** | **LEA** (2303.14730); decoder-as-regularizer for encoding |
| Predictive-coding dynamics improve fMRI predictivity | **DEAD** | ESANN 2025 (VGG16 + PC dynamics); deep PC nets vs temporal adaptation |
| Masked-ROI self-supervised objectives | **DEAD** | Region-Aware fMRI FM (2511.00443); MaskROI; Brain-DiT |
| Per-level uncertainty decomposition | **DEAD ON ARRIVAL** | not prior art — **our heads never trained** (T8/T13) |

**Every enumerated contribution in the original PCD design and in the Phase-2 seed
candidates A and B is already published.** This is not a close call.

### The two findings that reshape the program

**(a) Hi-DREAM pre-empts PCD almost exactly.** Same ROI hierarchy, same early/mid/late
grouping, same random-and-reversed control — published eight months ago, with a positive
result. `03_HYPOTHESES.md` H2 is answered, and not by us.

**(b) The `03_HYPOTHESES.md` H1/H2 program is now near-worthless as a *contribution*.**
It retains value only as an internal *diagnostic* (does our model behave sanely?), not as a
paper. The Level-3 bypass measurement should still run — it is one forward pass and it tells
us whether our own model is broken — but it is **not a research programme**, exactly as the
Phase 2 mission warned.

## 2. What the literature does **not** own

Three gaps survive. Two are weak; one is real.

### Gap 1 (weak) — Parameter-matching of the hierarchy controls
Hi-DREAM's random/reversed controls degrade performance, but whether they were
**parameter- and capacity-matched** is unverified [ABSTRACT-ONLY]. A degradation under
random ROI-depth assignment is equally consistent with *"any coherent structure helps"* as
with *"this anatomy is right"*.
**Why it's weak:** this is a referee's comment on someone else's paper, not a thesis. It
could at best support a short methods note, and requires reading their full text first.

### Gap 2 (weak) — Uncertainty in fMRI retrieval
Conformal prediction is mature; brain-decoding applications surfaced were brain-*age*, not
visual retrieval. There may be a real gap in **conformal retrieval for fMRI decoding**.
**Why it's weak:** the repo already has a separate conformal line
(`docs/paper/calibrated_uncertainty_paper.md`), our kappa is degenerate (T9/F-003), and the
search on this family was the thinnest. Not a PCD contribution.

### Gap 3 (REAL) — The mechanism behind the imagery-generalization failure

**NSD-Imagery (CVPR 2025) established a benchmarked negative result:** simple linear +
multimodal decoders transfer to mental imagery **better** than complex architectures, which
**overfit to data recorded exclusively from vision**. Seen-image performance does not
predict imagery performance.

What that paper established: **THAT** complexity hurts imagery transfer.
What nobody has established: **WHY**, and **what to do about it**.

Specifically unclaimed, as far as this review can determine:

1. **No mechanistic account** of *what* complex decoders learn that fails to transfer.
   "Complexity" is a correlate, not a mechanism. Parameter count is not an explanation.
2. **No model designed to control the shared perception/imagery component**, rather than
   discovering it accidentally.
3. **No test of identifiability as the operative variable.** The hypothesis that complex
   decoders overfit *because their intermediate representations are unconstrained* — free to
   exploit perception-specific voxel patterns with no requirement to be neurally meaningful —
   is untested. Joint encoding–decoding (LEA) exists, but as a *performance* device, not as
   an **identifiability constraint evaluated on transfer**.

## 3. The surviving thesis

> **Identifiability hypothesis.** Complex fMRI decoders fail to transfer from perception to
> imagery not because they are large, but because nothing constrains their intermediate
> representations to model cortical structure. Adding a leakage-safe **held-out neural
> prediction objective** (masked-ROI prediction) constrains representations to capture
> structure shared between perception and imagery, and should improve perception→imagery
> transfer **at matched perception performance**.

**Why this is defensible where everything else is not:**

- **It is not a SOTA race.** The endpoint is a *transfer ratio*, not absolute R@1. Our
  16.4% vs the frozen 77.2% (T12) stops being disqualifying — a 167M-parameter model that
  overfits by 82 pp is the *ideal case study* for the pathology NSD-Imagery documented.
- **It is falsifiable and cheap to kill.** If masked-ROI prediction does not improve the
  transfer ratio, the identifiability hypothesis is wrong and we say so.
- **The negative result is publishable.** "Identifiability does not explain the imagery gap"
  meaningfully narrows an open problem that a CVPR benchmark paper explicitly flagged.
- **It rewards simplicity**, which is what our own evidence (F-001: four rounds of
  regularization failed to close an 82 pp gap) and NSD-Imagery independently both demand.
- **It reuses the repo's real asset** — clean, verified, image-level splits with a sealed
  SHARED1000 (T10) — rather than its liability (a mislabelled architecture).

**Novelty status: TENTATIVE, not established.** The review is abstract-level and
single-index (`13` §7). Before any "first"/"novel" wording:
1. Read NSD-Imagery in full; confirm they did not test an auxiliary neural-prediction objective.
2. Read LEA in full; confirm joint encode–decode was never evaluated on imagery transfer.
3. Read Hi-DREAM in full; check parameter-matching of controls (also settles Gap 1).
4. Search families not yet run: OpenReview sweep, citation-graph forward search from
   NSD-Imagery, imagery-decoding literature predating NSD-Imagery.

**Defensible framing today:** *"we test whether X improves Y"* — never *"we are the first to"*.

## 4. Consequences for the seed candidates

| Seed candidate | Verdict |
|---|---|
| **A — Identifiable predictive cortical inference** | **REJECT as primary.** Direction claim dead (T6); PC-dynamics-improve-predictivity dead (ESANN 2025); enters a literature using laminar fMRI and encoding models where retrieval accuracy is not currency. **Salvage:** the neural-prediction objective, as an identifiability *mechanism* rather than a biological *claim*. |
| **B — Cross-subject cortical world model** | **REJECT.** Every component is published: MindEye2 (shared latent), MindTuner (low-rank adapters), ZEBRA (shared/subject disentanglement, claims first zero-shot), MindAligner (functional alignment), Region-Aware FM (masked ROI). Saturated and moving monthly. We cannot win this race. |
| **C — Perception-to-imagery cortical decoder** | **SELECT, reframed.** Not as "a decoder for imagery" (that is just NSD-Imagery's benchmark) but as **a test of the identifiability hypothesis for the imagery gap**. |
| **"NeuroPC" (mission's tentative preference)** | **REJECT the name and the framing.** "Predictive Cortical" reasserts precisely the label Gate 0 disproved (T6/D-002) and Hi-DREAM owns the architecture. The mission's own thesis — *"a cortical inference model trained to predict both held-out neural activity and visual representations learns more transferable representations"* — is **retained**, with "cortical inference" demoted to "identifiability constraint" and transfer specialised to **perception→imagery**. |

## 5. Hard blocker

**NSD-Imagery is not on the pod.** Verified 2026-07-16: `find /home/jovyan/work/data -ipath
"*imagery*"` returns nothing. 181 TB free, so capacity is not the issue — this is a public
dataset requiring download and a compatibility audit (subject overlap with NSD's 8, trial
counts, ROI availability, preprocessing parity, licensing). **The selected thesis is
contingent on this.** See `17_PRIMARY_THESIS_SELECTION.md` §5 for the fallback if the data
proves unusable.
