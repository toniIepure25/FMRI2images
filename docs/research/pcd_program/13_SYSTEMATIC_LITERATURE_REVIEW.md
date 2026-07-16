# 13 — Systematic Literature Review

**Search executed:** 2026-07-16 · **Tool:** WebSearch (US index) · **Coverage:** through July 2026
**Status:** SUBSTANTIVE BUT INCOMPLETE — see §7 for the honest limits.

**Epistemic rule applied throughout:** every work below was surfaced by a live search this
session, and claims about it are drawn from search-returned abstract/summary text. Where I
have not opened the full PDF, the entry is marked **[ABSTRACT-ONLY]**. Nothing here is
recalled from model memory. **No claim in a paper may cite any of these until the full text
is inspected** — abstract-level reading is sufficient to establish *novelty overlap*, which
is this document's purpose, and insufficient for *technical citation*.

---

## 1. Query families executed

| # | Family | Yield |
|---|---|---|
| 1 | MindEye2 shared-subject latent / few-shot | high |
| 2 | MindAligner cross-subject functional alignment | high |
| 3 | MindTuner cross-subject fine-tuning | high |
| 4 | MindHier hierarchical decoding / CLIP | high |
| 5 | NSD-Imagery mental imagery benchmark | **decisive** |
| 6 | Predictive-coding networks / top-down prediction error / fMRI | high |
| 7 | Masked brain modeling / fMRI foundation models | medium |
| 8 | DREAM / Hi-DREAM reverse visual pathway | **decisive** |
| 9 | Joint encoding–decoding objectives | high |
| 10 | ZEBRA zero-shot cross-subject disentanglement | **decisive** |
| 11 | Conformal prediction / brain decoding uncertainty | low (gap) |
| 12 | Noise ceiling / voxelwise predictivity NSD | medium |
| 13 | Brain-inspired ablation vs random ROI grouping | **decisive** |
| 14 | Biologically-inspired components / scrambled-ROI critique | **decisive** |

---

## 2. Cross-subject decoding — a crowded, fast-moving field

**MindEye2** (ICML 2024, arXiv 2403.11207). Maps all subjects into a **shared-subject latent
space via subject-specific ridge regression**, then a shared non-linear mapping to CLIP,
then fine-tunes SDXL. Pretrains on 7 subjects, fine-tunes on 1 hour of a new subject's data.
SOTA retrieval + reconstruction. [ABSTRACT-ONLY]

**MindTuner** (AAAI 2025, arXiv 2404.12630). Pretrains across 7 subjects, fine-tunes with
**LoRA + Skip-LoRA** to capture a subject "visual fingerprint"; Pivot module bridges fMRI and
text via images. 1 hour of new-subject data. [ABSTRACT-ONLY]

**MindAligner** (ICML 2025, arXiv 2502.05034). **Explicit** functional alignment: a Brain
Transfer Matrix projects a new subject's signals onto a known subject; fine-grained
functional correspondence. Claims cross-subject neuroscience insight. [ABSTRACT-ONLY]

**ZEBRA** (NeurIPS 2025, arXiv 2510.27128). Claims the **first zero-shot** cross-subject
decoder — no subject-specific adaptation at all. Core insight: fMRI representations
**decompose into subject-related and semantic-related components**; uses **adversarial
training to explicitly disentangle** them. Code public (`xmed-lab/ZEBRA`). [ABSTRACT-ONLY]

**Also surfaced, not yet read:** The Pictorial Cortex (arXiv 2601.15071, zero-shot via
compositional latent modeling), StableMind (arXiv 2605.02586, source-free adaptation),
MindAdapter (arXiv 2605.24679, few-shot parameter-efficient residual calibration).

> **Implication.** The cross-subject axis is saturated and moving monthly. Shared latent
> space (MindEye2), low-rank subject adapters (MindTuner), explicit alignment (MindAligner),
> and shared/subject-specific disentanglement (ZEBRA) are all **published**.

---

## 3. Hierarchical / ROI-aware decoding — the PCD premise is occupied

**DREAM** (WACV 2024, arXiv 2310.02265). Crafts **reverse pathways** emulating the
hierarchical and parallel structure of the human visual system; separate pathways decode
semantics, color, depth. [ABSTRACT-ONLY]

**Hi-DREAM** (arXiv 2511.11437, Nov 2025). **This is the closest prior art to PCD.** An ROI
adapter groups fMRI into **early / mid / late streams** — explicitly *"V1/V2 for edges, V3/V4
for color and parts, LOC/FFA for semantics"* — and builds a multi-scale cortical pyramid
aligned to U-Net depth. **Critically, it already ran the control experiment PCD was built to
run:** *"Control experiments with random and reversed ROI-depth assignments degrade
performance, indicating the importance of matching cortical hierarchy to architecture
depth."* [ABSTRACT-ONLY — the parameter-matching of their random control is unverified and
is the one thing worth checking in full text.]

**MindHier / Hierarchy-to-Hierarchy Autoregression** (arXiv 2510.22335). Hierarchical fMRI
Encoder extracting multi-level neural embeddings + **Hierarchy-to-Hierarchy Alignment
enforcing layer-wise correspondence with CLIP features** + scale-aware coarse-to-fine
guidance. Explicitly motivated by the claim that mapping fMRI to a *single* high-level
embedding "collapses hierarchical neural information". [ABSTRACT-ONLY]

**BrainMCLIP** (arXiv 2510.19332). Multi-layer CLIP feature fusion for brain decoding.
**Validates its neuro-inspired ROI partitioning with permutation tests — shuffling
functional labels of visual ROIs 1 000 times** to build a random-grouping null via RSA.
[ABSTRACT-ONLY]

**Also:** DecoFuse (arXiv 2504.00432, decomposes what/where/how for fMRI-to-video, with
ablations tying representations to biological counterparts); STpGCN (HBM 2023, bottom-up
pyramidal pathways mimicking brain information flow, with pathway-removal ablations).

> **Implication — this is the finding that reshapes the program.** PCD's ROI-hierarchy
> grouping (V1/V2 → V3/V4 → category-selective) is **exactly Hi-DREAM's early/mid/late
> grouping**, published November 2025. PCD's `reversed` and `random` ablation arms are
> **exactly Hi-DREAM's control experiments**, already run, with a **positive** result.
> BrainMCLIP additionally ran a 1 000-permutation ROI-label null. The hypothesis H2 in
> `03_HYPOTHESES.md` ("anatomical hierarchy beats degree-matched random grouping") is
> **not novel and has already been answered affirmatively by two independent groups**.

---

## 4. Predictive coding in neural models of vision

Predictive coding is well-formalized and actively modelled, with the canonical direction
unambiguous: *"higher-order brain areas generate predictions sent to lower-order sensory
areas, where top-down predictions are compared to bottom-up sensory data and mismatches
evoke prediction errors encoded in layer 2/3 pyramidal neurons."* [ABSTRACT-ONLY]

**This independently confirms Gate 0 finding T6:** PCD's lower→higher prediction is not
predictive coding, on the field's own definition.

Relevant modelling work surfaced:
- **Predictive Coding Dynamics Enhance Model-Brain Similarity** (ESANN 2025). VGG16 endowed
  with predictive-coding dynamics predicts fMRI across visual ROIs **better over recurrent
  timesteps than the feedforward model**. [ABSTRACT-ONLY]
- **Deep predictive coding networks partly capture neural signatures of short-term temporal
  adaptation in human visual cortex** (bioRxiv 2024). [ABSTRACT-ONLY]
- **High-level visual prediction errors in early visual cortex** (PLOS Biology): prediction
  errors in both low- and high-level visual cortex respond to **high-level, not low-level,
  visual surprise**. [ABSTRACT-ONLY]
- Laminar fMRI localizes prediction error to superficial (L2/3) voxels. [ABSTRACT-ONLY]

> **Implication.** "Predictive-coding dynamics improve fMRI predictivity" is **already
> published**. A PCD-style claim in that space would be entering an occupied, more
> methodologically mature literature — one that uses laminar fMRI and encoding models, not
> retrieval accuracy.

---

## 5. Joint encoding–decoding

**LEA — Joint fMRI Decoding and Encoding with Latent Embedding Alignment** (arXiv
2303.14730). Unified framework doing both directions via aligned latent spaces.
Encoder-decoder models for naturalistic video **use the decoder as a regularizer to improve
encoding**. [ABSTRACT-ONLY]

> **Implication.** "Do encoding and decoding jointly" is established. A successor cannot
> claim it as a contribution *per se*. What is **not** established is using a neural-prediction
> objective specifically as an **identifiability constraint** on interpretable intermediate
> quantities — see `14_NOVELTY_GAP_ANALYSIS.md`.

## 6. Perception → mental imagery: the standout open problem

**NSD-Imagery** (CVPR 2025, arXiv 2506.06898). Benchmark of fMRI paired with **mental**
images, released to complement NSD. Evaluated MindEye1, MindEye2, Brain-Diffuser, iCNN.

**The headline finding, quoted in substance:** *architectural choices significantly impact
cross-decoding performance — models employing **simple linear decoding architectures and
multimodal feature decoding generalize better to mental imagery, while complex architectures
tend to overfit training data recorded exclusively from vision**.* High performance on
seen-image decoding **does not imply** generalization to imagined images. [ABSTRACT-ONLY]

> **Implication.** This is a published, benchmarked, ~14-month-old **negative result about
> complexity**, on a public dataset, with an explicit call that imagery generalization is
> "critical for real-world applications in medical domains and BCI". It is a documented open
> problem where **simplicity and identifiability win** — the opposite of a scale race. It is
> also a direct indictment of PCD's profile (167M params, 82 pp overfit gap).

## 7. Honest limits of this review

1. **Abstract-level only.** No full PDF was opened. Sufficient for novelty triage; **not**
   sufficient for citation. Full-text inspection is required before any paper claim.
2. **One index, one language, one session.** US WebSearch only. No OpenReview/proceedings
   sweep, no citation-graph traversal, no backward/forward snowballing.
3. **No systematic inclusion/exclusion protocol** was preregistered before searching.
4. **Recency asymmetry.** Several 2026 preprints surfaced incidentally (2601, 2604, 2605,
   2606, 2607). The field is moving monthly; this review has a **shelf life of weeks**.
5. **Not searched:** EEG/MEG decoding, non-visual decoding, clinical BCI, RSA methodology
   critiques, variance-partitioning methodology, NSD preprocessing literature.

**Consequence:** no "first" claim is authorized by this document. It is strong enough to
**refute** novelty (which it does, repeatedly, below) and too weak to **establish** it.
