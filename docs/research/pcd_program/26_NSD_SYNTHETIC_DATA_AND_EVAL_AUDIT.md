# 26 — NSD-Synthetic: Data and Evaluation Audit

**Date:** 2026-07-16 · **Source:** Gifford et al., *A 7T fMRI dataset of synthetic images for
out-of-distribution modeling of vision*, **Nature Communications 2026** (arXiv 2503.06286);
read via open-access PMC12901131. Code: `github.com/gifale95/NSD-synthetic`.
**Status:** full-text-level read of methods/results sections. **Data not yet obtained.**

---

## 1. Why this becomes the *primary* OOD test

| | NSD-Imagery (`22`) | **NSD-Synthetic** |
|---|---|---|
| Subjects | **4** (subj01/02/05/07) | **8 (subj01–subj08)** |
| Stimuli | **18** | **284** |
| Licence | CC-BY-**NC-ND** 4.0 | **CC-BY 4.0** |
| Purpose | imagery benchmark | **purpose-built for OOD generalization testing** |
| ROIs | NSD-standard | NSD-standard (V1, V2, V3, hV4, PPA, VWFA, FFA, EBA) |

It fixes the n = 4 power crisis (`22` §5) and the 18-stimulus endpoint problem in one move,
and its licence is permissive. **This inverts Phase 2's priority: NSD-Synthetic is primary,
imagery is a sealed secondary test.**

## 2. Composition — 284 images, 8 classes, 71 subclasses

| Class | n | CLIP-semantic content? |
|---|---|---|
| Spiral gratings | **112** | **none** |
| Chromatic noise | **64** | **none** |
| Single words (4/6-letter × 5 positions) | **40** | orthographic only |
| Noise (white, large-block, pink, achromatic) | **16** | **none** |
| Contrast modulation (scenes × 5 levels) | **16** | **graded, scene-derived** |
| Phase-coherence modulation (scenes × 4 levels) | **16** | **graded, scene-derived** |
| Manipulated scenes (inverted, Mooney, line-drawing) | **12** | **scene-derived** |
| Natural scenes (grayscale) | **8** | **yes** |

Acquisition: one session post-NSD-core; 8 runs (4 fixation, 4 one-back); 744 stimulus trials;
236 images ×1 per task, 32 images ×2 per task; 16 scene images ×4 per task.

## 3. Their key finding — it is the argument for this whole thesis

> OOD tests **revealed differences between models not detected by ID tests**. AlexNet vs.
> vit_b_32: **ID difference peaked at 10%** absolute explained variance; **OOD difference
> peaked at 25%**. ID-vs-OOD performance drop: **P = 10⁻⁷** (paired t-test). The **degree of
> OOD is predictive of the magnitude of model failure**.

This is independent, published evidence that **in-distribution benchmarks cannot discriminate
models that OOD benchmarks can** — exactly the premise of `24`. It also implies a graded
design: OOD degree is a continuous predictor, not a binary.

## 4. The problem nobody should skip: it was built for *encoding*, not CLIP decoding

Their metrics are **noise-ceiling-normalised explained variance**, NCSNR, RSA, and
**zero-shot image identification**. The first three are encoding metrics (image → brain). We
decode (brain → CLIP).

> **232 of 284 stimuli (82%) have essentially no CLIP-semantic content.** Spiral gratings,
> chromatic noise, pink noise, and isolated words are not scenes. A CLIP-retrieval decoder
> asked to rank pink noise against white noise is near-chance **by construction**, not
> because it fails to generalise. **That is a measurement artefact, not a finding, and
> reporting it as OOD failure would be misleading.**

**Only 52 stimuli (18%) are scene-derived:** 8 natural + 12 manipulated + 16 contrast + 16
phase-coherence.

## 5. The design this actually affords — better than a binary OOD test

The 32 contrast- and phase-modulated images are **the same natural scenes at graded
degradation levels** (5 contrast levels, 4 phase levels). That is a **parametric OOD axis on
semantically meaningful stimuli**:

> **Primary NSD-Synthetic endpoint:** decoding performance as a **function of OOD degree**
> along the contrast and phase-coherence continua, on scene-derived stimuli.
> **Prediction under the thesis:** ARM-B's advantage over ARM-A/F **grows with OOD degree**.
> A constant offset across levels is a capacity effect, not a generalization effect.

This is far stronger than "does it work OOD?" — it is a **dose-response test**, and it maps
directly onto their own "degree of OOD predicts failure magnitude" finding.

**Two endpoints, declared now:**
1. **Confirmatory:** graded degradation on scene-derived stimuli (52 images; families =
   contrast, phase, manipulated, natural).
2. **Exploratory, reported separately:** the 232 non-semantic stimuli, where **CLIP-decoding
   near-chance is the expected null and proves nothing either way**. Also available: neural
   *predictivity* on all 284 (ARM-B predicts held-out ROI activity — an encoding-side metric
   valid on all stimuli, and directly comparable to their noise-ceiling-normalised EV).

> The neural-predictivity endpoint is the one place where all 284 stimuli are usable, and it
> is the natural home for our masked-ROI head. **This may be the strongest evaluation in the
> entire program** and it did not exist before this audit.

## 6. Stimulus families and correction groups (preregistered)

**Confirmatory family set (scene-derived, 52):** `natural` (8), `manipulated` (12),
`contrast` (16), `phase` (16). **FDR (BH) across the 4 families, q = 0.05.**
Per `24`, an effect driven by a single family is **not** generalization — cross-family
consistency is required.

**Exploratory set (232):** `spiral`, `chromatic_noise`, `words`, `noise`. Reported, never
used for confirmation, never pooled with the confirmatory set.

## 7. Access and risk

- Public at `naturalscenesdataset.org`, **CC-BY 4.0**. One extra session/subject — small
  relative to NSD-core, which is already on the pod (181 TB free).
- **Not yet obtained.** Status: **DATA_ACCESS_UNVERIFIED**.
- **R-18:** the 52-stimulus confirmatory set is small. With 8 subjects × 52 stimuli and a
  graded design, power comes from the **slope across OOD levels**, not from n per cell. The
  SAP must model it as a slope, not 4 independent cell comparisons.
- **R-19:** NSD-Synthetic is **one session** — lower SNR than NSD-core's 40. Use their NCSNR
  and noise ceilings; do not compare raw accuracies across datasets.
- **R-20:** stimuli are grayscale/synthetic; our CLIP cache is built for NSD's colour natural
  scenes. **The target-embedding pipeline must be verified to handle them** before any claim.
