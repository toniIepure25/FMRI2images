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

## 5b. DUAL-ENDPOINT PROTOCOL (Phase 2.6) — the 232 are not waste

An earlier draft treated the 232 non-semantic stimuli as merely exploratory. **That
undersold them.** They are not a nuisance subset — they are a **parametrically controlled
low-level probe**, and they are the sharpest available test of whether ARM-B's
neural-prediction constraint learned anything real.

The logic: CLIP is useless on gratings and noise, so *semantic retrieval* there is
meaningless. But **neural predictivity is meaningful on every stimulus**, and ARM-B's entire
mechanism *is* neural prediction. The 232 vary orientation, spatial frequency, contrast,
colour and texture **systematically and by design** — exactly the axes a constraint that
"models cortical structure" should capture. If ARM-B's constraint is real, it should show up
here most cleanly; if it shows up only on the 52 semantic stimuli, that points to a semantic
effect, not a neural-structure one.

### Endpoint 1 — Semantic (52 scene-derived stimuli)

`natural` (8) · `manipulated` (12) · `contrast` (16) · `phase` (16)

- retrieval (R@1, both gallery definitions), **rank**, CLIP similarity
- **contrast and phase-coherence dose–response** — the parametric OOD axis (§5)
- **Confirmatory.** FDR (BH) across the 4 families, q = 0.05. Cross-family consistency
  required (§6).

### Endpoint 2 — Non-semantic (232 controlled stimuli)

`spiral` (112) · `chromatic_noise` (64) · `words` (40) · `noise` (16)

- **held-out neural predictivity**, noise-ceiling-normalised (comparable to Gifford et al.)
- **ROI-level explained variance**
- **prediction-residual calibration**
- **representational geometry** (cross-validated RDMs with noise ceilings)
- **low-level model targets**: orientation, spatial frequency, colour, texture

**Confirmatory for the *mechanism* question** (does the neural constraint predict held-out
activity out of distribution?), **never for the decoding question**.

### The rule that keeps this honest

> **CLIP-semantic retrieval failure on non-semantic stimuli is NOT evidence of
> neural-decoder failure** (C-019). It is a measurement artefact of using a semantic target
> on a non-semantic stimulus. The two endpoints answer different questions and are **never
> pooled, never traded off, and never presented as one number.**

### CLIP-cache audit (O-9 / R-20) — required, and now correctly scoped

The cache must still be audited on grayscale/Mooney/line-drawing stimuli, because **Endpoint 1
depends on it**. But a *failure* of that audit does **not** sink Endpoint 2, which needs no
CLIP embeddings at all. **O-9 therefore blocks Endpoint 1 only** — a narrower and more
accurate scoping than the earlier "upstream of every synthetic number".

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
