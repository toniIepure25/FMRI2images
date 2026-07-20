# 01 — Roy et al. (2025) Full-Text Audit, and Correction of My Prior Errors

**Date:** 2026-07-17 · **Source:** PMC12424947 (open full text) · PMID 40950062
**Roy, T.S., Breedlove, J., St-Yves, G., Kay, K., Naselaris, T.** *A transformation from vision
to imagery in the human brain.* bioRxiv 2025.09.02.672180 — **preprint, not peer-reviewed.**

---

## 0. THREE ERRORS IN MY PREVIOUS GATE M0 REPORT — corrected

I reported these from the abstract alone. The full text shows all three were wrong, and each
error pushed in the same direction: **making the pre-emption look more total, and the
transformation look more hostile to MINDIR, than it is.**

### ERROR 1 — I misdescribed the "25–50% variance" quantity

**I said:** *"reconstructions of visual activity in terms of imagery dimensions explain only
25–50% of the variance"* — implying 50–75% of neural variance is destroyed.

**Actual:** it is an **alignment ratio** (their Eq. 10), not raw variance. *"In V1, the imagery
subspace explains 25%–30% of the variance in visual activity **that is explained by the visual
subspace**. This percentage increases to 50% with progression to V4."* It is a **ratio of
variance-explained to variance-explained**, measuring **subspace misalignment**: 0 = orthogonal,
1 = fully aligned. Not an information-loss figure.

### ERROR 2 — I claimed the map is "strongly non-invertible, contradicted by measurement"

**Wrong twice over.** (a) They **never ran the inverse** (img→vis) — so nothing about
invertibility is measured, let alone contradicted. (b) The contraction is **early-visual only**.
In **ventral, lateral and parietal ROIs the alignment ratio is near 100% — "imagery and visual
subspaces occupy identical subspaces."**

**I stated as measured fact something they did not test, and generalised an early-visual
finding to the whole hierarchy.** That is the exact error class this program exists to catch.

### ERROR 3 — I said dimensionality "halves", globally

**Actual:** halving is **early visual only** (`d_im ≈ 2–3` vs `d_vis ≈ 4–6` in V1/V2). In
higher areas dimensionality is **near parity**. Their striking result is that **imagery
dimensionality is nearly CONSTANT across ROIs** while visual dimensionality *rises* through
the hierarchy — so the *gap* closes because vision expands, not because imagery contracts.

> **Consequence: my "the evidence points away from an algebra" conclusion was overstated.**
> The evidence points away from a *uniform* transformation. It is consistent with a
> **structured, ROI-dependent contraction** — which is precisely MINDIR's premise. Their
> result **supports** the MINDIR framing more than it refutes it. It also pre-empts part of it.

## 1. What they did — verified

| | Dataset 1 (NSD-Imagery) | Dataset 2 (Spatial Imagery) |
|---|---|---|
| Subjects | **8** (2M/6F, 19–32) | **3** (26–40) |
| Stimuli | **12** — 6 simple (4 bars 0/45/90/135°, 2 crosses) + 6 naturalistic (5 scenes, 1 artwork) | **64 objects × 8 locations = 512 unique conditions/subject** |
| Repeats | vision 8/stim; imagery 16 across 2 runs (imgA-1/imgB-1 dropped, low SNR) | each condition ×2 per run; 13–16 runs/state/subject |
| Pairing | same stimulus seen then imagined; **separate runs** | objects cued to bracket locations |

**Note:** NSD-Imagery here is **8 subjects**, not the 4 I carried forward from the CVPR
benchmark paper (which used 4 who completed full NSD training). Corrected.

**Model** — linear voxel-to-voxel **reduced-rank** regression, `R_im = R_vis · W_im + ε`, as
two layers: `W1` (n → d latent), `W2` (d → n, orthogonal rows defining the imagery subspace).
**Rank `d` is a hyperparameter chosen by 4-fold cross-validated line search** maximising
Pearson r on validation.

**vis2vis** — maps a visual trial to *a different visual trial of the same stimulus*. It is a
**noise-ceiling / denoising control**, and is the right one.
**vis2img** — maps a visual trial to an imagery trial of the same stimulus.

**Performance:** median Pearson r ≈ **0.3–0.4 early**, **0.4–0.5 higher visual**. Notably *"at
several locations in higher visual ROIs, the vis2img model predicted imagery activity **more
accurately than the vis2vis model**."* Prediction is **good**, not marginal.

**Vividness (Dataset 1 only):** VVIQ vs **V1 imagery-subspace dimensionality: r = 0.71,
p < 0.05, bootstrap 95% CI [0.15, 0.98]**. Null elsewhere (no alignment-ratio correlation in
any ROI; no dimensionality correlation outside V1).

## 2. What they did NOT do — verified absences

| Test | Status | Consequence for MINDIR |
|---|---|---|
| **Semantic / stimulus-feature control** | **NOT PERFORMED.** *"All analysis is purely voxel-activity-to-voxel-activity."* | **The single largest gap.** Nobody has shown the "imagery transformation" is *neural* rather than explainable by stimulus features. |
| **Cross-subject transfer** | **NOT PERFORMED.** Entirely within-subject; subjects averaged, no LOSO | H1/H2 (universal spectrum, shared subspaces) untested |
| **Inverse (img→vis)** | **NOT PERFORMED** | Invertibility genuinely open |
| **Composition (3rd state)** | **NOT PERFORMED.** No WM, no delay, no fading condition | H4/H5 untouched |
| **Prospective** | Retrospective throughout | Flagship prospective test untouched |

**Author-stated limitations:** results give **relative**, not total, dimensionality (capped by
12 conditions in Dataset 1); low trial counts forced them to feed **denoised** vis2vis outputs
into vis2img. Future work framed around **generative feedback** from higher to lower areas.

## 3. Corrected pre-emption ledger

**Genuinely pre-empted — remove from MINDIR:**
- The concept of a vision→imagery transformation as a formal object
- Within-subject linear prediction of imagery activity from perception (r ≈ 0.3–0.5)
- ROI-specific transformation analysis
- Dimensional reduction + subspace reorientation in early visual cortex
- Refutation of "imagery = weak vision"
- **Partially: H6** — vividness ↔ V1 imagery dimensionality already reported

**NOT pre-empted:**
- **Semantic-shortcut validity of the transformation itself** (H7) — untested by anyone
- **Cross-subject universality of the contraction spectrum** (H1/H2)
- **Invertibility / retained-vs-removed subspace structure** (img→vis)
- **≥3-state transitions, path dependence, lossy composition** (H4/H5)
- **Prospective prediction of future internal-state degradation**

## 4. Adjacent pre-emption found this session

- **"Universal scale-free representations in human visual cortex"** (arXiv 2409.06843) —
  visual cortex latent dimensions show a **universal scale-free spectrum**; detecting shared
  dimensions at higher ranks **requires subject-specific functional alignment** (anatomical
  alignment finds only ~10 shared dimensions). **Partially pre-empts H1 for perception**, and
  is a **methodological warning**: any cross-subject spectrum claim must use functional
  alignment or it will find nothing for trivial reasons.
- **Episodic-memory dimensionality transformation** (ScienceDirect S136466132500021X) and
  **"A compressed code for memory discrimination"** (arXiv 2510.10791; PMC12632993) —
  lossy compression perception→memory, dimensionality reduction through V4/IT into
  hippocampal DG/CA3/CA1, *"the original event can never be fully reconstructed."*
  **The general principle of directional information contraction is established.**
  MINDIR's H3 is therefore **not a new idea**; only its *quantitative, operator-level,
  cross-subject* form would be.
- **Inter-individual / inter-site neural code conversion without shared stimuli**
  (arXiv 2403.11517) — relevant to cross-subject transport machinery.

## 5. The honest position

**MINDIR's core phenomenon — directional, structured information contraction between mental
states — is real, published, and converging from two independent literatures** (Roy et al. for
perception→imagery; the memory-compression literature for perception→memory).

**What nobody has done is establish that these transformations are (a) neural rather than
semantic, and (b) shared across people.** Those are validity and universality questions, they
are cheap, and (a) is answerable on public data **today**.

**But neither is a flagship.** They are a strong *validity study* of someone else's finding.
The flagship — multi-state contraction laws, path dependence, prospective degradation
prediction — requires data that does not exist.
