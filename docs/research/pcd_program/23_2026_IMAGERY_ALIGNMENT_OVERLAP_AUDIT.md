# 23 — Overlap Audit: Spera et al. 2026, "Seeing the imagined"

**Date:** 2026-07-16 · **Source:** arXiv 2604.15374 (q-bio.NC, submitted 2026-04-15)
**Authors:** Spera, Boccato, Olak, Cammarota, Ciferri, Tronti, Toschi, Ferrante
**Status:** ABSTRACT + arXiv landing page read. Full PDF not opened — sufficient to settle
overlap, insufficient to cite.

---

## 1. What they do

| Component | Their approach |
|---|---|
| Base model | **DynaDiff** — a pretrained *perception* decoder |
| Adaptation | **Latent functional alignment**: maps imagery-evoked activity into the pretrained model's **conditioning space**, all other components **frozen** |
| Supervision | **Matched imagery–perception supervision** |
| Data scarcity fix | **Retrieval-based augmentation** — selects semantically related NSD *perception* trials to supplement limited matched imagery supervision |
| Evaluation | Imagery-NSD benchmark; semantic imagery reconstruction |
| Neuroscience | Above-chance decoding from **multiple cortical regions**; per-region contribution analysis |

## 2. The decisive fact

> **They fit on imagery data.** The method *adapts* a perception decoder using matched
> imagery–perception supervision. Imagery is training signal, not a sealed test.

## 3. What this kills

**"Improving perception→imagery transfer" is no longer an open contribution.** Phase 2's
thesis (`17`, D-006) framed the imagery gap as an unexplained phenomenon we would attack.
Spera et al. attack it directly — with alignment, imagery supervision, and augmentation — and
report improved semantic imagery reconstruction. They also already do per-cortical-region
contribution analysis, which was on our roadmap.

Combined with NSD-Imagery (CVPR 2025) owning the benchmark and the "complex models overfit"
finding (`22`), **the imagery-transfer axis as an end in itself is closed.**

`06_CLAIM_EVIDENCE_REGISTRY.csv`: the claim *"we improve perception→imagery transfer"* moves
to **FORBIDDEN**.

## 4. What this does **not** kill — and why the distinction is real, not rhetorical

Their setting and ours are **different problems**, not different methods for one problem:

| | Spera et al. 2026 | This program (revised) |
|---|---|---|
| Imagery data in model fitting | **Yes** (matched supervision) | **No — sealed** |
| Imagery data in hyperparameter / λ selection | Yes (implied) | **No — sealed** |
| Adaptation to the target state | **Yes**, that is the method | **None. Zero-shot.** |
| Augmentation using target-related trials | **Yes** (retrieval-based) | **No** |
| Evidence base | Imagery only | **Five shifts; imagery is one sealed test** |
| Question | *How do we adapt a decoder to imagery?* | *Does a perception-only neural-prediction constraint produce representations that generalize across shift?* |

An adaptation method and a **zero-shot generalization property** are not competing claims.
Spera et al. cannot answer whether a constraint learned *only from perception* transfers,
because they use imagery to fit. We cannot answer whether alignment helps, because we never
adapt.

**But the honest consequence is a demotion, not a rescue:** imagery can no longer carry the
paper. If imagery were our only evidence, the reviewer question *"why not just align like
Spera et al., which works?"* has no good answer. Imagery survives **only** as one sealed
instance of a broader generalization claim — which is precisely the Phase 2.5 correction.

## 5. Consequences adopted

1. **Thesis reframed** to multi-shift generalization (`24`). Imagery is demoted from *the*
   endpoint to *one* sealed test among five.
2. **NSD-Synthetic becomes the primary OOD evaluation** (`26`): 8 subjects vs imagery's 4,
   284 stimuli vs 18, CC-BY 4.0 vs CC-BY-NC-ND, and purpose-built for OOD. This simultaneously
   repairs the novelty position *and* the n=4 power crisis flagged in `22` §5.
3. **The zero-shot constraint becomes a hard protocol rule, not a preference.** No imagery
   sample may touch training, model selection, λ selection, checkpoint selection, early
   stopping, or representation design. Any leak collapses our setting into theirs and
   forfeits the distinction entirely. This is now a testable invariant, not an intention.

## 6. Outstanding

- **Full PDF read required** before citing: confirm they report no zero-shot (imagery-naive)
  arm. If they *do* include one, our distinction narrows sharply and `17` reopens.
- Check whether DynaDiff itself (their base) uses any auxiliary neural-prediction objective.
- Their per-region contribution analysis may pre-empt parts of our planned per-ROI analysis;
  read before designing ours.
