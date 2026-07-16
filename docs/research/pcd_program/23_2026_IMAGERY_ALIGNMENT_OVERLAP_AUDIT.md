# 23 — Overlap Audit: Spera et al. 2026, "Seeing the imagined"

**Date:** 2026-07-16 · **Source:** arXiv 2604.15374 (q-bio.NC, submitted 2026-04-15)
**Authors:** Spera, Boccato, Olak, Cammarota, Ciferri, Tronti, Toschi, Ferrante
**Status:** **FULL TEXT READ 2026-07-16 (Phase 2.6).** Supersedes the abstract-level audit below.

---

## 0. THE DECISIVE FINDING — the zero-shot arm exists, and it is AT CHANCE

**They report a frozen zero-shot DynaDiff baseline** — a perception-trained decoder applied
to imagery with **no imagery fitting**. This is exactly our setting. Table 2:

| Metric | Frozen zero-shot DynaDiff | Chance |
|---|---|---|
| **CLIP** | **48.94%** | 50% |
| **Alex(5)** | **50.21%** | 50% |
| **Alex(2)** | **51.03%** | 50% |
| PixCorr | 0.0295 | — |
| SSIM | 0.3431 | — |

> **A strong perception decoder, applied zero-shot to imagery, performs at chance.**

**This closes shift 3 (zero-shot imagery) as an endpoint — not on novelty grounds, on
empirical ones.** DynaDiff is a far stronger perception decoder than NCD (our model is at
16.4% R@1 in-distribution, T11). If DynaDiff is at chance zero-shot, NCD will be at chance
zero-shot. **You cannot detect "ARM-B > ARM-F" at a floor.** Both arms would sit on 50% and
the comparison would carry no information.

This is the **perception OOD kill test** predicted in `22` §7 — and Spera et al. have already
run it, with a better model, and published the answer. **It costs us nothing to accept it.**

**Consequences, adopted immediately:**
1. **Shift 3 is removed from the thesis as a positive endpoint** (`24` revision). It was
   already excluded from the primary statistical family (`20` §4) for a *different* reason
   (n=4, p-floor 1/16). It is now excluded on *empirical* grounds too: there is no signal
   to improve.
2. Shift 3 may still be **reported as a null**, corroborated by Spera's baseline — *"as
   Spera et al. show and we confirm, perception-only decoders do not transfer to imagery
   zero-shot; the state shift is too large."* That is honest and cheap, and it is **not a
   contribution**.
3. **Imagery adaptation works only with imagery data.** Their aligned model beats the frozen
   baseline substantially. That is their contribution and it is well earned.

**This is a genuine blow.** The multi-shift thesis loses its most distinctive shift — the one
that separated it from generic "auxiliary objectives improve robustness". See §7 for the
honest impact assessment.

---

## Superseded abstract-level audit (retained for provenance)

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

## 6. Full-text record (Phase 2.6)

| Item | Finding |
|---|---|
| **Frozen zero-shot baseline** | **YES — at chance** (CLIP 48.94%, Alex(5) 50.21%, Alex(2) 51.03%). See §0. |
| **Imagery data used for fitting** | ~24 imagery runs/subject (3 attention runs excluded); ~50% of vision trials discarded for cue–image mismatch → ~24 usable trials/run. 80% of augmented data for training. |
| **Latent-alignment objective** | **MSE** between the brain module's imagery output and the corresponding **visual target**: maps imagery fMRI → CLIP-Image embeddings taken from vision trials. Module is an MLP (one hidden layer, LayerNorm, GELU). All other DynaDiff components **frozen**. |
| **Matched vision trials** | Only trials where the displayed image matched the cue letter (~50%) — these supply the direct imagery↔vision supervision. |
| **Retrieval-based NSD augmentation** | Complex stimuli: CLIP-Image embeddings + k-NN on cosine distance → **180 nearest NSD trials per stimulus**. Conceptual stimuli: CLIP-**Text** embeddings of the target word → 180 nearest NSD images. Original imagery trials replicated with Gaussian noise (σ²=0.002) to match augmented counts. |
| **Train/val/test** | 4 subjects (**1, 2, 5, 7**). Test: 20 held-out trials/subject (10 simple, 10 complex). Val: 20% of augmented training data. Batch 36; max 8 epochs (complex/conceptual), max 3 (simple, non-visual ROIs). |
| **Cortical regions** | Primary **nsdgeneral**; secondary from HCP_MM1: prefrontal, frontal, temporal, parietal. Per-subject voxel counts in their Table 1. |
| **Inferential unit** | **Per-subject** scores; **Wilcoxon signed-rank** paired within subjects; metrics averaged over 4 subjects with SEM. |
| **Reconstruction seeds** | **10 reconstructions per test image** (different VD seeds). **SEM computed across 10 seeds × 4 subjects = 40 "reconstructions" per condition.** |
| **Auxiliary neural-prediction / masked-ROI / encoding objective** | **NONE.** Only the alignment MSE. The frozen VD diffusion loss is not used during alignment. |

### A methodological note we may make, carefully

Their SEM is computed across **10 reconstruction seeds × 4 subjects = 40 units**. Seeds are
not independent stimulus evidence — they are repeated draws from one model on one stimulus.
Pooling them into the SEM **understates uncertainty**. Their *significance* tests are
subject-level Wilcoxon (n=4), which is correct; it is the **SEM error bars** that are
inflated-precision.

Our own SAP already forbids this (`20` §1: "reconstruction seeds are not independent stimulus
evidence"). **This is a legitimate observation, not a rebuttal** — it does not touch their
central claim, which rests on the Wilcoxon tests. Do not overstate it, and do not lead with it.

## 7. Impact on the NCD thesis — honest assessment

**What survives:** they use **no auxiliary neural-prediction objective** (confirmed at full
text). Our mechanism remains untested by them. Shifts **1, 2, 4, 5, 6** are untouched by this
paper.

**What dies:** shift 3. Zero-shot imagery is at chance for a stronger model, so there is no
headroom for ARM-B to beat ARM-F there. The multi-shift thesis loses the shift that most
distinguished it from the generic claim *"auxiliary objectives improve robustness"*.

**The uncomfortable question this forces** — and it must be answered before the pilot, not
after: with imagery gone, the thesis is *"a masked-ROI auxiliary objective improves robustness
across stimulus/data/noise/subject shift"*. That is a **robustness-regularization claim**, and
ARM-F is precisely the arm that tests whether the *neural* target is doing any work. **The
entire scientific content now rests on B vs C and B vs F.** If those come back null — which
`24` §6 already names as the most likely outcome — the honest report is
`GENERIC_REGULARIZATION_SUPPORTED`, and the paper is a robustness-regularization paper with a
careful negative on neural specificity.

That is a smaller paper than the one this program set out to write. It is also, on current
evidence, the one the evidence supports.

## 8. Forbidden claims (registry-enforced)

- ❌ *"first zero-shot perception-to-imagery decoder"* — **they published the zero-shot arm**
- ❌ *"first application of a perception decoder to Imagery-NSD"*
- ❌ *"first method improving perception-to-imagery transfer"*
- ❌ *any* "first" — the review remains partial (`13` §7)

**Candidate surviving claim, and it is not yet supported:**
> A neural-prediction constraint learned exclusively from perception data is **evaluated** as
> a source of zero-shot robustness across stimulus, noise, data, and subject shifts.

Note "**evaluated as**", not "**is**". Nothing has been run.
