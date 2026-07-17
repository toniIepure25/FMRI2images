# 02 — Frontier Literature Review (Gate M0)

**Date:** 2026-07-17 · **Tool:** WebSearch + WebFetch · **Coverage:** through July 2026
**Status:** **PARTIAL.** Four of the five mandated families searched; the decisive result was
found and pursued. Scope limits in §5 — they are real, and this review does not license a
"novel" claim on its own.

---

## 1. THE PRE-EMPTION — read this first

> **Roy, T.S., Breedlove, J., St-Yves, G., Kay, K., & Naselaris, T. (2025).**
> *A transformation from vision to imagery in the human brain.*
> bioRxiv 2025.09.02.672180 · PMID 40950062 · preprint, not yet peer-reviewed.

**This is Naselaris's lab, with Kendrick Kay — an NSD author. It is the strongest possible
source in this exact space, and it published MINDCOMPILER's core conceptual move.**

Verbatim from the abstract:

> *"we introduce the concept of an **imagery transformation** — a mapping from visual to
> imagery activity patterns evoked by **the same stimulus**. […] Using **two 7T fMRI
> datasets**, we estimated imagery transformations across different visual areas and found
> they **accurately predict imagery activity**."*

| MINDCOMPILER element | Roy et al. 2025 |
|---|---|
| `T_s` / `T_{a→b}` as a *mapping between mental states* | **Introduced as "the imagery transformation"** |
| Same content across states | **Yes — "evoked by the same stimulus"** |
| Predict **measured** target-state activity | **Yes — a "vis2img" model** |
| Held-out content | **Yes — held-out test stimuli, accuracy = Pearson r** |
| Region-dependent transformation (our **E8**) | **Yes — "across different visual areas"** |
| 7T fMRI, NSD-Imagery + a spatial-imagery dataset | **Yes — both** |

**Our E2 (direct perception→imagery neural prediction) and E8 (region-dependent
transformation) are done. Candidate C (neural counterfactual world model) is largely
pre-empted in its retrospective form.**

## 1b. Worse than a novelty problem — their result argues *against* the algebra

This is not merely "someone got there first". **It is evidence against H3/H5 and against the
algebra thesis:**

> *"in early visual cortex, they **halve the number of active dimensions and reorient them**,
> such that reconstructions of visual activity in terms of imagery dimensions explain only
> **25–50% of the variance**. […] imagery activity patterns […] **encode fewer features and
> occupy a distinct subspace**."*

Consequences to accept, not argue around:

1. **The transformation is strongly non-invertible.** Half the dimensions are gone. An
   approximate inverse `T_{b→a} ∘ T_{a→b} ≈ I` is **contradicted by measurement**, not merely
   untested. Our §8 inverse test has a published answer, and it is *no*.
2. **Composition would compound the loss.** If perception→imagery destroys 50–75% of
   variance, a path routed through imagery is degraded by construction. A composed path
   beating a direct path is *a priori* implausible on this substrate.
3. **The shared-content-manifold hypotheses (H1–H3) are weakened.** Imagery occupies a
   **distinct subspace**, not a rescaled or offset version of the visual one. H1 (additive
   offset) is close to dead on their evidence.

**Taken with Spera et al. 2026** (frozen zero-shot perception→imagery decoder **at chance**:
CLIP 48.94% vs 50%), the picture is coherent and unfavourable: perception→imagery is a
**lossy, dimension-halving, subspace-shifting map** — learnable *within subject, with paired
data* (Roy), carrying **no zero-shot transfer** (Spera).

## 2. Cross-subject alignment — `S_p` is a mature, occupied field

Hyperalignment projects response/connectivity pattern vectors into a **common
high-dimensional information space** rather than aligning anatomical topographies. Shared
Response Models use **response-tuning basis functions common across brains** plus
**individual-specific topographic basis functions** — *precisely* the
`shared + subject-residual` factorisation of our H5 / §9.3.

**Cross-subject neural prediction is already their standard validation**: *"using one
subject's latent representation with another subject's decoder to reconstruct neural
responses […] directly tests whether the common space captures genuinely shared neural
structure"*. Also surfaced: Hybrid Hyperalignment (bioRxiv 2020.11.25.398883); geometry-aware
whole-brain functional alignment (arXiv 2607.10931); task-guided cross-subject latent
alignment VAE (arXiv 2606.15989); VoxelFormer (arXiv 2509.09015).

> **`S_p` is not a contribution.** Subject transport is solved-enough machinery to *use*.

## 3. Neural / Koopman operators — `T_s` machinery is occupied

Neural Koopman models learn a latent space with approximately linear dynamics, applied to
brain dynamics (MICCAI 2024 structure-function coupling; PMC12628071 with a control
mechanism; NeuroBRIDGE behaviour-conditioned Koopman; scaling laws arXiv 2602.19943).
**Compositional Koopman operators for model-based control already exist.**

> Caveat we must respect (mission §3: *"do not call a method an operator merely because it
> maps one latent vector to another"*): the Koopman literature concerns **temporal** dynamics
> of one system; our `T_s` is a **between-state** map. That distinction is real — but it must
> be *earned*, and the machinery is not ours.

## 4. Counterfactual neural prediction — term occupied, referent differs

Surfaced work (GCAN arXiv 2403.01758; CounterSynth) is **counterfactual augmentation /
explanation on clinical images and connectomes** — label-driven diffeomorphic deformation,
attention-guided region attribution. **None predicts an unobserved *cognitive-state* neural
response for specific content.** This is the one family where terminology collides but the
referent genuinely does not.

## 5. Honest scope limits

1. **Roy et al. read at abstract + PubMed-extraction depth only.** bioRxiv full text returned
   **HTTP 403**. **Their composition / inversion / cross-subject / semantic-control status is
   UNCONFIRMED** — extraction reported "not mentioned", which is **not** "not done".
   **Reading this paper in full is the single highest-priority action in the program.**
2. **Not searched:** mental-state families (memory reinstatement, vividness, aphantasia,
   dreams, working memory); causal representation learning / identifiability; multimodal
   fusion; intracranial; real-time fMRI / closed-loop.
3. One index, one session; no citation-graph traversal, no OpenReview sweep.
4. **No "first" claim is authorized by this document.** It is strong enough to **refute**
   novelty — which it does — and far too weak to establish it.

## 6. What survives

Not pre-empted by anything found:

- **Composition** across ≥3 states (`T_{a→c}` vs `T_{b→c}∘T_{a→b}`) — **but §1b argues it is
  unlikely on a lossy map, and no public dataset has ≥3 states on the same content**.
- **Cross-subject transport of the state operator itself** (not of content). Roy et al. are
  within-subject; hyperalignment transports *content*, not *transformations*. This is the
  cleanest surviving idea found.
- **Probabilistic / calibrated** target-state prediction — Roy's accuracy is Pearson r.
- **Non-commutativity, interpolation, extrapolation** — need states public data lacks.

**Every survivor requires data that does not exist publicly.** That is the core of the Gate M0
verdict, and it is a data fact, not a taste judgment.
