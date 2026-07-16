# 25 — Matched Auxiliary Control Protocol

**Date:** 2026-07-16 · Resolves reviewer objection **O-1**, the objection that decides whether
any positive result survives review.

> **The central scientific problem.** Does the *neural-prediction target* matter, or does
> **any** matched auxiliary objective regularize the decoder? A masked-ROI objective is an
> auxiliary task, and auxiliary tasks regularize. Without this family, a positive ARM-B result
> is uninterpretable.

---

## 1. Matching contract — binding on every arm

| Held identical | Tolerance |
|---|---|
| NCD backbone (d_model, layers, heads, ROI tokenisation) | exact |
| Total trainable parameters | **≤ 2%** spread across arms, **documented per arm** |
| Auxiliary-head capacity | exact (same shapes; a disabled head is still allocated) |
| Optimizer, LR, schedule, weight decay | exact |
| Batch size, grad accumulation, number of updates | exact |
| Masking frequency and mask ratio | exact (arms without masking still draw the mask) |
| Auxiliary loss **scale** | normalised to comparable magnitude at init; recorded |
| Hyperparameter-selection budget | equal trials/arm, logged in the manifest |
| Checkpoint-selection rule | exact (`val_r@1`, declared) |
| Seeds | same set |
| Target embedding dimensionality | 768, exact (confound — `22` §2) |

**Anything not on this list must be identical too; the list is illustrative, not permissive.**

## 2. The arms

| Arm | Auxiliary objective | Isolates |
|---|---|---|
| **A** | none (retrieval only) | the floor |
| **B** ★ | **true masked neural prediction** — predict held-out ROI activity from remaining ROI context | the hypothesis |
| **C** | **stimulus-shuffled neural targets** — real ROI activity, but permuted across *unique images* within the training split, preserving subject and ROI marginals | Is the target's **stimulus-specific** content doing the work, or its **marginal statistics / connectivity structure**? (reviewer O-3) |
| **D** | **random fixed targets** — fixed random vectors, matched dimensionality, variance, head capacity | Does *any* prediction target regularize? |
| **E** | **generic self-reconstruction** — non-anatomical reconstruction, same masking rate, comparable information content | Does **anatomical** target structure matter vs generic reconstruction? |
| **F** | **tuned standard regularization** — retrieval-only + dropout, weight decay, input noise, stochastic depth, tuned under the same budget | **The arm that most likely explains a positive B.** |
| **G** | **random pseudo-ROI structure** — random voxel groups matched to real ROIs in group count, dimensionality, parameter budget, masking probability | Does **anatomical grouping** matter, or just *having* groups? |
| **H** | **true ROI structure, no cross-ROI prediction** — ROI tokenisation, auxiliary task that never predicts one ROI from others (per-ROI self-encoding) | Separates **ROI organisation** from **predictive dependency** — the two things PCD conflated |

**C vs D** separates *stimulus-specific structure* from *any target*.
**G vs B** separates *anatomy* from *grouping*.
**H vs B** separates *tokenisation* from *cross-ROI dependency*.
**F vs B** is the one that decides publication.

## 3. Preregistered survival criteria for ARM-B

The neural-prediction hypothesis survives **only if all six hold**:

1. Improves **≥ 2 independent shift regimes** (`24` §4).
2. Outperforms **C, D, E, F, G, H** under matched compute.
3. Improves **held-out neural predictivity**.
4. **Consistent subject-level direction** (not driven by 1–2 subjects).
5. Does **not** trade a large in-distribution loss for a tiny OOD gain.
6. Survives **correction across the preregistered primary comparisons**.

**Any failure → the hypothesis does not survive.** These are conjunctive by design: five of
six is a negative result, and we report it as one.

## 4. Predefined conclusions

| Outcome | Conclusion — fixed now |
|---|---|
| B ≈ F (best generic regularization) | **"Auxiliary regularization improves robustness, but neural target structure is not specifically supported."** Publishable; likely. |
| B ≈ C | The effect is marginal/connectivity structure, **not** stimulus-specific neural content. |
| B ≈ D | Any target regularizes. The neural target is irrelevant. |
| B ≈ G | Anatomical grouping irrelevant; random groups suffice. |
| B ≈ H | Predictive dependency inert; only tokenisation matters. |
| B > all, neural predictivity ↑, OOD decoding flat | **Route to computational neuroscience**, not ML. |
| B > all on both | Full thesis. |

## 5. Implementation requirements

- **Deterministic target shuffling (C).** Permutation seeded from `(split_hash, seed)`, over
  **unique image IDs**, computed once and stored in the run manifest. All trials of one image
  receive the same permuted target — otherwise the shuffle leaks the true target through
  repetitions.
- **Pseudo-ROI grouping (G).** Random voxel partition matched to real ROI **sizes**, seeded,
  registered as buffers, reproducible from the manifest. **Multiple seeds** — a single fixed
  shuffle is an anecdote (the `random.Random(42)` defect, `03` H2).
- **Head capacity parity.** Every arm allocates the same auxiliary heads even when unused, so
  parameter counts match and are asserted by test.
- **Run manifest carries treatment-arm identity**: `arm`, `aux_objective`, `aux_weight`,
  `shuffle_seed`, `pseudo_roi_seed`, `param_count`, plus source hashes (never `git rev-parse`
  on a dirty tree — F-005).
- **Config-diff validation.** A test asserts arm configs differ **only** in permitted keys.
- **Control-specific gradient tests.** Each arm's auxiliary head must be reachable **iff** its
  objective is enabled — the F-002 contract, per arm.
