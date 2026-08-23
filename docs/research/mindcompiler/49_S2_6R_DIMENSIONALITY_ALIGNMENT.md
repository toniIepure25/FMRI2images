# S2.6R — Roy-defined dimensionality & visual↔imagery subspace alignment

**Gate:** S2.6R · **Branch:** `research/mindcompiler-neural-state-operators` · **Input HEAD:** `4aa5a8c`
**Class:** `PUBLIC_RESULT_RECONSTRUCTION_AND_CLAIM_CONCORDANCE` (NOT prospective confirmation — the
S2.5M test curves/data are already observed).

Reconstructs the two central Roy scientific quantities that S2.5M deferred, on the exact
public-method-aligned **B0** (b2-compatible) pipeline frozen in S2.5M — no re-optimization of
prediction, no new beta, no proxy geometry, **no final Roy reproduction verdict** (that is S2.7R).

Primary source re-opened this session (PMC open access, DOI `10.1101/2025.09.02.672180`):
<https://pmc.ncbi.nlm.nih.gov/articles/PMC12424947/>. Short quotes only; no copyrighted text committed.

---

## PART A — source reverification (verbatim short quotes)

### Dimensionality (Figure 4)
- Operational model rank: *"The value of dim is determined by 4-fold cross-validated line search."*
- Reported dimensionality estimate: *"The estimate of the optimal number of visual subspace dimensions
  (dvis) for each subject is the average of the values at 99% of the peak (dots) of each curve."*
- Hidden-layer interpretation: *"The number of hidden nodes in the hidden layer, dim and dvis
  respectively, may be less than or equal to the number of voxels."*

### Dimensionality result
- *"In early visual areas the number of imagery subspace dimensions was on average slightly less than
  half the number of visual subspace dimensions."*
- *"In higher visual areas the number of dimensions of the imagery and visual subspaces was close to
  parity, a finding consistent with pure reactivation."*

### Alignment definition (Figure 5)
- *"The alignment ratio is the total variance (across all stimuli) in visual activity explained by the
  dim-dimensional imagery subspace divided [by] the total variance in visual activity explained by the
  first dim dimensions of visual subspace."*

### Alignment result
- *"In V1, the imagery subspace explains 25%-30% of the variance in visual activity that is explained by
  the visual subspace. This percentage increases to 50% with progression to V4, and converges on 100%
  in the parietal ROI."*

### Alignment null (Figure 5 colored band)
- *"The null distribution for every subject (colored band) is constructed by replacing the dim imagery
  dimensions in the alignment ratio calculation with a random set of dim singular vectors of the visual
  activity space 100 times."*

---

## PART B — rank semantics (the critical distinction)

Two DIFFERENT quantities, never overwritten one with the other:

| symbol | source | our object |
|---|---|---|
| `r_model_fold` (Roy "dim by line search") | **validation** argmax mean r | S2.5M operational rank — **unchanged** |
| `d_report_fold` (Roy Figure-4 dot)        | **test** accuracy vs retained rank, 99%-of-peak | **new** in S2.6R |
| subject dimensionality                    | average of the four fold-level `d_report` | reported here |

`r_model` stays frozen from S2.5M and is NOT touched. `d_report` is derived only from the TEST rank
curve. `OPERATIONAL_MODEL_RANK != REPORTED_DIMENSIONALITY_ESTIMATE`.

### Frozen d_report rule (declared BEFORE computing tables)
```
D99_RULE = FIRST_INTEGER_RANK_REACHING_99_PERCENT_OF_TEST_PEAK
peak = max(finite test_score[r]);  thr = 0.99 * peak
d_report_fold = smallest integer r (1..rank_max) with test_score[r] >= thr
```
- Ranks `1 … min(12, supported rank)`. No splines, no inter-integer interpolation.
- `D99_EXACT_TIE_INTERPOLATION_NOT_PUBLICLY_SPECIFIED` (first-crossing is the parsimonious reading).
- **Non-positive / no-finite peak** → `NOT_EVALUABLE_NONPOSITIVE_TEST_PEAK` (argmax kept descriptively);
  a dimensionality is never forced from an uninformative curve.
- Fold support per participant×ROI: 4/4 → primary; 3/4 → `PARTIAL_FOLD_SUPPORT`; ≤2/4 →
  `DIMENSIONALITY_NOT_RELIABLY_EVALUABLE`. No post-hoc rescue.

---

## PART F/G/H — bases, alignment space, alignment ratio

- **Bases are model OUTPUT-space singular vectors** (no proxy — no PCA of test/raw data, no principal
  angles / CCA / Procrustes / RSA). `V = right singular vectors of the fitted TRAIN prediction
  X_train @ W_Λ`; the first r columns are exactly the reduced-rank projector basis
  `W_RRR = W_Λ V_r V_rᵀ`. `V_vis` from the vis2vis solution, `V_img` from the vis2img solution.
  Sign-invariance handled by hashing/using the projector `P = V_d V_dᵀ`.
- **Alignment input** `X_test_vis` = held-out VISUAL source trials of the fold, standardized by the
  train-fitted **visual** z-score. `ALIGNMENT_SPACE = S2_5M_TRAIN_STANDARDIZED_MODEL_SPACE` (the paper
  states centered/scaled matrices but the author scaler scope is not exposed).
- **Ratio** uses `d = d_img_fold` in BOTH numerator and denominator:
  ```
  a_g_fold = TV( X_test_vis · P_img_d ) / TV( X_test_vis · P_vis_d )
  ```
  `ddof=1`; never clipped (finite noisy data may give `a_g > 1`); non-evaluable if the visual
  denominator ≤ ε. Sanity: identical subspaces → 1, orthogonal (with `X` in the visual subspace) → 0.

## PART I — alignment null
`N_ALIGNMENT_NULL = 100`. Per fold, 100× draw `d_img` distinct columns of the **full `V_vis`** basis
(the visual model's output-space singular vectors — the most method-consistent reading of "singular
vectors of the visual activity space"), project the SAME `X_test_vis`, divide by the SAME `TV_vis`.
DOI-derived per-iteration seeds `SHA256(DOI|S2.6R|participant|ROI|fold|null=k)`; the observed value
never enters the seed. `ALIGNMENT_NULL_BASIS_SOURCE_NOT_FULLY_IDENTIFIABLE`.

---

## Predeclared public claims (frozen before group results)

**Dimensionality** — DIM-C1: early visual `d_img < d_vis` (~half). DIM-C2: `d_img/d_vis` → parity in
higher cortex. DIM-C3: imagery dimensionality relatively stable across visual ROIs vs visual
dimensionality. Status ∈ {DIRECTIONALLY_CONCORDANT, MIXED, DIRECTIONALLY_DISCORDANT, NOT_EVALUABLE}.

**Alignment** — ALIGN-C1: V1 ≈ 0.25–0.30. ALIGN-C2: ≈0.5 by hV4. ALIGN-C3: parietal ≈ 1.0. ALIGN-C4:
early-visual imagery subspace reoriented, not a mere lower-dim subset. Status ∈ {NUMERICALLY_CLOSE,
DIRECTIONALLY_CONCORDANT, MIXED, DIRECTIONALLY_DISCORDANT, NOT_EVALUABLE}.

Final claim-by-claim reproduction rubric and verdict are **deferred to S2.7R**.

---

## Remaining public ambiguities carried into S2.6R
- `AUTHOR_RANDOM_SEED_NOT_PUBLICLY_IDENTIFIABLE` (pairings; DOI-derived here).
- `D99_EXACT_TIE_INTERPOLATION_NOT_PUBLICLY_SPECIFIED`.
- `CENTER_SCALE_SCOPE_NOT_IDENTIFIABLE_PUBLICLY` (alignment standardization scope).
- `ALIGNMENT_NULL_BASIS_SOURCE_NOT_FULLY_IDENTIFIABLE`.
- `ALIGNMENT_STIMULUS_SET_SCOPE` — paper says "across all stimuli"; we use held-out TEST visual trials
  for a leakage-safe independent reconstruction (recorded, not hidden).
- `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE`, `BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS`.
