# Metric redundancy audit (MINDIR-PROX / P2)

Relationships among registry metrics computed on **synthetic** fixtures (no historical outcomes). Classifies
metrics as equivalent / near-equivalent / complementary / unstable / misleading-in-edge-cases.

## Geometry family
- `subspace_overlap`, `grassmann_distance`, `chordal_distance`, `projector_frobenius` are all monotone functions
  of the principal angles → **near-equivalent** for ranking two subspaces (they agree on ordering). Recommend
  **`subspace_overlap`** as the primary reportable (bounded [0,1], intuitive); report a distance only when a
  metric geometry is needed. `grassmann` and `chordal` **diverge only when ranks differ** → use with care
  (unstable across unequal ranks).
- `canonical_correlations` are the raw ingredients; the scalar metrics summarize them.

## Prediction family
- `pattern_correlation` vs `cosine_similarity`: **complementary** — cosine is magnitude-insensitive and
  sign-folded; correlation removes mean and keeps sign. Cosine is more robust for direction-only claims (used by
  M2). `R^2` is **misleading** at N~12 / low-dim and is only reported where genuinely valid.

## Support family
- `support_fraction` and `outside_support_residual` are **exact complements** (sum to 1) → report one.

## Calibration family
- `total_recovery` and `full_capacity_fraction` are **complementary** (different denominators: native oracle vs
  full-resource ceiling); both are informative and both required by the frozen O2.16 criterion. `sample_efficiency`
  and `calibration_regret` are **complementary** adaptive-acquisition views.

## Stability family
- `repeat_agreement` and `split_half_reliability` are **near-equivalent** proxies for reliability;
  `repeat_agreement` generalizes split-half to >2 repeats → prefer it when repeats ≥ 3.

## Edge-case warnings
- All geometry metrics are **unstable when a subspace is near rank-deficient** (tiny singular values) — the
  numerical-stability report documents the safe regime.
- Ratio calibration metrics are **misleading when the denominator (oracle−baseline or ceiling−baseline) is
  small**; guard with a floor and report the raw components.

## Recommendation
Primary reportables: `subspace_overlap` (geometry), `cosine_similarity` (direction), `outside_support_residual`
(support), `total_recovery` + `full_capacity_fraction` (calibration), `repeat_agreement` (stability). Others are
diagnostics.
