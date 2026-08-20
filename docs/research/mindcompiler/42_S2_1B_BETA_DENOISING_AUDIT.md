# 42 — S2.1B: Beta-preparation × denoising interaction audit (subj01 × V1)

**Gate:** S2.1B · **Input commit:** `af6532e` · **Branch:** `research/mindcompiler-neural-state-operators`.

```
ANALYSIS_CLASS = EXPLORATORY_MECHANISM_AUDIT
```

**Motivating observation (already made in S2.0 — hence this is exploratory, not confirmatory):**

```
B0 D1 vis2img aggregate mean r ≈ +0.2028
B1 D1 vis2img aggregate mean r ≈ -0.0300   (subj01 × V1, 4 folds)
```

Because the B0/B1 divergence was **already observed on subj01 × V1**, every mechanistic analysis
built here is **exploratory**: no p-hacking, no policy selection to make B0 or B1 look better, no
population inference from folds. Hypotheses generated here must be **frozen** and later tested on
**untouched** ROIs/participants — those become the independent confirmation (PART K).

**Scope:** subj01 × V1 only (27 voxels, 4 folds, same folds for B0 and B1). No other ROI, no other
participant, **no Roy reproduction verdict**, no author dependency. Normal forward commits.

## Scientific question

Not "which beta version gives larger r", but **which stage of the pipeline causes B0 and B1 to
diverge**: (1) beta preparation, (2) raw vision reliability, (3) raw imagery reliability,
(4) vis2vis predictability, (5) D1 denoising, (6) vision–imagery alignment, (7) covariance/
dimensionality, (8) the specific LOTO implementation of D1.

## S2.0 preservation

S2.0 is frozen and untouched: `s2_frozen_config.json` (`e3f32a0d`), fold assignments
(`fold_manifest.csv` `243b2e07`), `B0_fold_results.json`, `B1_fold_results.json`, the S2.0 D1
dependency manifest. S2.1B is a **new diagnostic layer** under `artifacts/mindcompiler/roy_s2_1b/`,
clearly separated.

## Matched 2×2 factorial (PART B)

`{B0, B1} × {RAW_S2_MATCHED, D1_STRICT_CROSSFIT}` — all four cells use the **same** S2 four folds,
4/2/2 assignments, P1 train-only z-score, vis2img index-aligned pairing, ridge 1e-3..1e5×100,
rank rule, finite-fraction criterion, one-shot test policy. The RAW cell is **re-run under the
matched S2 protocol** (NOT the historical single-split D0). Descriptive per-fold effects:
`Δdenoise = r(D1) − r(RAW)`; `interaction = Δdenoise_B1 − Δdenoise_B0`. **No fold p-values.**

## D1b prespecified sensitivity (PART H)

`D1b = symmetric k-fold crossfit` was registered in S2.0 *before* observing results. It is
implemented to that registered intent (symmetric cross-fit for train vision; no trial denoises
itself; val/test leakage-safe; same outer folds, P1, vis2img contract). Not promoted to primary if
it performs better.

## Interpretation boundary (PART J item 24)

Forbidden: fold p-values; voxels as independent subjects; "B0 superior" / "B1 wrong";
"GLMdenoise damages imagery"; "Roy used B0"/"Roy failed". This stays a subj01 × V1 exploratory
mechanism audit; the causal mechanism is **not** proven by a PASS.
