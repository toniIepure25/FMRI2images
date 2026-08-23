# S2.7R — Final claim-by-claim independent-reproduction verdict (Roy et al., Dataset 1)

**Gate:** S2.7R (AUDIT / VERDICT — no modelling) · **Branch:** `research/mindcompiler-neural-state-operators`
**Input HEAD:** `932b94f` · **Rubric freeze:** `a718faf` (sha `9acc7e1a…`, committed before synthesis).

**Scope:** `DATASET_1_NSD_IMAGERY_INDEPENDENT_REPRODUCTION_VERDICT` — Dataset 2 (Spatial Imagery)
was **not** reconstructed in Track R. **Source:** Roy et al., bioRxiv **PREPRINT** v1 (2025-09-03),
PMC12424947, *not peer-reviewed*. Supplement 1 (`media-1.pdf`, sha `67afe50a…`, 3 pp) retrieved via
Europe PMC: **three figures only, no material method detail** contradicting the reconstruction.

## FINAL DATASET-1 VERDICT: `INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`

The Dataset-1 public method was independently reconstructed to a degree sufficient for scientific
comparison. The perception→imagery **prediction** result and the cortical **increase** in
visual↔imagery subspace **alignment** were directionally reproduced. However, the paper's **central
early-visual dimensional-compression result was not reproduced**: the independent reconstruction
estimated imagery dimensionality *greater* than visual dimensionality in early visual cortex
(d_img/d_vis ≈ 1.47 vs the paper's ≈ 0.5). Early-cortex **alignment magnitudes** were also
substantially higher than reported (V1 ≈ 0.70 vs ≈ 0.25–0.30). Feature/tuning correspondence
(Fig 5C–E) was **not evaluated**. Therefore the Dataset-1 scientific result is **only partially
reproduced.** (This is not a claim that the paper is wrong, that reproduction failed, or that any
discrepancy is due to author error; possible causes remain unresolved.)

---

## Method sufficiency: `METHOD_PARTIALLY_IDENTIFIABLE_BUT_ASSESSMENT_POSSIBLE`
17 method rows: **9 MATCH, 8 CLOSE_DEFENSIBLE_RECONSTRUCTION, 0 MATERIAL_MISMATCH** (see
`final_method_concordance.csv`). Remaining ambiguities (do not block assessment):
author random seed; λ tie-break; center/scale scope; exact D1 train-dependency; prediction-null N;
B0 b2-compatible-not-bitwise; original code unavailable; bitwise replication unavailable.

## Claim ledger

| claim | class | importance | paper | our result | status |
|---|---|---|---|---|---|
| **P-C1** | P | CENTRAL | vis2img predicts imagery across cortex & subjects above null (Fig 3, Fig S1) | median r 0.117, 7/7 positive, 17.3% voxels > null p95, pairing-robust | **DIRECTIONALLY_CONCORDANT** |
| **D-C1** | D | CENTRAL | early visual d_img < d_vis (~half) (Fig 4) | early d_img/d_vis ≈ **1.47** (imagery HIGHER-dim) | **DIRECTIONALLY_DISCORDANT** |
| D-C2 | D | SUPPORTING_CENTRAL | d_img/d_vis → parity in higher cortex | parietal ratio 0.996 (parity, from above) | MIXED |
| D-C3 | D | SUPPORTING | imagery dimensionality stable across visual ROIs | d_img 3.5–5.7 across visual ROIs | MIXED |
| **A-C1** | A | CENTRAL | V1 substantially reoriented (~0.25–0.30) (Fig 5) | V1 alignment **0.70** (above null 0.19) | MIXED (reorientation present but weaker) |
| **A-C2** | A | CENTRAL | alignment rises across hierarchy (~0.5 hV4) | V1 0.70 → hV4 0.77 → parietal 0.92, monotone | **DIRECTIONALLY_CONCORDANT** |
| A-C3 | A | SUPPORTING_CENTRAL | parietal ~1.0 | parietal 0.92 | NUMERICALLY_CLOSE |
| **F-C1** | F | SUPPORTING_CENTRAL | imagery dims preserve coarse feature/tuning correspondence (Fig 5C–E) | not reconstructed | **NOT_EVALUATED** |

## Domain summary
- **METHOD** — sufficient for assessment (9 MATCH / 8 CLOSE / 0 mismatch).
- **PREDICTION** — `ROY_PREDICTION_DIRECTIONALLY_CONCORDANT` (7/7 new-cohort positive, above shuffle null, pairing-robust; Fig S1 corroborates). B1 is a sensitivity branch, never used to downgrade.
- **DIMENSIONALITY** — `ROY_DIMENSIONALITY_MIXED`. **The central early-visual dimensional-compression claim (D-C1) is DIRECTIONALLY DISCORDANT** (imagery higher-dimensional than visual); parietal parity (D-C2) agrees, hence the domain is MIXED rather than fully discordant.
- **ALIGNMENT** — `ROY_ALIGNMENT_DIRECTIONALLY_CONCORDANT`. Direction concordant (rises to parietal, all above the reconstructed random-subspace null); early magnitude discordant (V1 0.70 vs 0.25–0.30, coupled to inflated d_img); parietal NUMERICALLY_CLOSE.
- **FEATURE_CORRESPONDENCE** — `NOT_EVALUATED` (Fig 5C–E not reconstructed; no proxy substituted).

## Explicit numerical comparisons
- **Dimensionality (d_vis, d_img, ratio):** V1 (3.53, 4.48, 1.33); V2 (4.34, 5.31, 1.40); V3 (3.03, 5.14, 1.69); hV4 (2.56, 3.47, 1.37); ventral (3.28, 5.66, 2.02); lateral (4.34, 5.99, 1.48); parietal (6.45, 5.45, **0.996**). Early-visual mean ratio **1.475** vs paper ≈ 0.5 → **OPPOSITE direction**.
- **Alignment (ours vs paper):** V1 0.70 vs 0.25–0.30; hV4 0.77 vs ~0.50; parietal 0.92 vs ~1.0. Direction concordant, early magnitude discordant, parietal close.
- **Dimension-ratio ↔ alignment** (S2.6R): median per-participant Spearman −0.14 (no consistent within-subject relation).

## Discrepancy registry (central disagreements — possible explanations UNRANKED)
- **Dimensionality** (early ratio ~1.47 vs ~0.5, OPPOSITE): possible unresolved causes — author seed; center/scale scope; exact D1 train dependency; numerical/SVD details; beta not bitwise-confirmed; d_report tie/interpolation unspecified; **low imagery SNR (Supplement Fig S2)** may inflate noisy imagery reduced-rank curves. Possible explanations only; not ranked without evidence.
- **Alignment** (early magnitude high): directional progression concordant, early magnitude discordant, consistent with the inflated d_img (a larger imagery subspace captures more visual variance).

## Limitations
- **Exact replication:** original author code and random seeds unavailable; B0 is b2-*compatible*, not bitwise author-confirmed. `BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS`.
- **Dataset 2** (Spatial Imagery) not reconstructed — no claim about it; this is **not** a full two-dataset paper reproduction.
- **Preprint** status: evaluated source is a non-peer-reviewed bioRxiv preprint (provenance note only).
- **F-C1 not evaluated** — not every major paper result has been independently reproduced.

## Immutable
`H_A_CROSS_PARTICIPANT_PARTIAL` unchanged; `S2_5R_VERDICT_WITHHELD` and `S2_6R_RESULTS_IMMUTABLE` preserved.
No models fit, no beta extraction, no new null, no threshold/rank tuning, no participant exclusion,
no B1 promotion/demotion by outcome, no numerical weighted score.

## Track-R closure & next action
Track R (Roy Dataset-1 independent reproduction) **closes at `INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`.**
Optional handoff **O1 — perception→imagery neural-state operator characterization** (not executed here;
the S2.6R dimensionality contradiction must be preserved — dimensionality treated as an empirical
property of the operator, not assumed). Optional **R-DIAG1** dimensionality-discrepancy diagnostic may
later probe *predeclared* unresolved ambiguities only, and must not be presented as part of this verdict.
