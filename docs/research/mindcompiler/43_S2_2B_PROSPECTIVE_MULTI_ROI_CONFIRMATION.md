# 43 — S2.2B: Prospective multi-ROI confirmation of the beta-preparation reliability hypothesis

**Gate:** S2.2B · **Input commit:** `b4f472f` · **Branch:** `research/mindcompiler-neural-state-operators`.

```
ANALYSIS_CLASS = PROSPECTIVE_WITHIN_SUBJECT_SPATIAL_CONFIRMATION
NO_POPULATION_INFERENCE
```

S2.1B (exploratory, on V1) suggested **H-A**: B1 reduces useful identity/repeat reliability,
especially for imagery, and B1 is already weak under matched RAW. The prospective prediction was
frozen there: *B1/B0 vis2img divergence should covary across ROI with B1−B0 repeat-reliability loss.*
S2.2B tests that prediction on **six previously untouched ROIs** of subj01 — changing epistemic
status from exploratory to prospective spatial confirmation.

## Data separation

- **Discovery (V1)** — `DISCOVERY_ROI`. Permanently discovery data; **excluded from the primary
  six-ROI correlation/confirmation**. May appear only as an appendix reference after the prospective
  analysis is complete.
- **Prospective (six)** — `PROSPECTIVE_CONFIRMATION_ROIS`: V2, V3, hV4, ventral, lateral, parietal.

Six ROI observations are **NOT six independent participants**. No population p-values, no
pseudoreplication (folds/voxels are not subjects). This remains one participant.

## Primary prediction (spatial profile)

For ROI *j*: `L_rel_img(j) = R0_img(j) − R1_img(j)` (B0−B1 imagery repeat reliability);
`L_perf_RAW(j) = P0_RAW(j) − P1_RAW(j)` (B0−B1 four-fold RAW_S2_MATCHED vis2img mean r). Positive =
B1 worse. Predict `Spearman ρ(L_rel_img, L_perf_RAW) > 0`.

## Frozen contract

- Reliability estimator = **exactly** the S2.1B implementation (mean over unordered repeat-pair
  Pearson r, averaged over 12 identities). Unchanged.
- Primary endpoint = **RAW_S2_MATCHED** (isolates the beta-preparation/raw stage), NOT historical D0.
  D1 is a secondary endpoint. **No D1b, no alternate preprocessing/pairing/refit.**
- Same four S2 folds, 4/2/2, P1 train-only z-score, vis2img index-aligned, ridge 1e-3..1e5×100,
  rank rule, train-only fit, test once.
- ROI voxel selection is **outcome-independent** (NSD-core ncsnr; strict > ROI 98th pct); B0 and B1
  use identical voxel indices. No threshold adaptation, no ROI merging, no manual voxels.
  `pattern_reliability_min_voxels = 3`; below that → `NOT_EVALUABLE` (no rescue). If fewer than four
  prospective ROIs are evaluable → `H_A_SPATIAL_PROFILE_INCONCLUSIVE`.

## Chronology (freeze then execute)

**Phase 1** (this + the freeze commit): reconcile, build six ROI masks, select voxels from ncsnr
only, write `roi_selection_manifest.json`, freeze `s2_2b_frozen_config.json` with Criteria A–D, then
commit + push **before** extracting/evaluating any B0/B1 ROI outcome. Violating this →
`S2_2B_PROSPECTIVE_FREEZE_VIOLATION`.

**Phase 2**: extract B0/B1 selected matrices, value QC, reliability, matched RAW, D1, secondary
geometry, results, classification.

## Interpretation boundary

Allowed: "H-A spatial-profile prediction supported / not supported **within subj01**". Forbidden:
mechanism proven; GLMdenoise_RR damages imagery; B1 wrong; B0 is Roy's version; Roy reproduced/
failed; population generalization; causal claim; six ROIs as independent replications.
