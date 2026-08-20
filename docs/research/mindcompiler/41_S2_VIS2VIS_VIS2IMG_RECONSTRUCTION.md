# 41 — S2.0: Independent paper-compatible vis2vis→vis2img reconstruction (subj01 × V1)

**Gate:** S2.0 · **Input commit:** `bdbeabc` · **Branch:** `research/mindcompiler-neural-state-operators`.
**Scope:** subj01 × V1 only. No other ROI, no other participant, **no Roy reproduction verdict.**
Normal forward commits only.

This project **does not depend on an author response.** An author clarification may later
resolve an ambiguity, but its absence must never block the independent reconstruction.

---

## 3. Independent-reproduction policy

```
AUTHOR_RESPONSE_REQUIRED = false
AUTHOR_RESPONSE_ROLE     = optional ambiguity resolution only
STATUS                   = INDEPENDENT_METHOD_RECONSTRUCTION_ACTIVE
```

Rules (binding for S2.0):
1. Every unresolved method choice is implemented as one or more **defensible variants**, frozen
   before test evaluation.
2. All variants use identical data partitions where comparable.
3. Results from **all executed variants** are reported.
4. **No variant may be promoted because its test score is closer to Roy.**
5. A later author response is mapped onto the already-defined variant registry, never used to
   retro-justify a post-hoc choice.

---

## 2. Epistemic contract — three categories

Category C ambiguities are **never** silently converted into facts.

### A. Paper-supported facts (Roy et al. 2025 / NSD-Imagery documentation)

> **Provenance honesty:** these items are recorded in the project's prior extraction
> (`contracts.py`, `30_ROY_IMPLEMENTATION_AMBIGUITY_REGISTRY.csv`) attributed to Roy et al. 2025
> Methods and the NSD-Imagery release. Where a primary-source page was **not** independently
> re-verified in this session, the item is tagged `[project-recorded]` — it must be treated as a
> working citation to confirm against the paper PDF, not as a freshly audited quote.

| # | Fact | Source |
|---|------|--------|
| A1 | Dataset-1 analysis uses **12 identities** | Roy et al. 2025 Methods `[project-recorded]` |
| A2 | **6 simple + 6 naturalistic** stimuli | Roy et al. 2025 `[project-recorded]`; corroborated by NSD-Imagery set A/B structure in `s3_nsdimagery_listing.json` |
| A3 | **8 analyzed repeats** per identity per state (vision, imagery) | Roy et al. 2025 `[project-recorded]`; corroborated by the built trial table (`_validate_trial_table`, 8/identity/state) |
| A4 | **4 / 2 / 2** train/validation/test split of the 8 repeats | Roy et al. 2025 `[project-recorded]`; encoded in `contracts.SPLIT_*_REPEATS` |
| A5 | Ridge grid **1e-3 … 1e5, 100 log-spaced candidates** | Roy et al. 2025 `[project-recorded]`; encoded in `contracts.RIDGE_GRID_*` |
| A6 | **Reduced-rank** regression models | Roy et al. 2025 `[project-recorded]`; implemented in `reduced_rank_ridge.py` |
| A7 | Dimensionality constrained by analyzed stimulus structure | Roy et al. 2025 `[project-recorded]` (motivates rank ≤ #conditions) |
| A8 | Vision trials are **denoised via vis2vis predictions before vis2img** | Roy et al. 2025 `[project-recorded]` — the central two-stage structure |
| A9 | **Multiple cross-validation folds** are used | Roy et al. 2025 `[project-recorded]` (historical smoke used ONE 4/2/2 realization; S2.0 uses four) |
| A10 | Responses are described as **centered / scaled** | Roy et al. 2025 `[project-recorded]` — scope (which rows, which axis) is Category C (see C2) |

### B. Released-data facts (independently verified this session)

| # | Fact | Evidence |
|---|------|----------|
| B1 | **B0 = `fithrf`**, sha256 `31485ff0…`, 1,052,494,008 B, verified | `subj01_B0_download.json` |
| B2 | **B1 = `fithrf_GLMdenoise_RR`**, sha256 `cd42e680…`, 1,052,494,008 B, verified | `subj01_B1_download.json` |
| B3 | NSD betas are **int16 scaled ×300** → divide by 300 for PSC | `smoke_pipeline.extract_v1_matrix`, verified in `test_extraction_fixtures.py` |
| B4 | HDF5 layout is **(trial, Z, Y, X)** — reverse of NIfTI (X,Y,Z); NIfTI (x,y,z) → betas[:,z,y,x] | verified in `test_extraction_fixtures.py` |
| B5 | Beta trial order follows the behavioural **TSV `TRIAL`** sequence | `smoke_pipeline.build_trial_table`, `subj01_trial_table.csv` |
| B6 | V1 = prf-visualrois {1,2}; SNR from shared **NSD-core `betas_fithrf/ncsnr`**, strict >98th pct | `select_v1_voxels`, `test_extraction_fixtures.py`; voxel_hash `a1bc56fe7c55` (27 voxels), identical B0/B1 |

### C. Unresolved reconstruction choices (Category C — NOT facts)

| # | Ambiguity | S2.0 handling |
|---|-----------|---------------|
| C1 | Exact **beta version** Roy used (fithrf vs GLMdenoise_RR vs other) | run **both** B0 and B1; report sensitivity; no claim which Roy used |
| C2 | Exact **centering/scaling scope** (center-only vs z-score; per-voxel; which rows) | variants **P0 = historical_center_only**, **P1 = train_only_zscore** (primary) |
| C3 | Exact generation of **denoised TRAIN vision inputs** | **D1 strict cross-fit** (leakage-safe); D2 only if publicly identifiable |
| C4 | Exact generation of **validation/test denoised inputs** | governed by the D1 dependency graph (item 10/12) |
| C5 | **Fold dependency graph** for denoising | made explicit in `D1_dependency_manifest.csv` |
| C6 | **vis2vis pairing** policy | variants **V2V-P0 all_ordered_distinct**, **V2V-P1 single_deterministic_derangement** |
| C7 | **Random seeds** | frozen fold + pairing seeds recorded in `s2_frozen_config.json` |
| C8 | **Fold aggregation** rule | frozen before evaluation (item 18): mean/median/std across fold summaries |
| C9 | Any **train+validation refit** | historical is train-only; train+val refit remains a labeled sensitivity, not the primary |
| C10 | **Tie-breaking** not explicitly documented | explicit registry-resolved policies (rank/ridge), no silent defaults |

---

## 21. Project-direction record (reproduction vs MINDCOMPILER discovery)

**Stage R — Independent reproduction foundation.** vision → imagery under the closest publicly
reconstructable Roy pipeline. **← S2.0 is here.**

**Stage O — Neural-state operator characterization.** After reproduction is sufficiently
established, characterize `T_{vision→imagery}` across ROI, participant, stimulus family, rank,
geometry, and shared vs subject-specific components. *(Not performed in S2.0.)*

**Stage M — MINDCOMPILER prospective discovery.** Only after Stage R/O: test whether neural-state
transformations form reusable/composable operators across additional cognitive states. *(Not
performed in S2.0.)*

---

## 19/20. Interpretation boundary

**Allowed S2.0 conclusions** (subj01 × V1 engineering/method-reconstruction only):
`D1_PIPELINE_EXECUTION_PASS`, `D1_PIPELINE_EXECUTION_FAILURE`,
`BETA_VERSION_SENSITIVITY_OBSERVED`, `..._REDUCED_UNDER_D1`, `..._PERSISTS_UNDER_D1`.

**Forbidden:** Roy reproduced / failed to reproduce; B0 or B1 *is* Roy's version; D1 *is* Roy's
denoising; imagery generated by a causal transformation; population generalization; mechanistic
neural claims. **No reproduction verdict at this gate.**
