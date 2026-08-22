# 47 — S2.5R: Public-paper concordance audit and independent-reproduction verdict gate

**Gate:** S2.5R · **Input commit:** `1568ae4` · **Branch:** `research/mindcompiler-neural-state-operators`.

**Purpose:** determine, against the **actual public Roy et al. paper**, which parts of the NSD-Imagery
analysis have been independently reproduced, which differ materially, and whether a defensible
reproduction verdict can be issued. `H_A_CROSS_PARTICIPANT_PARTIAL` remains unchanged. No author
dependency. No forced verdict. No beta selection by score.

## Source of truth (re-opened this session)

Primary source **independently re-verified from the open-access PMC full text**
([PMC12424947](https://pmc.ncbi.nlm.nih.gov/articles/PMC12424947/), DOI `10.1101/2025.09.02.672180`,
bioRxiv v1, 2025-09-03; authors Saha Roy, Breedlove, St-Yves, Kay, Naselaris). Short exact quotes are
recorded in `public_method_claims.json` / `public_result_claims.json`; the full PDF is **not**
committed (copyright). Supplementary Information was **not** separately retrieved this session
(`supplement_claim_inventory.json` → `NOT_RETRIEVED_THIS_SESSION`). Prior project notes were **not**
relied upon.

## Method concordance (verified, quoted)

| Item | Roy (public) | Ours | Status / materiality |
|---|---|---|---|
| Voxel selection | ">98th percentile … in that ROI" (NSD-core SNR); Dataset-1 (distinct from Dataset-2 80th) | strict >98th, NSD-core ncsnr, per ROI | **MATCH** (LOW) |
| ROIs | V1,V2,V3,hV4,ventral,lateral,parietal | same 7 | **MATCH** (LOW) |
| Split / folds | "4-fold … 50/25/25" | 4-fold 4/2/2 | **MATCH** (LOW) |
| Ridge grid | "100 values between 10⁻³ and 10⁵" | same | **MATCH** (LOW) |
| Centering/scaling | "centered and scaled" (scope not stated) | train-only z-score | **CLOSE** (scope NOT_IDENTIFIABLE; MOD) |
| Reduced-rank ridge / two-stage | vis2vis → denoised vision → vis2img | same family | **CLOSE_DEFENSIBLE** (LOW) |
| **vis2img pairing** | "vision trial repeats were paired with **randomly-selected imagery trial repeats**" | **index-aligned (I0)** | **MATERIAL_MISMATCH (HIGH)** |
| vis2vis pairing | "**shuffling … within each data split**" | all_ordered_distinct (P0) | **MATERIAL (MODERATE)** — our P1 derangement is closer; P0 is primary |
| Rank/dimensionality semantics | "average of the values at 99% of the peak of each curve" (validation, averaged over 4 folds) | per-fold smallest-rank at 99% of selected-λ peak | **NOT_DIRECTLY_COMPARABLE (MOD)** |
| Denoise train dependency | not precisely stated | D1 LOTO (leakage-safe) | **NOT_IDENTIFIABLE** — D1 not penalized, not claimed identical |

## Beta evidence

Paper: betas "**similar to b2**" → NSD b2 = `betas_fithrf` = **B0**. `PUBLIC_EVIDENCE_STRONGLY_IDENTIFIES_
B0_AS_B2_COMPATIBLE`. "Similar to b2" ≠ bitwise-identical released file. **B1 (=b3, GLMdenoise_RR) is
NOT the publicly described preparation** → B1's weaker results are a sensitivity branch, **not** a
reproduction failure.

## Result-claim status

- **Prediction (P):** our B0 D1 gives positive vis2img (median r≈0.13, 6/7 new participants) — but we
  have **no Roy-compatible shuffle null**, so `PREDICTION_ABOVE_PAPER_NULL_NOT_DIRECTLY_TESTED`
  (r>0 ≠ above-null). At most **DIRECTIONALLY_CONCORDANT**.
- **Dimensionality (D):** Roy averages 99%-of-peak validation-curve ranks over folds; we did not
  compute the Roy-style d_vis/d_im estimate → `DIMENSIONALITY_NOT_DIRECTLY_COMPARABLE`.
- **Alignment/geometry (G):** Roy's variance-projection alignment ratio (V1≈25-30%, hV4≈50%,
  parietal≈100%) was **not** computed in S2.4R → `ROY_ALIGNMENT_RATIO_NOT_YET_RECONSTRUCTED`. No proxy
  is substituted.

## Verdict

A **HIGH-materiality explicit method mismatch** (vis2img index-aligned vs the paper's explicit random
pairing) means the current S2.4R is **not yet a fair implementation of the public vis2img method**;
additionally the central dimensionality and alignment results were not reconstructed. Per the frozen
rubric's `MATERIAL_GAP_rule`, the gate issues **`S2_5R_MATERIAL_METHOD_GAP_IDENTIFIED`** and does **not**
force a reproduction status — preferable to an invalid reproduction claim.

**Terminology preserved:** `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE`;
`BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS`. No "exact Roy reproduction" claimed.

## Next

**S2.5M — public-method-aligned pairing and rank reconstruction:** random within-identity
vision↔imagery pairing; within-split vis2vis shuffling; paper-compatible rank-selection semantics;
B0/b2-compatible primary (B1 retained as sensitivity). Then replay. Alignment/dimensionality remain
for a later **S2.6R**. Track O (operator research) stays deferred.
