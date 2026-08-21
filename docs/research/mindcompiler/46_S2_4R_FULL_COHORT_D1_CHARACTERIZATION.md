# 46 — S2.4R: Full-cohort cross-participant D1 Roy-pipeline characterization

**Gate:** S2.4R · **Input commit:** `be84dea` · **Branch:** `research/mindcompiler-neural-state-operators`.

```
ANALYSIS_CLASS = FROZEN_FULL_COHORT_INDEPENDENT_METHOD_RECONSTRUCTION_CHARACTERIZATION
```

First full-cohort execution of the independently reconstructed two-stage pipeline
`raw vision → vis2vis reduced-rank → D1 leakage-safe denoised vision → vis2img reduced-rank →
predicted imagery`, carrying **B0 (fithrf)** and **B1 (fithrf_GLMdenoise_RR)** as parallel
beta-preparation sensitivity branches. **No beta winner is selected. No Roy reproduction verdict.**

Scope: 8 participants × 7 ROIs × 2 betas × 4 folds = **448 outer-fold D1 cells**.

## Chronology (partial-prospectiveness — stated honestly)

- `subj01` D1 already observed (S2.0) → `DEVELOPMENT_REFERENCE_D1`.
- `subj02–08` RAW already observed (S2.3); their **D1 outcomes are NEW** →
  `FROZEN_NEW_D1_CHARACTERIZATION_COHORT`. The D1 contract is frozen **before** inspecting any
  subj02–08 D1 outcome (violation → `S2_4R_D1_FREEZE_VIOLATION`). The whole dataset is **not**
  described as fully untouched.

## Frozen (immutable) contract

D1 = `D1_STRICT_CROSSFIT` (S2.0): train=LOTO, val/test=full-train, pooled within-identity vis2vis,
inverse-transform to raw units; leakage invariants all 0. **No D1b, no D2.** Preprocessing =
`PREPROC_ZSCORE_TRAIN_ONLY` (both stages, train-only). vis2vis pairing = `V2V-P0 all_ordered_distinct`
(expanded pair rows are **not** inferential N). vis2img = within-identity, within-split, index-aligned.
Ridge 1e-3..1e5×100; rank ≤ min(12, conditions, dims, effective) with 99%-of-peak selection. Four
S2 folds reused per participant (same for B0/B1). **No alternate preprocessing/pairing/threshold; no
participant exclusion; no beta selection; no H-A rescue.**

## Frozen descriptive rules

- Participant is the inferential unit. `D1_subject(s,b)` = median over 7 ROIs of the four-fold D1
  vis2img aggregate r. `S_D1_beta_loss(s)` = median over ROIs of `D1_B0−D1_B1`.
- D1-vs-RAW gain `G(s,r)=D1−RAW` (reuse frozen S2.3 RAW). `NEAR_NEUTRAL_ABS_THRESHOLD=0.02`:
  G>+0.02 `D1_GAIN`, G<−0.02 `D1_LOSS`, else `D1_NEAR_NEUTRAL`.
- `Delta_beta_sensitivity = D1_loss − RAW_loss` (descriptive; NOT causal interaction).
- Cohort beta-sensitivity (subj02–08): `D1_BETA_SENSITIVITY_PERSISTS` if median S_D1_beta_loss>0 and
  ≥5/7 participants >0; `REVERSES` if median<0 and ≥5/7 <0; else `MIXED`. No significance requirement;
  not used to select a beta.

## Known vs unknown before S2.4R

Known: subj01 D1; all-participant RAW; H-A partial; reliability heterogeneity; beta-prep sensitivity.
Unknown: subj02–08 D1 vis2vis/vis2img; full-cohort D1 ROI profiles; participant-level D1 beta
sensitivity; D1-vs-RAW gain across the cohort.

## Boundary

Allowed (if supported): the reconstructed D1 pipeline executed across the full cohort/7 ROIs; D1 gave
positive perception→imagery mapping in X/7 new participants under B0/B1; the beta-prep sensitivity
persisted/attenuated/reversed under D1; heterogeneity remained visible. Forbidden: Roy reproduced/
failed; B0 or B1 is Roy's; B0 correct because higher r; B1 invalid; GLMdenoise harmful; D1 equals the
authors' pipeline; causal transformation; population claim. Roy concordance is the separate S2.5R gate.
