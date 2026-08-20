# 44 — S2.3: Prospective cross-participant confirmation of the beta-preparation reliability mechanism

**Gate:** S2.3 · **Input commit:** `977b3cd` · **Branch:** `research/mindcompiler-neural-state-operators`.

```
ANALYSIS_CLASS = PROSPECTIVE_CROSS_PARTICIPANT_CONFIRMATION
N_PRIMARY_PARTICIPANTS = 7
```

The **participant** — not ROI, voxel, fold, or trial — is the primary unit of analysis.

- **Development/reference:** `subj01` (MUST NOT enter the primary cross-participant test).
- **Primary confirmation:** `subj02 … subj08` (7 previously untouched participants).

Evidence chain so far: V1 exploratory discovery → six-ROI prospective confirmation within subj01
(ρ=0.771, all criteria) → **this** seven-participant prospective test. No author dependency; no Roy
reproduction verdict; primary hypothesis is not changed after observing any participant.

## Primary hypothesis `H-A-CROSS-PARTICIPANT`

For participant *s*, ROI *r*: `L_rel_img(s,r) = B0−B1 imagery reliability`;
`L_perf_RAW(s,r) = B0−B1 RAW_S2_MATCHED four-fold vis2img mean r` (positive = B1 worse).
Reduce the seven ROIs per participant to one summary (equal ROI weight, **median**):
`S_rel(s) = median_r L_rel_img`, `S_perf(s) = median_r L_perf_RAW`.
Primary: `Spearman ρ(S_rel, S_perf) > 0` across subj02–08, with an **exact one-sided permutation
p** over 7! = 5040 orderings (α = 0.05). Not an asymptotic p-value.

## Frozen criteria (before outcomes)

- **A**: median_s S_rel > 0. **B**: median_s S_perf > 0. **C**: ρ_primary > 0 AND p_exact ≤ 0.05.
- **D**: ≥ 5 of 7 participants with S_rel > 0 AND S_perf > 0.
- Class: SUPPORTED (all A–D) / PARTIAL (2–3) / NOT_SUPPORTED (0–1) / INCONCLUSIVE (< 5 evaluable).

## Frozen contract (unchanged from S2.1B/S2.2B)

Seven ROIs (V1 prf{1,2}, V2 {3,4}, V3 {5,6}, hV4 {7}, ventral streams{5}, lateral {6},
parietal {7}). Outcome-independent voxel selection (participant NSD-core ncsnr; strict > ROI 98th
pct; identical B0/B1 voxels; min 3 voxels; no adaptation). Reliability estimator = S2.1B unchanged.
Primary endpoint = **RAW_S2_MATCHED** (NOT D0, NOT D1). Four deterministic 4/2/2 folds via the
S2 algorithm/root-seed contract, built per participant from their own trial table (row indices need
not match subj01). **No D1/D1b, no alternate preprocessing/pairing/threshold in the primary path.**

## Chronology

Phase-1 freeze (this + `s2_3_frozen_config.json`) committed and pushed **before** inspecting any
subj02–08 B0/B1 outcome. Violation → `PROSPECTIVE_PARTICIPANT_FREEZE_VIOLATION`. Then acquire
(certified downloader; anonymous public bucket; locally-computed content SHA, never the S3 ETag),
validate, select ROIs, execute, classify.

## Interpretation boundary

Allowed (if supported): "across seven previously untouched participants, larger B1 imagery-
reliability loss was associated with larger B1 matched RAW vis2img loss"; "the subj01 spatial
mechanism generalized across participants." Forbidden: causal GLMdenoise effect; B1 wrong; B0 is
Roy's version; Roy reproduced/failed; GLMdenoise damages imagery; human-population prevalence beyond
this sample; "choose B0 because higher performance". Beta-version identity ≠ measurement reliability.
