# O1 — Stimulus-invariant perception→imagery neural-state operator (charter)

**Gate:** O1 (Track O — original research, NOT Roy reproduction) · **Branch:** `research/mindcompiler-neural-state-operators`
**Input HEAD:** `e378288` · **Analysis class:** `FROZEN_ORIGINAL_OPERATOR_DISCOVERY_ON_PREVIOUSLY_USED_DATA`
(pre-registered/frozen secondary discovery on NSD-Imagery — **not** independent-dataset confirmation).

## Immutable Track-R seal
- `TRACK_R_STATUS = INDEPENDENT_METHOD_REPRODUCTION_PARTIAL` · `TRACK_R_REOPEN_ALLOWED = false`.
- `H_A_CROSS_PARTICIPANT_PARTIAL` immutable. Preserved permanently: Roy prediction directionally
  reproduced; alignment progression directionally reproduced; **early-visual dimensional compression
  NOT reproduced** (our early d_img/d_vis ≈ 1.47, not ≈ 0.5); Dataset 2 not reproduced; exact author
  implementation unavailable. O1 does **not** reopen or repair any of these.

## Scientific question
From *"Can we reproduce Roy?"* to *"Does a reusable transformation map perception-state neural
representations to imagery-state representations for stimuli **never used to estimate it**, and if so,
what mathematical class describes it?"* Core property:
**`STIMULUS_IDENTITY_HELD_OUT_OPERATOR_GENERALIZATION`** — imagery data for a test identity never
enters operator fitting, scaler fitting, hyperparameter/rank/model-class selection, or validation.

## Measurement contract
- **B0 (fithrf, b2-compatible) primary** — `B0_SELECTION_BASIS = PRE_O1_MEASUREMENT_QUALITY_AND_PUBLIC_METHOD_EVIDENCE`
  (not chosen from O1 outcomes). B1 optional sensitivity only, never selects the operator family.
- **Primary source/target = RAW vision / RAW imagery B0 centroids** (NOT Roy D1-denoised vision).
- 8 participants; 7 ROIs (existing validated B0 voxel selections; no new ROI/threshold).

## Representation & splits
- Identity centroid = mean of the 8 repeats. **Fitting N = number of training identities** (repeats
  never inflate N); within-identity repeat reliability retained for diagnostics only.
- **Common vision scaling** (`COMMON_VISION_SCALE_TRAIN_ONLY`): μ_vis/σ_vis from TRAIN vision centroids
  applied to **both** states (preserves state-dependent gain); zero-variance vision dims dropped and
  recorded. Models may carry an intercept b.
- 12 identities = 6 simple (stim_set A) + 6 naturalistic (stim_set B). **Outer:** 6 folds, each holds
  out 1 simple + 1 naturalistic (every identity tested exactly once; 10 train / 2 test). **Inner:** 5
  identity folds within each outer-train (1 simple + 1 naturalistic each). Leakage certificate: all True.

## Operator ladder (linear only; no NN/kernel/diffusion/transformer)
O0 IMAGERY_MEAN (ŷ=b) · O1 GLOBAL_GAIN (ŷ=b+a·x) · O2 DIAGONAL_AFFINE (ŷ=b+D·x) ·
O3 LOW_RANK_RESIDUAL (ŷ=b+x+Δ_r·x) · O4 FULL_RRR_OPERATOR (ŷ=b+W_r·x).

## Selection, metric, inference
- Ridge grid 50 log 1e-4..1e4; rank 1..min(8, n_inner_train−1, eff_source_rank, tgt_dim); selected by
  **inner identity-held-out CV**; tie → larger ridge, then smaller rank.
- Primary metric: **HELD_OUT_IDENTITY_PATTERN_R** (Pearson across voxels, common-vision-scaled space);
  each of the 12 identities contributes to test exactly once. Secondary: normalized-MSE, cosine,
  explained-variance, centroid-norm-ratio; 2×2 identity discrimination.
- Inference unit = **participant (N=8)**; exact 2⁸ sign-flip one-sided test for O4−O0, O4−O1, O4−O2;
  Holm across 7 ROIs per family; α=0.05. Never folds/identities/voxels as subjects.

## Frozen hypotheses
H-O1 O4>O0 (reusable operator) · H-O2 O4>O1 (more than global weakening) · H-O3 O4>O2 (cross-voxel
mixing) · H-O4 O3 competitive with O4 if median(O4−O3)≤0.02 and not worse by >0.05 in >2/8.

## Secondary & structural
Cross-family transfer (simple→naturalistic and reverse; hyperparameters chosen inside source family).
Final descriptive O4 on all 12 identities (geometry only, not generalization evidence): supported
visual-centroid span B_v; identity deviation; out-of-visual-span fraction; restricted spectrum;
effective ranks; residual operator; off-diagonal mixing; polar decomposition (rotation × stretch);
outer-fold operator stability (HIGH≥0.80 / MODERATE≥0.50 / LOW). Dimensionality is an **outcome**,
never assumed lower for imagery.

## Prohibited
Nonlinear models; D1 as primary source; identity leakage; trial-level random validation; inflating N
by repeats; cross-subject raw-W comparison (deferred to O2); assuming imagery lower-dimensional; causal
operator claim; reopening Roy reproduction; B1 selecting the operator family.

## Next
O2 (shared + subject-specific operator, LOSO zero-shot) if SUPPORTED; O1.1 heterogeneity audit if
PARTIAL; stop if NOT_SUPPORTED. Cross-subject common-space (hyperalignment/SRM) is O2's problem, not O1's.
