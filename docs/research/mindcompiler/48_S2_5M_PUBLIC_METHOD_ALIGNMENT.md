# 48 — S2.5M: Public-method-aligned pairing, per-voxel ridge, prediction null

**Gate:** S2.5M · **Input `34b70f6`.** `ANALYSIS_CLASS = FROZEN_PUBLIC_METHOD_ALIGNMENT_RECONSTRUCTION`.
Repairs ONLY method components explicitly supported by the public paper (S2.5R gaps). No beta
selection by score; `H_A_CROSS_PARTICIPANT_PARTIAL` immutable; no reproduction verdict.

`B0 = PUBLIC_B2_COMPATIBLE_PRIMARY_RECONSTRUCTION_BRANCH` (b2-compatible, not bitwise-confirmed);
`B1 = PREDECLARED_BETA_PREPARATION_SENSITIVITY_BRANCH` (b3).

| Component | public evidence | old | new S2.5M | remaining ambiguity |
|---|---|---|---|---|
| vis2vis pairing | "shuffling within each data split" | all_ordered_distinct | within-split derangement | author seed unknown |
| vis2img pairing | "randomly-selected imagery repeats" | index-aligned I0 | random within-identity | exact realization unknown |
| ridge lambda | n ridges, voxel-specific lambda_i | single scalar lambda | **per-target Lambda** | tie-break not public |
| operational rank | validation-optimized | smallest at 99% of peak | argmax validation r (smallest on tie) | 99%-of-peak dimensionality -> S2.6R |
| prediction null | Fig-3 shuffle | r>0 only | test-fold input<->output shuffle N=1000 | author N unknown |
| D1 | denoised vision inputs (train dep not stated) | — | D1_STRICT_CROSSFIT kept | exact train dependency unknown |

Seeds are DOI-derived (`SHA256(DOI|S2.5M|PAIRING_V1|realization)`), excluding ROI/beta so pairings are
identical across ROIs/betas (controlled). Preprocessing train-only z-score (scope not public). Rank
cap min(12,...). Full rank curves saved for S2.6R (dimensionality DEFERRED).

Even after a clean pass, these remain: AUTHOR_RANDOM_SEED_UNKNOWN, EXACT_DENOISING_TRAIN_DEPENDENCY_
UNKNOWN, CENTER_SCALE_SCOPE_UNKNOWN, ORIGINAL_CODE_UNAVAILABLE, BITWISE_REPLICATION_UNAVAILABLE, and
ROY_DIMENSIONALITY/ALIGNMENT_NOT_YET_RECONSTRUCTED -> FULL_INDEPENDENT_REPRODUCTION_VERDICT STILL
DEFERRED to S2.6R. Next: S2.6R dimensionality + alignment (no proxies).
