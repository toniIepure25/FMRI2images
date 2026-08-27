# O2.2 — Cross-state transport geometry, imagery-residual sharedness & vision-only sufficiency (charter)

**Gate:** O2.2 (post-hoc geometry audit — no refit, no new common-space) · **Input HEAD:** `760a54b`
**Class:** `POST_HOC_CROSS_STATE_REPRESENTATIONAL_GEOMETRY_AUDIT`.

## Immutable
O2 `SHARED_OPERATOR_PARTIAL`; O2.1 `O2_TRANSFER_LIMIT_CROSS_STATE_SPACE_DOMINANT`; **O3 `O3_NOT_READY`
locked through O2.2**; O1 `STIMULUS_INVARIANT_OPERATOR_PARTIAL`; `SHARED_COMPONENT_DOMINANT` qualified
by `DECOMPOSITION_CONFIDENCE_MODERATE`. O2.2 diagnoses, never rescues.

## Question
O2.1 showed vision alignment works but the vision-derived common space preserves ~10% of imagery
identity contrast. O2.2 asks: is imagery lost because (A) SRM truncated an otherwise adequate **visual
identity span**, or (B/C) imagery occupies neural directions **outside** the visual identity span that
are shared / subject-specific, or (D) the missing component is **measurement noise**? And: does the
within-subject O1 operator predict the *discarded* component? Can a new subject's residual orientation
be identified from **vision only**?

## No remedy fitting
No imagery-SRM / joint-SRM / hyperalignment / connectivity-SRM; no K change; no imagery target
calibration; no new mapper; no T_shared/baseline change; no alternative-common-space competition.

## Three nested subspaces (TRAIN identities only, per subject×ROI×fold)
- **P_SRM** = W Wᵀ (exact frozen O2 vision-derived target mapping).
- **P_VIS_FULL** = projector of all nonzero right-singular vectors of centered train **vision**
  identity centroids (rank ≤ 9).
- **P_IMG_FULL** = same for train **imagery** (post-hoc geometry only, never O2 fit).

## Primary discriminator
On held-out identity contrasts Δx, Δy (target std native): `R_Y_SRM` (reproduces O2.1), **`R_Y_VISFULL`**
(imagery retention in the full visual span — the new critical quantity), `SRM_TRUNCATION_LOSS =
R_Y_VISFULL − R_Y_SRM`, `CROSS_STATE_SPAN_LOSS = 1 − R_Y_VISFULL`. Plus a descriptive vision-rank
retention curve R_Y_VIS(k), visual↔imagery principal angles / overlap, and the outside-visual fraction
`||Δy(I−P_VIS_FULL)||²/||Δy||²`.

## Is the lost component real & shared?
Fixed **odd/even** repeat split → cross-half identical-identity pattern correlation per component
(inside-SRM / inside-visual / outside-visual / full): `RELIABILITY_*`. Representational **RDMs**
(1−Pearson, 12 identities) per component → **LOSO RDM sharedness** (Spearman of RDM_s vs mean of other
7; participant is the unit). Categories HIGH ≥0.40 / MODERATE 0.20–0.40 / LOW.

## Does O1 predict the discarded component?
Exact frozen O1 O4 replay (deterministic re-selection, verified vs committed held-out r); decompose
Ŷ_O1 and Y_true by P_VIS_FULL → `G_O1_PARALLEL`, `G_O1_PERP` (O4−O0 per component). Relate frozen O1
`out_of_visual_span` to `G_O1_PERP` / outside fraction (descriptive Spearman across N=8, per ROI).

## Regimes (frozen, Part P) & statuses
A SRM-truncation-dominant · B shared-geometry-outside-vision · C subject-specific-imagery-geometry ·
D imagery-measurement-limited · MULTIREGIME. Overall ∈ {SRM_TRUNCATION / SHARED_IMAGERY_SPACE_OUTSIDE
_VISION / SUBJECT_SPECIFIC_IMAGERY_SPACE / MEASUREMENT_LIMITED / MULTIREGIME / UNRESOLVED}_DOMINANT.

## Identifiability (Part S — crucial)
`TARGET_STATE_ORIENTATION_IDENTIFIABLE_FROM_VISION` requires a **frozen observed relationship**
predicting the residual orientation from vision-derived quantities **without target imagery** — shared
RDM geometry alone is **not** sufficient. Else NOT_IDENTIFIABLE / UNKNOWN.

## Future-gate decision (design-readiness only, does not modify O2)
∈ {O2_3_RICHER_VISION_SPACE_FEASIBLE / O2_3_STATE_AWARE_SHARED_SPACE_FEASIBLE /
O2_3_SUBJECT_SPECIFIC_STATE_MAPPING_REQUIRED / O2_3_MEASUREMENT_BOTTLENECK / O2_3_NO_CLEAR_REMEDY}.
`future_common_space_requirements.json` records what a future representation would need — but O2.2 fits
none of it. Per-ROI mechanisms kept separate. O3 stays NOT_READY.

## Forbidden interpretations
Not "SRM fails for imagery" / "parietal has no shared transformation" / a new space "is a remedy" /
shared geometry ⇒ identifiable orientation — unless the specific diagnostic supports the bounded claim.
