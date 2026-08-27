# O2.1 — Shared-operator transfer bottleneck & cross-state common-space audit (charter)

**Gate:** O2.1 (post-hoc bottleneck audit — no refit, no new common-space) · **Input HEAD:** `adcb244`
**Class:** `POST_HOC_SHARED_OPERATOR_TRANSFER_BOTTLENECK_AUDIT`.

## Immutable
O2 `SHARED_OPERATOR_PARTIAL` / `SHARED_COMPONENT_DOMINANT` unchanged; O2 primary result, DetSRM
common-space method, and K-selection immutable. O1/O1.1/Track-R/H-A unchanged. O2.1 is explanatory
only — never upgrades PARTIAL, never repairs anything.

## Question
Where is the ~95% loss of the within-subject O1 incremental operator advantage incurred when moving
to cross-subject transfer (O1→O2 median transfer fraction ≈ 0.0478)? Why does the *identical* frozen
SRM + shared-operator pipeline transfer in **ventral** cortex (G_shared +0.0145 Holm-significant,
G_beyond_gain +0.0197, 7–8/8) but not in lateral (+0.0017 ns) / parietal (−0.004 ns)? Ventral
succeeds with the *smallest* K (K=2 in 48/48), so **low K alone cannot explain failure**.

## Bottleneck axes
B1 target vision calibration · B2 vision common-space capacity · **B3 cross-state common-space
adequacy** (does a vision-derived W preserve *imagery* identity structure?) · B4 shared-operator
failure inside common space · B5 native back-projection · B6 strong group-imagery-mean baseline · B7
subject-specific heterogeneity · B8 stimulus/identity heterogeneity · **B9 regularization-induced
inflation of the shared-component estimate** (λ_delta at the 1e4 upper boundary shrinks Δ→0).

## Source & replay
Committed O2 artifacts + frozen O1/O1.1 summaries. Deterministic frozen-SRM **replay** allowed only to
derive non-persisted projection diagnostics, reproducing the committed per-cell K and λ (from
`outer_fold_predictions.csv`), same subjects/identities/scaler/seeds; replay must reproduce the
committed `r_Tshared` within tolerance. No new fitting choice; imagery used only as evaluation data.

## Key new diagnostics (formulas frozen in config)
- **Contrast retention**: vision `||Δx·WWᵀ||²/||Δx||²`, imagery `||Δy·WWᵀ||²/||Δy||²`; cross-state gap
  = vision − imagery. Identity-*contrast* retention is primary (total energy hides it).
- **Common-space oracle** (capacity, not a predictor): `Y_img·WWᵀ` → inverse vision scaler → native
  pattern r / NMSE / projection energy.
- **Shared-space contrast capture** (K often 2, so NOT Pearson-over-K): `Δŷ_shared` vs `Δy_shared`,
  relative error and cosine — does T_shared transfer the stimulus-dependent imagery *difference*?
- **Gauge-invariant T stability** across LOSO folds (singular values, spectral/Frobenius norm,
  effective rank, NTI, contraction — never raw-W).
- **Δ regularization audit**: DELTA_UPPER_BOUNDARY_RATE (≥0.25 → concern) + DELTA_CV_INCREMENT
  (held-out identity score of T+Δ vs T at committed λ_delta) → decomposition confidence HIGH/MOD/LOW.
- **SMD** = (median_A−median_B)/pooled_MAD: ≥1 strong, 0.5–1 moderate.

## Statuses (frozen)
Per-ROI case audits (ventral transfer, lateral/parietal failure — kept distinct). Overall bottleneck
∈ {CALIBRATION / CROSS_STATE_SPACE / SHARED_OPERATOR_HETEROGENEITY / BASELINE / MULTIFACTORIAL /
UNRESOLVED}_DOMINANT. Decomposition confidence HIGH/MODERATE/LOW. O3 readiness A–E (broad ventral
transfer, not a projection artifact, genuine shared-space contrast, confidence ≥ MODERATE, no leakage).

## Q1–Q5 (Part X)
Q1 is visual alignment sufficient for operator transfer? Q2 does vision-derived W preserve imagery
identity structure? Q3 does T_shared encode held-out imagery contrasts in common space? Q4 is ventral
special because more shared or because projection/reconstruction easier? Q5 is the 70% shared
decomposition robust to Δ regularization? Each: SUPPORTED / NOT_SUPPORTED / MIXED / INCONCLUSIVE.

## Forbidden interpretations
Not "SRM fails for imagery" / "parietal has no shared transformation" / "ventral contains the
universal imagination operator" / "70% proves biological universality" / "low K caused the result" —
unless the specific diagnostic supports the precise bounded claim. No p-values on N=8. No status change.
