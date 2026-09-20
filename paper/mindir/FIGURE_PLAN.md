# Figure plan (max 6 main figures + supplement)

**Rule:** every panel uses only already-computed, sealed results. No figure requires new inference, new models,
or new statistics. Panels cite the artifact CSV/JSON they render.

## Figure 1 — Conceptual framework
Schematic (no data inference): perception-support subspace → target-imagery geometry, showing (a) shared
within-support component, (b) outside-support residual (D = 2), (c) subject-specific orientation of the
residual. Purely explanatory; frames Results 1–2.

## Figure 2 — Where does perception fail?
*From O2.5, O2.6.* Panel A: perception-support ceiling vs native oracle per ROI (bar/point). Panel B: low-
dimensional missing-basis spectrum (outside-support dimensionality). Message: a real, low-dimensional residual
lies beyond perception support.

## Figure 3 — Subject-specificity
*From O2.10, O2.11, O2.13.* Panel A: cross-subject donor transfer near null (O2.10 sharedness). Panel B:
perception-only prediction of target orientation fails (O2.11). Panel C: covariate-corrected prediction fails
(O2.13). Message: the residual orientation is participant-specific and not recoverable from the tested
predictors.

## Figure 4 — Minimal calibration frontier
*From O2.9, O2.12.* Panel A: M × T recovery grid (median TOTAL) per ROI with the M4T2 / 8-observation cell
highlighted; annotate ventral 0.569 (7/8), lateral 0.530 (6/8), FCF 0.596/0.591 (8/8). Panel B: O2.12
perception-prior improves recovery but does not reduce the eight-observation minimum. Message: eight
observations is the smallest common frontier under the tested estimator.

## Figure 5 — Schedule fragility and geometric quality
*From O2.14, O2.15.* Panel A: schedule-coverage distribution per ROI (median 0.656 ventral, 0.579 lateral) —
fragility. Panel B: schedule-level Q_OUT vs robustness margin, participant medians (ρ 0.423 ventral, 0.477
lateral; 8/8). Message: outside-support repeat geometry predicts calibration success.

## Figure 6 — Mechanistic constraints (exploratory; compact multi-panel)
*From X1–X4.* Panel A: X1 invariant survival — B_angles positive, other blocks pipeline-trivial. Panel B: X2
held-out K2−K1 (−0.113 ventral, −0.100 lateral; 0/8) — no discrete states. Panel C: X3 scalar reliability →
margin (ρ 0.004 / 0.064) — negative. Panel D: X4 ANISO_CENTRED (+0.093 / +0.101; 8/8, Holm) positive vs
MODE_GAP→margin (−0.128 / −0.099) negative. Message: repeat variability is structured but distributed.

## Supplementary figures (illustrative, not exhaustive)
- S1 Provenance/freeze timeline (config SHA → server-verify → compute → seal per gate).
- S2 O2.9 full M × T grids per participant.
- S3 O2.14 leave-one-fold-out and jackknife stability.
- S4 O2.15 Q_IN vs Q_OUT comparison; within- vs outside-support.
- S5 X4 isotropic-null vs real top-fraction; cross-half reproducibility per fold.
- S6 Null distributions and pipeline-triviality controls for X1/X3/X4.
