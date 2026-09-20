# Subject-specific geometry and calibration limits in human visual imagery
### (MINDIR discovery manuscript — draft for co-author review)

**Cohort classification.** Historical **N = 8** = **DISCOVERY / DEVELOPMENT COHORT**. Independent replication
(O2.16) = **PREREGISTERED / FROZEN BUT NOT YET EXECUTED**. Nothing here is a completed replication.

**Central question.** *How much of the human perception→imagery transformation is shared, how much is
subject-specific, and what limits minimal target-state calibration?*

This document assembles the manuscript package. Full sections live in companion files:
`ABSTRACT.md`, `METHODS.md`, `RESULTS.md`, `DISCUSSION.md`, `LIMITATIONS.md`, `FIGURE_PLAN.md`,
`SUPPLEMENT.md`, `replication_box.md`, `reviewer_risk_register.md`, `claims_matrix.csv`, `evidence_table.csv`.

---

## Abstract
See `ABSTRACT.md` (structured, ~300 words; Background / Methods / Results / Interpretation). It distinguishes
discovery cohort, operational findings, exploratory mechanistic findings, and pending independent replication.

## Introduction (outline)
1. Imagery and perception engage overlapping visual cortex, but **similarity does not imply identical
   geometry**; recent work frames imagery as a transformation of perceptual activity (Saha Roy et al., 2025;
   cf. general neural transformations, Ward, Isik & Chun, 2018).
2. Open questions this leaves: cross-subject sharedness of the transformation (cf. hyperalignment, Haxby et al.,
   2011/2020; SRM, Chen et al., 2015); which target-state components lie outside perception support; the minimal
   subject-specific calibration to recover them; and the robustness of that calibration. Vision-trained
   *decoding* of imagery is established (MIRAGE, Kneeland et al., 2026) but concerns brain→image reconstruction
   rather than the brain-state→brain-state transformation studied here.
3. This project began as an **independent method reproduction and extension** of Saha Roy et al.; original
   code/seeds for a bitwise replication are unavailable.
4. Framing (no priority claim): *rather than asking only whether perception predicts imagery, we ask which
   components of imagery geometry perception fails to specify, how those components vary across participants, and
   how much direct imagery data are required to recover them.* We study, in an N = 8 discovery cohort with
   prospective freezes: (i) how much target geometry lies outside perception support; (ii) whether it is shared
   or subject-specific; (iii) the minimal direct calibration; (iv) its schedule robustness; (v) the geometric
   predictor of calibration success; and (vi) the structure of the repeat variation (exploratory).

*References for the positioning above are verified in `review/p3/literature_primary_source_audit.csv`.*

## Results
See `RESULTS.md`. Five sections:
1. Perception predicts imagery structure but does not fully determine target-state geometry (O1/O2/O2.1/O2.2/O2.5).
2. The missing imagery geometry is low-complexity but subject-specific (O2.6/O2.10/O2.11/O2.13).
3. Minimal direct target-state calibration recovers substantial geometry — 8 observations at M4T2 (O2.9/O2.12).
4. The eight-observation frontier is an average result, not a robust guarantee (O2.14/O2.15).
5. *Exploratory mechanistic analyses:* the failure-predictive repeat geometry is structured but distributed
   (X1/X2/X3/X4).

## Discussion
See `DISCUSSION.md` (A–F): imagery ≠ perception replay; a subject-specific but low-complexity target component;
calibratable from few observations; quality governed by repeat geometry; that geometry is distributed;
replication is the critical next step.

## Limitations
See `LIMITATIONS.md` — N = 8; discovery cohort; repeated-cohort exploratory reuse; fMRI limits; estimator-
specific minimum; ROI-specific; no independent replication yet; no causal inference; no bitwise Roy replication;
bounded trial-level analyses; imagery-task provenance partly unresolved.

## Independent replication
See `replication_box.md`. O2.16 primary (frozen, not executed) + O2.16-SEC secondary endpoints (S1 B_angles,
S2 Q_OUT→margin, S3 ANISO) preregistered. No independent participant acquired.

## Methods
See `METHODS.md` (17 numbered sections). Participant is the inferential unit; exact sign-flip + Holm; prospective
freezes with server-verify; estimator replay to ≤ 1e-10.

## Data and code availability
Artifacts under `artifacts/mindcompiler/<gate>/` with per-gate `hashes.json` / `execution_provenance.json` /
`test_report.json`. NSD is a public dataset; no raw beta files are redistributed here.

## Author contributions / competing interests
(To be completed by the human authors.)
