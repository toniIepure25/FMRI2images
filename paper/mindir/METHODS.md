# Methods

Numbered to match the required Methods structure. This is a synthesis of the sealed pipeline; no analysis was
re-run. Exact parameters and nulls are in `SUPPLEMENT.md`; provenance in `provenance_manifest.json`.

## 1. Dataset and participants
Natural Scenes Dataset (NSD; Allen et al., 2022). **N = 8** participants, treated as a **discovery / development
cohort** and as the sole inferential unit. Perception and imagery conditions as released by NSD / NSD-Imagery.

## 2. Perception and imagery tasks
Perception: viewed natural scenes. Imagery: cued mental imagery of a fixed identity set. The imagery-task field-
level provenance used for the future replication is documented from sealed NSD-Imagery provenance (see
`replication_box.md`); some run/session-layout details remain `O2_16_DATA_IMAGERY_TASK_PROVENANCE_BLOCKER` and
are stated as such.

## 3. ROI definitions
ROIs were frozen before outcomes (`roi_selection_manifest.json`, `PHASE_1_FROZEN_BEFORE_OUTCOME`). Early visual
ROIs came from the `prf-visualrois` atlas (V1 = labels 1,2; V2 = 3,4; V3 = 5,6; hV4 = 7); the ventral, lateral,
and parietal ROIs came from the `streams` atlas (ventral = label 5, lateral = label 6, parietal = label 7). The
atlas is always named because label "5" denotes V3 under `prf-visualrois` but ventral under `streams`. The two
**primary** ROIs are the **ventral** and **lateral** streams; **parietal** was reserved as a post-seal
secondary and is not a primary conclusion, and V1 served only as a reference/discovery ROI.

## 4. Neural response estimation and voxel selection
Single-trial betas were the NSD-Imagery **B0** preparation, `nsdimagerybetas_fithrf`, corresponding to the
public NSD **`betas_fithrf`** (b2-compatible) preparation; the public description is "similar to b2," so we do
**not** claim the released public file is bitwise identical to the file used in the original Roy et al.
analysis. A GLMdenoise/RR (b3-compatible) preparation (`…_GLMdenoise_RR`, branch B1) was examined only as a
measurement-preparation **sensitivity** branch and is **not** the primary preparation. All analyses were in the
native **`func1pt8mm`** 1.8-mm functional grid (NIfTI shape 81×104×83, single shared affine); there was **no
spatial resampling** for the O1/O2 representation, and the HDF5 imagery layout uses reversed storage axes
(Z,Y,X) relative to NIfTI — a data-layout convention, not spatial interpolation.

**Voxel selection (frozen before outcomes).** For each ROI, eligible voxels were the intersection of the ROI
mask, valid NSD-Imagery voxels, and voxels with finite NSD-core noise-ceiling SNR (ncsnr, from each subject's
`…/func1pt8mm/betas_fithrf/ncsnr.nii.gz`). Within each ROI we retained voxels with ncsnr **strictly greater
than the ROI-specific 98th percentile** of that eligible distribution (percentile computed separately per ROI;
strict `>` comparator; the same selected voxels used for the B0/B1 comparison; no threshold adaptation after
outcomes). This is a per-ROI strict-98th-percentile rule on an explicitly defined eligible set, not a generic
"top 2% of voxels." Thresholds and counts are per subject (for example, in subj01: ventral 7604 eligible →
threshold ≈ 0.624 → 153 selected; lateral 7799 → ≈ 0.808 → 156; these are implementation examples, not
cohort-wide counts). We note voxel selection could in principle shape the outside-support residual, and flag
this as a bounded interpretation (Limitations). BLAS threads pinned to 1 for determinism; all heavy computation
executed on a fixed CPU node under committed source (ConfigMap-injected), never with ad-hoc code.

## 5. Perception support construction
A perception-support subspace was constructed per participant × ROI and frozen as `W_target` (orthonormal
within-support basis) plus a fixed **D = 2** outside-support basis, with orthogonal ranges (composite projector
P = P_in + P_out). No Procrustes alignment.

## 6. Perception→imagery operator representation
The perception→imagery transformation was represented on the frozen support geometry (direct-SVD within-support,
q = min(r_best, M); fixed two-dimensional outside-support component). All ranks/dimensions frozen before
outcomes.

## 7. Cross-subject shared-operator analyses
Donor-transfer / leave-one-subject-out analyses tested whether the composite target-state geometry transfers
across participants (O2.10). **Cross-subject comparisons were made only on coordinate-invariant or donor-mean
quantities — never on raw native voxel spaces, which are not directly comparable across participants.**
Consequently, the "no reliable cross-subject transfer" result cannot be an artifact of incomparable voxel
spaces; it is a statement about the invariant quantities that *are* comparable.

## 8. Target-state residual decomposition
The target-imagery state was decomposed into within-support and outside-support components. The residual
(outside-support) geometry was quantified against the perception-support ceiling and the native oracle (O2.5,
O2.6), and its dimensionality assessed.

## 9. Minimal target-state calibration
An M-identities × T-repeats grid (balanced identity subsets; repeat-position subsets) was fit with the **exact
frozen estimator** from each schedule alone and evaluated on held-out identities (O2.9). A perception-only prior
control (O2.12) tested whether a prior reduces the minimum.

## 10. Recovery metrics
Per cell: **TOTAL recovery** = (R − R0)/(R_native − R0) and **full-capacity fraction (FCF)** = (R − R0)/(R111 −
R0). A cell is sufficient iff median ≥ 0.50 with ≥ 6/8 participants ≥ 0.50 on both metrics (frozen before
outcomes). `N_TRIALS_STAR` is the smallest common-sufficient burden across both ROIs.

## 11. Schedule robustness
For each schedule (one identity subset × one repeat-position subset) the exact estimator was refit and scored;
**schedule coverage** = fraction of schedules meeting the joint sufficiency criterion. A cell is schedule-robust
only if the average criterion holds AND median coverage ≥ 0.75 with ≥ 6/8 ≥ 0.75 (frozen). Leave-one-fold-out
and participant jackknife diagnostics accompany it (O2.14).

## 12. Repeat-geometry diagnostics
Outside- and within-support repeat-to-repeat agreement metrics (Q_OUT, Q_IN) were related to the O2.14
robustness margin by participant-level Spearman correlation with the historical fold/Fisher-z procedure (O2.15).
Participant is the inferential unit; schedules are not.

## 13. Exploratory invariant / state / drift analyses
Four exploratory gates on the same cohort: coordinate-invariant blocks with a constrained-random-operator
pipeline-triviality control (X1); Gaussian / two-state / Student-t held-out modelling (X2); principal-angle
stability and scalar-reliability→margin (X3); repeat projector-perturbation anisotropy, isotropic matched null,
cross-half reproducibility, and a cross-fit drift-mode→margin test (X4). Each froze its config before outcomes.

## 14. Statistical inference
Participant-level **exact one-sided sign-flip** tests over 2^8 = 256 sign assignments (minimum one-sided p =
1/256 ≈ 0.00390625) with participant-first aggregation. Ratio-type effect statistics were centred so the null
expectation is 0 before entering a sign-flip test (X4 disclosure).

## 15. Multiple-testing correction and statistical scope
Holm correction within each gate's prospectively frozen test family (e.g. X4: exactly four tests; X-secondary
preregistration: exactly six). No cross-gate pooling; each family declared before outcomes.

**Statistical scope and multiplicity (explicit).** The inferential unit is the participant (N = 8); no analysis
treats folds, schedules, identity subsets, or repeats as independent subjects. Correction is applied *within*
each gate's declared family; the **program-wide** multiplicity across gates is **not** corrected, because the
program is a sequential discovery process on one cohort. This is precisely why the three surviving findings are
preregistered for an independent cohort (O2.16-SEC) rather than treated as confirmed, and why the X-series is
labelled exploratory. Sufficiency criteria (O2.9 median ≥ 0.50 with ≥ 6/8; O2.14 coverage ≥ 0.75 with ≥ 6/8)
are **pre-registered decision rules, not hypothesis tests**, and carry no p-value.

## 16. Leakage prevention and prospective freezes
Every gate froze its config (SHA-hashed), driver, and tests **before** any outcome, committed and pushed, then
server-verified (remote HEAD == local HEAD) before compute. Reused estimators replay upstream results to ≤ 1e-10
(e.g. O2.14/O2.15/X4 replay O2.9 margins to ≤ 5.6e-17). Outside-support leakage was monitored.

## 17. Reproducibility and provenance
All results trace to committed artifacts under `artifacts/mindcompiler/<gate>/`, each with a `hashes.json`,
`execution_provenance.json`, and `test_report.json`. Seal commit SHAs are listed in `provenance_manifest.json`
and `evidence_table.csv`. No raw beta files are committed; no GPU was used for MINDCOMPILER.
