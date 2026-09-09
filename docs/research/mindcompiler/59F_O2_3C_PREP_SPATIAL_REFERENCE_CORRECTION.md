# O2.3C-PREP-SPATIALFIX — Mean-EPI Spatial-Reference Correction

**Analysis class:** `PROSPECTIVE_CORRECTION_OF_CONTRAST_MISMATCHED_SPATIAL_REFERENCE`
**Source HEAD:** `1d580a9` · **Frozen config SHA:** `5e3954d9`

Corrects **one** spatial-registration design flaw and nothing else: the BENCHFIX bridge registered the
candidate against `R2.nii.gz` (a GLM variance-explained map — contrast-mismatched for EPI→EPI
registration), which yielded strong ROI-mean temporal fidelity (0.945–0.988) but failed voxel-level gates
(Dice 0.846, spatial r 0.851, median voxelwise temporal r 0.374). This gate replaces the fixed reference
with the official NSD **mean-EPI** and uses an EPI→EPI registration. The choice is fixed **before** any
new metric — not a threshold change, registration leaderboard, or post-outcome optimization.

## The single correction (frozen)

- **Fixed reference:** `NSD_FUNC1PT8MM_MEANFIRST5` — `nsddata/ppdata/subj01/func1pt8mm/meanFIRST5.nii.gz`
  (SHA256 `d94d96db…`, 1,328,438 B). The NSD Data Manual documents `meanFIRST5` (a mean EPI) as used in
  co-registration. `mean.nii.gz` is inventoried only, **not** a competitor; `R2` is **not** used.
- **Moving image:** `TEMPORAL_MEAN_OF_EXISTING_FMRIPREP_DESC_PREPROC_BOLD` — the mean over all 188
  candidate volumes (no temporal filtering, no GT involvement). This makes it a **mean-EPI → mean-EPI**
  registration.
- **Candidate:** the **byte-identical** BENCHFIX/R4 derivative (`desc-preproc_bold` SHA256 `3838946f…`);
  fMRIPrep is **not** rerun.
- **Transform:** ANTs **rigid+affine**, Mattes MI, no nonlinear/SyN/B-spline — pyramid/convergence/shrink/
  smoothing/init reused **byte-for-byte from BENCHFIX**. Interpolation unchanged (BOLD `LanczosWindowedSinc`,
  masks `NearestNeighbor`).
- **Temporal correction:** reused **exactly** from BENCHFIX and **not reopened** — candidate effective
  timestamps `0.8 + 1.6·n`, GT-only cubic (`scipy CubicSpline`) resample to those timestamps, no lag search,
  candidate never resampled.

## No-optimization rule

The only evaluated pipeline is **candidate temporal mean → meanFIRST5 → rigid+affine**. No selection among
rigid-only / affine-only / `mean.nii` / `R2` / T1 / other metric / other interpolation. If it fails, the
gate fails.

## Frozen quality gates (thresholds UNCHANGED from BENCHFIX)

Geometry (`81×104×83×188`, affine) PASS required; brain-mask Dice ≥ 0.90; mean-BOLD spatial r ≥ 0.95;
ROI-mean temporal r ≥ 0.90 (ventral/lateral/parietal = NSD streams 5/6/7); median voxelwise temporal
r ≥ 0.70; tSNR ratio ∈ [0.50, 2.00]. Same ROI/mask definitions as BENCHFIX.

## Status & next gate

Exactly one of `O2_3C_PREP_SPATIALFIX_{PASS,FAILURE,REFERENCE_UNAVAILABLE,TECHNICAL_FAILURE}`; PASS needs
**all** gates (no partial pass). On **PASS** → `O2_3C_PREP_TASK_FIDELITY_CERTIFIED_AFTER_PROSPECTIVE_REFERENCE_CORRECTION`,
then cohort Phases 2–4 (same runtime + frozen temporal + meanFIRST5 bridge, no imagery) → `O2_3C_PREP_PASS`
→ `O2.3C-RESUME` (`528f23eb`). On **FAILURE** → close the fMRIPrep-based PREP path
(`FMRIPREP_DERIVED_REST_FUNC1PT8MM_NOT_CERTIFIED_FOR_FINE_SCALE_CONNECTIVITY`), no further preprocessing-method
search, return to the scientific-program decision. Historical failures remain immutable; `O2 = SHARED_OPERATOR_PARTIAL`,
`O2.3A`, `O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O3 = O3_NOT_READY` preserved.
