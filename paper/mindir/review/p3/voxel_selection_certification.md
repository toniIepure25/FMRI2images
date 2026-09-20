# Voxel-selection & ROI certification (P3 Methods closure)

**Verified against sealed artifact** `artifacts/mindcompiler/roy_s2_2b/roi_selection_manifest.json`
(`phase = PHASE_1_FROZEN_BEFORE_OUTCOME`) and `roy_s1/subj01_spatial_alignment.json`. No new analysis.

## Selection policy (verbatim sealed field)
`selection_policy = "ROI mask ∩ valid_nsdimagery ∩ finite NSD-core ncsnr; strict > ROI-specific 98th pct;`
`identical voxels for B0/B1; NO threshold adaptation"`
- `pattern_reliability_min_voxels = 3`
- `ncsnr_source = "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz"` (per-subject path; the
  manifest example is subj01 — do **not** extrapolate subj01 threshold values to the cohort).

## Certified Methods wording (approved)
> For each ROI, eligible voxels were the intersection of the ROI mask, valid NSD-Imagery voxels, and voxels with
> finite NSD-core noise-ceiling SNR (ncsnr, from `…/func1pt8mm/betas_fithrf/ncsnr.nii.gz`). Within each ROI we
> selected voxels with ncsnr **strictly greater than the ROI-specific 98th percentile** of the eligible
> distribution (percentile computed separately per ROI; strict `>` comparator; the same selected voxels used
> for the B0/B1 preparation comparison; no threshold adaptation after outcomes). This is a per-ROI strict-98th-
> percentile rule on an explicitly defined eligible set — **not** a generic "top 2% of voxels."

## ROI definitions (verbatim sealed labels)
- Early visual (`prf-visualrois`): **V1** = labels 1,2; **V2** = 3,4; **V3** = 5,6; **hV4** = 7.
- Streams atlas: **ventral** = label 5; **lateral** = label 6; **parietal** = label 7.
- Primary analyses use ventral + lateral (streams); parietal reserved as secondary; V1 is a reference/discovery
  ROI excluded from the primary set. (Note: label "5" means V3 under prf-visualrois but ventral under streams —
  the atlas must always be named.)

## Spatial space (verbatim sealed fields)
- `nifti_shape = [81,104,83]`, `voxel_mm = [1.8,1.8,1.8]`, single shared affine across all NIfTI artifacts.
- B0 HDF5 `b0_spatial_shape_in_hdf5 = [83,104,81]`, `correspondence = "axis-reversed (Z,Y,X); direct voxel`
  `mapping betas[:,z,y,x]; NO resampling"`, `status = "ALIGNMENT_PASS"`.
- Certified wording: native **`func1pt8mm`** 1.8-mm functional grid; **no spatial resampling** for the relevant
  O1/O2 representation; the HDF5 imagery layout uses **reversed storage axes** relative to NIfTI — a data-layout
  convention, **not** spatial interpolation.

## subj01 QC example (implementation example ONLY — not cohort-wide counts)
| ROI (streams) | eligible voxels | 98th-pct threshold | selected |
|---|---|---|---|
| ventral | 7604 | 0.62417 | 153 |
| lateral | 7799 | 0.80783 | 156 |
| parietal | 3548 | 0.54147 | 71 |

These subj01 numbers may appear only as an implementation/QC example; per-subject counts and thresholds differ.

**Status: RESOLVED — voxel-selection + ROI + spatial placeholders closed against sealed evidence.**
