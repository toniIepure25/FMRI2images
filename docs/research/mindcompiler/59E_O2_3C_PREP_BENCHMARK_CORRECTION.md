# O2.3C-PREP-BENCHFIX — Temporal-Grid-Corrected Task Benchmark Comparator

**Analysis class:** `PROSPECTIVE_CORRECTION_OF_INVALID_TECHNICAL_BENCHMARK_COMPARATOR`
**Source HEAD:** `c3143c1` · **Frozen config SHA:** `4b384f31`

Repairs **only the benchmark comparator**. Does not modify fMRIPrep, preprocessing, registration, SDC,
slice-time correction, the spatial-mapping policy, the candidate derivative, or scientific Track-O
methodology. The historical `O2_3C_PREP_TASK_BENCHMARK_FAILURE` is **permanent and not rewritten**.

## Why the historical comparator was invalid

The historical R4 compared samples that did not represent the same temporal grid:

- **Candidate** (fMRIPrep 25.2.5): 188 volumes at TR = 1.6 s. fMRIPrep's slice-time correction (AFNI
  `3dTshift`) aligns to the **middle of each TR**, so the candidate's effective signal timestamps are
  `t_candidate[n] = 0.8 + 1.6·n`, n = 0…187 → **0.8 … 300.0 s**.
- **Official NSD prepared** product: 226 volumes cubic-resampled to **4/3 s** (= 1.33333…), first sample at
  the start of the first RF pulse, so `t_nsd[m] = m·(4/3)`, m = 0…225 → **0.0 … 300.0 s**.

All candidate effective timestamps lie inside the NSD temporal support, so the two products can be placed
on one grid by resampling the reference — **not** by touching the candidate.

## The correction (frozen before any metric)

- **Direction:** resample **only** the official NSD GT → the candidate effective timestamps
  (`NSD_GT_MATCHED_1P6_EFFECTIVE`, 188 volumes). The candidate is **never** temporally resampled or
  reprocessed (byte-identical to the R4 product; fMRIPrep is not rerun).
- **Interpolation:** one deterministic `scipy.interpolate.CubicSpline` (not-a-knot, `extrapolate=False`),
  input `arange(226)·(4/3)`, query `0.8 + arange(188)·1.6`, in-bounds required.
- **No lag search:** the 0.8 s offset is **prospectively derived** from fMRIPrep's documented middle-of-TR
  target — never estimated. No ±TR shift, fractional-offset/phase search, sample dropping, DTW, or warping.
- **Synthetic certification (Part E):** cubic interpolation is validated on constant / linear / 0.01 /
  0.05 / 0.10 Hz sinusoids against the analytic values at the candidate timestamps before any candidate
  metric.
- **Spatial mapping (Part G/H):** the already-frozen policy the historical R4 stopped before — ANTs
  rigid+affine, Mattes MI, moving = fMRIPrep native BOLD reference, fixed = imagery-independent func1pt8mm
  reference (`R2.nii.gz`), BOLD `LanczosWindowedSinc`, atlas `NearestNeighbor`, no nonlinear, GT never used
  for registration → candidate at `81×104×83×188`.

## Corrected geometry + frozen quality gates (thresholds UNCHANGED)

After (candidate spatial mapping) + (GT temporal harmonization), both products are `81×104×83×188` with
identical spatial affine. Then the **unchanged** gates: brain-mask Dice ≥ 0.90; mean-BOLD spatial r ≥ 0.95;
ROI-mean temporal r ≥ 0.90 for ventral/lateral/parietal (NSD streams 5/6/7); median voxelwise temporal
r ≥ 0.70; tSNR ratio ∈ [0.50, 2.00] (candidate tSNR on its 188 samples vs matched-GT tSNR on 188).

## Status & relation to history

Exactly one of `O2_3C_PREP_CORRECTED_BENCHMARK_{PASS,FAILURE,TECHNICAL_FAILURE}`. Even on corrected PASS,
`O2_3C_PREP_TASK_BENCHMARK_FAILURE` is preserved as historical truth and reported as
`HISTORICAL_BENCHMARK_INVALID_FOR_PIPELINE_QUALITY_DUE_TO_TIMEGRID_MISMATCH` — the corrected benchmark does
**not** turn the old result into a pass; it answers a newly well-posed question.

**Interpretation boundary:** a PASS means only that the fMRIPrep-derived product is sufficiently
technically faithful (after the known deterministic timing difference) for the planned resting-connectivity
analysis. It does **not** mean fMRIPrep reproduces NSD preprocessing, that outputs are bitwise equivalent,
or that the pipelines are the same. Immutable: `O2 SHARED_OPERATOR_PARTIAL`, `O2.3A`,
`O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O3_NOT_READY`, Phase-0 `2d1a26a5`, contract `24e6ca83`, methodology `528f23eb`.
