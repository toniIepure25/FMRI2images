# O2.3C-PREP — Raw Resting-State Preprocessing, Native func1pt8mm Product Certification, DK Target Mapping

**Gate:** O2.3C-PREP (TECHNICAL_PREPROCESSING_AND_REGISTRATION_CERTIFICATION)
**Input HEAD:** `e215f3f` · **Frozen config SHA:** `2d1a26a5` · **Resumes frozen O2.3C SHA:** `528f23eb`
**Phase-0 status:** `PHASE0_FROZEN` · **Terminal execution status:** *pending pod compute (not issued)*

This gate builds and certifies only the **data products** the already-frozen O2.3C methodology
requires. It performs **no** connectivity, cSRM, orientation test, or O2.3C scientific QC. Historical
`O2_3C_REST_PRODUCT_INCOMPATIBLE` is preserved unchanged.

## Part A — Reconcile (verified)

| Item | State |
|---|---|
| Branch | `research/mindcompiler-neural-state-operators` |
| Local HEAD = remote HEAD | `e215f3f` = `e215f3f` ✓ |
| Tracked tree | clean ✓ |
| Ancestry from `e215f3f` | HEAD is `e215f3f` ✓ |
| Frozen O2.3C config SHA | `528f23eb` recomputes ✓; `O2_3C_REST_PRODUCT_INCOMPATIBLE` intact ✓ |
| Raw rest inventory (e215f3f) | subj01/05 = 36 runs; others = 20; 188 TR × 1.6 s; raw grid `[120,120,84]` |
| Raw fieldmaps | present (phasediff + magnitude1/2 per run) ✓ |
| Prepared main-NSD func1pt8mm timeseries | present (`timeseries_sessionNN_runMM`) ✓ |
| FreeSurfer / DK source | `aparc+aseg.mgz` present ✓ |
| Released transform assets | `anat*-to-func1pt8.nii.gz` present (nsdcode-applicable) ✓ |
| MATLAB runtime | **absent** (local); not expected on pod |
| Container runtime | **absent** (local: no docker/singularity/apptainer) |
| Pod | **network-unreachable this session** (port 22 timeout ×3) |

## Part B — Public preprocessing provenance audit

All five public NSD preprocessing repositories are reachable and pinned
(`author_preprocessing_audit.json`):

| Repo | HEAD SHA | Runtime |
|---|---|---|
| cvnlab/nsddatapaper | `80e891b` | MATLAB |
| kendrickkay/preprocessfmri | `ab09d30` | MATLAB |
| kendrickkay/alignvolumedata | `b513116` | MATLAB |
| kendrickkay/knkutils | `c321b29` | MATLAB |
| cvnlab/nsdcode | `bfd36a5` | **Python** (transform applier) |

The `preprocess_nsd*.m` driver scripts (slice-time, motion, fieldmap/gradient-unwarp, EPI→reference
registration, single-resampling second-stage func1pt8mm) are **MATLAB**. Siemens gradient-nonlinearity
coefficients are **not** in the public release (vendor-proprietary).

## Parts C–E — Technical lane decision (frozen before any QC metric)

- **Lane A — NSD_AUTHOR_LINEAGE (preferred): INELIGIBLE.** Part D criterion A fails — the MATLAB
  runtime + licensed Kendrick-Kay toolchain is not available locally nor on the designated pod; and
  criterion D fails — gradient-nonlinearity vendor coefficients are not public. Recorded:
  `NSD_AUTHOR_LINEAGE_NOT_EXECUTABLE_FROM_PUBLIC_RELEASE`.
- **Lane B — FMRIPREP_BIDS_RECONSTRUCTION: ACTIVATED & FROZEN.** fMRIPrep **25.2.5**, official
  container (`nipreps/fmriprep`, digest recorded at pull), SDC from acquired phasediff fieldmaps (no
  SyN-SDC), slice-timing from BIDS, rigid motion, BOLD reference, **no** smoothing / AROMA / denoising
  / GSR, confounds exported. Raw→func1pt8mm via ANTs **rigid+affine** (Mattes MI; BOLD
  LanczosWindowedSinc; atlas NearestNeighbor; **no** nonlinear SyN rest→func).

## Frozen contracts (Part Y)

Committed before any preprocessing output:
`artifacts/mindcompiler/operator_o2_3c_prep/` — `o2_3c_prep_frozen_config.json` (SHA `2d1a26a5`),
`author_preprocessing_audit.json`, `preprocessing_provenance_contract.json`, `lane_selection_contract.json`,
`lane_selection_result.json`, `task_benchmark_contract.json`, `task_benchmark_manifest.csv`
(resolves to `session01_run01` for all 8), `func1pt8mm_mapping_contract.json`, `motion_contract.json`,
`tissue_mask_contract.json`, `DK_mapping_contract.json` + `DK_parcel_manifest.csv` (canonical 68),
`product_certification_contract.json`, `execution_provenance.json`, `hashes.json`.

**Benchmark pass thresholds (frozen):** geometry exact; brain-mask Dice ≥ 0.90; mean-volume spatial
r ≥ 0.95; ROI-mean temporal r ≥ 0.90; voxelwise temporal r ≥ 0.70; no subject primary-ROI voxel
r < 0.50; group tSNR ratio ∈ [0.50, 2.00]. On fail: STOP, no tuning, no lane switch.

## Phases 1–4 — compute (PENDING pod infrastructure)

The benchmark (Phase 1), rest preprocessing (Phase 2), motion/tissue/DK products (Phase 3), and cohort
certification (Phase 4) require the **container runtime on the pod**, which is network-unreachable this
session. **No benchmark or product metric is fabricated** — result artifacts are explicit
`PENDING_POD_COMPUTE` placeholders. This is a transient infrastructure hold, **not** a gate verdict:
`O2_3C_PREP_PUBLIC_ASSET_FAILURE` does not apply (all public assets are present) and
`O2_3C_PREP_LANE_SELECTION_FAILURE` does not apply (a lane was selected and frozen). The pod itself is
demonstrably capable — it executed the O2.3A 131 GB acquisition + compute — so this is a reachability
outage, not an environmental inability.

## Guards

`target_imagery_files_opened_by_PREP = 0`; the O2.2 residual is hash-checked only, never read for a
metric; no imagery-derived score influences any stage. Immutable statuses (TRACK_R, O1, O2, O2.2,
O2.3A, O2.3C first-attempt + orientation, O3) carried forward unchanged.

## Next action

When the pod is reachable with a container runtime: pull `nipreps/fmriprep:25.2.5` (record digest),
run Phase 1 benchmark on the frozen `session01_run01` selection, and — only on
`FMRIPREP_TASK_BENCHMARK_PASS` — proceed to Phases 2–4, then issue a terminal Part AC status. If the
benchmark fails: STOP with `O2_3C_PREP_TASK_BENCHMARK_FAILURE` (no tuning, no lane switch).
