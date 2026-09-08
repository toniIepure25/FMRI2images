# O2.3C-PREP-RUNTIME — R3/R4 Execution Result on orchestraiq (ORCHESTRAIQ_DIRECT_K8S)

**Source HEAD:** `b27ea3b` · **Runtime:** official `nipreps/fmriprep@sha256:15cbf8dc…` as a native K8s
batch Job on `k8s-worker-cpu-node-r770`. Technical runtime only — no scientific methodology changed.

## Runtime outcome (SUCCESS end-to-end)

The `ORCHESTRAIQ_DIRECT_K8S` path executed fully: `o23c-` PVCs (local-path, r770), FreeSurfer Secret
(hash `6f7afab5…`), staged subj01/ses-nsd01/run-01 inputs from public S3, pulled the pinned official
image, and **fMRIPrep finished successfully (exit 0, crashfiles = 0)** with the complete derivative set
(preproc BOLD in func space, HMC, phasediff SDC, boldref→T1w coreg, confounds/motion).

## R3 — `PERSISTENT_NIPYPE_RESUME_CERTIFIED`

The run completed before a mid-run pod delete could be staged, so resume was certified with a **fresh
identical Job** (new pod UID `9d86f7cc…` ≠ original `aa24e087…`, same `/work` PVC on r770, same image
digest + command). Two evidence classes: (A) the replacement finished in **~260 s vs the original
~35 min**, re-executing only 1 report node (`conf_plot`), every heavy compute node reused from cache;
(B) 4 pre-sealed completed-node result caches were **byte-identical** afterward (same SHA256 **and** same
nanosecond mtime) — reused, not regenerated, no corruption.

## R4 — `O2_3C_PREP_TASK_BENCHMARK_FAILURE`

Measuring the actual products revealed a decisive, frozen-metric-level incompatibility:

| | spatial grid | voxel | volumes | TR |
|---|---|---|---|---|
| Candidate (fMRIPrep `preproc_bold`, func) | 120×120×84 | 1.8 mm | **188** | **1.6 s** |
| Ground-truth (NSD prepared func1pt8mm `timeseries_session01_run01`) | 81×104×83 | 1.8 mm | **226** | **1.3333 s** |

NSD's prepared product is **temporally upsampled** to 1.3333 s (226 = 188 × 1.6/1.3333, ×1.2). The frozen
O2.3C-PREP methodology performs **no** temporal resampling (`temporal_cleaning_here: false`,
`no_resampling_beyond_single_design`), so the candidate is intrinsically 188 vols @ 1.6 s. Consequences,
using **only** the frozen metrics/guards:

- **Geometry (4D shape equality): FAIL** — candidate `(81,104,83,188)` vs ground-truth `(81,104,83,226)`.
- **Temporal-alignment guard: FAIL** — 188 ≠ 226 volumes, 1.6 s ≠ 1.3333 s (a deterministic NSD
  resampling convention, reported not "optimized" — no lag search, no resample).
- **Temporal-agreement thresholds** (ROI r ≥ 0.90, voxel r ≥ 0.70) are **unmeetable**: the two series live
  on incompatible temporal grids, and the only way to make them comparable is a temporal resampling that
  is outside the frozen methodology and explicitly forbidden.

Therefore the frozen PASS contract cannot be satisfied. Downstream spatial metrics (Dice, mean-BOLD
spatial r, tSNR) were **not computed** — non-determining given the failing geometry/temporal gate, and the
only imagery-independent func1pt8mm reference (`R2.nii.gz`) is contrast-mismatched for a fair spatial
registration; no transform was applied (so no inverse-direction risk).

**This is a prepared-product temporal-grid incompatibility surfaced at execution — not a preprocessing
deficiency.** fMRIPrep ran correctly and the runtime path is fully validated. It echoes the prior
`O2_3C_REST_PRODUCT_INCOMPATIBLE` pattern, now for the task product. Per the frozen contract this is the
authorized non-PASS terminal.

## STOP (per brief) — nothing tuned, nothing rescued

No threshold weakening, no temporal resampling / lag search, no alternate registration, no different
fMRIPrep version, no changed flags, no AWS. **Cohort Phases 2–4 NOT started** (authorized only on PASS).
**`O2.3C-RESUME` (SHA `528f23eb`) NOT run** (requires `O2_3C_PREP_PASS`).

## Immutable scientific state — preserved

`O2 = SHARED_OPERATOR_PARTIAL`, `O2.3A = CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE`,
`O2.3C (first) = O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O3 = O3_NOT_READY`. Phase-0 `2d1a26a5`, contract
`24e6ca83`, and blockers `fbab942`/`4377100` unchanged.
