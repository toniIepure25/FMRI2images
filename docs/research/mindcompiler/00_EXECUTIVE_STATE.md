# 00 — MINDCOMPILER Executive State

**Last updated:** 2026-07-17 · **Gate M0 — VERDICT ISSUED (§Verdict below)**
**Branch:** `research/mindcompiler-neural-state-operators`
**Bootstrap commit:** `3c4ebcd` (the earlier `c1095de` self-reference in this file and in `19`
was **stale**: it named the *parent* commit, not the commit that contained these files.
Corrected 2026-07-17.)
**Parent:** `feature/predictive-cortical-decoder` @ `c1095de` (PCD/NCD archived, not deleted)
**Active runs: NONE.** Pod idle. **No GPU has been spent on MINDCOMPILER.**

---

## O2.3A-RD / O2.4R (2026-09-09): PROSPECTIVE RE-DERIVATION -- BOTH CONFIGS FROZEN (RD `c1b2ddb0`, R4 `ad959446`)

Exact O2.3A replay is closed (PROVREC), so a NEW prospective re-derivation is authorized: `O2.3A-RD` (dense
perception -> subject-specific imagery-residual orientation, ZERO target imagery) then conditionally `O2.4R`
(calibration frontier). NEW analyses -- historical O2.3A/O2.4/PROVREC remain immutable; never relabel historical
O2.3A as reproduced. **Both configs frozen + committed + pushed BEFORE any outcome** (two-stage preregistration:
O2.4R frozen before seeing the new M=0). Every PROVREC-missing detail is now fully specified: NEW deterministic
SRM (SVD init, max_iter=200, tol=1e-7, float64, canonical gauge); K-blocks = anchor_order mod 5; rank inner-CV =
sorted-family identity-pair folds; null RNG = SHA256->first16hex->uint64->PCG64; tie rules <=1e-12; K{2,4,8,16,32,64};
r{1..6}; reuse frozen 512 anchor identity list (re-acquire DATA only) + O1/O2 B0 ROI voxels; immutable O2_2 small-span
residual; 6 outer folds; gauge cert <=1e-10. O2.4R: M{0,2,4,6,8,10} identities, subsets 25/100/100/25/1, orthogonal
Procrustes only, identity-correspondence null (100), N=8 sign-flip + Holm(10), M_STAR, "coherently positive" defined
now, M0==P_ZERO_RD certified, internal RD-M0->M>0 comparison (never vs unreconstructable historical). Access order
strict; no refreeze after M0; leakage asserts.

**Status: frozen (steps 1-3 done).** Next: commit generator source + data-free determinism/gauge certification, then
the data-acquisition (public S3 -> orchestraiq persistent, ~131GB b2 core + 7/8 imagery betas) + fit + frontier
(substantial cluster compute). Compute artifacts are honest PENDING placeholders; nothing fabricated. **O3 remains
`O3_NOT_READY`.** Artifacts: `operator_o2_3a_rd/` + `operator_o2_4r/` + `docs/.../62`. 10 data-free freeze tests pass.

## O2.4-PROVREC (2026-09-09): FORENSIC O2.3A REPLAY -- `O2_4_PROVREC_HISTORICAL_DETAIL_MISSING`

Forensic attempt to reconstruct + exactly replay the O2.3A machinery so O2.4 can reuse it (config `9e3060af`).
**P0 proven: the O2.3A generator source was never committed** -- execution commit `5f13e4a` added only artifacts+
tests; across ALL git history the only O2.3A `.py` are the two test files. All 21 historical O2.3A artifacts
inventoried+hashed (anchor list 512 present).

**Exact replay cannot be certified.** The frozen contracts specify the method FAMILY, not the numerically-
determining implementation. Genuinely-absent details precluding the required categorical-EXACT + scalar-1e-6
replay: (1) generator source; (2) DetSRM init/n_iter/convergence (fixes gauge -> all scalar metrics); (3)
K-selection nested-CV block partition (n_blocks=5 but no partition rule -> selected K); (4) null random-subspace
RNG (seed rule known, seed->subspace map absent -> median_null). Contract-compatible implementations diverge
beyond 1e-6 -> gate STOP (no tolerance-loosening, no closest-scorer pick). Compounded: **131GB b2 core anchor
data absent** (local subj01-only; 7a0f653 cache gone; pod 3.5G) + 7/8 NSD-Imagery betas absent.

**`O2_4_PROVREC_HISTORICAL_DETAIL_MISSING`.** Replay NOT run; no reusable state; **M=0 NOT certified; no target
imagery opened; nothing fabricated.** O2.4 stays `TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE`, stronger
reason `HISTORICAL_O2_3A_STATE_NOT_RECONSTRUCTABLY_REPRODUCIBLE` -- a reproducibility limitation, NOT evidence
against the calibration hypothesis. Historical O2.3A execution valid; **O3 remains `O3_NOT_READY`**. Resolution
(user's call): recover the original generator code and commit it, OR authorize a freshly-frozen re-derivation
gate (supersedes rather than bit-replays). Artifacts: `artifacts/mindcompiler/operator_o2_4_provrec/` + `docs/.../61`.
7 data-free tests pass. Preserves all immutable Track-O states.

## O2.4 (2026-09-09): TARGET-STATE CALIBRATION FRONTIER -- `TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE` (provenance)

Prospective identity-diversity sample-complexity design (config `321b42f9`) frozen+pushed BEFORE any outcome:
budgets M={0,2,4,6,8,10}, balanced subsets 25/100/100/25/1, 6 outer folds, **orthogonal-Procrustes-only**
estimator, identity-correspondence null (100 perms), N=8 sign-flip (2^8) + Holm over 10 primary tests, M_STAR
rule, claim + prior-art boundaries, zero-target closeout `ZERO_TARGET_IMAGERY_ORIENTATION_NOT_IDENTIFIED_UNDER_
TESTED_PUBLIC_ANCHORS` (bounded; NOT "impossible").

- **Method certified (data-free)**: gauge-invariance to machine precision (|dR_CAL|=6.7e-16); estimator recovers
  orientation to oracle in the identifiable case; leakage detector fires on injection. Design/estimator SOUND.
- **Real frontier NOT run -- PROVENANCE block**: O2.4 must REUSE the exact frozen O2.3A dense-perception common
  space (W_target/K/vision alignment) + residual-orientation template, and M=0 must reproduce O2.3A to hash-level
  agreement. Those are NOT present in reusable form -- **no in-repo O2.3A generating code** (only the O2 small-span
  phase code exists) and **no persisted intermediate matrices** (only summary CSVs). The gate FORBIDS rebuilding
  (no new core preprocessing / no refit). (O2.3A itself was validly executed via public S3; block is reusability,
  not raw data.) **No calibration outcome computed or fabricated; no target imagery opened.**

**`TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE`.** ventral/lateral M_STAR = NOT_REACHED. Resolution
(user's call, not auto-taken): persist+commit the O2.3A code+matrices to enable reuse, then re-run; or authorize a
fresh re-derivation gate. Artifacts: `artifacts/mindcompiler/operator_o2_4/` (frozen config, closeout, prior-art,
budget/fold/subset/template/procrustes contracts, synthetic_controls, gauge_certification, provenance, status) +
`docs/.../60`. 13 data-free tests pass. **O3 remains `O3_NOT_READY`.** Preserves TrackR/O1/O1.1/O2/O2.1/O2.2/O2.3A/
O2.3C/O2.3C-PREP immutable.

---

## O2.3C-PREP-SPATIALFIX (2026-09-09): MEAN-EPI REFERENCE CORRECTION -- `O2_3C_PREP_SPATIALFIX_FAILURE` -> fMRIPrep PREP path CLOSED

Prospective correction (config `5e3954d9`, frozen BEFORE metrics, pushed+verified) of the ONE spatial flaw:
R2 (variance map, contrast-mismatched) -> official NSD **meanFIRST5** mean-EPI (public S3, sha d94d96db..);
moving = temporal mean of the **byte-identical** candidate (mean-EPI->mean-EPI). Transform params + interpolation +
temporal correction reused byte-for-byte from BENCHFIX (no lag search; candidate never resampled/reprocessed).
Ran on r770 in the certified fMRIPrep image. No-optimization rule honored (single pipeline only).

- **Correction VALIDATED the R2 diagnosis**: voxelwise temporal r **0.374->0.673**, mean-BOLD spatial r
  **0.851->0.893**, ROI-mean temporal r **0.983-0.995**, registration sanity warped-vs-fixed r 0.942, tSNR 0.964.
- **Still FAILS fine-scale gates**: Dice **0.816**<0.90; spatial r **0.893**<0.95; voxel temporal r **0.673**<0.70.
  Residual gap consistent with sub-voxel geometric differences (gradient-nonlinearity + NSD's own coregistration)
  that only nonlinear/gradient-unwarp correction could close -- FORBIDDEN, NOT attempted.

**`O2_3C_PREP_SPATIALFIX_FAILURE` -> CLOSE fMRIPrep-based PREP path:
`FMRIPREP_DERIVED_REST_FUNC1PT8MM_NOT_CERTIFIED_FOR_FINE_SCALE_CONNECTIVITY`.** No mean.nii/nonlinear/
gradient-unwarp/T1/surface/version/threshold/AWS rescue. Cohort Phases 2-4 NOT started; O2.3C-RESUME (528f23eb)
NOT run. Return to the scientific-program decision. Author-asset inventory (provenance only): released anat/MNI<->func
warps AVAILABLE, per-session EPI coreg .mat + gradient-unwarp coeffs NOT public -> PARTIALLY_AVAILABLE. Artifacts:
`artifacts/mindcompiler/operator_o2_3c_prep_spatialfix/` + `docs/.../59F`. Bounded interpretation: fMRIPrep product is
ROI-faithful (region temporal r >=0.98) but NOT certified at the frozen voxel scale. Historical failures immutable;
preserves 2d1a26a5/24e6ca83/528f23eb/O2 SHARED_OPERATOR_PARTIAL/O2.3A/O2_3C_REST_PRODUCT_INCOMPATIBLE/O3_NOT_READY.

## O2.3C-PREP-BENCHFIX (2026-09-09): TEMPORAL-GRID-CORRECTED BENCHMARK -- `O2_3C_PREP_CORRECTED_BENCHMARK_FAILURE`

Prospective correction (config `4b384f31`) of the historically-invalid comparator (frozen BEFORE any metric,
pushed+verified). Reason known a priori: candidate = fMRIPrep 188@1.6s with middle-of-TR STC -> effective
timestamps 0.8+1.6n; NSD GT = 226@ exactly 4/3s. Fix resamples **GT-only** (scipy CubicSpline, not-a-knot,
no-extrapolate) to candidate timestamps; **candidate byte-identical to R4** (preproc_bold sha 3838946f..,
fMRIPrep NOT rerun); no lag search. Spatial map = frozen ANTs rigid+affine Mattes MI (moving=coreg_boldref,
fixed=R2 func1pt8mm), Lanczos/NN -> [81,104,83,188]. Ran in the certified fMRIPrep image on r770.

- **Temporal correction VALIDATED**: geometry now PASS ([81,104,83,188] both, affine equal); synthetic cubic
  interp PASS (max err 0.0083); **ROI-mean temporal r PASS** ventral 0.988 / lateral 0.980 / parietal 0.945;
  tSNR ratio 1.05 PASS. The historical fatal timegrid mismatch WAS the comparator flaw.
- **Genuine quality FAIL (3 frozen gates)**: brain-mask Dice **0.846**<0.90; mean-BOLD spatial r **0.851**<0.95;
  median voxelwise temporal r **0.374**<0.70. Signature (high ROI-mean r, low voxel r + sub-threshold Dice/
  spatial-r) = imperfect voxel-level spatial registration; frozen fixed reference R2 is a variance map
  (contrast-mismatched vs mean-BOLD; no independent mean-EPI ships in func1pt8mm). Per brief an alternate
  registration is FORBIDDEN and was NOT attempted.

**`O2_3C_PREP_CORRECTED_BENCHMARK_FAILURE` -> STOP PREP.** No tuning/interp-change/alt-registration/fMRIPrep-
change/AWS. Cohort Phases 2-4 NOT started; O2.3C-RESUME (528f23eb) NOT run. Historical
`O2_3C_PREP_TASK_BENCHMARK_FAILURE` preserved (corrected benchmark does NOT turn it into a pass). Artifacts:
`artifacts/mindcompiler/operator_o2_3c_prep_benchfix/` (frozen config, temporal/spatial contracts, candidate/gt
immutability, synthetic test, transform+matched-grid cert, corrected metrics+status) + `docs/.../59E`. 25 benchfix
tests pass. Preserves 2d1a26a5/24e6ca83/528f23eb/O2 SHARED_OPERATOR_PARTIAL/O2.3A/O2_3C_REST_PRODUCT_INCOMPATIBLE/O3_NOT_READY.

## O2.3C-PREP-RUNTIME (2026-09-08): ORCHESTRAIQ EXECUTION -- R3 CERTIFIED, R4 `O2_3C_PREP_TASK_BENCHMARK_FAILURE`

Ran the frozen benchmark on orchestraiq/r770 via the official image `nipreps/fmriprep@sha256:15cbf8dc...` as a
native K8s Job (ORCHESTRAIQ_DIRECT_K8S). **Runtime succeeded end-to-end**: o23c- PVCs (local-path r770),
FreeSurfer Secret (hash 6f7afab5..), staged subj01/ses-nsd01/run-01 from public S3, **fMRIPrep finished exit 0,
crashfiles=0**, full derivatives (preproc_bold func, HMC, phasediff SDC, boldref->T1w coreg, confounds).

- **R3 `PERSISTENT_NIPYPE_RESUME_CERTIFIED`**: fresh identical Job (new pod 9d86f7cc vs aa24e087, same /work PVC
  r770, same digest+cmd); replacement finished ~260s vs ~35min, only 1 report node re-ran; 4 pre-sealed node caches
  byte-identical (SHA256+ns-mtime) -> reused, uncorrupted. Two evidence classes.
- **R4 `O2_3C_PREP_TASK_BENCHMARK_FAILURE`**: decisive frozen-metric incompatibility. Candidate (fMRIPrep) =
  188 vols @ TR 1.6s; NSD prepared ground-truth `timeseries_session01_run01` = **226 vols @ TR 1.3333s** (NSD
  temporally UPSAMPLED x1.2). Frozen methodology does NO temporal resampling -> geometry 4D-shape equality FAILS and
  the temporal-alignment guard FAILS (188!=226, 1.6!=1.333s); PASS-critical temporal-r thresholds unmeetable without
  a FORBIDDEN temporal-resample. Prepared-PRODUCT temporal-grid incompatibility (echoes O2_3C_REST_PRODUCT_INCOMPATIBLE),
  NOT a pipeline deficiency. STOP per brief: no tuning/resample/alt-registration/version-change/AWS.

**Cohort Phases 2-4 NOT started** (only on PASS). **O2.3C-RESUME (528f23eb) NOT run.** Artifacts: r3_pre_restart_evidence,
resume_certification (CERTIFIED), fmriprep_run_provenance, r4_transform_certification, r4_benchmark_metrics,
r4_benchmark_status, k8s_execution_provenance + `docs/.../59C`,`59D`. 42 data-free tests pass. Preserves 2d1a26a5,
24e6ca83, O2 SHARED_OPERATOR_PARTIAL, O2.3A CORE_ANCHOR..NOT_IDENTIFIABLE, O2_3C_REST_PRODUCT_INCOMPATIBLE, O3_NOT_READY,
blockers fbab942/4377100.

## O2.3C-PREP-RUNTIME (2026-09-08): ORCHESTRAIQ AUDIT -- `ORCHESTRAIQ_DIRECT_K8S_FMRIPREP_FEASIBLE` (avoids AWS)

Read-only Kubernetes/Run:ai audit (kubeconfig `antoniu_iepure.yaml`; **no cluster mutation**) to qualify orchestraiq
against Path-A host spec. **Verdict: DIRECT_K8S FEASIBLE -- run the official image as a native pod; no AWS.**
Cluster `10.130.123.31:10443`, ns `runai-romania-dev`, project `romania-dev`, on-prem bare metal.

- **orchestraiq = interactive jupyter Deployment** (`schedulerName=runai-scheduler`); this IS the previously-unstable
  resource -- unstable because INTERACTIVE (preemptible/idle-timeout), NOT node recycling. Fix = a separate batch Job.
- **Node `k8s-worker-cpu-node-r770`: 256 CPU / ~503 GiB / ~6.4 TiB, no taints, no pressure** -- vastly exceeds
  16 vCPU/64 GiB/500 GiB. (GPU node xe9680 EXCLUDED: tainted unreachable.)
- **Direct pod image works (no DinD):** egress CONFIRMED (public Docker Hub images running); server-dry-run of a
  Job with `nipreps/fmriprep:25.2.5` ACCEPTED by admission (created nothing).
- **RBAC:** create pods/jobs/pvcs/secrets/configmaps = yes; create `trainingworkloads.run.ai` = NO (can't make the
  Run:ai native non-preemptible CR); metrics/priorityclasses forbidden.
- **Stability:** batch Jobs run to completion here -- `eval-index` ran **~2.9 days** to Completed; multiple multi-hour
  `phase2-*`; NO preemption/OOM/eviction events. Interactive=preemptible, batch Job=stable.
- **Storage:** `local-path` (node-local, **no root-squash**, used by prior 200-500 Gi fmri2img jobs) recommended for
  workdir/derivatives/TemplateFlow; `nfs-client` default has root-squash risk. License -> K8s Secret (never committed).

**Recommendation: option A** -- native K8s batch Job (image = `nipreps/fmriprep:25.2.5`) on r770 + local-path PVCs +
license Secret. Proposed manifest `k8s_fmriprep_workload_PROPOSED.yaml`; audit `orchestraiq_infra_audit.json`;
`docs/.../59C_ORCHESTRAIQ_INFRA_AUDIT.md`. Audit mutated nothing; **execution needs one-time `kubectl apply` approval**
(harness auto-mode gated the write). No scientific methodology changed; preserves Phase-0 `2d1a26a5`, contract
`24e6ca83`, `O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O2 SHARED_OPERATOR_PARTIAL`, `O3_NOT_READY`, blockers `fbab942`/`4377100`.
39 data-free tests pass.

## O2.3C-PREP-RUNTIME (2026-09-08): PATH A CHOSEN -- OFFICIAL DOCKER-HOST MIGRATION -- `BLOCKED_PENDING_HOST_PROVISIONING`

User chose **Path A** (official `nipreps/fmriprep:25.2.5` on a non-preemptible Docker host); bare-metal branch
closed historically at `4377100`. TECHNICAL host/runtime migration ONLY -- no methodology/benchmark/registration/
output-space/Track-O change; Phase-0 `2d1a26a5`, `O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O3_NOT_READY` preserved.

**Ready (host-independent, verified):** FS license in hand (hash `6f7afab5...`, hash-only record, never committed);
ALL feasibility-first benchmark inputs fetchable from public NSD S3 (verified 2026-09-08: `task-nsdcore run-01`
bold+sbref+events, run-01 phasediff+magnitude fmaps, `ses-nsdanat`, `freesurfer/subj01 aparc+aseg.mgz`, prepared
`func1pt8mm/{timeseries,motion,design}` ground-truth -> `benchmark_input_manifest.json`); frozen benchmark command
derived from `2d1a26a5` recorded. **Sole blocker = a qualifying host.** Local Windows workstation INADEQUATE
(12 vCPU / 13.9GB RAM / ~38GB free SSD; Docker not operational) -- fails vCPU/RAM/SSD; old RunAI pod EXCLUDED
(preemptible + container-blocked). Host provisioning/access is the user's to provide.

**Status `BLOCKED_PENDING_HOST_PROVISIONING`.** On host: record digest+provenance (before benchmark) -> R3 resume
cert -> R4 frozen `subj01/ses-nsd01/run-01` benchmark (unchanged thresholds) -> on PASS Phases 2-4 -> on
`O2_3C_PREP_PASS` the scientific `O2.3C-RESUME` (SHA `528f23eb`). Artifacts: `host_migration_provenance.json`,
`benchmark_input_manifest.json`, `docs/.../59B_O2_3C_PREP_HOST_MIGRATION.md`. 38 data-free tests pass. Nothing fabricated.

## O2.3C-PREP-RUNTIME (2026-09-08): FS LICENSE RESOLVED -> R2 RE-SEAL `O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE`

**FS-license blocker RESOLVED by user.** `license.txt` placed -> copied to declared PVC path, hash-verified
(`sha256 6f7afab5...`, 133B, perms 600), structurally valid (5 lines, email, keys). Recorded **hash-only**
(`fs_license_status.json`); contents never printed/committed; file stays on PVC, never in git. `FS_LICENSE_REQUIRED`
is no longer the blocker.

**New binding constraint = external-dependency container-equivalence.** The authoritative fMRIPrep 25.2.5 spec
(`pixi.lock` + base image `ghcr.io/nipreps/fmriprep-base:20251006`) uses versions that **materially differ** from the
R0-frozen (docs-page-derived) pins: **ANTs 2.5.1->2.6.2**, **workbench 1.5.0->2.0.1**, **FSL monolithic 6.0.7.7 ->
componentized `fsl-*`** (definitive mismatches); **AFNI 24.0.05** and **FreeSurfer 7.3.2** target versions
**UNVERIFIABLE** (base-image ghcr pull blocked). The R0 pins came from fMRIPrep's stale *Manually Prepared
Environment* docs page. The frozen bare-metal dependency set **cannot be certified container-equivalent**, and
installing the container's *actual* versions would deviate from the frozen inventory (forbidden). No env built past
the python layer; **no benchmark run; nothing fabricated.**

**Terminal `O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE`** (brief: "versions cannot be pinned defensibly / not
container-equivalent enough"). **Recommended resolution `O2_3C_PREP_RUNTIME_HOST_REQUIRED`**: run the official
`nipreps/fmriprep:25.2.5` on a **non-preemptible Docker host** (>=16 vCPU / >=64GB / >=500GB SSD, no GPU) -- the only
defensibly container-equivalent runtime; FS license already in hand. Alt: user authorizes a corrected dependency
re-freeze matching `pixi.lock`. Artifacts: `fs_license_status.json`, `baremetal_dependency_inventory.json`
(authoritative-vs-frozen table), `runtime_amendment_status.json` (terminal + resolution). 36 data-free tests pass.
Preserves contract `24e6ca83`, Phase-0 `2d1a26a5`, `O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O2 SHARED_OPERATOR_PARTIAL`,
`O3_NOT_READY`, container-path blocker `fbab942`. Scientific terminal status unchanged (`null`).

## O2.3C-PREP-RUNTIME (2026-09-07): BARE-METAL RUNTIME AMENDMENT -- SEALED `O2_3C_PREP_FS_LICENSE_REQUIRED`

**Input HEAD `fbab942`; amendment contract SHA `24e6ca83`; Phase-0 `2d1a26a5` preserved.** TECHNICAL
runtime-packaging amendment ONLY (no scientific/benchmark/registration/Track-O change): try fMRIPrep 25.2.5
as a **manually-prepared bare-metal env** (container runtime blocked by pod securityContext) resumable via a
persistent Nipype `-w` on the PVC. Frozen (R0) + committed **before** install (`660777d`).

- **Runtime IS installable, container blocker bypassed.** On the pod, user-space (no sudo) on the **107TB
  persistent PVC**: micromamba(static) -> conda-forge `python=3.12` (**3.12.14**) -> `pip fmriprep==25.2.5`
  (**fmriprep 25.2.5**, **nipype 1.12.0**), importable. Packaging is not the obstacle. Network open
  (conda-forge/PyPI/TemplateFlow-S3/GitHub all reachable).
- **Binding constraint = FreeSurfer license, PROVEN from the installed 25.2.5 source (not assumed):**
  (1) `init_fsl_bbr_wf` (the `--fs-no-reconall` branch) initializes BOLD->anat coreg with FreeSurfer
  `mri_coreg`/`MRICoreg` -- `fmriprep/workflows/bold/registration.py:283,333`; (2) unconditional license gate
  `if not check_valid_fs_license(): return_code=126` -- `fmriprep/cli/workflow.py:119,137`, **no `run_reconall`
  guard** in that file. Cannot be avoided within the frozen methodology (gate precedes all nodes; swapping the
  coreg initializer = forbidden registration-policy change).
- **License absent** on PVC/home (`FS_LICENSE`/`FREESURFER_HOME` unset); **neither fabricated nor downloaded**
  per brief. Heavy binaries (ANTs/AFNI/FSL/workbench) not installed -- the hard stop preempts them. R3/R4
  `NOT_REACHED`; no benchmark/product metric fabricated. Scientific `terminal_execution_status` stays `null`.
- **Unblock:** user places a valid FreeSurfer `license.txt` at the declared `FS_LICENSE` PVC path; the proven
  env then completes (heavy binaries) + R3 resume-cert + R4 frozen `subj01/ses-nsd01/run-01` benchmark, unchanged.

Artifacts: `runtime_amendment_contract.json` (`24e6ca83`), `baremetal_dependency_inventory.json`,
`baremetal_environment_manifest.json`, `runtime_amendment_status.json` (terminal `O2_3C_PREP_FS_LICENSE_REQUIRED`),
R3/R4 `NOT_REACHED` placeholders + `docs/research/mindcompiler/59A_O2_3C_PREP_RUNTIME_AMENDMENT.md`. 35 data-free
tests pass. Preserves `O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O3_NOT_READY`, container-path blocker `fbab942`.

## O2.3C-PREP (2026-09-04): RAW REST PREPROCESSING PRODUCT GATE -- `PHASE0_FROZEN` (Lane B fMRIPrep 25.2.5); compute PENDING pod

**Input HEAD `e215f3f`; frozen config SHA `2d1a26a5`; resumes frozen O2.3C SHA `528f23eb`.** TECHNICAL
gate (no science): build+certify native func1pt8mm rest timeseries + motion + WM/CSF + Desikan-Killiany
68-parcel products for the frozen O2.3C input contract. Historical `O2_3C_REST_PRODUCT_INCOMPATIBLE`
preserved unchanged.

- **Audit (Part B)**: 5 public NSD preprocessing repos pinned; core (`preprocessfmri`/`alignvolumedata`/
  `knkutils`/`nsddatapaper`) is **MATLAB**, only `nsdcode` Python. Fieldmaps present (SDC feasible),
  FreeSurfer `aparc+aseg` present (DK source), **released `anat->func1pt8mm` transforms present**
  (DK mappable in pure Python via nsd_mapdata, NN), prepared main-NSD func1pt8mm timeseries present.
- **Lane decision (frozen before any QC)**: Lane A **INELIGIBLE** -- no MATLAB runtime locally/on pod +
  gradient-nonlinearity vendor coeffs not public -> `NSD_AUTHOR_LINEAGE_NOT_EXECUTABLE_FROM_PUBLIC_RELEASE`.
  **Lane B activated+frozen**: fMRIPrep **25.2.5** (no `latest`), phasediff SDC (no SyN), rigid motion,
  no smoothing/AROMA/denoise/GSR; raw->func1pt8mm ANTs rigid+affine (Mattes MI, BOLD LanczosSinc, atlas NN).
- **Benchmark (frozen selection)**: `session01_run01` all 8; thresholds Dice>=0.90, spatial r>=0.95,
  ROI-temporal r>=0.90, voxel-temporal r>=0.70, tSNR ratio in [0.5,2.0]; STOP-on-fail, no tuning/lane-switch.
- **Compute Phases 1-4 PENDING**: pod is **network-unreachable this session** (port22 timeout x3). NO benchmark/
  product metric fabricated -- result artifacts are explicit `PENDING_POD_COMPUTE`. NOT `PUBLIC_ASSET_FAILURE`
  (assets all present) nor `LANE_SELECTION_FAILURE` (lane selected); a transient infra hold, no terminal Part-AC
  status issued. Guard: `target_imagery_files_opened_by_PREP = 0`. 25 data-free tests pass.

Artifacts: `artifacts/mindcompiler/operator_o2_3c_prep/` (frozen_config + 8 contracts + audit + lane result +
benchmark manifest + DK 68 manifest + PENDING result placeholders + hashes) + `docs/research/mindcompiler/59_O2_3C_PREP_REST_PRODUCT.md`.

**RUNTIME PROBE (2026-09-04, pod recovered)** -- `runtime_probe.json`: pod back (256 CPU / 1TB RAM / 107TB free /
262GB NSD present). Verified: **no MATLAB** (Lane A out); **apptainer 1.5.3 installs** (conda-forge, userns enabled)
**but cannot execute** -- pod securityContext blocks mount propagation (`mount --make-rprivate /` -> Permission denied);
**udocker runs containers via proot only** (fakechroot F1-F4 fail) = not a supportable fMRIPrep runtime (too slow/fragile,
correctness risk); **fMRIPrep 25.2.5 requires a FreeSurfer license** even with `--fs-no-reconall` (authoritative docs),
**absent on PVC** (external credential). => Lane B not executable in a supported way. `compute_state = BLOCKED_PENDING_USER_ENABLERS`;
terminal-if-unresolved `O2_3C_PREP_PUBLIC_ASSET_FAILURE`. No benchmark/product fabricated.

**COMPUTE FEASIBILITY (empirically tested, same session):** attempted to establish the Lane B runtime end-to-end. Findings:
udocker pull FAILS on the ~15GB fMRIPrep layer (NFS root_squash 'delete not owner' -> hang; local overlay -> zombie curl @30MB);
`skopeo` 1.24.0 transfers robustly (after v1->v2 registries.conf + `--insecure-policy`) BUT the **pod preempted mid-copy at ~2.7/20GB**
and the ephemeral overlay (`/home/jovyan`) is wiped on rollout -> tar lost; NFS PVC root_squash breaks container image stores. Pod cycled
up/down repeatedly this session (uptime ~tens of min). **Decisive:** fMRIPrep 25.2.5 cannot be executed to completion here -- no supported
container runtime (apptainer blocked by securityContext mount restriction; no docker/singularity/sudo), udocker only via proot (slow/unsupported),
sustained compute defeated by preemption + ephemeral-wipe, FreeSurfer license absent. Infrastructure limitation, NOT a scientific result.
`compute_state=BLOCKED_PENDING_STABLE_CONTAINER_HOST_AND_FS_LICENSE`. **Two enablers needed (user-domain):** (1) a STABLE non-preemptible
container-capable host (Docker or Apptainer/Singularity permitted, persistent storage for the ~20GB image + outputs); (2) FreeSurfer `license.txt`
on persistent storage. **Next:** on enablers -> Phase 1 benchmark (session01_run01) -> Phases 2-4 -> O2.3C-RESUME `528f23eb`.

---

## O2.3C (2026-09-01): RESTING-STATE CONNECTIVITY ANCHOR -- `O2_3C_REST_PRODUCT_INCOMPATIBLE` / `CONNECTIVITY_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE`

**Input HEAD `c73f304`; frozen methodology sha `528f23eb`.** Tests whether intrinsic resting-state
functional connectivity (deterministic connectivity-SRM / cSRM) can identify the native orientation of
the O2.2 imagery residual in an imagery-unseen subject, without target imagery/perception. Methodology
FROZEN before any outcome (Parts A-D, Desikan-Killiany 68 parcels in native func1pt8mm, FD>0.25mm QC,
>=30 usable rest min, NO GSR, one anchor family, no leaderboard). O2.2 residual imported unchanged.

**Gate halted at the frozen DATA-AVAILABILITY precondition (brief Part C / F.6).** Probe of public NSD
Open Data (unsigned, no credentials):
- **0 prepared resting-state TIMESERIES runs** in native func1pt8mm for all 8 subjects (the 548 native
  runs are nsdimagery/nsdsynthetic/prffloc/main-NSD only).
- **`restingbetas_fithrf` are 3D single-volume** GLM betas + `R2` maps (verified ndim=3, [81,104,83]) --
  unusable for temporal connectivity.
- **Raw BIDS `task-rest` exists in abundance** (20-36 runs/subj, 188 TR x 1.6s = 5.0 min/run =>
  100-180 min/subj) but only in the raw acquisition grid [120,120,84], no cleaned/motion product;
  **Part F.6 forbids** building a raw preprocessing pipeline in this gate.
- Desikan-Killiany aparc is surface/anatomical only, **not** a native func1pt8mm volume.

**Verdict = `O2_3C_REST_PRODUCT_INCOMPATIBLE`** (NOT insufficient -- raw volume exceeds 30 min; NOT
unavailable -- raw exists). No fitting performed => orientation `INCONCLUSIVE`, leakage vacuously clean.
Technical blocker, not a scientific result on connectivity. Immutable carried forward: TRACK_R
`INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`, O1 `STIMULUS_INVARIANT_OPERATOR_PARTIAL`, O2
`SHARED_OPERATOR_PARTIAL`, O2.3A `FEASIBILITY_PASS`/`ORIENTATION_NOT_IDENTIFIABLE`, O3 `O3_NOT_READY`.
Artifacts: `artifacts/mindcompiler/operator_o2_3c/` (frozen_config, contracts, rest_data_inventory,
scientific_status, provenance) + `docs/research/mindcompiler/58_O2_3C_CONNECTIVITY_ANCHOR.md` + 8
data-free tests (`tests/mindcompiler/operator_o2_3c/`).

---

## O2.3A-EXECUTE (2026-08-31): NSD-CORE PERCEPTUAL ANCHOR -- `O2_3A_CORE_ANCHOR_FEASIBILITY_PASS` / `CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE`

**`d2b2bc7`(blocker) -> acquire `7a0f653`/`66d48fa` -> execute `5f13e4a`, server-verified.** Continuation
of the FROZEN O2.3A (methodology sha `39d7bc97`, unchanged) after acquiring NSD-core b2 for all 8
subjects via **public unsigned AWS Open Data** (s3://natural-scenes-dataset, --no-sign-request; 131 GB,
284 session files on the PVC). Historical `O2_3A_CORE_DATA_UNAVAILABLE` preserved as first-attempt truth.
O2 `SHARED_OPERATOR_PARTIAL` + O3 `O3_NOT_READY` immutable; no B1; C3R untouched.

- **Phase A**: core<->imagery voxel mapping CERTIFIED (8 subjects, native func1pt8mm, affine matches
  ncsnr, no resampling, existing O1/O2 ROI xyz). Anchor = **512 shared NSD-core images** (leakage-safe:
  0 NSD-Imagery overlaps; the 3-image leak that would have entered a naive 515-set was caught+removed).
- **Phase B (VISION-ONLY)**: DetSRM anchor common space VALIDATED STRONGLY -- cross-subject native
  reconstruction r **0.61 ventral / 0.37 lateral** (K median 8-16); far better than the 12-identity O2 space.
- **Phase C**: residual-orientation prediction (from vision-only anchor) vs random-anchor null, Holm
  ventral/lateral.

### KEY FINDINGS
- **Dense perception CAPTURES far more imagery structure than the small visual span**: R_Y_CORE 0.39 /
  0.34 / 0.62 (ventral/lateral/parietal) vs O2.2 R_Y_VISFULL 0.25 / 0.31 / 0.57 (V1 0.92, V3 0.74) ->
  `PARIETAL_RICH_VISION_RECOVERY_SUPPORTED`; confirms O2.2 small-vision insufficiency.
- **BUT target-state ORIENTATION is NOT identifiable**: predicted residual retention ~= random-anchor
  null in every ROI (ventral 0.081 vs 0.077; lateral 0.107 vs 0.121; parietal 0.109 vs 0.112; no Holm
  significance) -> `CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE`. The shared cross-subject
  orientation template does not recover a new subject's residual orientation from vision-only calibration.

**Bounded conclusion:** even dense (512-image) NSD-core perception validates a strong cross-subject
common space and recovers much of the imagery structure's PRESENCE, yet still cannot IDENTIFY the
subject-specific native ORIENTATION of the imagery residual without target imagery. Imagery-zero-shot
subject transfer remains structurally blocked under vision-only calibration. O2/O3 unchanged. Leakage-
clean. **Next: O2.3C -- target-state-independent CONNECTIVITY/ANATOMICAL anchor feasibility** (one new
anchor family; needs that ancillary NSD data). 9 data-free tests.

---

## O2.3A (2026-08-28): NSD-CORE PERCEPTUAL ANCHOR FEASIBILITY -- `O2_3A_CORE_DATA_UNAVAILABLE` / `CORE_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE`

**Input `4498f65` -> `d2b2bc7`, server-verified.** Doc `57_...`; artifacts
`artifacts/mindcompiler/operator_o2_3a/`. Config sha `39d7bc97`. **Full methodology FROZEN for future
execution**; execution HALTED at the Part-C data-availability gate. O2 `SHARED_OPERATOR_PARTIAL` +
O3 `O3_NOT_READY` immutable; C3R untouched.

Question: can dense target-state-INDEPENDENT NSD-core PERCEPTION identify the native orientation of
the O2.2 out-of-visual-span imagery residual in an imagery-unseen subject (zero target imagery)?
Frozen: B0/b2 core only (B1/b3 prohibited); ventral/lateral primary (parietal secondary); shared-scene
anchor excluding all NSD-Imagery/shared1000 overlap; DetSRM; vision-only K in {2..64}; residual-
orientation template from TRAINING subjects only; predicted-residual retention vs random-anchor null +
full-core oracle; Holm ventral/lateral; gauge-invariant. Prior art NOVELTY_CANDIDATE.

### DATA BLOCKER (technical, NOT a scientific failure -- Part AF.79)
B0-aligned NSD-core `betas_fithrf` (b2) is UNAVAILABLE for 7/8 subjects: local has subj01 only
(40 sessions b2); pod has none in b2. The only multi-subject core present (pod `nsd_betas_download`
subj02/05/07) is `betas_fithrf_GLMdenoise_RR` (b3/B1) -- WRONG beta version (prohibited by the frozen
B0 contract + Part AI) and belongs to the C3R track (do-not-touch). No NSD/AWS credentials. A LOSO
cross-subject anchor is not constructable; ~280 GB access-controlled acquisition is not feasible
autonomously. No fitting performed, no target imagery touched, no B1 used.

-> `O2_3A_CORE_DATA_UNAVAILABLE`; orientation `CORE_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE`; parietal
`INCONCLUSIVE`. O2.2's finding preserved; the dense-perception qualifier could NOT be tested.

**Admissible next (user-issued):** O2.3A-ACQUIRE (obtain NSD-core b2 for subj02-08 with NSD/AWS
credentials + storage, certify core<->imagery voxel mapping, then run the FROZEN O2.3A unchanged), or
O2.3C connectivity/anatomical anchor. Not admissible: pod b3/B1, subj01-only, interpolation rescue,
alignment-method competition. O3 stays NOT_READY. **Awaiting user brief / credentials.**

---

## O2.2 (2026-08-28): CROSS-STATE TRANSPORT GEOMETRY AUDIT -- `O2_2_STATE_TRANSPORT_DIAGNOSTIC_PASS` / `CROSS_STATE_TRANSPORT_MULTIREGIME` / `O2_3_SUBJECT_SPECIFIC_STATE_MAPPING_REQUIRED`

**Input `760a54b` -> freeze `76af537` -> execute `6b653cd`, server-verified.** Post-hoc geometry audit,
NO refit / no new common-space. Doc `56_...`; artifacts `artifacts/mindcompiler/operator_o2_2/`. New
`operator_o2_2` pkg (pure `audit_logic`) + 13 tests. Config sha `bbc9487c`. Frozen O2 SRM replay EXACT
(R_Y_SRM reproduces O2.1: ventral 0.096 / lateral 0.144 / parietal 0.171); O1 O4 re-selection err
1.5e-3 (thread-tie; descriptive conclusions robust). **O2 `SHARED_OPERATOR_PARTIAL` + O3 `O3_NOT_READY`
immutable.**

Question: is imagery lost by SRM truncation of an adequate visual span, or does it live outside the
visual identity span (shared? subject-specific? noise?)? THREE nested subspaces P_SRM / P_VIS_FULL /
P_IMG_FULL; held-out identity contrasts; odd/even reliability; RDM LOSO sharedness; exact O1 O4
parallel/perp decomposition.

### KEY FINDINGS
- **SRM truncation contributes** (R_Y_VISFULL > R_Y_SRM by +0.15/+0.15/+0.29) -- the low-K SRM space
  discarded imagery structure that IS in the visual span. BUT R_Y_VISFULL still LOW (0.25/0.31/0.57):
  **43-75% of imagery identity structure lies OUTSIDE the vision identity span** (outside fraction).
- **That out-of-visual-span component is RELIABLE** (odd/even 0.57-0.62, HIGH) **AND cross-subject
  SHARED** (LOSO RDM sharedness 0.70-0.75, HIGH) -> `SHARED_IMAGERY_GEOMETRY_EXISTS_OUTSIDE_VISION_
  IDENTITY_SPAN`. It is NOT measurement noise.
- **Within-subject O1 advantage lives MORE in the perpendicular (out-of-span) component**
  (G_O1_perp +0.06..+0.08 > G_O1_parallel +0.03..+0.05, all ROIs) -> O1 predicts exactly what the
  vision-only cross-subject space discards.
- **Regimes**: ventral B / lateral B (shared geometry outside vision) + parietal A (SRM truncation of
  in-span imagery) -> **`CROSS_STATE_TRANSPORT_MULTIREGIME`**.
- **Identifiability**: `TARGET_STATE_ORIENTATION_NOT_IDENTIFIABLE_FROM_CURRENT_VISION_ONLY_DATA` --
  shared RDM geometry does NOT imply a known native orientation in an unseen brain; the residual sits
  outside the visual span by construction and no vision-only predictor of its orientation exists.

**Bounded conclusion:** reliable, cross-subject-shared imagery identity geometry exists but lies
largely OUTSIDE the vision identity span; vision-only functional alignment (validated on vision)
discards it, and its orientation in a new subject cannot be recovered from vision alone. Imagery-
zero-shot subject transfer is structurally blocked under the current vision-only calibration contract
-> **`O2_3_SUBJECT_SPECIFIC_STATE_MAPPING_REQUIRED`**. Future representation would need a target-
state-independent anchor (imagery calibration, connectivity/anatomical, or richer multimodal data) --
O2.2 fits NONE of it. O2/O3 status unchanged. **Next: a separate O2.3 / O3.0-class gate (user-issued);
awaiting brief.**

---

## O2.1 (2026-08-27): SHARED-OPERATOR TRANSFER BOTTLENECK AUDIT -- `O2_1_TRANSFER_BOTTLENECK_AUDIT_PASS` / `O2_TRANSFER_LIMIT_CROSS_STATE_SPACE_DOMINANT` / `O3_NOT_READY`

**Input `adcb244` -> freeze `85d793f` -> execute `a9db401`, server-verified.** Post-hoc audit, NO
refit / no new common-space. Doc `55_...`; artifacts `artifacts/mindcompiler/operator_o2_1/`. New
`operator_o2_1` pkg (pure `audit_logic`) + 15 tests. Config sha `22bde42f`. Frozen-SRM replay EXACT
(r_Tshared err 1.1e-16, committed K/lambda). **O2 `SHARED_OPERATOR_PARTIAL` immutable.**

Question: where is the ~95% O1->O2 transfer loss, and why does the frozen pipeline transfer in
ventral but not lateral/parietal? **KEY FINDING -> `O2_TRANSFER_LIMIT_CROSS_STATE_SPACE_DOMINANT`:**
the vision-derived SRM common space is validated on VISION (identity-contrast retention ~0.49-0.55)
but preserves IMAGERY identity structure POORLY -- retention ventral 0.096 / lateral 0.144 /
parietal 0.171 (cross-state gap +0.28..+0.45). Native oracle r ~0.15 (low even for the ideal
in-subspace reconstruction). Shared-space contrast capture ~0 (NEGATIVE in ventral) -> T_shared does
NOT transfer stimulus-specific imagery even where vision alignment is perfect.
- **Low-K NOT the cause** (ventral transfers at the smallest K=2; failing ROIs use same/lower K).
- **Delta regularization**: upper-boundary rate 0.25 + median CV increment ~0 ->
  `DECOMPOSITION_CONFIDENCE_MODERATE` (70%-shared is partly a regularization artifact; the
  `SHARED_COMPONENT_DOMINANT` label is preserved but qualified).
- Baseline headroom not the primary limiter; calibration/vision-capacity NOT_SUPPORTED as causes.
- Identity breadth LIMITED (ventral increment concentrated); 7/8 target subjects BROAD, subj03 WEAK.
- Q1-Q5: visual alignment NOT sufficient for operator transfer; vision-W does NOT preserve imagery
  structure; T_shared does NOT encode imagery contrasts; ventral specialness INCONCLUSIVE.

**Bounded conclusion:** the visual common space aligned unseen subjects successfully, but cross-state
transfer was limited because imagery identity structure was poorly represented in that vision-derived
shared space; most within-subject operator advantage is lost during cross-subject STATE transport, not
calibration or low K. O2 status unchanged. **O3_NOT_READY** (ventral shared-space contrast <=0) ->
**Next: O2.2 -- VISION-DERIVED COMMON-SPACE STATE-TRANSPORT DIAGNOSTIC** (no alternative common-space
method rescue in this line).

---

## O2 (2026-08-26): SHARED + SUBJECT-SPECIFIC PERCEPTION->IMAGERY OPERATOR -- `O2_SHARED_OPERATOR_PASS` / `SHARED_OPERATOR_PARTIAL`

**Input `299df3c` -> freeze `e665cba` -> Phase A `c73be88` -> Phase B `3e06b28` -> Phase C `c48c7b7`,
server-verified.** Cross-subject + cross-stimulus. Doc `54_...`; artifacts
`artifacts/mindcompiler/operator_o2/`. New `operator_o2` pkg (det-SRM, gauge-equivariant scalar-ridge
shared operator, vision-only new-subject calibration) + 18 tests. Config sha `f26c12c1`. B0 only
(dependence is a limitation). Immutable: O1/O1.1/Track-R/H-A unchanged.

Claim class IMAGERY_ZERO_SHOT_SUBJECT_TRANSFER_WITH_VISION_ONLY_CALIBRATION. 8 LOSO subject folds x
6 global-identity-holdout folds x 3 primary ROIs (ventral/lateral/parietal; V2/hV4 prospectively
excluded). Common space = DetSRM fit on VISION ONLY; same W maps both states; new subject calibrated
from its VISION on training identities. Shared operator = FULL_SHARED_SCALAR_RIDGE (gauge-equivariant,
verified T'=Q^T T Q err 4e-16; NO per-target/diagonal). Baselines S0 group-imagery-mean, S1 global-gain.

- **Phase A `COMMON_SPACE_VALIDATED`** (vision-only): median VISION_IDENTITY_MARGIN ventral 1.57 /
  lateral 1.11 / parietal 1.17; 100% outer cells positive; no subject fails. K mostly 2 (low-dim
  identity-separating vision space).
- **Phase B `SHARED_OPERATOR_PARTIAL`** (144 cells, leak-clean -- target imagery + test identities
  never in SRM/lambda/T). G_shared=r(T_shared)-r(S0), exact 2^8 sign-flip + Holm/3:
  **ventral +0.0145 Holm-significant (7/8); G_beyond_gain +0.0197 Holm-significant (8/8)** -> shared
  operator beats BOTH group-mean and global-gain in ventral. lateral +0.0017 ns (6/8); parietal
  -0.004 ns (3/8). Holm-reject 1/3 -> PARTIAL. Only ~5% of within-subject O1 gain transfers.
- **Phase C**: secondary V1 +0.0044 (6/8 ns) / V3 +0.0008 (5/8 ns) -- cannot change primary. Delta_s
  (training subjects only): median shared-action fraction 0.702 -> **SHARED_COMPONENT_DOMINANT**
  (~70% shared, ~30% subject-specific). Ventral T_shared NTI=1.0 (pure reorientation), contractive,
  effective rank ~2.

**Bounded conclusion:** a shared perception->imagery operator learned from other participants predicted
imagery patterns in an imagery-unseen participant for held-out stimulus identities, after vision-only
calibration -- Holm-significant in ventral cortex (beating group-mean AND global-gain), with ~70% of
the transformation shared across training subjects; but transfer is region-specific (ventral only) and
much weaker than within-subject -> PARTIAL. No zero-calibration / universal-operator / brain-to-brain /
causal claim; raw SRM axes have no biological meaning. **Next: O2.1 -- SHARED-OPERATOR HETEROGENEITY /
CALIBRATION AUDIT** (per PARTIAL branch; no alternative common-space method rescue).

---

## O1.1 (2026-08-26): OPERATOR HETEROGENEITY / FAILURE-MODE AUDIT -- `O1_1_HETEROGENEITY_AUDIT_PASS` / `OPERATOR_HETEROGENEITY_REGION_STRUCTURE_DOMINANT` / `O2_READY`

**Input `64da8cf` -> freeze `12faabc` -> execute `e2618ac`, server-verified.** Post-hoc DESCRIPTIVE
audit (NO refit, no new models). Doc `53_...`; artifacts `artifacts/mindcompiler/operator_o1_1/`.
New `operator_o1_1` pkg (pure `audit_logic`) + 16 tests. Config sha `dbcbe657`, frozen before values.
**O1 status `STIMULUS_INVARIANT_OPERATOR_PARTIAL` UNCHANGED** (explanatory only). Only betas touched:
a one-time split-half repeat-reliability measurement summary (FM2/FM3/FM4); no operator refit.

Authoritative effect `G = O4 - O0` (raw O4 not evidence; O0 baseline ~0.71-0.78). Findings:
- **FM1 baseline-ceiling NOT primary** -- evidence ROIs (ventral/lateral/parietal) have LOWER O0
  baseline (0.65-0.71) / more headroom, not higher.
- **FM2 imagery measurement PARTIAL**, **FM3 vision measurement NOT_SUPPORTED** (vision reliable
  everywhere ~0.9; not the limiter), **FM4 beta-dependence PARTLY tracks reliability** (B0->B1
  operator loss vs imagery-reliability loss; Spearman 0.2-0.5). Does NOT reopen H-A.
- **FM5 instability CONTRIBUTES** (std-median-diff action-cosine FULL-vs-NO = 3.13; per-ROI median
  Spearman(G,stability)=0.38). **FM6 region-structure**: evidence ROIs have high out-of-visual-span
  (0.30-0.61) + stable operators; 4 geometry vars separate FULL-cross-voxel vs no-evidence cells.
- **V2 & hV4 MULTIFACTORIAL** (hV4: lowest imagery reliability 0.64, lowest stability 0.53, ~zero
  out-of-span). V1(+0.051, 6/8)/V3(+0.033, 5/8) positive trends not surviving Holm.
- O3 low-rank failure interpreted via identity-map anchor + strong contraction. Cross-family
  increment NOT supported (O4~=O0). Dimensionality context: high-rank/mixing operator (not
  compressing) -- consistent with S2.6R; Roy NOT reopened.

**OVERALL `OPERATOR_HETEROGENEITY_REGION_STRUCTURE_DOMINANT`** (frozen rule; transparently
near-multifactorial -- instability + imagery-measurement co-contribute). **`O2_READY`**: A) 3 ROIs
(ventral/lateral/parietal) Holm-significant B0 evidence; B) not baseline-ceiling; C) stability 0.739
>= 0.50; D) contract freezable; E) B0-dependence acknowledged. Candidate primary O2 ROIs
ventral/lateral/parietal; V1/V3 secondary; V2/hV4 prospectively justified out (not silently dropped).

**Next: O2 -- SHARED + SUBJECT-SPECIFIC PERCEPTION->IMAGERY OPERATORS** (T_s = T_shared + Delta_s;
critical test = LEAVE_ONE_SUBJECT_OUT + HELD_OUT_STIMULUS_IDENTITIES; common-space method to be
validated in O2, NOT selected here; no raw-W cross-subject comparison).

---

## O1 (2026-08-26): STIMULUS-INVARIANT PERCEPTION->IMAGERY OPERATOR -- `O1_OPERATOR_CHARACTERIZATION_PASS` / `STIMULUS_INVARIANT_OPERATOR_PARTIAL`

**Input `e378288` -> freeze `a4fe0fa` -> execute `84d6405`, server-verified.** FIRST Track-O
(original research, NOT Roy reproduction). Docs `51_...` (charter) + `52_...` (prior art);
artifacts `artifacts/mindcompiler/operator_o1/`. New `operator_o1` package (folds/operators/
evaluate/geometry) + driver + 22 data-free tests (incl negative controls). Config sha `c8ebd4af`,
frozen before cohort outcomes. TRACK_R sealed immutable (REOPEN_ALLOWED=false); H-A unchanged.

Question: does a reusable linear operator map perception-state -> imagery-state neural
representations for stimulus IDENTITIES never used to fit it? B0 primary, RAW vision/imagery
centroids (NO D1), identity = grouping+fitting unit. 6 outer x 5 inner identity folds (1 simple +
1 naturalistic each; leakage cert all-True). Operator ladder O0 imagery-mean / O1 global-gain /
O2 diagonal / O3 low-rank-residual / O4 full-RRR. Common vision scaling (both states). Primary
metric held-out-identity pattern r; inference = participant (N=8) exact 2^8 sign-flip + Holm/7 ROIs.
Prior art = NOVELTY_CANDIDATE (Roy same-identity; "General Transformations" 2018 held-out affine;
MIRAGE seen->imagined reconstruction; hyperalignment/SRM cross-subject -> O2). 56/56 cells.

### SCIENCE: `STIMULUS_INVARIANT_OPERATOR_PARTIAL`
- **O4>O0 (reusable operator vs imagery-mean) Holm-significant in 3/7 ROIs:** ventral +0.083,
  lateral +0.093, parietal +0.045; V1 +0.055 / V3 +0.046 positive but NOT Holm-significant;
  V2/hV4 no evidence. Cohort median O4-O0 +0.038. (Note: O0 baseline ~0.71-0.78 -- imagery
  centroids share strong common structure -- so O4-O0 is the meaningful signal.)
- **Where present, needs cross-voxel mixing:** O4>O1 in 5/7, O4>O2 in 3/7; O3 low-rank-residual
  NOT competitive (median O4-O3 +0.152). FULL_CROSS_VOXEL_OPERATOR_NEEDED modal in 5/7 ROIs.
- Operator far from identity (dev 1.09), CONTRACTS ~97% of supported directions (consistent with
  lower imagery amplitude/SNR, Supplement Fig S2), off-diagonal mixing 0.86, out-of-visual-span
  0.28; MODERATE fold stability (50/56). Cross-family transfer positive both directions but
  O4~=O0 (carried by shared imagery structure, not the vision-dependent operator).
- **B1 (b3) MEASUREMENT SENSITIVITY:** operator evidence does NOT replicate (median O4-O0 -0.006,
  0/7 Holm) -> the reusable-operator evidence is B0-specific. Honest caveat; B1 non-primary,
  never downgrades the B0 verdict (frozen rule).
- Operator-dimension context (NOT a Roy re-analysis): the operator is high-rank / strongly
  mixing (not low-rank-compressing), consistent with the S2.6R "imagery not lower-dimensional"
  finding; dimensionality treated as an empirical operator property, never assumed.

**Bounded conclusion:** a linear perception->imagery operator learned from one set of stimuli
generalizes to entirely unseen stimulus identities in higher-level ventral/lateral/parietal cortex
(above the imagery-mean baseline, participant-level Holm-significant), requires cross-voxel mixing
where it exists, but generalization is ROI/participant-dependent and B0-preparation-specific ->
PARTIAL. No causal/universal/thought-reading claim. **Next: O1.1 -- OPERATOR HETEROGENEITY AND
FAILURE-MODE AUDIT** (per PARTIAL branch; no shared-operator O2 claim yet).

---

## S2.7R (2026-08-23): FINAL CLAIM-BY-CLAIM REPRODUCTION VERDICT -- `S2_7R_FINAL_VERDICT_AUDIT_PASS`

**Input `932b94f` -> rubric freeze `a718faf` -> verdict `03b8fa3`, server-verified.** Audit/verdict
gate, NO modelling. Doc `50_...md`; artifacts `artifacts/mindcompiler/roy_s2_7r/`. Rubric (sha
`9acc7e1a`) frozen BEFORE synthesis. New: `roy_verdict.py` (pure rubric decision logic) + 18 tests.

Supplement 1 (`media-1.pdf`, sha `67afe50a`, 3 pp) retrieved via Europe PMC (PMC binary is behind a
JS proof-of-work gate): THREE figures only (S1 prediction maps subj02-08; S2 SNR maps -- imagery SNR
markedly lower than vision; S3 SIMULATED alignment illustration). **No material method gap** ->
S2_7R_NEW_MATERIAL_METHOD_GAP NOT raised. Source is a bioRxiv **PREPRINT** (not peer reviewed).

Method: 9 MATCH / 8 CLOSE_DEFENSIBLE / 0 MATERIAL_MISMATCH -> `METHOD_PARTIALLY_IDENTIFIABLE_BUT_
ASSESSMENT_POSSIBLE`.

### FINAL DATASET-1 VERDICT: `INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`
- PREDICTION `ROY_PREDICTION_DIRECTIONALLY_CONCORDANT` (B0 7/7 positive, 17.3% voxels > shuffle
  null, pairing-robust; Fig S1 corroborates). B1 NOT used to downgrade.
- DIMENSIONALITY `ROY_DIMENSIONALITY_MIXED`; **central D-C1 DIRECTIONALLY_DISCORDANT** -- early
  d_img/d_vis ~1.47 vs paper ~0.5 (imagery HIGHER-dim, OPPOSITE). A central directional discordance
  blocks SUPPORTED.
- ALIGNMENT `ROY_ALIGNMENT_DIRECTIONALLY_CONCORDANT` -- monotone V1 0.70 -> hV4 0.77 -> parietal
  0.92, all above null; early magnitude discordant (V1 0.70 vs 0.25-0.30, coupled to inflated d_img);
  parietal NUMERICALLY_CLOSE to ~1.0.
- FEATURE_CORRESPONDENCE (F-C1, Fig 5C-E) `NOT_EVALUATED` (no proxy).

Concordant prediction + alignment preclude NOT_SUPPORTED; sufficient method identifiability precludes
PUBLIC_METHODS_INSUFFICIENT; central D-C1 discordance precludes SUPPORTED -> **PARTIAL**. Scope =
**Dataset 1 (NSD-Imagery) ONLY** (Dataset 2 / Spatial Imagery NOT reconstructed). No exact/bitwise
replication claim. `H_A_CROSS_PARTICIPANT_PARTIAL` + `S2_5R_VERDICT_WITHHELD` + `S2_6R_RESULTS`
immutable. 40 tests (18 verdict + 22 geometry).

**TRACK R CLOSES at `INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`.** Optional next: **Track O -- O1
perception->imagery neural-state operator** (handoff prepared, NOT executed; S2.6R dimensionality
contradiction must be preserved, not assumed away). Optional `R-DIAG1` diagnostic = predeclared
ambiguities only, never a verdict rescue.

---

## S2.6R (2026-08-23): ROY DIMENSIONALITY + VISUAL<->IMAGERY ALIGNMENT -- `S2_6R_DIMENSIONALITY_ALIGNMENT_PASS`

**Input `4aa5a8c` -> output `efccf35`, server-verified. Freeze (config + semantics `5257c14`)
before any dimensionality/alignment outcome.** Doc `49_...md`; artifacts
`artifacts/mindcompiler/roy_s2_6r/`. New: `roy_geometry.py` (d99 rule, output-space bases,
alignment ratio + null, spectral diagnostics), `roy_geometry_engine.py` (replays S2.5M fold
+ derives geometry). B0 only; NO proxy geometry (no principal angles/CCA/Procrustes/RSA).

Reconstructed the two deferred Roy quantities on the frozen public-method-aligned B0 pipeline,
224/224 cells (8 subj x 7 ROI x 4 folds). **Ran on the S2.5M pod (identical LAPACK):**
engine-fidelity 0.0 vs frozen `run_fold_roy` AND committed-pod curve replay 0.0 on all 224;
0 reduced-rank-unstable cells; d_report platform-stable 1.0; pairing + voxel hashes match
S2.5M. (A local Windows-BLAS run first exposed cross-platform reduced-rank SVD-tie instability
in 2 cells -> resolved by same-platform pod execution.) Validator ALL_PASS.

- `r_model` (validation-selected operational rank, S2.5M) kept SEPARATE from `d_report`
  (99%-of-TEST-peak first-crossing). 209/224 folds evaluable (non-positive-peak rule).
- **Dimensionality DISCORDANT on the headline claim:** d_img/d_vis ~1.47 early visual,
  ~1.50 higher; imagery subspace is HIGHER-dimensional than visual (opposite of Roy ~0.5).
  Parietal alone at parity (0.996). DIM-C1 DIRECTIONALLY_DISCORDANT -> `ROY_DIMENSIONALITY_MIXED`.
- **Alignment DIRECTIONALLY concordant:** a_g = TV_img/TV_vis (d=d_img both terms), 100-draw
  random-visual-subspace null. V1 0.70, hV4 0.77, parietal 0.92; monotone up; all far above
  null (V1 null 0.19; 8/8 above null in most ROIs). Parietal NUMERICALLY_CLOSE to ~1.0; early
  ROIs ~2x Roy (coupled to inflated d_img) -> `ROY_ALIGNMENT_DIRECTIONALLY_CONCORDANT`.
- Dimension-ratio <-> alignment relation: median per-participant Spearman -0.14 (no consistent
  within-subject relation; 56 ROI rows NOT treated as independent).

**FULL INDEPENDENT REPRODUCTION VERDICT STILL DEFERRED -> S2.7R.** No beta selection, no B1
primary geometry, no D1b, no proxy. H_A_CROSS_PARTICIPANT_PARTIAL unchanged. 22 new geometry
tests (18+4); roy suite 215 local / 241 pod (1 pre-existing S1-manifest pod discrepancy).

**Next: S2.7R -- FINAL CLAIM-BY-CLAIM INDEPENDENT REPRODUCTION VERDICT** (freeze rubric from
public paper; combine PREDICTION + DIMENSIONALITY + ALIGNMENT + METHOD CONCORDANCE + ambiguities;
no new models).

---

## S2.5M (2026-08-21): PUBLIC-METHOD-ALIGNED RECONSTRUCTION -- `S2_5M_PUBLIC_METHOD_ALIGNMENT_PASS`

**Input `34b70f6` -> output `67e755f`, server-verified. Freeze (config `54e72e6` +
pairing manifests `1888cdd`) before any aligned outcome.** Doc `48_...md`; artifacts
`artifacts/mindcompiler/roy_s2_5m/`. New modules (`roy_public_rrr.py` per-target Lambda;
`roy_pairing.py` Roy pairings; `roy_engine.py`) leave historical S2 code immutable.

Repaired the S2.5R prediction-class method gaps: vis2vis within-split derangement,
vis2img RANDOM within-identity pairing, PER-TARGET Lambda (one ridge per voxel),
validation-selected operational rank, Fig-3 prediction null (N=1000). 448/448 cells,
leakage-clean, no scalar-lambda fallback. NON-INTERPRETIVE (no beta selection):
- B0 (b2-compatible PRIMARY) new-cohort median vis2img r = 0.1169, **7/7 positive**;
  median fraction voxels above shuffle-null p95 = 0.173.
- B1 (b3 sensitivity) median 0.0023 (4/7) -- weak, NOT a failure.
- S2.4R->S2.5M B0 0.1319->0.1169: corrections did NOT materially change B0 (7/7 vs
  6/7) -> earlier result not an artifact of index-aligned pairing / scalar lambda.
- Pairing-randomness sensitivity (B0, realizations 0-3): medians 0.117/0.115/0.126/
  0.107 (range 0.020), 7/7 positive each -> **PAIRING_REALIZATION_ROBUST**.

Prediction: PREDICTION_DIRECTIONALLY_CONCORDANT_ONLY (B0, with a paper-style null).
Full rank curves saved. **Dimensionality + alignment NOT reconstructed -> FULL
INDEPENDENT REPRODUCTION VERDICT STILL DEFERRED.** H-A unchanged. 204 tests.

**Next: S2.6R -- ROY-DEFINED DIMENSIONALITY AND ALIGNMENT RECONSTRUCTION** (method-
aligned B0; exact variance-projection alignment ratio; d_vis/d_img; no proxies).

---

## S2.5R (2026-08-21): PAPER-CONCORDANCE AUDIT -- `S2_5R_MATERIAL_METHOD_GAP_IDENTIFIED` (verdict withheld)

**Input `1568ae4` -> output `dfb2d76`, server-verified. Audit gate; no models rerun.**
Doc `47_...md`; artifacts `artifacts/mindcompiler/roy_s2_5r/`. Rubric frozen from public
claims first (`3be9303`).

Re-verified the ACTUAL Roy et al. paper from open-access PMC (PMC12424947), short
quotes only. Method MATCH: 8 subj, 12 identities (6+6), 8 repeats, 7 ROIs, >98th-pct
NSD-core voxel selection, 4-fold 4/2/2 (=50/25/25), ridge 100x1e-3..1e5, reduced-rank
two-stage. Beta: "similar to b2" -> b2=fithrf=**B0** (b2-compatible); B1(=b3) is a
NON-primary sensitivity branch (its weak results are NOT a reproduction failure).

**MATERIAL METHOD MISMATCHES:** vis2img pairing HIGH -- paper "randomly-selected
imagery trial repeats" vs our index-aligned I0; vis2vis "shuffling within each data
split" vs our all_ordered_distinct (MOD); rank-selection semantics (MOD). Central
RESULTS unreconstructed: dimensionality NOT_DIRECTLY_COMPARABLE; alignment ratio
NOT_YET_RECONSTRUCTED (no proxy). Prediction only DIRECTIONALLY_CONCORDANT for B0
(median r 0.13, 6/7) -- no Roy shuffle null (r>0 != above-null).

Verdict **WITHHELD** (HIGH-materiality method gap + unreconstructed central results)
-> S2_5R_MATERIAL_METHOD_GAP_IDENTIFIED, preferable to an invalid claim. Exact-
replication terminology preserved (ORIGINAL_CODE/BITWISE unavailable). H-A unchanged.
189 tests.

**Next (Path A): S2.5M -- PUBLIC-METHOD-ALIGNED PAIRING AND RANK RECONSTRUCTION**
(random vis2img pairing; within-split vis2vis shuffle; paper-compatible rank semantics;
B0/b2 primary, B1 sensitivity), then replay; alignment/dimensionality later in S2.6R.

---

## S2.4R (2026-08-21): FULL-COHORT D1 CHARACTERIZATION -- `S2_4R_FULL_COHORT_D1_PASS`

**Input `be84dea` -> output `73f2639`, server-verified. D1 contract frozen before any
new-D1 outcome (`4633e4f`).** Doc `46_...md`; artifacts `artifacts/mindcompiler/roy_s2_4r/`.
First full-cohort run of the reconstructed two-stage D1 pipeline (vis2vis ->
D1_STRICT_CROSSFIT -> vis2img): 8 participants x 7 ROIs x 2 betas x 4 folds = 448
cells, 8-way parallel on the pod. subj01 = DEVELOPMENT_REFERENCE_D1; subj02-08 new.

Validator: **448/448 cells, 0 dup, leakage-clean** (all 5 invariants 0). NON-
INTERPRETIVE (participant=unit; NO beta selection; NO Roy verdict):
- B0 D1 new-cohort median vis2img r = **0.1319 (6/7 positive)**
- B1 D1 new-cohort median r = **-0.0018 (3/7 positive)**
- **D1_BETA_SENSITIVITY_PERSISTS** (median S_D1_beta_loss 0.118, 6/7) -- the B0/B1
  divergence persists under the FULL D1 pipeline (consistent with the beta-prep-stage
  origin from S2.1B/S2.4H).
- D1-vs-RAW: mostly D1_NEAR_NEUTRAL for B0; mixed for B1. subj07 low-baseline regime
  persists under D1 (D1_B0=-0.026; still included, not excluded). reliability-context
  rho=-0.036 (H-A NOT reopened; H_A_CROSS_PARTICIPANT_PARTIAL immutable). 179 tests.

**Next (item 41): S2.5R -- PAPER-CONCORDANCE AND INDEPENDENT-REPRODUCTION VERDICT
AUDIT** (re-open Roy paper+supplement; freeze a concordance rubric INDEPENDENTLY
before comparing; compare both B0 and B1). Track O (operator research) deferred.

---

## S2.4H (2026-08-21): PARTICIPANT HETEROGENEITY AUDIT -- `S2_4H_HETEROGENEITY_AUDIT_PASS`

**Input `682a1e0` -> output `dc78877`, server-verified. POST-HOC DESCRIPTIVE; committed
S2.3 summaries ONLY (no raw data, no reruns, no models, no p-value rescue).** Doc
`45_...md`; artifacts `artifacts/mindcompiler/roy_s2_4h/`. **Frozen S2.3 primary
IMMUTABLE and UNCHANGED: H_A_CROSS_PARTICIPANT_PARTIAL (rho 0.6786, p 276/5040).**

Descriptive heterogeneity findings: subj07 = `NO_TECHNICAL_ANOMALY` ->
VALID_SCIENTIFIC_HETEROGENEITY (B1 imagery-reliability loss positive in 5/7 ROIs but
a LOW-BASELINE RAW regime: B0 RAW itself weak/negative -> B1 not worse, L_perf_RAW<0
in 6/7; participant-wide, not one ROI; NOT excluded). Imagery-vs-vision:
rho(img,perf)=0.679, rho(vis,perf)=0.786, rho(img,vis)=0.393 ->
GENERAL_RELIABILITY_FACTOR_PLAUSIBLE (label not renamed). Leave-one rho 7/7 positive
(robust direction). median 0.679 vs mean 0.393 (no sign reversal). rho(nvox,perf)
0.445 (MIXED). 172 tests.

Descriptive mechanism statement: H-A captured a robust DIRECTIONAL component of
beta-version sensitivity but does not fully explain participant heterogeneity; a
broader beta-preparation reliability factor (vision + imagery) remains plausible.
NOT causal; frozen criterion C not met.

**Next (Path A): freeze BETA_PREPARATION_RELIABILITY_SENSITIVITY_WITH_PARTICIPANT_
HETEROGENEITY -> S2.4R FULL CROSS-PARTICIPANT D1 ROY-PIPELINE CHARACTERIZATION**
(both B0 and B1; no beta selection).

---

## S2.3 (2026-08-21): PROSPECTIVE CROSS-PARTICIPANT CONFIRMATION -- `S2_3_PROSPECTIVE_CROSS_PARTICIPANT_PASS` / `H_A_CROSS_PARTICIPANT_PARTIAL`

**Input `977b3cd` -> output `42974eb`, server-verified. PROSPECTIVE cross-participant;
freeze committed before outcomes (`56c2093`); participant = unit of analysis.** Doc
`44_...md`; artifacts `artifacts/mindcompiler/roy_s2_3/`. Acquisition on the pod
(subj02-08 B0/B1 verified, 16 HDF5, LOCALLY_COMPUTED SHA; raw gitignored). Identity
gate hardened for NFS worktree (`fe4d040`).

Tested the S2.2B-frozen H-A prediction on 7 untouched participants (subj02-08),
matched RAW endpoint, 7 ROIs each -> participant-level median losses:
  subj02 S_rel+0.100/S_perf+0.143  subj03 +0.102/+0.198  subj04 +0.125/+0.151
  subj05 +0.084/+0.147  subj06 +0.114/+0.173  subj07 +0.064/-0.033  subj08 +0.060/+0.144

Primary Spearman rho=0.6786; EXACT 5040-permutation one-sided p=0.0548 (276/5040).
Criteria A=T B=T C=F(p just >0.05) D=T (6/7 concordant) -> 3/4 ->
**H_A_CROSS_PARTICIPANT_PARTIAL** (near-miss on the frozen p<=0.05 bar; NOT rescued).
Leave-one-participant rho all 7 positive (0.60-0.71); mean-agg rho 0.393; vision
secondary rho 0.786 (label unchanged per contract); subj07 sole discordant. 165
tests. Bounded: directional cross-participant association present but not clearing
the predeclared threshold; not causal/population/Roy.

**Next (item 36, PARTIAL): freeze the ambiguity -> S2.4H PARTICIPANT HETEROGENEITY
AUDIT** (prespecified secondary summaries only; no participant-level optimization).

---

## S2.2B (2026-08-21): PROSPECTIVE MULTI-ROI CONFIRMATION -- `S2_2B_PROSPECTIVE_MULTI_ROI_PASS` / `H_A_SPATIAL_PROFILE_SUPPORTED`

**Input `b4f472f` -> output `1b4eb08`, server-verified. PROSPECTIVE within-subject
(subj01); freeze committed before outcomes (`b123072`); no population inference; no
verdict.** Doc `43_...md`; artifacts `artifacts/mindcompiler/roy_s2_2b/`.

Tested the S2.1B-frozen H-A prediction on SIX untouched subj01 ROIs (V2, V3, hV4,
ventral, lateral, parietal; V1 = discovery, excluded from primary). Outcome-
independent voxel selection (V1 regression-reproduced a1bc56fe7c55). Primary
endpoint = matched RAW_S2_MATCHED; losses L = B0-B1 (>0 = B1 worse):

  ROI/nvox: V2/29 V3/24 hV4/14 ventral/153 lateral/156 parietal/71
  L_rel_img: +0.113 +0.158 +0.060 +0.154 +0.119 +0.127
  L_perf_RAW:+0.062 +0.076 +0.012 +0.095 +0.041 +0.053

**All 6 ROIs concordant; Criteria A-D all pass; Spearman rho(L_rel_img,
L_perf_RAW)=0.771 -> H_A_SPATIAL_PROFILE_SUPPORTED.** Leave-one-ROI rho all positive
(0.6..0.9); voxel-count confound weak (~0.2). Secondary NOT supported (D1
rho=-0.086, cross-state H-B rho=-0.143) -> mechanism sits at the RAW/beta-prep
stage. Bounded claim: B1 imagery-reliability loss spatially tracks B1 RAW vis2img
loss ACROSS ROIs WITHIN subj01 (not causal/population/mechanism-proven). 155 tests.

**Next (item 29, H-A supported): freeze the subject-level prediction, then S2.3
PROSPECTIVE CROSS-PARTICIPANT RELIABILITY CONFIRMATION** (no new participant run
here).

---

## S2.1B (2026-08-21): BETA-PREP x DENOISING AUDIT -- `S2_1B_MECHANISM_AUDIT_PASS`

**Input `af6532e` -> output `4279446`, server-verified. EXPLORATORY (divergence
already observed); subj01 x V1 only; no fold p-values; no verdict.** Doc `42_...md`;
artifacts `artifacts/mindcompiler/roy_s2_1b/`. Frozen before eval: matched factorial
(`8f3fe9eb`) + hypotheses H-A..H-F.

Matched 2x2 {B0,B1}x{RAW,D1} (+D1b) under the SAME S2 folds; leakage-clean.
NON-INTERPRETIVE vis2img mean r: B0 RAW +0.138 / D1 +0.203 / D1b +0.179; B1 RAW
-0.042 / D1 -0.030 / D1b +0.020. **Primary descriptive finding: the divergence
originates at the BETA-PREPARATION / raw stage** -- B1 is already weak under matched
RAW, so D1 is not the cause. B1 largely PRESERVES B0's pattern (paired r 0.93, RDM
Spearman 0.885, rotation gain ~0.004 -> mostly rescaling not rotation) but has lower
repeat reliability, most strongly in imagery (0.144 vs 0.297) and lower vision SNR
(1.80 vs 2.72). Verdicts: H-A SUPPORTED; H-E PARTIAL; H-B/H-D PARTIAL; H-C/H-F
NOT_SUPPORTED. D1b = corrected symmetric k=2 (recorded deviation from degenerate
frozen k=n). 145 reproduction tests pass.

**Next (Path C, follows the result): S2.2B beta-preparation reliability confirmation
on UNTOUCHED ROIs** -- do NOT scale broadly on V1. Prospective predictions frozen.

---

## S2.0 (2026-08-21): vis2vis->D1->vis2img RECONSTRUCTION -- `S2_D1_FOUNDATION_PASS`

**Input `bdbeabc` -> output `bd26918`, server-verified. Author-independent; normal
forward commits.** Doc `41_S2_...md`; artifacts `artifacts/mindcompiler/roy_s2/`.

Independent reconstruction of the two-stage Roy analysis on subj01 x V1 only.
Frozen (before test eval, config sha e3f32a0d): 4 deterministic 4/2/2 folds
(balanced test coverage); P1 train-only z-score preprocessing; V2V-P0 vis2vis;
**D1_STRICT_CROSSFIT** (train=leave-one-trial-out, val/test=full-train, pooled
within-identity vis2vis, predictions inverse-transformed); ridge 1e-3..1e5x100;
rank cap min(12, conditions, dims, effective) w/ 99%-of-peak; vis2img within-
identity index-aligned. 136 reproduction tests pass.

Execution leakage-clean (0 self-target across the D1 dependency manifest); every
fold/both beta pass finite_frac 1.00. **NON-INTERPRETIVE** (never vs the paper):
B0 D1 vis2img agg r 0.2028 (folds 0.176/0.172/0.192/0.272, positive/consistent);
B1 D1 agg r -0.0300 (folds 0.088/-0.036/-0.107/-0.064). D0 ref: B0 0.165, B1 0.007.

Observation (allowed): **BETA_VERSION_SENSITIVITY_PERSISTS_UNDER_D1** -- B0 works,
B1 near-zero/negative; divergence persists/widens under cross-fit denoising. No
reproduction verdict; D2 = NOT_IDENTIFIABLE_FROM_PUBLIC_METHODS; D1b k-fold
registered-not-run. **Next (follows the result): S2.1B beta-preparation/denoising
interaction audit** -- B0/B1 radically divergent, so do NOT scale to seven ROI yet.

---

## B1 (2026-08-20): SECOND BETA VERSION -- `B1_ENGINEERING_SMOKE_PASS__NON_INTERPRETIVE`

**User-authorized. Output HEAD `104a6e5`, server-verified. Normal forward commits.**
Charter: `40_B1_GATE_CHARTER.md`. Status: `subj01_b1_status.json`.

Acquired the pre-registered second beta version via the hardened downloader:
subj01 func1pt8mm `nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5`
(1,052,494,008 B, sha256 `cd42e680...`, verified; raw HDF5 gitignored, never
committed; anonymous HTTPS, no credentials). Runner now uses a SHA-pinned
beta-version registry (fithrf=B0, fithrf_GLMdenoise_RR=B1); subj01xV1xD0 stay
fixed. Cross-artifact validator generalised to certify `beta_sha` per version.
106 reproduction tests pass; B0 replay preserved bit-for-bit.

**Controlled comparison** (voxel selection uses shared NSD-core ncsnr, so
voxels/splits/pairings are IDENTICAL across B0/B1 -- voxel_hash `a1bc56fe7c55`
for both; only beta VALUES differ). B1 smoke: P0/I0, cross-artifact validation
19/19, finite_frac 1.00. **NON-INTERPRETIVE numbers** (never vs the paper):
B0 vis2vis mean 0.6141 / vis2img 0.1650; B1 vis2vis mean 0.5042 / vis2img 0.0070.

**Explicitly NOT done:** no Roy reproduction verdict; no other ROI/subject; the
`beta_version` ambiguity (GLMdenoise_RR is itself denoised, interacting with
Roy's separate vis2vis denoising) remains UNRESOLVED, deferred to the author
request. See `B0_vs_B1_comparison.json`.

---

## S1.9 (2026-07-31): PRE-B1 HARDENING COMPLETE -- `S1_PRE_B1_HARDENING_PASS`

**Output HEAD `5d17c76`, server-verified. Normal forward commits only.** The subj01 x
V1 x B0 x D0 engineering smoke is now leakage-safe by construction, tested, and
provenance-complete. Authoritative status: `artifacts/mindcompiler/roy_s1/subj01_s1_9_status.json`.

**Done (items 6,7,8,9,10,11,12,13,14):** FittedPipeline lifecycle (leakage prevented
structurally) + structured metrics module now the single execution path (runner migrated,
inline path retired); rank caps (`rank_max`/`r_max`) enforced + recorded; synthetic HDF5 and
ROI/SNR fixtures pin the reversed `(trial,Z,Y,X)` extraction and strict-98th-pct V1 selection;
`pairing.py` child-seed derivation (SHA-256, PYTHONHASHSEED-independent) + P0/P1/I0/I1 row-level
CSV manifests (P0/I0 byte-parity with `smoke_pipeline`); real HTTP-server downloader
resume/interruption tests; CLI pairing-seed contract (P1/I1-without-seed rejected, unused-seed
warned); `artifact_validation.py` cross-artifact validator (catches the S1.8 defect class) with
19/19 checks on the hardened dir. **102 reproduction tests pass (Lane E).**

**Replay preserved bit-for-bit** through the migrated, cross-artifact-validated path:
vis2vis lam=148.49682622544665 rank=5 mean=0.6140929522115459; vis2img lam=2915.0530628251818
rank=1 mean=0.1650281390649757. P1/I1 are sensitivity variants only (no reported number uses
them). Numbers remain NON-INTERPRETIVE (never compared to the Roy paper).

**This PASS authorizes the USER to open a SEPARATE future B1 gate.** It does NOT itself download
B1, run another ROI/subject, or issue a Roy reproduction verdict -- all still unauthorized here.

---

## S1.4 (2026-07-23): first real-data touch -- downloader shipped, B0 acquiring

**Input commit 4765b82 -> output commits 142e32b (downloader+fold-hardening+env fix) and this.**
Server-verified. Normal forward commits only.

**Shipped, tested (29 tests, Lane E, normal collection):** atomic S3 downloader
(`downloader.py`: .part, Range resume + server-ignores-Range fallback, streamed SHA-256,
expected-byte + Content-Length checks, HDF5 signature + read-only h5py open, atomic rename,
.corrupt quarantine, ETag as metadata NEVER a checksum); hardened `select_with_folds`
(>=2 folds for one-SE, grid 1-D/ascending/positive/length-matched, NaN-cell exclusion);
env provenance corrected (real host date 2026-07-23, Europe/Bucharest, session_input_commit).

**B0 acquisition IN PROGRESS** at session end: `subj01/func1pt8mm/nsdimagerybetas_fithrf/
betas_nsdimagery.hdf5`, expected **1,052,494,008 bytes**, downloading via the committed
downloader (background). On completion it atomically renames and writes
`artifacts/mindcompiler/roy_s1/subj01_B0_download.json` with the SHA-256. **Verify that file
exists with status=verified before using the HDF5.** Raw HDF5 is gitignored (`data/nsd/`).

**EXACT RESUME POINT (next session), using the committed plumbing:**
1. Confirm B0 download finished (`subj01_B0_download.json` status=verified; size + sha256).
2. HDF5 schema audit -> `subj01_B0_beta_schema.json` (chunked scan, no full RAM load).
3. Spatial alignment (nibabel): B0 dims vs prf-visualrois/streams/valid/mean/ncsnr -> affine
   hashes; require direct compatibility, no resampling.
4. V1 label resolution + NSD-core 98th-pct SNR selection -> subj01_v1_snr_selection.json.
5. Trial table from design matrices, reconcile to HDF5 trial axis (12 ids, 6+6, 8+8 repeats,
   excluded imgA-1/imgB-1).
6. Fitted preprocessing pipeline object (train-only, item 4) + structured metric report
   (item 5) -- NOT yet built.
7. Freeze one 4/2/2 split + within-identity pairing; run V1 x B0 x D0 vis2vis/vis2img smoke;
   status only, non-interpretive.

**Deferred from S1.4 (budget):** items 4 (pipeline object) and 5 (metric report) as separate
tested modules; downloader edge-case unit tests (the real download exercised the happy path +
HDF5 validation live). No B1, no other ROI, no verdict.

---

## S1.1 (2026-07-17): provenance repaired; real-data smoke BLOCKED by missing env libs

**h5py and nibabel are absent on this Windows CPU host**, so ROI schema, SNR selection, HDF5
audit, trial table and the V1 smoke (objectives 3-11) cannot run here. Downloading 1.8 GB of
betas that cannot be opened was deliberately deferred. Two clean paths (see
`31_S1_1_PROVENANCE_AND_ENV_STATUS.md`): (A) `pip install h5py nibabel` locally, or (B) run S1
on the pod (has the stack; idle; needs a code sync + branch checkout -- inspect first).
**Neither chosen autonomously** (side effects; interactive host decision).

**Provenance repaired:** stale `acquisition_manifest.json` path fixed; duplicate manifest
under `data/` removed; `.gitignore` deduplicated (raw data still ignored, artifacts trackable);
manifest-reference validation test added (3 passing). **Force-push note (§1 of 31):** b2b3ac0
-> bc92c4c was a message-only amend, identical trees; **no further force-pushes on this branch.**

**Next code step (host-independent):** the NumPy model core (reduced-rank ridge, 100-value log
grid, rank@99%-of-peak, per-voxel Pearson) with synthetic-recovery tests -- needs no real data.

---

## GATE M1 S1: DATA ACCESS UNBLOCKED (user completed NSD DUA 2026-07-17)
## Taxonomy: data-access UNBLOCKED · original author code NOT publicly located · original-code reproduction UNAVAILABLE · independent method reproduction AUTHORIZED+FEASIBLE · bitwise replication UNAVAILABLE (no original code/seeds) · **no reproduction verdict** (no data analyzed)

**S3 inspected, minimum subset resolved, 51 small files downloaded + SHA-256 verified (11.4 MB).**
- Space: **func1pt8mm** (func1mm hdf5 is 93 GB, unnecessary). ROIs: **prf-visualrois** (V1-hV4) + **streams** (ventral/lateral/parietal) reproduce Roy's 7 ROIs exactly. SNR: NSD-core **ncsnr**.
- Downloaded (subj01): design matrices, behavioural, ROIs, ncsnr, valid/mean masks. Manifest: `data/manifests/mindcompiler/nsdimagery_metadata_manifest.json`.
- **Staged, not downloaded:** imagery beta HDF5 (~0.9-1.2 GB/subject/version; 16.39 GB for all 8 x both versions). Exact keys/sizes in `artifacts/mindcompiler/roy_s1/acquisition_manifest.json`. Not pulled this session because they are unusable within remaining budget and a half-verified multi-GB artifact would violate the provenance rule.
- **New ambiguity (beta_version):** fithrf vs fithrf_GLMdenoise_RR unconfirmed for Roy; both to be analyzed as a sensitivity variant (GLMdenoise_RR interacts with Roy's separate vis2vis denoising).

**Next session:** download subj01 both-version betas (~1.8 GB) with size+sha256 checks; open the HDF5; one-subject/one-ROI smoke running D0/D1/D2 side by side.

## (superseded) GATE M1 S1 VERDICT: `ROY_S1_BLOCKED_BY_DATA_ACCESS` (execution)
## Reproducibility: `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE` + `INDEPENDENT_METHOD_REPRODUCTION_PREPARED`
## Awaiting: `AWAITING_USER_DUA_COMPLETION`, `AWAITING_AUTHOR_CLARIFICATION`

**S1.0 correction (2026-07-17).** My prior "impossible in principle / no author code / four artifacts unavailable / dataset is CC-BY-NC-ND" statements were overstated or wrong and are retracted (M-030, M-031). The paper **specifies** voxel selection (98th-pct NSD-core SNR), ridge grid (100 log 1e-3..1e5), rank candidates (<=12), the 4/2/2 split, within-identity pairing, and the metric. Denoising **architecture** is specified; only its **fold-level implementation** is ambiguous. Dataset reuse is governed by the **NSD Data Access Agreement**. **Independent method reproduction is feasible after data access** with preregistered sensitivity variants (`30_..._AMBIGUITY_REGISTRY.csv`). Scaffold `roy_method_reproduction/` built (paper-specified pieces only; ambiguities default to `author_clarification_required` and fail loudly); 7 tests passing.

**Issued 2026-07-17** before any download, per the early-status rule. Detail in
`27_ROY_S1_DATA_AND_CODE_INVENTORY.md`; machine-readable in
`artifacts/mindcompiler/roy_s1/S1_VERDICT.json`. **No data downloaded, no reproduction run.**

**[SUPERSEDED by the S1.0 correction above -- retained only as the record of my error.]** The
text originally here said "no author code exists", "exact reproduction impossible in principle",
a "permanent" approximate ceiling, "four artifacts UNAVAILABLE", and a CC-BY-NC-ND dataset
license. All of those are retracted (M-030/M-031); see the S1.0 correction block above and
`27_ROY_S1_DATA_AND_CODE_INVENTORY.md` §0. User action to unblock: complete the NSD Data
Access Agreement personally and share the non-secret access route; optionally send the author
request in `21` (Request 3).

---

## (superseded) GATE M1.3 STATUS: `GATE_M1_S0_FRAMEWORK_VALIDATED__OPERATING_CHARACTERISTICS_UNRESOLVED`

**Corrected 2026-07-17.** Detail in `26_GATE_M1_3_CORRECTIONS.md`. Two M1.2 overreaches of
mine are **retracted**:

1. **"max FPR 0.000, criterion met" was statistically invalid.** It came from 15 evaluation
   seeds. Clopper-Pearson 95% CI for 0/15 is **[0.000, 0.218]** -- the true FPR could be 21.8%.
   **n >= 72 zero-event trials are needed merely to bound FPR <= 0.05**; the target is 1000.
   Every FPR must now carry an exact binomial interval.
2. **"the nuisance is stronger than the transformation" was an unmatched comparison.** W1f and
   W2 were not matched on source/target reliability, predictive correlation, effective rank,
   or explainable variance. Withdrawn. The legitimate claim is only that **some
   latent-common-cause configurations are observationally indistinguishable from a
   condition-level transformation.**

**Task P (predictive, Level B) and Task C (latent-confound sensitivity) are now separate.** The
M1.2 error was using an *unrestricted* worst-case latent confound to set the primary threshold
for a *predictive* test -- which guarantees zero power by construction. Task C is a
**robustness frontier**, not a threshold.

**W1f reclassified** as `W1f_condition_locked_latent_common_cause` -- a stable latent *content*
property (salience, memorability, unmeasured semantics). **It is not "attention"**; that label
was mine and was wrong.

**Unchanged and important:** W1c (incomplete observed features) yields positive `dR2` with **no
transformation**, so any positive real result is always W1c-compatible; NSD-Imagery reaches
**Level B** at best.

**S1 is unblocked and is the next action** -- exact reproduction does not depend on Task C.

---

## (superseded) GATE M1.2 STATUS

**Downgraded 2026-07-17** from `..._CHECK_PASSED`. Detail in `25_GATE_M1_2_CALIBRATION.md`.
The M1.1 detector had ~0.10 FPR against W1 (short of 0.05) and reported power 1.00 at
n_id=64 where the effect mean was **-0.002** -- "detecting" a transform for being merely
*less negative* than the null. Both fixed: least-favourable per-null calibration plus a
**positive** smallest effect of interest (`delta_min`).

**Canonical estimand:** *incremental source-state predictive value beyond the preregistered
stimulus-feature battery*, `dR2_source|F = R2(Y|F,X) - R2(Y|F)`. Positive does **not** prove a
neural mechanism and does **not** exclude unmeasured common causes. Never write "unique neural
contribution" unqualified.

**THE KEY RESULT -- valid but powerless.** Against a 6-world least-favourable null family:
max FPR **0.000** (criterion met) but **power 0.00**. W1f (one unobserved shared
condition-level attentional nuisance) yields mean dR2 **+0.470**, *larger than the genuine
transform's +0.213*, so the honest threshold sits above the alternative. W1c (incomplete
observed features) yields **positive** dR2 with **no transform**, and since no real battery is
exhaustive, a positive real result is always W1c-compatible.

> An honest E-M1 **cannot** separate a real condition-level transformation from a shared
> unobserved nuisance at n_id=12 unless that nuisance is **measured or excluded by design**.
> A design finding, not a tuning problem -- and the strongest argument yet that the flagship
> requires prospective acquisition.

**Claim ceiling:** NSD-Imagery reaches **Level B** (incremental value over *measured*
features) at best. **Level C** (transformation-specific) needs MindStates.
**S1 NOT STARTED** -- no data downloaded; provenance-verification budget unavailable.

---

## (superseded) GATE M1.1 STATUS

**Issued 2026-07-17.** No Roy verdict — no real data has been run. Detail in
`24_GATE_M1_1_CORRECTED_ESTIMAND.md`.

**Corrected the estimand.** The Gate M1 sim modelled *single-trial* coupling, which Roy's
random cross-run pairing cannot observe. Four condition-level worlds now: W0 (null), W1
(feature mediation), W2 (genuine transform), W3 (mixture). **Primary estimand: unique neural
contribution beyond stimulus features (LOIO)** — because W1 generalizes *without* a
transformation, so raw held-out prediction is not enough. Measured (12 seeds, n_id=12):
W0 −0.32, **W1 −0.15**, **W2 +0.28**, W3 +0.07 → the estimand separates null from transform.
13 ground-truth tests passing.

**Three of my own claims retracted/corrected** (M-017/M-019, and the withdrawn 0.447/0.93
sensitivity numbers): the 12-identity "validity" claim is narrowed to "sensitivity unknown";
**single-trial coupling does NOT average away** — a linear single-trial map survives as a
condition-level transform (M-019), making the identifiability limit *stronger*, not weaker.

**Next gates (S1–S3 required before any Roy verdict):** real-data reproduction, identity-
baseline eval, content-held-out eval. S4 = Roy Dataset 2 (512 conditions) confirmation.

**Not artifactual.** The simulation shows only that the repeat-level protocol does not by
itself identify cross-content generalization (M-020). Roy's result has not been reproduced.

---

## ✅ GATE M0.2 VERDICT: `PUBLIC_TWO_STATE_PROGRAM_FEASIBLE__THREE_STATE_ACCESS_REQUIRED`

**Supersedes M0.1.** Issued 2026-07-17. **M0.1's `ONLY_PROSPECTIVE_DISCOVERY_REMAINS` was too
restrictive and rested on a false statement of mine.**

### The correction

**I wrote:** *"No public dataset has ≥3 mental states on the same content at trial level."*
**False as an existence claim.** **Oedekoven et al. (2017)** measured **21 participants
watching, immediately retrieving, and retrieving after one week, the same 24 videos** — three
states, identical content. The true statement is:

> Three-state data **exist and are verified**; they are **not openly downloadable at trial
> level** (NeuroVault 2814 = group t-maps only); trial-level is **available on reasonable
> request**.

*"No dataset exists"* terminates a retrospective program. *"Data exist but need an author
request"* makes it an **access task**. I conflated them, and it changed the verdict.

### Verdict, separated as required

| | |
|---|---|
| **Downloadable NOW** | **Li, Yang & Bao 2026** (Dryad 7.37 GB) — perception + working memory · **NSD-Imagery** — perception + imagery · **ds001132** — movie + spoken recall (confounded) |
| **Requires author approval** | **Oedekoven 2017 trial-level** — the only verified encoding→immediate→delayed sequence. Draft in `21`, **NOT SENT** |
| **Supports a retrospective paper** | **E-M1** (is the Roy transformation neural or semantic? — nobody has tested this) + **E-M2** (perception→WM reorganization). Both public, both runnable now |
| **Requires prospective scanning** | causal path dependence · controlled multi-state factorial · vividness/delay manipulation · prospective counterfactual prediction · closed-loop |
| **Could be A\*-competitive** | **E-M3** (encoding→immediate→delayed predictive factorization, 21 subjects) **if access is granted**; otherwise only Track P |

### The theory changed — and the data forced it

Roy et al.: perception→imagery **contracts** early-visual dimensionality.
Li, Yang & Bao: perception→WM **expands spatially** — ipsilateral representation across
**70–90% of ipsilateral LOC**, exceeding unilateral perception.

> **Two internally-generated states reorganize information in opposite geometric directions.**
> A pure "information contraction" thesis is **refuted by already-published data**. MINDIR's
> object becomes **reorganization** — contraction, expansion, rotation, redistribution, noise —
> and the discovery target is whether *which* information is retained follows reproducible
> regularities. **Do not force "contraction" onto expansion.**

### Strongest fatal objection (still unresolved)

> *"E-M1 is a control experiment on someone else's preprint; E-M2 is a re-analysis; E-M3 needs
> data you may not get. Where is the discovery?"*

Honest answer: **E-M3 is the only retrospective candidate for a discovery-level result, and it
is gated on an email the user must send.** Everything else is validity work — worth doing,
publishable, not a flagship.

### Immediate next action

**E-M1 on NSD-Imagery.** Public, no GPU, no access needed, and both outcomes publish: either
the imagery transformation is neural (validating the substrate) or it is a stimulus-identity
effect (a substantive correction to an actively-cited preprint).

---

## (superseded) GATE M0.1 VERDICT: `ONLY_PROSPECTIVE_DISCOVERY_REMAINS`

**Supersedes the M0 verdict below.** Issued 2026-07-17 after the **full** Roy et al. text
(PMC12424947), which **corrected three errors in my own prior report** (`01` §0).

### The corrections matter — I had overstated the pre-emption

1. The **"25–50% variance"** figure is an **alignment ratio** (subspace misalignment), *not*
   an information-loss figure.
2. **I claimed the map was "non-invertible, contradicted by measurement". They never ran the
   inverse.** I stated as measured fact something untested — and generalised an early-visual
   result to the whole hierarchy. In **ventral/lateral/parietal, alignment is ~100%: "imagery
   and visual subspaces occupy identical subspaces."**
3. **Dimensionality halving is early-visual only.** Imagery dimensionality is nearly *constant*
   across ROIs; the gap closes because *vision expands*, not because imagery contracts.

> Roy et al. **support** MINDIR's premise (structured, ROI-dependent contraction) more than
> they refute it. They also pre-empt part of it. My previous "evidence points away from an
> algebra" was too strong and is withdrawn.

### Verdict, separated

| Dimension | Status |
|---|---|
| **Conceptual novelty** | **Substantially pre-empted.** Roy et al. own the vision→imagery transformation, within-subject prediction (r ≈ 0.3–0.5), ROI-specific analysis, and the refutation of "imagery = weak vision". The memory-compression literature (episodic dimensionality transformation; *A compressed code for memory discrimination*) owns directional perception→memory contraction. **H3 is not a new idea.** |
| **Methodological novelty** | **REAL but narrow.** Nobody has tested (a) whether the transformation is **neural or semantic** (H7), (b) **cross-subject universality** of contraction spectra (H1/H2), (c) the **inverse**, (d) **≥3-state composition** (H4/H5). |
| **Public-data feasibility** | **INSUFFICIENT for discovery.** **No audited public dataset has ≥3 states on the same content at trial level.** NSD-Imagery has **12 content identities** — too few for serious held-out-content work. |
| **Prospective necessity** | **YES** for every discovery-level claim (path dependence, contraction laws, prospective degradation prediction). |
| **A\* potential** | **Only via MindStates-7T.** Not reachable by modelling public data. |

**`ONLY_PROSPECTIVE_DISCOVERY_REMAINS` is not termination.** One genuine public-data
experiment survives and should run — it is a **validity study, not a discovery**.

### Strongest surviving gap — and it is a validity question about someone else's result

**Roy et al. never ran a semantic control.** Their analysis is *"purely
voxel-activity-to-voxel-activity."* **Nobody has shown the imagery transformation is neural
rather than a stimulus-identity effect.** With only 12 conditions in Dataset 1, a model could
predict imagery activity by implicitly identifying *which of 12 stimuli* it was — with no
state-transformation content whatsoever.

That is testable **now**, on public data, with no GPU: **experiment E-M1 (H7)**.
Both outcomes publish. A negative would be a substantive correction to an actively-cited
preprint.

### Strongest fatal objection to MINDIR (unresolved)

> *"Your surviving contribution is a control experiment on a preprint, and your flagship needs
> a scanner you do not have. The contraction phenomenon is already established from two
> directions. What is the discovery?"*

**There is currently no answer that does not require MindStates-7T.** Recorded, not resolved.

### Verdict caveat — 4 dataset families remain UNAUDITED

Generic Object Decoding (Horikawa/Kamitani), **memory-reinstatement**, **working-memory**, and
MEG/EEG imagery datasets were not audited. **Memory-reinstatement and working-memory are the
highest-value remaining audits**: either could supply a *third state* on shared content and
would upgrade this verdict toward `PARTIALLY_PREEMPTED_BUT_FLAGSHIP_SURVIVES`. **The verdict
is provisional pending those two audits.**

---

## (superseded) GATE M0 VERDICT: `PROSPECTIVE_PROGRAM_REQUIRED`

**The retrospective flagship is pre-empted. The surviving claims are untestable on public data.**

| Dimension | Verdict |
|---|---|
| **Conceptual novelty** | **LARGELY PRE-EMPTED.** Roy, Breedlove, St-Yves, Kay & Naselaris (2025) introduce *"the imagery transformation — a mapping from visual to imagery activity patterns evoked by the same stimulus"*, estimate it per visual area on **two 7T datasets**, and **predict held-out imagery activity**. That is MINDCOMPILER's core move, our E2 and our E8, from the field's leading lab, on our own substrate. |
| **Methodological novelty** | **PARTIAL.** Composition, cross-subject transport *of the operator itself*, and calibrated probabilistic target-state prediction were not found. `S_p` (hyperalignment/SRM) and `T_s` machinery (neural/Koopman operators, incl. compositional variants) are mature and **not ours to claim**. |
| **Public-data feasibility** | **INSUFFICIENT for the flagship.** No public dataset has **≥3 mental states on the same content at trial level** — composition, non-commutativity and interpolation are therefore **unmeasurable**. NSD-Imagery is 4 subjects / 18 stimuli, already used by Roy et al. |
| **Prospective necessity** | **YES.** Every surviving claim requires MindStates-class acquisition. |
| **A\* potential** | **REDUCED, not zero** — and now conditional on new measurements, not on modelling. |

**`PROSPECTIVE_PROGRAM_REQUIRED` is not termination** (mission §12). The platform and
retrospective tests may proceed; the *flagship claim* needs new data.

### The finding that decides it — and it is scientific, not bibliographic

Roy et al. measured what the perception→imagery transformation *is*: in early visual cortex it
**halves the active dimensions and reorients them**; reconstructions explain only **25–50% of
variance**; imagery occupies a **distinct subspace**.

1. **The map is strongly non-invertible.** `T_{b→a}∘T_{a→b} ≈ I` is **contradicted by
   measurement**, not untested. Our inverse test has a published answer: *no*.
2. **Composition compounds the loss.** A path routed through imagery is degraded by
   construction; a composed path beating a direct one is *a priori* implausible here.
3. **H1 (additive state offset) is close to dead**; H2/H3's "shared manifold" is weakened —
   imagery is a *different subspace*, not a rescaled one.

With **Spera et al. 2026** (zero-shot perception→imagery **at chance**, CLIP 48.94% vs 50%),
the picture is coherent: perception→imagery is **lossy, dimension-halving, subspace-shifting**
— learnable within-subject with paired data, carrying **no zero-shot transfer**.

> **Do not retrofit an algebra to this.** The published measurements point away from
> composability and invertibility. That is the answer arriving early and cheaply — which is
> what Gate M0 is for.

### Mandatory next action before *any* redesign

**Read Roy et al. in full.** bioRxiv returned **HTTP 403**; only abstract + PubMed extraction
were obtained. Their composition / inversion / cross-subject / semantic-control status is
**UNCONFIRMED** — "not mentioned" is not "not done". If they tested composition, the last
survivors die too. **Zero cost, maximal information.**

---

## Mission

Investigate whether mental states are related by structured, composable transformations over
a shared neural content manifold:
`y_{p,s,m} = O_{p,m}( T_s(c) ) + ε`

**The object of study is the transformation between mental states**, not the decoder. The
existing decoding stack is substrate.

**The decisive prospective demonstration:** given neural activity from one state, predict the
neural activity the *same* participant would produce for the *same* content in *another*
state — validated on **held-out measured neural data**. A reconstructed image or a semantic
match is **not** a neural counterfactual.

## Terminology status — WORKING HYPOTHESES ONLY

`universal`, `causal`, `algebra`, `operator`, `counterfactual`, `brain-to-brain`,
`mental-state compiler` are **project vocabulary, not claims**. None has passed its test.
None may appear as a scientific claim until it does.

## Gate M0 progress

| § | Task | State |
|---|---|---|
| 3.1 | Reconcile local / remote / pod | ✅ **DONE** |
| 3.2 | Archive PCD/NCD | ✅ **DONE** — `docs/research/archive/PCD_NCD_TERMINATION_MEMO.md` (`c1095de`) |
| 3.3 | Create + publish branch | ✅ **DONE** — verified on server via `git ls-remote` |
| 4 | Research OS scaffold | 🔶 **PARTIAL** — this file + `19_SESSION_HANDOFF.md` only |
| 5 | Frontier literature review + novelty verdict | ❌ **NOT STARTED — blocks everything** |
| 6 | Formal operator algebra (H0–H7) | ❌ NOT STARTED |
| 7 | Dataset/modality matrix + adapters | ❌ NOT STARTED |
| 8 | Falsification ladder B0–B5 | ❌ NOT STARTED |
| 18 | Synthetic operator-recovery benchmarks | ❌ NOT STARTED |

**No novelty verdict has been issued.** The permitted set is
`MOONSHOT_NOVELTY_SUPPORTED` / `PARTIALLY_OVERLAPPING_REQUIRES_REDESIGN` /
`NOVELTY_INSUFFICIENT` / `DATA_INSUFFICIENT_FOR_FLAGSHIP` / `PROSPECTIVE_PROGRAM_REQUIRED`.
Issuing one before the review would repeat exactly the error this program is built to avoid.

## Verified state

| | |
|---|---|
| Remote | `origin/research/mindcompiler-neural-state-operators` = `c1095de` — **server-verified** |
| Parent published | `origin/feature/predictive-cortical-decoder` = `c1095de`; **nothing unpushed** |
| Tests | 115 program tests passing (inherited); 10 pre-existing env failures, verified unrelated |
| Pod | `orchestraiq-jupyter-54644cff87-gz6n2` — no processes, GPU idle. Pod git HEAD `b96affa` **matches no commit** (hand-copied files, F-005). **Pod is NOT synced to this branch.** |

## Inherited assets (from the archive memo §4)

Verified clean NSD splits (train ∩ val = 0; ∩ SHARED1000 = 0) · `partial_conjunction.py`
(r-of-n compound test + 21 simulations — directly applicable to multi-property algebra
claims) · `arm_c_partner.py` (content-shuffled controls + 17 invariant tests — mandatory for
any transport claim) · matched-control family + `assert_param_parity` · per-ROI tokenisation
with low-rank subject adapters (reusable as an **observation model** `O_{p,m}`).

**Inherited rule, program-wide:** *every interpreted quantity must have an identifying
objective, enforced by a gradient test.* PCD's kappa heads violated it and produced stable,
plausible figures from random weights. `T_s`, `S_p` and all uncertainties are interpreted
quantities and inherit this rule.

## The two risks that most likely kill this program

1. **Semantic shortcut (§H6).** Apparent neural transport explained entirely by shared
   semantic labels or pretrained embeddings. **B5 must be built before any operator claim.**
2. **Data insufficiency.** Neural counterfactual prediction needs *the same content measured
   in multiple states, paired, at trial level*. NSD-Imagery has **4 subjects, 18 stimuli**,
   and Spera et al. showed a stronger decoder is **at chance zero-shot** on it — so the
   obvious substrate may be too thin for the flagship. **This is the likeliest route to
   `PROSPECTIVE_PROGRAM_REQUIRED`, and finding that out is a legitimate M0 outcome.**

## Next five actions

1. **§5 frontier literature review** → `02_FRONTIER_LITERATURE_REVIEW.md`,
   `03_NOVELTY_AND_OVERLAP_MATRIX.csv`, adversarial novelty verdict. **Zero GPU. Blocks all.**
   Priority families: hyperalignment / shared response models; neural/Koopman operators;
   causal representation learning; perception-vs-imagery transformation work; cross-subject
   neural translation; optimal transport for neural data.
2. **§7 dataset matrix** → `07_DATASET_AND_MODALITY_MATRIX.csv`. Decide honestly whether
   *any* public data supports trial-level paired multi-state content.
3. **§6 formal theory** → `06_FORMAL_OPERATOR_ALGEBRA.md` with H0–H7 and per-property
   estimand / null / baseline / threshold / kill criterion.
4. **§18 synthetic recovery** — must recover *absence* of composition when absent. **A model
   that always finds an algebra fails this gate.** Build before touching real data.
5. **§8 B0–B5**, with **B5 (semantic shortcut) first** among the transport baselines.

## Standing prohibitions (inherited + new)

- All PCD/NCD FORBIDDEN claims carry forward (archive memo §5) and do not expire.
- **No "first" claims.** Novelty must be conceptual and experimental, never nominal.
- **Cycle consistency alone is not evidence** — degenerate solutions satisfy it.
- **Never call a reconstruction or semantic match a neural counterfactual.**
- **Never call simulation prospective validation.**
- Do not treat trials, voxels, seeds, or reconstruction samples as independent participants.
- Do not begin full-scale GPU training before the relevant promotion gate.
