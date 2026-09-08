# O2.3C-PREP-RUNTIME — Path A: Official Docker-Host Migration

**Migration class:** `TECHNICAL_HOST_RUNTIME_MIGRATION_ONLY` · **Source HEAD:** `4377100`
**Resolution path:** `O2_3C_PREP_RUNTIME_HOST_REQUIRED` · **Status:** `BLOCKED_PENDING_HOST_PROVISIONING`

Path A (user-selected) runs the **official** `nipreps/fmriprep:25.2.5` container on a stable
non-preemptible Docker host. This is a host/runtime migration only — it changes **no** scientific
methodology, Phase-0 freeze `2d1a26a5`, benchmark manifest/thresholds, registration or output-space
policy, Lane-B algorithm, runtime-amendment history, `O2/O2.2/O2.3A` states, historical
`O2_3C_REST_PRODUCT_INCOMPATIBLE`, or `O3_NOT_READY`. The bare-metal terminal
`O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE @ 4377100` stands unchanged as history. No bare-metal re-freeze.

## What is ready (host-independent, verified this session)

- **FreeSurfer license — in hand.** Placed on the old PVC, hash `6f7afab5…` (133 B), structurally valid.
  Recorded by **hash only**; contents never printed or committed; it will be transferred securely to the
  new host and validated in-container.
- **Benchmark inputs — all fetchable from public NSD S3** (unsigned, verified 2026-09-08):
  raw `task-nsdcore run-01` bold+sbref+events; run-01 `phasediff`+`magnitude1/2` fieldmaps; `ses-nsdanat`
  anatomy; `freesurfer/subj01/mri/aparc+aseg.mgz`; prepared `func1pt8mm/{timeseries,motion,design}`
  ground-truth. See `benchmark_input_manifest.json`. **Feasibility-first:** fetch only these subj01 inputs
  directly from S3 to the host — not the full 262 GB, and not via the old pod.
- **Frozen benchmark command** derived from Phase-0 `2d1a26a5` (BOLD-only: phasediff SDC, BIDS
  slice-timing, rigid motion, no smoothing/AROMA/GSR, confounds; `--fs-no-reconall`; func1pt8mm via the
  separate frozen ANTs rigid+affine step) is recorded in `host_migration_provenance.json`, to be finalized
  on the host against `fmriprep 25.2.5 --help`.

## The one blocker — a qualifying host

The migration needs a **non-preemptible Linux host with a working Docker Engine** meeting the frozen spec
(≥16 vCPU, ≥64 GB RAM, ≥500 GB SSD — 1 TB preferred; linux/amd64; internet + public-S3 access; **no GPU**).
Neither host reachable from this session qualifies:

| Candidate | Verdict |
|---|---|
| This Windows workstation | **INADEQUATE** — 12 logical CPU, **13.9 GB RAM**, ~38 GB total free SSD; Docker not operational (no CLI, service unregistered, `docker-desktop` WSL stopped). Fails vCPU/RAM/SSD by wide margins. |
| Old RunAI pod | **EXCLUDED** — preemptible, container runtime blocked by pod securityContext (the host we are migrating away from). |

So Path A is authorized and fully prepared up to the host boundary, but **provisioning/accessing the host
is the user's to provide.** I will not run fMRIPrep on the inadequate laptop, fabricate a benchmark, or
provision paid cloud resources on my own authority.

## Resume plan (the moment a host is provided)

1. **Provenance (before R4):** record host OS/kernel/Docker version, CPU/RAM/storage, `nipreps/fmriprep:25.2.5`
   **tag + full repo digest** (`@sha256:…`), in-container `fmriprep --version == 25.2.5`, license
   destination path + hash + validation, data source paths, workdir/outdir, exact command line. Commit +
   push **before** any benchmark outcome.
2. **R3 — resume certification:** image starts; version 25.2.5; FS license validates; mounts readable;
   workdir/outdir writable; TemplateFlow resolves; restart with the same persistent workdir preserves
   Nipype intermediates; a completed node is reused, not recomputed → `PERSISTENT_NIPYPE_RESUME_CERTIFIED`.
3. **R4 — frozen benchmark:** exactly `subj01 / ses-nsd01 / run-01`, unchanged thresholds/metrics/
   references, 8–16 effective CPUs, persistent `-w`, no `--clean-workdir`. PASS → O2.3C-PREP Phases 2–4
   (cohort rest preprocessing, motion, WM/CSF, DK mapping, cohort certification), then — only on
   `O2_3C_PREP_PASS` — the original scientific gate `O2.3C-RESUME` (frozen SHA `528f23eb`). FAIL →
   `O2_3C_PREP_TASK_BENCHMARK_FAILURE` (no tuning, no method switch).

## What I need from you

SSH access (host/IP, user, key) to a host meeting the spec **or** explicit authorization to provision a
specific cloud instance (provider, region, instance type, and how billing is handled). With that, I resume
at step 1 immediately — license and data paths are already settled.
