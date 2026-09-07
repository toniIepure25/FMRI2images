# O2.3C-PREP-RUNTIME — Bare-Metal Manual-Environment Runtime Amendment

**Amendment class:** `TECHNICAL_RUNTIME_PACKAGING_ONLY` · **Input HEAD:** `fbab942`
**Contract SHA:** `24e6ca83` · **Phase-0 config preserved:** `2d1a26a5` · **Resumes frozen O2.3C SHA:** `528f23eb`
**Status:** `RUNTIME_AMENDMENT_FROZEN` (terminal status pending R1–R4)

This amendment changes **only the executable packaging/runtime** of the already-frozen O2.3C-PREP Lane B.
It does **not** alter any scientific PREP methodology, benchmark threshold, registration policy, lane
algorithm (**fMRIPrep 25.2.5**), O2/O3 status, or the historical infrastructure blocker committed at
`fbab942`. The historical first-attempt verdict `O2_3C_REST_PRODUCT_INCOMPATIBLE` and `O3_NOT_READY`
are carried forward unchanged.

## Why admissible

- Official fMRIPrep documentation supports a **Manually Prepared Environment (Python 3.10+)**.
- Nipype reuses completed intermediate nodes after interruption when the same persistent `-w WORK_DIR`
  is reused — enabling resumable execution on a preemptible pod when the workdir lives on the persistent PVC.
- No Phase-1 benchmark outcome has yet been produced, so nothing scientific is being re-decided.

**Change:** FROM `nipreps/fmriprep:25.2.5` container → TO `fMRIPrep 25.2.5 manually prepared environment
with container-equivalent pinned external dependencies`. Algorithm, version, input data, command options,
and benchmark are **unchanged**.

## Runtime probe evidence (pod, this session)

| Fact | Value |
|---|---|
| Privileges | `uid=1000(jovyan)`, **no sudo** (password required) → user-space only |
| PVC `/home/jovyan/work` | **107 TB free**, persistent NFS, writable → survives pod replacement |
| Pre-existing neuro tools | **none** (clean slate) |
| Network | conda-forge `200`, GitHub `200`, TemplateFlow S3 `200`, PyPI `200`, FSL `302` |
| Container runtime | still blocked — `unshare` mount-propagation `Permission denied` (bare-metal only route) |
| Capacity | 256 CPU, ~970 GB RAM available |

## R1 — external dependency audit (targets frozen; observed pending install)

Container-equivalent pins from the official fMRIPrep 25.2.5 manual-environment docs + Dockerfile
(base `ghcr.io/nipreps/fmriprep-base:20251006`, `pixi.lock`):

| Dependency | Pinned version | Executable | Source | User-space installable |
|---|---|---|---|---|
| Python | 3.12 | `python` | conda-forge (micromamba) | yes |
| fmriprep | 25.2.5 | `fmriprep` | PyPI | yes |
| FSL | 6.0.7.7 | `flirt` | `fslinstaller.py` | yes |
| ANTs | 2.5.1 | `antsRegistration` | conda-forge | yes |
| AFNI | 24.0.05 | `3dTshift` | conda-forge | yes |
| FreeSurfer | 7.3.2 | `mri_convert`/`bbregister` *(only if invoked)* | official tarball | yes (execution license-gated) |
| connectome-workbench | 1.5.0 | `wb_command` | conda-forge | yes |
| bids-validator | 1.14.10 | `bids-validator` | npm | yes |
| templateflow | resolved-by-fmriprep | *(pkg)* | PyPI + S3 cache | yes |

All sources are reachable and installable without elevated privileges. Exact version-match verification
is recorded in `baremetal_environment_manifest.json` after install.

## FreeSurfer license — determination rule (no assumption)

The frozen Lane-B path runs fMRIPrep BOLD-only (SDC, slice-timing, rigid motion, confounds) with
`--fs-no-reconall`; WM/CSF and DK-68 come from **existing** NSD FreeSurfer `aparc+aseg` mapped **outside**
fMRIPrep via `nsdcode` (reads of released derivatives + transforms, not `recon-all`). Whether any
**licensed FreeSurfer binary** is invoked inside fMRIPrep hinges on the BOLD→anat coregistration node
selection under `--fs-no-reconall`. This will be **proven from the actual workflow graph/runtime** before
the requirement is asserted or removed. The amendment will **not** fabricate or download a license; if the
frozen path genuinely requires one, it stops at `O2_3C_PREP_FS_LICENSE_REQUIRED` and reports the exact tool.

## Persistent layout (all on PVC — survives pod replacement)

- env prefix: `/home/jovyan/work/envs/fmriprep-25.2.5-baremetal`
- work dir (`-w`): `/home/jovyan/work/o2_3c_prep_fmriprep/work`
- output dir: `/home/jovyan/work/derivatives/o2_3c_prep`
- TemplateFlow cache: `/home/jovyan/work/o2_3c_prep_fmriprep/templateflow`

## Phases & hard stops

- **R0** freeze (this doc + `runtime_amendment_contract.json`) — committed before install. ✓
- **R1** dependency audit — targets frozen; observed versions pending install.
- **R2** persistent env install on PVC.
- **R3** restart/resume certification (non-scientific): CLI starts; binaries resolve; TemplateFlow
  resolves; workdir/outdir writable; workdir survives pod replacement; a completed Nipype node is reused
  after restart → `PERSISTENT_NIPYPE_RESUME_CERTIFIED`.
- **R4** single-run feasibility — the predeclared `subj01 / ses-nsd01 / run-01` only, unchanged frozen
  benchmark, persistent `-w`, ≤16 effective CPUs, no `--clean-workdir`; rerun identical command on preempt.

**Hard stops (no looping):** a single indispensable Nipype node whose uninterrupted runtime exceeds the
pod uptime window → `O2_3C_PREP_BAREMETAL_PREEMPTION_BLOCKED`; a required binary not installable
user-space, versions not pinnable defensibly, or NFS-induced Nipype cache/correctness failure →
`O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE`; required FreeSurfer license absent →
`O2_3C_PREP_FS_LICENSE_REQUIRED`; bare-metal not certifiable / not container-equivalent enough →
`O2_3C_PREP_RUNTIME_HOST_REQUIRED` / `O2_3C_PREP_RUNTIME_AMENDMENT_FAILURE`.

On `subj01` benchmark completion it is evaluated against the **unchanged** frozen thresholds: PASS →
continue the original O2.3C-PREP workflow with this runtime; FAIL → genuine
`O2_3C_PREP_TASK_BENCHMARK_FAILURE` (no tuning). This amendment modifies no scientific Track-O status.

## Determination (SEALED) — `O2_3C_PREP_FS_LICENSE_REQUIRED`

**The bare-metal runtime is genuinely installable and the container blocker is bypassed.** On the pod
this session, a manually-prepared environment was built **user-space on the persistent PVC** with no
sudo: micromamba (static) → conda-forge `python=3.12` (observed **3.12.14**) → `pip install
fmriprep==25.2.5` (observed **fmriprep 25.2.5**, **nipype 1.12.0**), all importable. Runtime packaging is
therefore **not** the obstacle.

**The binding constraint is the FreeSurfer license**, proven from the *installed* fMRIPrep 25.2.5 source
(the actual runtime — corroborated by the 25.2.5 GitHub tag), not assumed:

- **FACT 1 — a licensed FreeSurfer binary is in the coregistration path.** `init_fsl_bbr_wf` — the branch
  used when FreeSurfer is disabled (`--fs-no-reconall`) — initializes BOLD→anat coregistration with
  FreeSurfer's `mri_coreg` (`nipype.interfaces.freesurfer.MRICoreg`).
  `site-packages/fmriprep/workflows/bold/registration.py:283` (import), `:333` (node); the docstring notes
  this is "equivalent to running `bbregister --init-coreg`".
- **FACT 2 — the license gate is unconditional.** `build_workflow()` runs
  `if not check_valid_fs_license(): return_code = 126` (`niworkflows.utils.misc.check_valid_fs_license`)
  at `site-packages/fmriprep/cli/workflow.py:119` → `:137`. There is **no `run_reconall` guard** in that
  file (grep: 0 matches) — fMRIPrep 25.2.5 refuses to run at all without a valid license, regardless of
  `--fs-no-reconall`.

**It cannot be avoided within the frozen methodology:** the gate fires before any node executes, and
replacing the `mri_coreg` initializer would be a forbidden registration-policy/methodology change.

**License availability:** none present on the PVC or home; `FS_LICENSE`/`FREESURFER_HOME` unset; the
declared `FS_LICENSE` path is absent. Per the brief the license is **neither fabricated nor downloaded**.

**Terminal status:** `O2_3C_PREP_FS_LICENSE_REQUIRED`. Heavy binaries (ANTs/AFNI/FSL/workbench) were not
installed because this hard stop preempts them. Scientific `terminal_execution_status` stays `null`;
Phase-0 `2d1a26a5`, `O2_3C_REST_PRODUCT_INCOMPATIBLE`, `O3_NOT_READY`, and the container-path blocker at
`fbab942` are all preserved.

**Exact unblock:** place a valid FreeSurfer `license.txt` at the declared `FS_LICENSE` path on the PVC
(`/home/jovyan/work/o2_3c_prep_fmriprep/freesurfer_license.txt`). The bare-metal env already proven here
then completes with the heavy binaries and proceeds through R3 (resume certification) → R4 (frozen
`subj01/ses-nsd01/run-01` benchmark) under the unchanged methodology. The FreeSurfer license is free for
individual use from the FreeSurfer project; it is the user's to obtain and place — not mine to supply.
