# 40 — B1 GATE CHARTER (subj01 × V1 × D0, second beta version)

**Opened:** 2026-08-20 · **Authorized by the user.** · **Predecessor:** `S1_PRE_B1_HARDENING_PASS`.
**Branch:** `research/mindcompiler-neural-state-operators`. Normal forward commits only.

## What B1 is

B1 brings in a **second beta version** of the SAME subj01 NSD-Imagery data and runs
the already-hardened, leakage-safe pipeline on it, **non-interpretively** (pipeline
validation only). B1 is the pre-registered second beta version — not a new guess:
the acquisition manifest staged both versions for exactly this purpose
(`beta_version` sensitivity row in the ambiguity registry).

- **B0 (done):** `…/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.hdf5`
  (1,052,494,008 B, sha256 `31485ff0…`, verified).
- **B1 (this gate):** `…/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5`
  (expected 1,052,494,008 B; sha256 computed on download — NSD publishes none).

Because V1 voxel selection uses the **NSD-core** `betas_fithrf/ncsnr.nii.gz` (shared,
not imagery-derived), the 27 V1 voxels, trial table, splits, and pairings are
**identical** across B0 and B1. Only the beta *values* differ → a clean controlled
comparison of the pipeline on a different GLM estimation.

## In scope

1. Download the B1 HDF5 via the hardened atomic downloader (size + streamed SHA-256
   + HDF5 signature + read-only h5py open + atomic rename + `.corrupt` quarantine).
   Record `subj01_B1_download.json`. **Never commit the raw HDF5** (gitignored).
2. Extend the single-command runner to accept `fithrf_GLMdenoise_RR` as the second
   fixed beta version (still subj01 × V1 × D0), with **per-beta-version SHA pinning**
   and per-beta provenance. Add CLI tests (accept B1, still reject truly-unsupported
   subject/ROI/denoising/beta-version). No change to the historical B0 path.
3. Run the B1 smoke through the frozen historical policies (vis2vis P0, vis2img I0),
   leakage-safe FittedPipeline, producing `smoke_result.json` + split/pairing
   manifests + cross-artifact validation (must pass 19/19).
4. Produce a **B0-vs-B1 engineering comparison** (same voxels/splits/pairings; report
   both non-interpretive finite-fraction/mean/rank/lam side by side).

## Explicitly NOT in scope (unchanged hard constraints)

- **No Roy reproduction verdict.** B1 numbers are pipeline validation, never compared
  to the paper.
- **No other ROI, no other subject.** Fixed to subj01 × V1 × D0.
- **No resolution of the `beta_version` ambiguity** (GLMdenoise_RR is itself denoised,
  interacting with Roy's separate vis2vis denoising — deferred to the author request).
- **No raw beta / matrix / HDF5 / credentials committed.** Public bucket, anonymous
  HTTPS, no credentials.

## Deliverables

`subj01_B1_download.json` (provenance), runner + tests supporting B1, a B1 smoke
artifact directory with cross-artifact validation, a `B0_vs_B1_comparison.json`, and
a gate-outcome status. B1 numbers remain NON-INTERPRETIVE.
