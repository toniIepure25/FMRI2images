# 31 — S1.1: Provenance Repair, Force-Push Note, and Environment Blocker

**Date:** 2026-07-17 · **Input commit:** `bc92c4c` (output commit differs).
**No beta data downloaded or analyzed. No reproduction verdict.**

---

## 1. Force-push process note (recorded, not repeated)

| | |
|---|---|
| Original pushed tip | `b2b3ac0` |
| Replacement tip | `bc92c4c` |
| Operation | message-only `git commit --amend` + `git push --force-with-lease` |
| Tree/content impact | **none** — verified: `b2b3ac0` and `bc92c4c` have identical trees; only the commit message differed (the amend fixed a message that had grabbed a stale message file) |
| Reason | the commit accidentally used a leftover `/tmp` message file from the prior S1.0 session |

> **Corrective policy, binding henceforth:** **no further force-pushes on this research branch,
> including message-only amendments.** Any future message fix is made with a **normal forward
> commit** (e.g. an empty commit noting the correction), never by rewriting a pushed tip. This
> session and all subsequent ones use forward commits only.

## 2. Provenance repairs (this session)

- **Stale manifest path fixed.** `acquisition_manifest.json` referenced
  `data/manifests/mindcompiler/nsdimagery_metadata_manifest.json`; the file was moved to
  `artifacts/mindcompiler/roy_s1/metadata_manifest.json`. Corrected.
- **Duplicate manifest removed.** The leftover copy under `data/manifests/` was deleted; the
  tracked copy under `artifacts/` is authoritative.
- **`.gitignore` deduplicated.** Removed the redundant bare `data/` rule. Retained `/data/`
  (root data ignore), `data/nsd/` (explicit raw-NSD exclusion), and the `!src/fmri2img/data/`
  package negations. Verified: raw data still ignored; `artifacts/` manifests trackable.
- **Manifest-reference validation test added**
  (`tests/mindcompiler/test_manifest_references.py`): every local manifest entry with
  `status == "verified"` must resolve to an existing file.

## 3. ENVIRONMENT BLOCKER — objectives 3–11 cannot run on this host

**`h5py` and `nibabel` are both absent** on this Windows CPU box (import fails). Consequences:

| Objective | Requires | Status on this host |
|---|---|---|
| §3 ROI schema (`prf-visualrois`, `streams`) | nibabel | **BLOCKED** |
| §4 NSD-core SNR / 98th-pct selection | nibabel | **BLOCKED** |
| §6 HDF5 beta schema audit | h5py | **BLOCKED** |
| §7 trial table (uses HDF5 trial dim) | h5py | **partially BLOCKED** |
| §5 beta download | — (urllib) | *possible, but pointless without h5py to open them* |
| §9 model core (ridge / reduced-rank / Pearson) | numpy only | **feasible** (deferred, §5 below) |

> **Downloading 1.8 GB of beta HDF5 that cannot be opened here would waste bandwidth and leave
> large unusable files.** Deferred deliberately. The exact keys/sizes are already staged in
> `acquisition_manifest.json`.

## 4. Two clean paths forward (user choice)

**Option A — install the libraries locally:**
```bash
pip install h5py nibabel
```
then resume S1.1 on this host (74 GB free; adequate for subj01 betas).

**Option B — run S1 on the pod**, which already has the scientific stack (h5py, nibabel,
CUDA). The pod is idle. This is the natural home for the beta-heavy work; it needs a code sync
(the pod is on an old commit and **not** on this branch — inspect before syncing).

**Neither was chosen autonomously**: installing packages and pod sync both have side effects,
and the smoke needs an interactive decision on host. Recorded for the user.

## 5. What is still feasible here and was deferred only for budget

The **model core** (§9: reduced-rank ridge, 100-value log ridge grid, rank selection at 99%
of peak validation, per-voxel Pearson) is **pure NumPy** and needs neither h5py nor nibabel.
It can be implemented and tested against synthetic low-rank ground truth **without any real
data**, exactly as the split-protocol and policy contracts already were. Deferred this session
only because provenance repair + the environment determination consumed the budget; it is the
**highest-value next code step** and is host-independent.

## 6. Status

- Provenance: **repaired.**
- `S1_SMOKE_ENGINEERING_FAILURE`? **No** — nothing failed; the smoke was **not attempted**
  because the host lacks h5py/nibabel. Correct status: **smoke not run; blocked by missing
  environment libraries**, not by data schema or alignment.
- **No S1 reproduction verdict.** No beta data analyzed.

## 7. Exact next action

Choose Option A or B (§4). Then: download subj01 both-version betas with size+SHA-256 checks;
build `subj01_beta_schema.json`, `subj01_roi_schema.json`, `subj01_trial_table.csv`; implement
the NumPy model core with synthetic-recovery tests; run the V1 smoke with B0/B1 × D0 (add D1
after D0 contracts pass). **Do not start all eight subjects.**
