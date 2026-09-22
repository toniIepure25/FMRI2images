# Real-site BIDS validation procedure

Status until real data exist: **`REAL_SITE_BIDS_NOT_YET_VALIDATED`**. The engineering stack currently produces
only an internal structural schema check (`INTERNAL_BIDS_SCHEMA_PASS`) on synthetic events; that is never
equated with an official validator PASS.

## Procedure (at the site, on a dry-run acquisition; no participant required for the engineering dry run)
1. **Acquire dry-run DICOM** (short synthetic run or phantom, per facility policy).
2. **Convert** DICOM → NIfTI with pinned `dcm2niix`.
3. **Generate events** with the experiment stack (`*_events.tsv` / `.json`, `*_scansync.tsv`).
4. **Run official `bids-validator`** on the dataset (record version + output).
5. **Inspect volume count** against expected functional volumes.
6. **Compare trigger count** to acquired volumes.
7. **Compare events to BOLD volumes** (onsets fall within the acquired series).
8. **Verify first-volume timing** (task start aligned to the first retained trigger).
9. **Verify dummy-volume handling** (non-steady-state volumes excluded consistently).
10. **Certify** → only then set the real-site status to validated; until then it remains
    `REAL_SITE_BIDS_NOT_YET_VALIDATED`.

## Conversion mapping (to be finalized with the site)
- `task-perception` / `task-imagery` functional series → BIDS `func/`.
- `T1w`, `T2w` → BIDS `anat/`.
- Fieldmap / opposite-PE distortion series → BIDS `fmap/`.
- Physio (if recorded) → BIDS `*_physio.tsv.gz` (+ json).

No conversion is run on fabricated "real" scanner data; synthetic testing only until a real dry run occurs.
