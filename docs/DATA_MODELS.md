# Data + Models

## NSD dataset

This repo **does not redistribute NSD**. You must obtain it separately and mount/copy it.

### Required layout (local mirror)

Supported layouts:

- Layout A:
  - `$NSD_DATA_ROOT/nsddata/...`
  - `$NSD_DATA_ROOT/nsddata_stimuli/...`
  - `$NSD_DATA_ROOT/nsddata_betas/...`

- Layout B:
  - `$NSD_DATA_ROOT/natural-scenes-dataset/nsddata/...` (same for the other folders)

### What we validate

`make data` runs `scripts/verify_dataset.py` and checks for:

- `nsddata/experiments/nsd/nsd_stim_info_merged.csv`
- `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5`
- Subject session design CSV(s) under `nsddata/ppdata/<subject>/behav/session01/`
- Subject beta NIfTI under `nsddata_betas/ppdata/<subject>/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz`

If you want to stream betas from S3 instead of mounting them locally, run:

```bash
python scripts/verify_dataset.py --allow-s3-only
```

## Hugging Face models (diffusion)

### Environment variables

- `DIFFUSION_ENABLED=1` enables diffusion dependency checks.
- `DIFFUSION_MODEL_ID` selects the model (default: `stabilityai/stable-diffusion-2-1`).
- `DIFFUSION_MODEL_REV` optionally pins a revision/commit on the Hub.
- `HF_TOKEN` optional (required for gated/private models).

### Caches

For reproducible pods, set HF caches under `$CACHE_ROOT` (recommended in `.env.example`).

### Fetching

`make models` runs `scripts/fetch_models.py`:

- tries offline-only first (no network)
- downloads/resumes if missing
- keeps artifacts in HF cache
