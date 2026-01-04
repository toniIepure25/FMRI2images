# Reproducibility (draft)

## What we record

All major runs should write a `manifest.json` recording:

- environment (python/torch/cuda)
- git commit and dirty flag
- key configuration values
- hashes of critical inputs (index, checkpoints, cache)

The implementation lives in `fmri2img.utils.manifest`.

## How to reproduce a run

1. Checkout the recorded `git_commit`.
2. Recreate the environment (`environment.yml` / `requirements.txt`).
3. Ensure the same index/cache/checkpoint hashes.
4. Re-run the command line captured in `manifest.json`.

## Verification checklist

- The evaluator must match reconstructions to GT (no skipped samples).
- CLIP embedding dimensions must match.
- `manifest.json` must exist in the output directory.
