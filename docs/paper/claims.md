# Claims (draft)

This document lists claims we intend to support with experiment cards + artifacts.

## Claims

1. **Reconstruction evaluation is stimulus-safe**: reconstructions are deterministically matched to GT stimuli, and evaluation fails fast on mismatches.
   - Evidence: evaluator mapping logic + unit tests + debug output.

2. **Manifests enable run-level reproducibility**: key environment/config/input hashes are recorded for major runs.
   - Evidence: `fmri2img.utils.manifest` + tests, plus manifests produced by scripts.

3. **Baseline models train and evaluate end-to-end** (ridge/MLP at minimum).
   - Evidence: experiment cards + output reports.

Add new claims only if you can point to a reproducible artifact path.
