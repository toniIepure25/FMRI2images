# MINDIR API stability & deprecation

## Public (STABLE) API
- `fmri2img.mindcompiler.prox.metrics` — metric registry + geometry/prediction/support/calibration/stability metrics.
- `fmri2img.mindcompiler.prox.statistics` — participant sign-flip, Holm, permutation, bootstrap.
- `fmri2img.mindcompiler.prox.objects` — typed object model.
- `fmri2img.mindcompiler.prox.governance` — immutability, firewall, reproducibility contract.
- `fmri2img.mindcompiler.prox.synthworld`, `.benchmark`, `.sample_complexity` — synthetic evaluation.
- `fmri2img.mindcompiler.prox.falsification` — F1-F10 battery.
- `fmri2img.mindcompiler.prox.release`, `.hardening` — demo/reproduce/provenance + R1 hardening.
- `fmri2img.mindcompiler.prox.errors` — actionable exceptions.
- CLI: `mindir version|validate|metrics|simulate|demo|benchmark|phase|falsify|reproduce`.

## EXPERIMENTAL
- `hardening.cohort_c_plan`, `sample_complexity` bounds — interfaces may change as real data inform them.

## DEPRECATED_DO_NOT_USE_FOR_NEW_ANALYSIS
- Per-gate modules under `operator_o2_*`, `x1_*`..`x4_*`, `moonshot_triad`, `o2_16_experiment` are SEALED research
  gates. They remain for provenance and are **not** deleted, but new analysis must route through `prox`. They must
  not be imported to bypass the firewall or immutability checks.

## Rules
No accidental reliance on internal gate modules; no import of historical-cohort loaders; schema-versioned
artifacts are never silently reinterpreted (a schema mismatch raises `SchemaMismatchError`).
