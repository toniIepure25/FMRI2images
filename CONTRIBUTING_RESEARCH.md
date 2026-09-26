# Contributing research to MINDIR

How to extend MINDIR **without violating frozen science, protected outcomes, or historical-cohort closure**.
Route all new work through the canonical platform layer (`src/fmri2img/mindcompiler/prox/`), never through the
sealed per-gate modules (those are `DEPRECATED_DO_NOT_USE_FOR_NEW_ANALYSIS`; see `docs/research/mindcompiler/
API_STABILITY.md`).

## Hard rules (CI-enforced)
1. **Do not read the historical N=8 cohort.** The `governance` firewall blocks it; do not work around it.
2. **Do not modify any `*_frozen_config.json`.** Their `config_sha256` is load-bearing; immutability CI will fail.
3. **Do not open protected Stage-B outcomes** in predictor-side code. Stage B needs a human release token.
4. **Participant is the inferential unit.** Never treat schedules/folds/repeats as independent N.
5. **No biological claim from synthetic simulations.** Synthetic results establish software/method feasibility only.

## Add a metric
Register it in `prox/metrics.py` with FULL metadata (question, definition, range, direction, null, failure modes,
invariances, min sample, aggregation, allowed/forbidden use) + a `prox-metric` version bump. Add a property test
(invariance/edge case) in `tests/mindcompiler/prox/`.

## Add a null / falsifier
Add it to `prox/falsification.py` (a label-destroying transform + collapse check) and a row to
`artifacts/mindcompiler/prox/claim_falsification_matrix.csv`. Every new claim must name its falsifier.

## Add a prospective experiment
Add it to the experiment catalog + `innovation_value_matrix.csv` with question / falsifier / cohort / min-N /
value-if-positive / value-if-negative. New discoveries require Cohort B then an independent Cohort C.

## Add an internal state (cross-state)
Extend `synthworld`/`benchmark` (B8) and, for real acquisition, follow the M3 human-review protocol (ethics
amendment; does not modify O2.16).

## Add a benchmark task or site
Add a task to `prox/benchmark.py` with ground-truth scoring; add a site config template under
`configs/mindcompiler/o2_16/sites/` (verified public values only; `TBD_SITE_OPERATOR` otherwise;
`CANDIDATE_NOT_VALIDATED`).

## Tests + CI
Every contribution ships with tests. CI runs core (lint/type/tests/CLI), science-integrity (immutability +
firewall + leakage), docs, and benchmark smoke. Keep coverage high for geometry / metrics / falsification /
governance / firewalls / hashing / randomization / benchmark scoring.
