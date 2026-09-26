# MINDIR — subject-specific geometry & calibration of internal visual states

*This is the MINDIR sub-program's external entry point. The repository root `README.md` documents the broader
FMRI2images / Brain-to-Image project and is intentionally preserved unchanged. MINDIR lives under
`src/fmri2img/mindcompiler/` with its platform layer in `.../prox/`.*

## What is MINDIR?
A research platform studying how the brain represents seen vs imagined images: how much of the imagined
("target") neural state perception explains, how much is subject-specific, how little direct imagery data
calibrates to it, and whether that structure generalizes across people, sessions, and internal states.

## What is actually established?
On a **development/discovery cohort (N=8, permanently CLOSED):** perception is informative but incomplete; a
low-dimensional but subject-specific residual exists; an 8-observation calibration frontier **under the frozen
estimator** (not a universal minimum); the frontier is schedule-fragile; repeat-geometry predicts calibration
success. Exploratory follow-up: the failure-predictive geometry is structured but distributed.

## What is NOT established?
Any generalization beyond the development cohort. The independent replication is **preregistered and frozen but
NOT executed**. Statuses: O2.16 `O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE`; O2.16-DATA
`O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION`; O2.16-SEC
`O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT`; M0
`MINDIR_MOONSHOT_TRIAD_PREREGISTERED_AWAITING_NEW_DATA`; **O3 `O3_NOT_READY`**.

## What data are required?
None to use the platform (synthetic worlds included). The science requires a **new, independent** fMRI cohort
(the historical Natural Scenes Dataset development cohort is closed). No raw participant data ship with this repo.

## Install (clean)
```
git clone <repo> && cd FMRI2images
python -m venv .venv && . .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -e .                                     # core MINDIR (numpy/scipy/pandas/pyyaml)
pip install -e ".[dev]"                              # + test/lint/type/coverage tooling
```
No manual `PYTHONPATH` editing is needed after install; the `mindir` command is installed as a console script.

## Run a synthetic demo
```
mindir demo                 # generate world -> recover support -> private correction -> few-shot -> falsification -> report
```

## Run tests
```
python -m pytest tests/mindcompiler/prox -q
```

## Run the benchmark
```
mindir benchmark synthetic  # B1-B8 synthetic ground-truth benchmark, fixed seeds
```

## Reproduce an analysis
```
mindir reproduce <manifest.json>   # verifies code/config/input/metric/seed/schema; reports
                                   # EXACT_REPRODUCTION | METHOD_REPRODUCTION | REPRODUCTION_UNAVAILABLE (never conflated)
```

## How is scientific leakage prevented?
Predictor-side (Stage-A) processing is frozen and hashed before any held-out outcome (Stage-B) is opened; a
human release token gates Stage-B. A historical-N8 firewall blocks any code path from reading the closed
development cohort. Immutability checks verify the frozen O2.16 / O2.16-SEC / M0 config hashes. These run in CI
(`.github/workflows/ci-science-integrity.yml`).

## Current research status
Stage **R1 — independent replication (blocked on human acquisition)**. See `docs/research/mindcompiler/
MINDIR_ROADMAP.md`. The exact next scientific action is: **complete the frozen O2.16 independent replication on
new participants.**

## Citation / license
See `CITATION.cff`. Code license = repository MIT (`LICENSE`); dataset and stimulus licenses are separate (see
`artifacts/mindcompiler/o2_16_auth_prep/stimulus_permission_status.json`). No internal agent/session history is
exposed in this documentation.
