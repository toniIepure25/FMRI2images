# MINDIR — subject-specific geometry & calibration of internal visual states

*MINDIR-specific overview. The repository-root `README.md` (whole FMRI2images project) is unchanged; this is the
MINDIR platform entry point.*

## What MINDIR is
A research platform studying how the brain represents seen vs imagined images: how much of the imagined
("target") neural state perception explains, how much is subject-specific, how little direct imagery data
calibrates to it, and whether that structure generalizes across people, sessions, and internal states.

## Current scientific status
- **Established (development cohort, N=8, CLOSED):** perception is informative but incomplete; a low-dimensional
  but subject-specific residual exists; an 8-observation calibration frontier under the frozen estimator (not a
  universal minimum); the frontier is schedule-fragile; repeat-geometry predicts calibration success. Exploratory
  follow-up (X1–X4): the failure-predictive geometry is structured but distributed.
- **NOT established:** any of the above generalizing beyond the development cohort. Independent replication is
  **preregistered and frozen but not executed**.
- **Statuses:** O2.16 `O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE`; O2.16-DATA
  `O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION`; O2.16-SEC
  `O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT`; M0
  `MINDIR_MOONSHOT_TRIAD_PREREGISTERED_AWAITING_NEW_DATA`; **O3 `O3_NOT_READY`**.

## What is / is not established (one line)
Established = development-cohort discovery + a frozen, engineering-ready replication + preregistered future
program. **Not** established = any biological generalization; that requires the independent cohort.

## Architecture (platform layer)
`src/fmri2img/mindcompiler/prox/`: `objects` (typed model), `metrics` (canonical registry), `statistics`
(participant inference), `synthworld` + `benchmark` (ground-truth synthetic evaluation), `sample_complexity`
(phase transitions), `falsification` (F1–F10), `governance` (immutability + historical-N8 firewall + repro
contract), `cli` (`mindir`). Sealed gate code is untouched (see `artifacts/mindcompiler/prox/migration_plan.md`).

## Quickstart
```
PYTHONPATH=src python -m fmri2img.mindcompiler.prox.cli validate     # immutability + env
PYTHONPATH=src python -m fmri2img.mindcompiler.prox.cli metrics      # registry
PYTHONPATH=src python -m fmri2img.mindcompiler.prox.cli benchmark    # synthetic ground-truth benchmark
PYTHONPATH=src python -m fmri2img.mindcompiler.prox.cli phase        # phase diagram
python -m pytest tests/mindcompiler/prox/ -q
```

## Reproducibility
Every platform output can carry a repro record (data hashes, commit, config hash, env, seed, participant set,
metric version, timestamp, frozen flag). Frozen scientific configs are content-hashed; the historical-N8 firewall
and immutability checks run in tests.

## Experimental pipeline
Acquisition software (`o2_16_experiment`) → BIDS → frozen preprocessing → Stage-A predictors → (human release
token) → Stage-B protected outcomes. Confirmatory acquisition fails closed until real authorization.

## Future research
See `MINDIR_ROADMAP.md`, the experiment catalog (E1–E12), and the moonshot preregistration (M1/M2/M3).

## Citation / license
Cite the MINDIR discovery manuscript (`paper/mindir/`) when published; dataset: Natural Scenes Dataset
(Allen et al., 2022). License: inherits the repository license.

*No internal agent/session/infra details appear in participant- or collaborator-facing documents.*
