# O2.16 experiment operator guide

A trained researcher should be able to operate the experiment from this guide **without reading source code**.
The stack presents the perception and imagery tasks, synchronises with the scanner, records exact timing and
responses, writes BIDS events, deterministically randomises trials, reconstructs every trial after acquisition,
and enforces the Stage-A/Stage-B firewall. Default mode is **SIMULATION**; **CONFIRMATORY** fails closed.

> This software changes no science. O2.16 remains `O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE`; O3 remains
> `O3_NOT_READY`. No participant acquisition is authorised by installing or running this software.

## Installation
- Python 3.10+ (this build validated on the repo's interpreter). Core stack needs only `numpy` (+ `pandas`
  where used) and the standard library; SIMULATION and all tests run with no presentation hardware.
- For real-site visual presentation, install and **pin** PsychoPy on the stimulus machine (see the environment
  manifest); validate flip timing on the actual display before any confirmatory use. CI uses the built-in
  `SimulationPresenter` and never PsychoPy.
- Run tests: `python -m pytest tests/mindcompiler/o2_16_experiment/ -q`.

## Simulation
- `mindir-o216 validate-config` — show protocol version/hash and confirmatory timing gaps.
- `mindir-o216 simulate-session --participant sub-SYN001` — full synthetic participant (1536 perception + 96
  imagery trials), replay-exact, balance-proven.
- `mindir-o216 simulate-cohort --n 12` — 12 synthetic participants (engineering certification, **not**
  scientific N).
- `mindir-o216 generate-session` / `replay` — prove balancing / reconstruct the exact sequence.

## Site configuration
- Copy `configs/mindcompiler/o2_16/sites/TEMPLATE_SITE.yaml` to a real per-site file and fill every `TBD`.
- `mindir-o216 validate-site --site <file>` — cross-checks against frozen targets (7T, ~1.8 mm, ~1.6 s TR,
  T1 ≤ 1 mm, T2 recommended). Differences are `REVIEW_REQUIRED`, decided by a PI/MRI physicist — never
  auto-approved.

## Participant setup
- Pseudonymous IDs only: `sub-SYN###` (simulation), `sub-PILOT###` (pilot), site-authorized pseudonym
  (confirmatory). **Never** store name/email/health in the research dataset; identity mapping lives outside it.

## Perception run
Presents each of 512 anchors ×3 (frozen), image 3 s + gap 1 s (config-driven). Records per trial: participant/
session/run/index, anchor, presentation, stimulus SHA, planned vs actual onset, volume index, response, RT,
frame QC, trigger QC. No scientific outcome is computed during presentation.

## Imagery run
Cues each of 12 identities ×8 (frozen 6 simple + 6 naturalistic), ~4 s trials (cue/imagery/response/ITI,
config-driven). Identity mapping is frozen — never adaptive, never response-dependent.

## Run QC
`run_qc.json` per run: expected vs observed trials, missing/extra, duplicate ids, repeat counts, response
completeness, timing deviation, missing/duplicate triggers, frame drops, stimulus-hash mismatch → PASS / WARN /
FAIL. QC is **operational only** — it never makes outcome-based scientific accept/reject decisions.

## Session QC
State machine: PRECHECK → PARTICIPANT_LOAD → STIMULUS_HASH_CHECK → DEVICE_CHECK → SCANNER_CHECK → RUN_READY →
WAIT_TRIGGER → RUN_ACTIVE → RUN_QC → RUN_SEALED → SESSION_COMPLETE. Any critical failure → SESSION_ABORTED_SAFE.

## Crash recovery
On a task crash the run is marked `RUN_CRASHED` with last-completed trial, timestamps, trigger count, manifest
SHA and error trace persisted. Do **not** restart silently; restart per the explicit run-level protocol. Partial
and restarted runs are never merged invisibly.

## BIDS export
Events written to `sub-<id>/ses-<id>/func/…_task-{perception,imagery}_run-XX_events.tsv` (+ `events.json`
sidecar, `_scansync.tsv`). In simulation a `SIMULATED_NO_BOLD` marker accompanies events (no NIfTI). Validation:
internal structural check always; official `bids-validator` only if installed — its PASS is never equated with
the internal PASS.

## Replay
`mindir-o216 replay …` reconstructs the exact expected sequence from {participant, session, task, protocol
version}; the round-trip test rebuilds the trial manifest from written BIDS events and requires an exact
identity/repeat/family/run/order match.

## Stage-A export
`mindir-o216 export-stage-a` prepares predictor-side artifacts (on synthetic fixtures here) with **no** read
access to `analysis/stage_b_protected/`. Stage B stays locked until a human-created `STAGE_B_RELEASE.json`
exists with all required fields. No command displays protected Stage-B outcomes during Stage A.

## Pilot procedure
Engineering pilot ≤ 2 participants, watermark `NON_CONFIRMATORY_ENGINEERING_PILOT`, output marker
`PILOT_ONLY_DO_NOT_POOL_WITH_CONFIRMATORY`. Pilot validates operations only; it may **not** touch thresholds,
the estimator, or primary hypotheses, and neural effect size is **never** a go/no-go criterion.

## Confirmatory authorization
CONFIRMATORY refuses to start unless an authorization manifest grants ethics_active, participant consent, MRI
safety, site authorization, stimulus authorization, and participant_independent — all true — with real ethics
values, a certified site/trigger/BIDS root, complete timing, non-placeholder stimuli, and a non-historical
participant id. **There is no override flag.**

## Failure modes
Simulate faults for testing: missing/duplicate triggers, corrupted stimulus hash, crash, missed responses. Each
is detected and surfaced by QC/guards (see the test suite). Real-site failures follow the site integration
checklist.
