# O2.3C — Resting-State Connectivity Anchor: Target-State Orientation Identifiability

**Gate:** O2.3C · **Input HEAD:** `c73f304` · **Frozen methodology SHA:** `528f23eb`
**Execution status:** `O2_3C_REST_PRODUCT_INCOMPATIBLE` (technical data blocker)
**Orientation status:** `CONNECTIVITY_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE`

## Question

Can *intrinsic resting-state functional connectivity* — a task/stimulus-independent anchor — identify
the **native orientation** of the O2.2 imagery-specific residual in an imagery-unseen participant,
*without* using that participant's target imagery or perception? The frozen common-space method is
deterministic **connectivity-SRM** (cSRM: `C_s ≈ W_s S_conn`, orthonormal `W_s`), the target
parcellation is FreeSurfer **Desikan-Killiany** (68 cortical parcels) in the native **func1pt8mm**
reference, QC censors frames with **FD > 0.25 mm**, with **≥ 30 usable rest minutes/subject**, **no
GSR**, and **no target perception/imagery in any fit**. The O2.2 imagery residual (component of the
imagery centroid perpendicular to the small 10-identity vision span, `P_VIS_SMALL`) is imported
unchanged.

## Methodology (frozen BEFORE any outcome — Part AP)

Frozen config: `artifacts/mindcompiler/operator_o2_3c/o2_3c_frozen_config.json`
(`frozen_methodology_sha = 528f23eb`). Contracts: `o2_3c_contracts.json`. Phases A–D as specified in
the brief (per-subject Fisher-z connectivity → vision-free cSRM with nested K-selection → shared
residual-orientation template → predicted native subspace vs matched random-subspace null + oracle →
participant-level exact 2^N sign-flip, Holm across {ventral, lateral}). **One anchor family only; no
method leaderboard.**

## Data-availability gate (Part C / F.6) — the gate halted here

The gate carries a hard data-availability precondition that must pass **before** any fitting. It does
not pass. Probe evidence (public NSD Open Data, `s3://natural-scenes-dataset`, unsigned, no
credentials) is recorded in `rest_data_inventory.json`:

| Product | Finding | Usable for cSRM connectivity? |
|---|---|---|
| Prepared resting **timeseries** in native func1pt8mm | **0 runs** for all 8 subjects (the 548 native runs are nsdimagery / nsdsynthetic / prffloc / main-NSD only) | No — none exist |
| `restingbetas_fithrf` (func1pt8mm) | 3D **single-volume** GLM `betas` + `R2` variance maps (verified `ndim=3`, dim `[81,104,83]`) | No — no temporal dimension |
| Raw BIDS `task-rest` bold | **Present, abundant**: 20–36 runs/subject, 188 TR × 1.6 s = 5.0 min/run ⇒ **100–180 min/subject** | Only in raw acquisition grid `[120,120,84]`; no cleaned/motion product; **Part F.6 forbids** building a raw preprocessing pipeline in this gate |
| Desikan-Killiany aparc | Surface/anatomical only; **not** a native func1pt8mm volume (func1pt8mm atlases are HCP_MMP1 / Kastner2015 / streams) | No native-space parcellation product |

**Key disambiguation.** The blocker is **product incompatibility**, not data scarcity:
- Not `O2_3C_INSUFFICIENT_REST_DATA` — raw resting volume (100–180 min/subject) exceeds the 30-min floor.
- Not `O2_3C_REST_DATA_UNAVAILABLE` — raw resting data exists for all subjects.
- It **is** `O2_3C_REST_PRODUCT_INCOMPATIBLE` — no prepared resting **timeseries** product exists in
  the native func1pt8mm reference (only single-volume betas/R2), and the target Desikan-Killiany
  parcellation is not provided as a native func1pt8mm volume. Per brief Part F.6, constructing an
  extensive raw-fMRI preprocessing + registration pipeline inside this gate is out of scope.

No connectivity was fit and no orientation test was run, so the orientation outcome is
`CONNECTIVITY_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE`. Because nothing was fit, the leakage contract
holds vacuously.

## What does NOT change

Carried immutable (unmodified by this gate):
`TRACK_R = INDEPENDENT_METHOD_REPRODUCTION_PARTIAL`, `O1 = STIMULUS_INVARIANT_OPERATOR_PARTIAL`,
`O2 = SHARED_OPERATOR_PARTIAL`, `O2.3A = FEASIBILITY_PASS / ORIENTATION_NOT_IDENTIFIABLE`,
`O3 = O3_NOT_READY`. The O2.2 residual is imported byte-for-byte unchanged.

## Integrity note

Freeze-before-outcomes is preserved: the methodology (SHA `528f23eb`) is frozen and committed, and
the gate stopped at the frozen data-availability precondition before any scientific quantity was
computed. The blocker is technical (data-product form), not a scientific result about connectivity —
it neither supports nor refutes connectivity-based orientation identifiability.
