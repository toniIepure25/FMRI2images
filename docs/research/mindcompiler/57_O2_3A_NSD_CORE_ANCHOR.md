# O2.3A — NSD-core perceptual anchor, target-state orientation identifiability (charter)

**Gate:** O2.3A · **Input HEAD:** `4498f65` · **Class:** `FROZEN_POST_HOC_TARGET_STATE_ORIENTATION_FEASIBILITY_ON_EXISTING_DATA`
**Execution status:** `O2_3A_CORE_DATA_UNAVAILABLE` (technical data blocker — **not** a scientific failure, Part AF.79).

## Immutable
O1 / O1.1 / O2 `SHARED_OPERATOR_PARTIAL` / O2.1 / O2.2 `CROSS_STATE_TRANSPORT_MULTIREGIME` / Track R —
unchanged. **O3 `O3_NOT_READY` locked through O2.3A.** C3/C3G/C3M/C3R untouched.

## Question
O2.2 found reliable, cross-subject-**shared** imagery identity geometry that lies **outside** the small
NSD-Imagery visual identity span, with orientation not identifiable from that small vision-only span.
O2.3A asks the qualified question: can **dense, target-state-independent PERCEPTION** from the original
NSD (hundreds of shared natural scenes) identify the subject-specific native **orientation** of that
residual in an **imagery-unseen** participant, with **zero target imagery** in any fitting? Not imagery
calibration; not decoding; a geometry-identifiability probe.

## Frozen methodology (preserved for future execution; `o2_3a_frozen_config.json` sha 39d7bc97)
- **B0 lineage only** (NSD-core `betas_fithrf` / b2). **B1/b3 (GLMdenoise_RR) prohibited.**
- Primary ROIs **ventral, lateral** (O2.2 regime B); parietal secondary (regime A, separate family).
- Anchor = shared NSD-core scenes with complete 8-subject repeat coverage, **excluding all NSD-Imagery
  scene identities (shared1000 overlap) globally**; train-only z-score; **DetSRM** common space (same
  family as O2 — tests information support, not an alignment-algorithm competition).
- K_anchor ∈ {2,4,8,16,32,64}, selected **vision-only** (LOSO subject × image blocks; metric =
  cross-subject native pattern reconstruction r; tie → smaller K).
- Retentions R_Y_CORE, R_Y_CORE_VISFULL vs frozen R_Y_SRM / R_Y_VISFULL_small; residual defined vs the
  **frozen O2.2 small visual span** (never redefined).
- Residual-orientation template = shared residual covariance in anchor coords from **training subjects
  only**; rank ∈ {1..6} nested-CV; predict target native residual subspace `Bhat = W_core_target·U_res`
  (zero target imagery). Primary test = predicted-residual retention vs random-anchor-subspace null
  (100 draws) + full-core oracle; participant unit N=8; **Holm across ventral/lateral**. Gauge-invariant
  (projectors only). Phases A (mapping cert) → B (vision-only anchor validation go/no-go) → C (imagery
  residual orientation), each frozen and committed before outcomes.

## Data-availability gate (Part C/AE) — the blocker
Required: NSD-core `betas_fithrf` (b2/B0) for **all 8 subjects**. Found:
- **Local:** subj01 core complete (40 sessions, b2/B0-aligned); **subj02–08 absent**.
- **Pod:** no b2 core for any subject. An `nsd_betas_download/` holds subj02/05/07 in
  `betas_fithrf_GLMdenoise_RR` (**b3/B1 — wrong version**, prohibited by the frozen B0 contract and
  Part AI, and belonging to the **C3R track** I was told not to touch).
- No AWS/NSD credentials. `nsd_expdesign.mat` + stim_info present.

⇒ B0-aligned core is unavailable for 7/8 subjects; a LOSO cross-subject anchor is not constructable;
acquiring ~280 GB access-controlled b2 core is not feasible autonomously. **Execution halts at Part C.**
No fitting performed, no target imagery touched, no B1 used, C3R untouched, no O2/O3 status change.

## Status
`O2_3A_CORE_DATA_UNAVAILABLE` · orientation `CORE_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE` · parietal
`INCONCLUSIVE`. O2.2's finding is preserved; O2.3A could **not** test the dense-perception qualifier.

## Admissible next (does not modify O2/O3)
- **O2.3A-ACQUIRE**: obtain NSD-core b2 for subj02–08 (user-provided NSD/AWS credentials + storage),
  certify core↔imagery voxel mapping, then run the **frozen** O2.3A methodology unchanged.
- **O2.3C**: one target-state-independent connectivity/anatomical anchor (also requires data acquisition).
- **Not admissible**: pod b3/B1 core, subj01-only demonstration, spatial-interpolation rescue,
  alignment-method competition. O3 stays NOT_READY.

## Forbidden interpretations
Dense NSD-core is still substantial target-subject calibration — not zero-calibration. No "imagery is
visually encoded," "perception determines imagination," "universal imagery coordinates," or "shared
operator supported."
