# 27 — Roy S1: Data and Code Inventory, and Feasibility Determination

**Date:** 2026-07-17 · **Revised 2026-07-17 (S1.0) to correct factual errors below.**
**No data downloaded. No reproduction run.**

---

## 0. CORRECTIONS to the first version of this file (my errors)

The first version overstated the barriers. Corrected:

| I wrote | Correct statement |
|---|---|
| "exact reproduction impossible **in principle**" | **Withdrawn.** Original-code/bitwise replication is *currently unavailable*; an **independent method reproduction is feasible after data access**, with specified sensitivity analyses. |
| "**no author code exists**" | **Withdrawn.** *No public author implementation was located as of the search date.* Private/unindexed code may exist; an author request has not been answered. |
| voxel-inclusion threshold "UNAVAILABLE" | **FALSE — it is specified.** 98th-percentile voxelwise SNR within each ROI, SNR from NSD-core (§2). |
| ridge grid "UNAVAILABLE" | **FALSE — it is specified.** 100 log-spaced values, 10⁻³–10⁵, selected on validation (§2). |
| rank grid "UNAVAILABLE" | **FALSE — specified.** Ranks 1…max (≤12 for NSD-Imagery); selection near 99% of peak validation performance (§2). |
| denoising "UNAVAILABLE" | **Overstated.** Architecture *is* specified (fit vis2vis; feed denoised outputs to vis2img). Only the **fold-level implementation** is ambiguous → `ARCHITECTURE_SPECIFIED__FOLD_IMPLEMENTATION_AMBIGUOUS`. |
| dataset is "CC-BY-NC-ND 4.0" | **Conflation.** That is the *manuscript/preprint* license. **Dataset reuse is governed by the NSD Data Access Agreement**, not yet read (§3). |

## 1. Corrected reproducibility taxonomy

| Status | Meaning |
|---|---|
| `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE` | no public author implementation located; private/unpublished code may exist; author request unanswered |
| `BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS` | RNG/seeds not published |
| `INDEPENDENT_METHOD_REPRODUCTION_FEASIBLE_AFTER_DATA_ACCESS` | the achievable and appropriate target |
| `ROY_S1_BLOCKED_BY_DATA_ACCESS` | **current execution state** |

**"Permanent" is withdrawn** — author code or clarification would change this.

## 2. Corrected artifact inventory (from the full paper)

| # | Artifact | Status |
|---|---|---|
| Voxel selection | **SPECIFIED**: ROIs V1, V2, V3, hV4, ventral, lateral, parietal; voxel SNR from **NSD-core**; include voxels **> 98th percentile** of voxelwise SNR within each ROI | explicit |
| Ridge grid | **SPECIFIED**: 100 log-spaced values, **10⁻³ … 10⁵**, selected on validation | explicit |
| Rank candidates | **SPECIFIED**: 1 … max; ≤ 12 conditions bound NSD-Imagery; selection near **99% of peak validation** | explicit; tie-break ambiguous |
| Split | **SPECIFIED**: per identity, **4 train / 2 validation / 2 test** of 8 repeats; all identities in every split | explicit |
| Pairing | **SPECIFIED**: vision paired with random imagery repeat of same stimulus; vis2vis shuffling within split | explicit; **seeds / #realizations / averaging ambiguous** |
| Denoising | **ARCHITECTURE SPECIFIED**: fit vis2vis; denoised vision = vis2vis outputs → vis2img inputs | `ARCHITECTURE_SPECIFIED__FOLD_IMPLEMENTATION_AMBIGUOUS` |
| Metric | **SPECIFIED**: per-voxel Pearson r, ROI/subject aggregation | explicit |

**No artifact is "unavailable".** The remaining gaps are **implementation ambiguities**, enumerated in `30_ROY_IMPLEMENTATION_AMBIGUITY_REGISTRY.csv`, each handled by a preregistered sensitivity variant rather than a guess.

## 3. License — corrected

- **Manuscript/preprint:** CC-BY-NC-ND 4.0 (governs the *text*).
- **Dataset access & reuse:** governed by the **NSD Data Access Agreement** + NSD Data Manual — **not yet read**. Redistribution/derivative/publication terms are **pending** that review.
- **No code or data redistribution decision** may be made until those terms are read.

## 4. Execution state

HEAD `dc75d0a` server-verified · tree clean but for `docs/CLAUDE_MEGA_PROMPT.md` · **74 GB free** on D: · no local NSD-Imagery data · pod not contacted/synchronized · no processes; no GPU.

## 5. Status after S1.0

`ROY_S1_BLOCKED_BY_DATA_ACCESS` (unchanged execution) · `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE`
· `INDEPENDENT_METHOD_REPRODUCTION_PREPARED` · `AWAITING_USER_DUA_COMPLETION`
· `AWAITING_AUTHOR_CLARIFICATION`. **No S1 empirical verdict.**

## 6. Exact user action to unblock

Complete the **official NSD Data Access Agreement** at naturalscenesdataset.org personally;
share the granted **non-secret access route** (S3 prefix / portal path). Optionally send the
author request in `21` (drafted, not sent). **I will not register, accept a DUA, or bypass
authentication.**
