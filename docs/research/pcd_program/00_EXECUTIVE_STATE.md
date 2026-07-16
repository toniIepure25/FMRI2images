# 00 — Executive State

**Last updated:** 2026-07-16 (Phase 2) · **Gate:** 2 complete → Gate 3 (implementation)
**Commit:** `feature/predictive-cortical-decoder` · **Active runs: NONE.** H100 idle.

---

## Current objective — CHANGED THIS PHASE

**The PCD research direction is abandoned** (D-006). A systematic literature search found
**every** PCD contribution already published — most damagingly **Hi-DREAM** (arXiv 2511.11437),
which uses PCD's exact ROI hierarchy *and already ran PCD's random/reversed control with a
positive result*.

**New direction: NCD (Neural-Constrained Decoder).** Test whether the perception→imagery
generalization gap documented by **NSD-Imagery (CVPR 2025)** — complex architectures overfit
to vision; simple linear models transfer better — is caused by **unconstrained intermediate
representations** rather than capacity. One manipulated variable: `λ_neural`, the weight on a
masked-ROI neural-prediction objective. See `17_PRIMARY_THESIS_SELECTION.md`.

Our 167M-param model with an 82 pp overfit gap is no longer an embarrassment — it is the
case study for the pathology NSD-Imagery documented.

## Blockers (ordered by value-of-information — both cost zero GPU)

1. **B-LIT — full-text reads: NSD-Imagery, LEA, Hi-DREAM.** Can kill the thesis outright (if
   either already tested an auxiliary neural-prediction objective) and is the only source for
   the SESOI and power analysis. **Highest VOI in the program; costs nothing.**
2. **B-DATA — NSD-Imagery is not on the pod.** Verified 2026-07-16. Public dataset, 181 TB
   free, so this is access + compatibility audit, not capacity. Fallback declared in advance
   at `17` §5 (transfer to reduced-data / low-reliability regimes).

## Verified results (unchanged; the only numbers that exist)

| Metric | Value | Caveat |
|---|---|---|
| PCD_v4 best val R@1 | **16.36%** @ epoch 101 | 6 956-image gallery |
| PCD_v4 val R@1 (trial-level) | **3.30%** | **5× fork — T11** |
| PCD_v4 train R@1 | **98.83%** | → **82 pp overfit gap** |
| `checkpoint_best.pt` | **epoch 98** | selected on **val_loss**, not R@1 (T14) |
| PCD_v4 params | 167 291 141 | vs **NCD's ~20.5M** (8× smaller) |
| Frozen retrieval system | 77.2% SHARED1000 | **NOT comparable** (T12) |

## What is solid

- **Splits clean, SHARED1000 sealed** (T10) — the asset. Reused wholesale by NCD.
- **~70% of the code survives** (`15`): splits, per-ROI tokenisation, encoders, training
  loop, retrieval head. Replaced: aggregator, subject projections, level grouping, kappa
  heads. Added: one masked-ROI objective. **This is not a rewrite.**
- 9 regression tests pin the Gate 0 findings.

## Next five actions

1. **Gate 3:** implement `src/fmri2img/models/neural_constrained_decoder.py` per `18`,
   with gradient tests **first** (every interpreted quantity must have an identifying
   objective — the T8/F-002 contract).
2. **B-LIT** full-text reads (zero GPU, can kill the thesis).
3. **B-DATA** NSD-Imagery compatibility audit.
4. **E-01** level-3 bypass diagnostic — one forward pass. **Now a diagnostic, not a
   contribution** (Hi-DREAM owns it).
5. **E-00** protocol standardisation: fix checkpoint selection (T14), declare the gallery,
   write `count_flops.py` (`FLOPs: NOT_MEASURED` must not appear twice).

## Standing prohibitions

- Do not resume PCD_v4 (D-001). Do not resurrect "predictive cortical" (D-002, D-007).
- Do not claim "identifiability" — claim a **neural-prediction constraint** (O-2).
- Do not report `val_r@1` without gallery size + trial-level figure (D-005). Never compare to 77.2% (T12).
- **No "first" claims** — the literature review is abstract-level and single-index (`13` §7).
- Do not escalate to 8 subjects if the 4-subject gate fails (`21` §4). F-001's lesson.
