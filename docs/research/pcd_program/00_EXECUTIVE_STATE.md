# 00 — Executive State

**Last updated:** 2026-07-16 (**Phase 2.5**) · **Decision: `CLEARED_FOR_MULTI_SHIFT_PILOT`**
(scoped — `27` §5) · **Active runs: NONE.** H100 idle.

---

## Current objective

**PCD is abandoned** (D-006): Hi-DREAM (arXiv 2511.11437) published its exact ROI hierarchy
*and already ran its random/reversed control, positively*.

**Direction: NCD (Neural-Constrained Decoder), multi-shift generalization** (`24`):

> A neural-prediction constraint learned **exclusively from perception** produces
> representations that generalize better than discriminative and generically regularized
> decoders across **distribution shift, subject shift, reduced data, neural noise, and
> zero-shot mental imagery**.

**Imagery is one sealed test of five, not the thesis.** Spera et al. 2026 (arXiv 2604.15374)
**fit on imagery** via latent functional alignment, so *"we improve imagery transfer"* is
**FORBIDDEN** (C-013). Our distinction is a *setting*: perception-only constraint, zero-shot
evaluation, evidence across shifts where imagery labels are irrelevant.

**NSD-Synthetic is now the primary OOD test** (`26`) — 8 subjects, 284 stimuli, CC-BY 4.0,
purpose-built for OOD. **This confines the n = 4 power crisis to one secondary test.**

## Blockers

1. **ARM-C needs a dataloader change.** It requires `shuffled_x` — the fMRI of the permuted
   partner image — which the dataset does not yet supply. The model **fails loudly** rather
   than degenerating into ARM-B, so this is safe but blocking for ARM-C. **Top action.**
   ARM-A/B/F are runnable now.
2. **B-DATA** — NSD-Synthetic + NSD-Imagery not obtained (both public; 181 TB free).
   Gates shifts 2–3 **only**; shifts 1/4/5/6 run on NSD already on the pod.
3. **O-9 / R-20** — CLIP cache never verified on grayscale/Mooney/line-drawing stimuli.
   Upstream of every synthetic number.
4. **O-8** — ARM-F's tuning budget must be equal and logged, or the parity claim is
   unfalsifiable and a reviewer will (rightly) assume a straw man.
5. Full PDF reads: **Spera et al.** (a zero-shot arm there voids the clearance), LEA, Hi-DREAM.

**Closed this session:** NCD wiring (`create_model` dispatch + `train_epoch` loss hook);
**O-7** (cross-shift multiplicity family frozen — `20` §4).

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

1. **Dataloader: supply `shuffled_x`** for ARM-C — load the permuted partner image's fMRI
   alongside each sample, using `DeterministicImagePermutation` seeded from
   `stable_seed(split_hash, seed)`. Unblocks ARM-C.
2. **Run E-P1-A/B/F now** (ARM-C follows once 1 lands): 2 seeds, subj01, ~6 GPU-h.
   **Read B vs F first** — it decides whether the thesis is about neural targets or about
   regularization.
3. **B-DATA** (parallel, 0 GPU): request NSD-Synthetic + NSD-Imagery.
4. **O-8**: log ARM-F's tuning trials in the manifest.
5. **Verify O-9/R-20**: CLIP cache on synthetic stimuli.

## Ready to run

**77 tests passing.** NCD is wired end to end: `create_model` dispatches `type: "ncd"` →
`NCDModel`; `train_epoch` adds `loss_weights["neural_prediction"] * model._last_aux_loss`
following the existing `_is_scfr` idiom; `loss.neural_prediction.weight` reaches the loop
through the generic weight extraction with no special-casing.

All 8 arms **exactly** parameter-matched. kappa is a **global learnable temperature**, pinned
constant across samples by test — NCD has no uncertainty head, and if someone makes kappa
input-dependent and starts reading it as confidence, that test fails first (F-002/F-003).

The config-diff test earned its place immediately: it caught ARM-F carrying a
`stochastic_depth` key the others lacked, which would have confounded the decisive B-vs-F
comparison invisibly.

## Standing prohibitions

- Do not resume PCD_v4 (D-001). No "predictive cortical" (D-002, D-007). No "identifiability".
- **Do not claim improved perception→imagery transfer** — Spera et al. own it (C-013).
- **No imagery data may touch training, selection, λ, checkpointing, early stopping, or
  representation design.** A leak collapses our setting into theirs and forfeits everything (R-22).
- **Never report CLIP near-chance on NSD-Synthetic's 232 non-semantic stimuli as OOD failure** —
  it is a measurement artefact by construction (C-019).
- Report both R@1 definitions with gallery size (D-005); never compare to 77.2% (T12).
- **No "first" claims** — review is abstract-level, single-index (`13` §7).
- **Do not escalate past the pilot if B ≈ F.** Report the negative (`24` §6). F-001's lesson.
