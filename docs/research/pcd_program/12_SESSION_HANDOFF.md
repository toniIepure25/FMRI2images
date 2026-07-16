# 12 — Session Handoff

**Written:** 2026-07-16 (end of Phase 2) · **Supersedes** `docs/HANDOFF_CONTEXT.md` entirely.

---

## Read this first

**The PCD research direction is dead** (D-006). Not because of the Gate 0 bugs — because
**Hi-DREAM** (arXiv 2511.11437, Nov 2025) published PCD's exact ROI hierarchy *and already ran
PCD's random/reversed control, with a positive result*. Every other PCD contribution is
likewise owned (`14` §1). Do not resume, repair, or re-argue PCD.

`docs/HANDOFF_CONTEXT.md` remains historically wrong (it says "resume from epoch 23, still
improving"; reality was epoch 113, plateaued, 82 pp overfit). Ignore it.

## Where the program now stands

**Direction:** NCD (Neural-Constrained Decoder). Test whether the perception→imagery
generalization gap documented by **NSD-Imagery (CVPR 2025)** is caused by unconstrained
intermediate representations rather than capacity. One manipulated variable: `λ_neural`.

**Phase 2 delivered:** literature review (`13`), novelty analysis (`14`), salvage/replace
(`15`), four successor architectures (`16`), adversarial selection (`17` + panel report),
formal spec (`18`), experiment plan (`19`), SAP (`20`), compute DAG (`21`), NSD-Imagery
full-text audit (`22`), and **the NCD model + 15 contract tests + 2 pilot configs, committed
and passing**.

## Exact current state

| | |
|---|---|
| Branch | `feature/predictive-cortical-decoder` @ `e706546` |
| Tests | **24 passing** (9 PCD contracts + 15 NCD contracts). 10 pre-existing failures on this CPU box (missing `diffusers`/`nibabel`/`h5py`, unset env) — verified unrelated, reproduce with changes stashed |
| Active runs | **NONE.** H100 idle. Nothing started, killed, or deleted this session |
| NCD size | **20 168 188 params — measured**, 8.3× smaller than PCD_v4 |
| New code | `src/fmri2img/models/neural_constrained_decoder.py`, `tests/test_ncd_gradient_flow.py`, `configs/experiments/NCD_v1_pilot_1subj{,_lam0}.yaml` |

## Next deterministic actions

1. **B-DATA — obtain NSD-Imagery.** Not on the pod (verified). Request via
   naturalscenesdataset.org; institutional access, CC-BY-NC-ND 4.0. **This is the top
   blocker and it is external.** 181 TB free, so capacity is not the issue.
2. **The cheap kill test, and it comes before everything else** (`22` §7): can any model
   identify NSD-Imagery's 18 stimuli from *perception* trials? They are 6 geometric shapes,
   6 word concepts, and 6 scenes — wildly out of distribution vs 73k natural scenes. **If
   vision 2AFC is at chance, imagery transfer is unmeasurable and the thesis dies on data
   grounds regardless of `λ_neural`.** Eval-only, no training.
3. **B-LIT remainder:** full-text LEA (arXiv 2303.14730) — the last novelty check — and
   Hi-DREAM (is their random control parameter-matched?).
4. **Wire NCD into `create_model`** (`unified_model.py`) and the training loop for
   `type: "ncd"`; add `count_flops.py` (`FLOPs: NOT_MEASURED` must not appear twice).
5. **E-03 pilot** once 1–4 clear.

## Decisions that must not be reopened without new evidence

| ID | Decision |
|---|---|
| D-001 | Do not resume PCD_v4 |
| D-002 | No predictive-coding framing; direction is a tested factor, not an assumption |
| D-005 | Never report `val_r@1` without gallery size + trial-level figure; never compare to 77.2% |
| **D-006** | **PCD direction abandoned; NCD/imagery-transfer thesis selected** |
| **D-007** | **No "NeuroPC"; no "identifiability" — claim a neural-prediction constraint** |

## Unresolved — carried forward

1. **Six reviewer objections, none resolved** (`11_REVIEWER_REPORTS/phase2_thesis_review.md`).
   O-1 (auxiliary task = regularizer; the matched-reg arm must be capacity-matched) is the
   one that decides whether any positive result survives review. **The matched-reg arm is not
   yet built.**
2. **n = 4, not 8.** Only NSD subj01/02/05/07 completed NSD training. SESOI = 6 pp on 2AFC.
   **Only large effects are detectable — this belongs in the abstract, not the limitations.**
3. **Distribution shift may sink the endpoint** (R-15). See action 2.
4. **Novelty is TENTATIVE, not established.** Review is abstract-level, single-index (`13`
   §7). **No "first" claims.**
5. Why PCD_v4 died at epoch 113: still UNKNOWN (F-004).
6. NCD is not yet wired into the training loop — the model exists and is tested in isolation.

## Context-budget note

`00_EXECUTIVE_STATE.md` + this file are sufficient to resume. Do not re-derive the Gate 0
findings from source — they are pinned by `tests/test_pcd_gradient_flow.py`. Do not re-run
the literature sweep — it is in `13`/`10`. Run the tests; read `14` and `22` for why the
direction changed.
