# 12 — Session Handoff

**Written:** 2026-07-16 (end of Phase 2.5) · Supersedes `docs/HANDOFF_CONTEXT.md` entirely.

---

## Decision issued this phase

### `CLEARED_FOR_MULTI_SHIFT_PILOT` — scoped and conditional (`27` §5)

**Cleared:** one-subject matched-control pilot (ARM-A/B/C/F), NSD only, ~8 GPU-h.
**NOT cleared:** the 8-subject confirmatory program, the ~450 GPU-h plan, any OOD or
imagery claim.

## What changed this phase

**Novelty was repaired, not asserted.** Spera et al. 2026 (arXiv 2604.15374) **fit on
imagery** — latent functional alignment, matched imagery–perception supervision,
retrieval-based augmentation. So *"we improve perception→imagery transfer"* is **FORBIDDEN**
(C-013). The surviving distinction is a **setting** difference: a perception-only constraint
evaluated **zero-shot** across **five shifts**, of which imagery is one sealed test.

**NSD-Synthetic replaces NSD-Imagery as the primary OOD test** (`26`): 8 subjects vs 4, 284
stimuli vs 18, CC-BY 4.0 vs CC-BY-NC-ND, purpose-built for OOD. **This confines the n = 4
power crisis to one secondary test** — the most valuable structural consequence of the phase.

**The key unblocking insight:** four of six shifts (in-distribution, reduced-data, noise,
subject) run on NSD, **already on the pod**. The matched-control pilot needs no external data.

## Exact current state

| | |
|---|---|
| Branch | `feature/predictive-cortical-decoder` |
| Tests | **62 passing** (15 NCD + 27 matched-control + 13 config-parity + 9 PCD (mislabelled file name, tests are contracts)). 10 pre-existing unrelated failures on this CPU box |
| Active runs | **NONE.** H100 idle. Nothing started, killed, or deleted |
| Arms | All 8 **exactly** parameter-matched (not merely within tolerance) |
| New code | `auxiliary_objectives.py`; `neural_constrained_decoder.py` (`aux_objective`, `auxiliary_loss`, `_aux_target`); `tests/test_ncd_matched_controls.py`; `tests/test_ncd_arm_config_parity.py`; 4 arm configs |

## Next deterministic actions

1. **Wire NCD into `create_model` (`unified_model.py`) and the training loop** for
   `type: "ncd"`, reading `loss.neural_prediction.objective` → `AuxObjective`.
   **The pilot cannot execute until this is done.** Engineering, not science.
2. **Declare the cross-shift multiplicity family (O-7).** Cheap, and until it exists
   *"improves ≥2 shifts"* is a garden of forking paths. **Do this before the pilot reads out.**
3. **Run E-P1**: ARM-A/B/C/F × 2 seeds, subj01, ~8 GPU-h. Read **B vs F** first.
4. **B-DATA** (parallel, 0 GPU): request NSD-Synthetic (CC-BY 4.0) and NSD-Imagery from
   naturalscenesdataset.org.
5. **Verify R-20/O-9**: the CLIP cache has never been checked on grayscale/Mooney/line-drawing
   stimuli. It sits upstream of every synthetic number.

## Decisions that must not be reopened without new evidence

| ID | Decision |
|---|---|
| D-001 | Do not resume PCD_v4 |
| D-002 / D-007 | No predictive-coding framing; no "NeuroPC"; no "identifiability" |
| D-005 | Report both R@1 definitions with gallery size; never compare to 77.2% |
| D-006 | PCD direction abandoned |
| **Phase 2.5** | Imagery is **sealed** and **secondary**; NSD-Synthetic is primary OOD; imagery-transfer improvement is FORBIDDEN (owned by Spera et al.) |

## Unresolved — carried forward

1. **O-7 (blocking any confirmatory read-out):** cross-shift multiplicity family undeclared.
2. **O-8:** ARM-F's tuning budget must be equal and logged, or the parity claim is
   unfalsifiable and a reviewer will assume a straw man.
3. **O-9 / R-20:** CLIP cache unverified on synthetic stimuli.
4. **O-10:** if B ≈ D (random targets), the word "neural" must be dropped from the mechanism.
5. **O-12:** if B ≈ H (per-ROI autoencoding suffices), that must be **reported, not buried**.
6. Full PDF reads outstanding: **Spera et al.** (if it contains a zero-shot arm, `23` §6
   reopens and the clearance is void), LEA, Hi-DREAM.
7. Why PCD_v4 died at epoch 113: still UNKNOWN (F-004).

## Pre-committed conclusions (do not renegotiate after seeing results)

- **B ≈ F** → *"auxiliary regularization improves robustness; neural target structure is not
  specifically supported."* Publishable. **The most likely outcome.**
- **B > all, neural predictivity ↑, OOD decoding flat** → route to computational neuroscience.
- **B ≈ G** → anatomical grouping irrelevant. **B ≈ H** → predictive dependency inert.

## Context-budget note

`00_EXECUTIVE_STATE.md` + this file suffice to resume. Do not re-derive Gate 0 findings
(pinned by tests). Do not re-run the literature sweep (`13`, `10`, `23`, `26`). Read `24` for
the thesis and `25` for why the control family exists.
