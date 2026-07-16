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

## PHASE 2.6 STATUS: **NO STATUS ISSUED — the pilot did not run**

Phase 2.6 asked for one of five statuses. **All five presuppose the E-P1 pilot ran. It did
not.** Sections 1, 3, 4, 5 are complete and committed; section 2 is half-complete; sections
6–7 were not started. Forcing a status onto an unrun experiment is precisely the failure this
program exists to prevent (it is how PCD's "25 pp gap" happened — F-001).

**Nothing was run. No GPU was used. No status is claimed.**

### Phase 2.6 completed (5 commits)

| § | Item | Commit | State |
|---|---|---|---|
| 1 | Spera full-text audit + claim repair | `202a99b` | **DONE — and it removed shift 3** |
| 4 | Partial-conjunction test + 21 simulations | `39656af` | **DONE** |
| 3 | O-8 tuning protocol frozen | `19d7f55` | **DONE** |
| 6 | NSD-Synthetic dual-endpoint protocol | `185dc5d` | **DONE** |
| 2 | ARM-C partner index + 17 tests | `8800db5` | **HALF** — index done; dataset wiring not |
| 5 | Optimization-parity measurement | — | **NOT DONE** |
| 7 | Pilot + adversarial review | — | **NOT RUN** |

### The finding that matters most this phase

**Spera et al. report a frozen zero-shot DynaDiff baseline AT CHANCE** (CLIP 48.94% vs 50%).
A perception decoder far stronger than NCD has **no zero-shot imagery signal**, so there is no
headroom for ARM-B to beat ARM-F there. **Shift 3 is removed on empirical grounds** (`23` §0).

Consequence, stated plainly: with imagery gone, the thesis is *"a masked-ROI auxiliary
objective improves robustness across stimulus/data/noise/subject shift"* — **a
robustness-regularization claim**. The entire scientific content now rests on **B vs C** and
**B vs F**. If those are null — already the pre-committed most-likely outcome — the honest
report is `GENERIC_REGULARIZATION_SUPPORTED`. That is a smaller paper than intended and it is
what the evidence currently supports (`23` §7).

### Statistical error found and fixed

The SAP's *"≥2 BH-FDR discoveries"* rule **does not calibrate the compound claim**. Measured
over 12k simulations: with **one** true shift (still the r=2 null) it rejects at **7.1–7.9%**
against a nominal 5%. The partial-conjunction test (Benjamini & Heller 2008, r=2/n=5,
Bonferroni — valid under arbitrary dependence) holds at **4.4–5.1%** and costs ~3 pp of power.

## Done since the Phase 2.5 report

- **NCD wired end to end** (`acfc68c`). `create_model` dispatches `type: "ncd"` → `NCDModel`;
  `train_epoch` adds `loss_weights["neural_prediction"] * model._last_aux_loss` via the
  existing `_is_scfr` idiom. No special-casing needed for the weight — `train_unified` builds
  `loss_weights` generically from `config["loss"][k]["weight"]`.
- **O-7 CLOSED** (`20` §4): cross-shift multiplicity family frozen before any read-out.
- **End-to-end smoke passed** (`68392e7`) from the shipped arm configs with real ROI dims:
  **19 727 997 params per arm, identical**; ARM-B's aux heads train, ARM-A/F's provably do not.
- **77 tests passing.**

## Next deterministic actions

1. **Dataloader: supply `shuffled_x` for ARM-C.** Load the permuted partner image's fMRI
   alongside each sample via `DeterministicImagePermutation`, seeded
   `stable_seed(split_hash, seed)`, recorded in the manifest. The model **raises** rather than
   degenerating into ARM-B, so the block is safe and explicit. **ARM-A/B/F are runnable now.**
2. **Verify the real data path provisions `roi_indices` for `type: "ncd"`** in
   `train_unified.py` — the smoke used hand-built indices; the pod path is unverified and is
   the most likely integration bug.
3. **Sync code to pod and run E-P1-A/B/F**: 2 seeds, subj01, ~6 GPU-h. **Read B vs F first.**
4. **B-DATA** (parallel, 0 GPU): request NSD-Synthetic (CC-BY 4.0) + NSD-Imagery.
5. **O-8**: log ARM-F's tuning trials in the manifest, or the parity claim is unfalsifiable.
6. **O-9/R-20**: CLIP cache on grayscale/Mooney/line-drawing stimuli — upstream of every
   synthetic number.

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
