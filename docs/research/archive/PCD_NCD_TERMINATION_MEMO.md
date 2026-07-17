# PCD / NCD — Termination Memo

**Date:** 2026-07-17
**Terminal branch:** `feature/predictive-cortical-decoder`
**Terminal commit:** `d4531e95fc8654e08d528bdba3396f83ac10c63c` (verified present on `origin` by
direct `git ls-remote` — all work published, nothing unpushed)
**Successor:** `research/mindcompiler-neural-state-operators`

This memo closes two programs. Nothing here is deleted: the full history, artifacts, tests,
and decision records remain on the branch above and on `origin`. **This is an archive, not a
purge.**

---

## 1. Why PCD was terminated

**PCD is not canonical predictive coding.** Verified by executable test
(`tests/test_pcd_gradient_flow.py::test_prediction_flows_low_to_high_not_rao_ballard`):
prediction flows **lower → higher** (`prediction_heads[k]` maps level *k* → the tokens of
level *k+1*). Rao & Ballard (1999) requires predictions to **descend** and residuals to
ascend; the field's own canonical statements confirm this. The module docstring cited Rao,
Friston and Clark and claimed a positive result would "validate predictive coding theory".
The implemented operation is **feed-forward cross-level residual extraction** — a defensible
ML inductive bias, wearing the wrong name and the wrong citations.

Compounding it: **decoding accuracy could not validate a biological theory even with the
direction corrected.** That is evidence about information present in the measurement, not
about cortical implementation.

**Per-level uncertainty claims are invalid.** `PCDModel.forward` returned only `(mu, kappa)`;
`level_kappas` went to `_last_pcd_extras`, read by **no loss** (only a smoke test). All four
`PerLevelKappaHeads` receive `grad is None` while every other module trains. Confirmed on the
**trained artifact**: across 14 epochs (checkpoint 98 → 112) the kappa heads moved **4.3e-07**
— 3 of 8 tensors bit-identical — while `vmf_decoder` moved 2.0e-01, a factor of ~5×10⁵. The
residue is EMA float accumulation; AdamW skips `grad is None` parameters outright.

**Why it survived:** `level_kappas.grad_fn is not None`, so the tensor *looks* connected. And
a random projection of *trained* features is not noise — it yields stable, reproducible,
category-varying figures. `pcd_neuroscience_analysis.py` would have produced publishable-
looking plots that meant nothing. **This is the most valuable thing the program found.**

**The overfitting was structural, not a tuning failure.** v1→v4 changed seven regularization
knobs across four runs and produced **no measurable gain**. v4's entire justification — "gap
25 pp vs v1's 82 pp" — was an artefact of reading the train/val gap at **epoch 20, before
convergence**. At epoch 110 the gap was **82 pp**, identical to v1 (train 98.83% vs val
16.36%). Regularization **delayed** overfitting; it never reduced it. Suspected location:
per-subject ROI projections (~57% of parameters). Best val R@1 **16.36% @ epoch 101**,
plateaued; v1 reached ~17.97% @ epoch 29 at 4× less compute.

**Novelty was independently dead.** **Hi-DREAM** (arXiv 2511.11437) uses PCD's *exact*
early/mid/late ROI hierarchy **and already ran PCD's random-and-reversed control, positively**.
**BrainMCLIP** ran a 1 000-permutation ROI-label null. MindHier owns hierarchy→CLIP-layer
alignment; DREAM owns reverse visual pathways; ESANN 2025 owns "predictive-coding dynamics
improve fMRI predictivity".

## 2. Why NCD was terminated

**NCD was methodologically sound.** ~20.2M params (8.3× smaller than PCD), per-ROI nodes
never pooled, low-rank subject adapters at **2.2%** of parameters (vs PCD's ~57%), no
unrestricted `nsdgeneral_other` bypass (263 680 learned params instead of 5 120 000, a cost
independent of its 10 000 voxels), no uncertainty head, and a **matched control family
(ARM-A…H) that was exactly parameter-matched and test-enforced**. The design answered
objection O-1 by construction. That work was not wasted — see §4.

**It was terminated for being scientifically too incremental once its distinctive shift died.**

**Spera et al. 2026 (arXiv 2604.15374) report a frozen zero-shot DynaDiff baseline AT
CHANCE**: CLIP 48.94%, Alex(5) 50.21%, Alex(2) 51.03% against 50%. DynaDiff is far stronger
than NCD. **A stronger perception decoder has no zero-shot imagery signal**, so there was no
headroom for ARM-B to beat ARM-F there. Shift 3 was removed on **empirical** grounds
(independently of its n=4 exact-test p-floor of 1/16).

With imagery gone, the thesis reduced to *"a masked-ROI auxiliary objective improves
robustness across stimulus/data/noise/subject shift"* — **a robustness-regularization claim**,
resting entirely on ARM-B vs ARM-C and ARM-B vs ARM-F. The pre-committed most-likely outcome
was `GENERIC_REGULARIZATION_SUPPORTED`. That is a legitimate small paper. **It is not a
flagship, and the program was honest about that before spending the GPU hours.**

## 3. What is NOT claimed — E-P1 never ran

> **No NCD scientific result exists.** The E-P1 screening pilot was **never executed**. No
> GPU was ever spent on NCD. No arm was ever compared to another. No status
> (`CLEARED_FOR_FOUR_SUBJECT_PILOT` / `GENERIC_REGULARIZATION_SUPPORTED` /
> `NEURAL_TARGET_IDENTITY_UNSUPPORTED` / `NCD_THESIS_FALSIFIED`) was ever issued, because all
> of them presuppose a pilot that did not happen.

Everything committed under NCD is **infrastructure, protocol, and negative literature
findings**. Anyone resuming must treat every NCD scientific claim as **untested**, not as
"promising".

## 4. Reusable assets (carry forward to MINDCOMPILER)

| Asset | Status |
|---|---|
| **Verified clean splits** — train ∩ val = 0; train/val ∩ SHARED1000 = 0, checked against NSD's own `nsd_stim_info_merged.csv` | **The single most valuable asset.** Reuse wholesale |
| `src/fmri2img/stats/partial_conjunction.py` + 21 simulation tests | Reuse directly — the r-of-n compound test is exactly what a multi-property algebra claim needs |
| `src/fmri2img/data/arm_c_partner.py` + 17 tests (derangement, repetition consistency, subject/split containment, restart stability, content-addressed manifest) | Reuse — shuffled-content controls are mandatory for any transport claim |
| `auxiliary_objectives.py` (matched-control family, `stable_seed`, pseudo-ROI grouping, `assert_param_parity`) | Reuse — the matched-control discipline transfers unchanged |
| `neural_constrained_decoder.py` (per-ROI tokenisation, low-rank subject adapters, budgeted context node) | Reuse as an **observation model** component |
| Gradient-reachability test pattern | **Reuse as a program-wide rule** |
| Config-diff parity testing | Reuse |
| Split/leakage audit tooling | Reuse |

**The transferable methodological lesson:** *every interpreted quantity must have an
identifying objective, enforced by a gradient test.* PCD's kappa heads violated it and would
have produced plausible figures from random weights. MINDCOMPILER's operators, transports and
uncertainties are all interpreted quantities and inherit this rule.

## 5. Claims that remain FORBIDDEN

Carried forward from `06_CLAIM_EVIDENCE_REGISTRY.csv`. These do not expire with the branch.

| Claim | Why |
|---|---|
| PCD implements / validates predictive coding | Direction contradicted (T6); category error (rules 10/11) |
| Per-level uncertainty decomposition | Heads untrained (T8/T13/F-002) |
| Prediction errors carry more information than raw activations | Magnitude ≠ information (rule 8); never tested |
| v4 regularization reduced overfitting | Contradicted: 82 pp at convergence (T4/F-001) |
| PCD 16.4% vs frozen 77.2% | Invalid comparison: different gallery/protocol/split; PCD never ran SHARED1000 (T12) |
| Any "first" (zero-shot perception→imagery; first on Imagery-NSD; first improving imagery transfer) | Spera et al. published the zero-shot arm (C-020…C-022) |
| We improve perception→imagery transfer | Owned by Spera et al. (C-013) |
| ARM-B beats ARM-F on zero-shot imagery | Floor effect — a stronger decoder is at chance (C-023) |
| CLIP near-chance on non-semantic stimuli = OOD failure | Artefact by construction (C-019) |
| Anatomical hierarchy novelty | Hi-DREAM / BrainMCLIP own the control |
| Any NCD scientific result | **E-P1 never ran** |

## 6. Final state

| | |
|---|---|
| Terminal branch | `feature/predictive-cortical-decoder` |
| Terminal commit | `d4531e9` — **verified on `origin` via `git ls-remote`** |
| Unpushed work | **None** |
| Working tree | clean but for `docs/CLAUDE_MEGA_PROMPT.md` (user-authored prompt doc, deliberately untracked) |
| Tests | 115 program tests passing; 10 pre-existing environment failures (missing `diffusers`/`nibabel`/`h5py`, unset env), verified unrelated |
| Pod | `orchestraiq-jupyter-54644cff87-gz6n2` — **no processes running, GPU idle (4 MiB, 0%)**. Pod git HEAD `b96affa` with hand-copied files; **matches no commit** (F-005) |
| PCD checkpoints | pod-only, hashed in `05_EXPERIMENT_REGISTRY.csv`: `best` (ep98) `9cece39b…`, `last` (ep112) `d716be3c…` |

**Nothing was deleted, squashed, rewritten, or force-pushed.** The PCD/NCD history stands as
the record of what was tried and what was found — including, and especially, the negatives.
