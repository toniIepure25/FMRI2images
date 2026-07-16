# 24 — Multi-Shift Generalization Thesis

**Date:** 2026-07-16 · **Revised Phase 2.6.** Supersedes `17`'s imagery-centred framing.

---

## 0. Phase 2.6 revision — shift 3 is removed on empirical grounds

The Spera et al. full text (`23` §0) reports a **frozen zero-shot DynaDiff baseline at
chance**: CLIP 48.94%, Alex(5) 50.21%, Alex(2) 51.03% (chance = 50%).

**A perception decoder far stronger than NCD is at chance zero-shot on imagery.** There is no
headroom for ARM-B to beat ARM-F at a floor. **Zero-shot imagery is removed as a positive
endpoint.** It may be reported as a null, corroborating Spera et al.; it is **not a
contribution** (C-023, C-024).

This is a real loss. Imagery was the shift that most distinguished this thesis from the
generic claim *"auxiliary objectives improve robustness"*. **With it gone, the entire
scientific content rests on ARM-B vs ARM-C (is the target's neural identity doing the work?)
and ARM-B vs ARM-F (is it doing anything a tuned regularizer would not?).** See `23` §7.

## 1. Thesis (revised)

> **A neural-prediction constraint learned exclusively from perception data produces
> representations that generalize better than discriminative and generically regularized
> decoders across stimulus-distribution shift, subject shift, reduced-data regimes, and
> neural noise.**

**Superseded wording** (retained for provenance): the same sentence ending *"...and zero-shot
mental imagery."* Removed per §0 — not because it was unfashionable, but because the endpoint
has no headroom.

**Unit of the claim is generalization, not imagery.** Imagery is one sealed test of a broader
principle. This is the Phase 2.5 correction and it is forced by evidence, not taste:
Spera et al. (2026) own imagery *adaptation* (`23`), so imagery alone cannot carry a paper.

## 2. What makes this distinct — and it is a *setting* difference, not a method tweak

| Constraint | Why it is load-bearing |
|---|---|
| **No imagery data in model fitting** | Spera et al. fit on imagery. If we leak, our setting collapses into theirs and the contribution is gone. |
| **No imagery in hyperparameter or λ selection** | Selecting λ on imagery is fitting on imagery with extra steps. |
| **Neural-prediction regularization learned on perception only** | The constraint must be *free* — inferable without ever seeing the target state. |
| **Zero-shot imagery evaluation** | The only way to claim a generalization *property* rather than an adaptation *method*. |
| **Additional shifts where imagery labels are irrelevant** | Proves the principle is about generalization, not about imagery. This is what makes it a thesis rather than a benchmark result. |

**A single sealed test would be a curiosity. Five shifts, from one perception-only
constraint, is a claim about representation learning.**

## 3. Independent support for the premise

NSD-Synthetic (Gifford et al., Nat Comms 2026, `26` §3) found, on their own data, that **OOD
tests reveal model differences invisible in-distribution** — ID difference peaked at 10%
explained variance, OOD at 25% — and that **degree of OOD predicts failure magnitude**.

Two consequences:
1. Our in-distribution numbers (16.4% R@1, `T11`) are **not the relevant evidence**, and we
   stop apologising for them.
2. The right test is **graded**, not binary. `26` §5 gives us a parametric OOD axis.

## 4. The five shifts

| # | Shift | Dataset | Primary metric | n |
|---|---|---|---|---|
| 1 | **In-distribution** | NSD sealed protocol | R@1 (image + trial), neural predictivity | 8 |
| 2 | **Stimulus distribution (OOD)** ★ primary | **NSD-Synthetic** | decoding vs **OOD degree** (scene-derived, 52); neural predictivity (all 284) | **8** |
| ~~3~~ | ~~Cognitive state (zero-shot)~~ | **REMOVED §0** — floor effect: a stronger decoder is at chance zero-shot (Spera et al.). Reportable as a null, never as a contribution | — | ~~4~~ |
| 4 | **Reduced data** | NSD | R@1 vs data fraction (~1h, 3h, 10, 25, 50, 100%) | 8 |
| 5 | **Noise / missing info** | NSD | R@1 vs corruption (additive noise, voxel/ROI dropout, session removal, low-reliability voxels, reduced averaging) | 8 |
| 6 | **Subject** | NSD | LOSO zero-shot, low-shot curves | 8 |

Shifts 1, 2, 4, 5, 6 use **8 subjects**. Only shift 3 is n = 4. **The power crisis of `22` §5
is thereby confined to one secondary test** — the single most valuable structural consequence
of this reframing.

**Note on shift 6:** cross-subject *methods* are a dead axis (`14` §1 — MindEye2, MindTuner,
ZEBRA). We are not proposing a transfer method. We use LOSO as **one shift among five**, to
test the generalization property. Framing it as a cross-subject contribution would walk into
ZEBRA and lose.

## 5. Predictions that could fail

| Prediction | Falsified if |
|---|---|
| ARM-B > ARM-A/F on ≥2 independent shifts | it wins on ≤1, or only where ARM-F also wins |
| ARM-B's margin **grows with OOD degree** (`26` §5) | margin is a constant offset → capacity effect, not generalization |
| ARM-B improves held-out neural predictivity | it does not → the constraint constrains nothing |
| Consistent subject-level direction | direction flips across subjects → n=8 heterogeneity, not an effect |
| ID cost is small relative to OOD gain | large ID loss for small OOD gain → a trade, not a mechanism |

## 6. Interpretation rules — fixed now, before any run

- **ARM-B ≈ best of ARM-D/E/F** → *"auxiliary regularization improves robustness; neural
  target structure is **not** specifically supported."* This is a legitimate, publishable
  finding and **the most likely outcome**.
- **ARM-B > controls on neural predictivity but not OOD decoding** → route to **computational
  neuroscience**, not an ML venue.
- **ARM-B > controls on both** → the full thesis; ML venue defensible.
- **ARM-B ≈ ARM-G** (random pseudo-ROI) → anatomical structure irrelevant; report that.
- **ARM-B ≈ ARM-H** (ROI structure, no cross-ROI prediction) → the *predictive dependency* is
  inert; only tokenisation matters. Report that.

## 7. Non-goals

Beating MindEye2/ZEBRA/DynaDiff on anything. Imagery adaptation (Spera et al. own it).
Cross-subject transfer methods (dead axis). Reconstruction. Uncertainty. Any claim about
cortical implementation.
