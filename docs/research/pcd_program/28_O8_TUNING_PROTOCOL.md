# 28 — O-8: Equal Tuning Budgets for ARM-B and ARM-F

**Date:** 2026-07-16 · **Status: FROZEN.** No pilot metric has been read.
**Binding rule:** the phrase *"ARM-B outperforms tuned regularization"* is **forbidden**
unless every clause below was satisfied and logged.

---

## 1. Why this document exists before any number

Reviewer 1: *"an under-tuned ARM-F is a straw man."* If ARM-F gets one lazy sweep and ARM-B
gets careful attention, the decisive B-vs-F comparison is worthless and a reviewer will
assume the worst — correctly. Parity of *architecture* (already exact, C-017) does not imply
parity of *search effort*. This freezes search effort.

**The asymmetry to guard against is not malice, it is attention.** ARM-B is the hypothesis;
it is natural to think harder about it. The protection is to remove the choice: both arms get
the same grid size, the same seeds, the same updates, the same selection rule, decided now.

## 2. What each arm may tune — exhaustive

### ARM-B — `lambda_neural` only

| Hyperparameter | Candidate values | n |
|---|---|---|
| `loss.neural_prediction.weight` | **{0.1, 0.25, 0.5, 1.0}** | **4** |

**Nothing else.** `mask_ratio` is fixed at 0.3 (it is a dose variable belonging to the
ablation family `18` §7, not the tuning budget). Architecture, optimizer, schedule and
regularization are frozen at the base config's values.

### ARM-F — standard regularization only

**4 preregistered configurations**, each a complete joint setting (not a per-knob grid — a
per-knob grid would silently give ARM-F a larger search than ARM-B):

| Config | dropout | weight_decay | fmri_noise_std | voxel_dropout | stochastic_depth |
|---|---|---|---|---|---|
| **F1** (light) | 0.15 | 0.03 | 0.15 | 0.10 | 0.00 |
| **F2** (moderate) | 0.25 | 0.05 | 0.25 | 0.20 | 0.10 |
| **F3** (strong) | 0.35 | 0.08 | 0.35 | 0.25 | 0.15 |
| **F4** (noise-led) | 0.15 | 0.03 | 0.45 | 0.30 | 0.05 |

`λ_neural = 0` for all four; `aux_objective = none`.

**F2 is the config already shipped** in `NCD_v1_armF_pilot_1subj.yaml`. The grid spans
light → strong, and F4 varies the *shape* of regularization rather than only its magnitude,
so ARM-F is not confined to one axis.

**Sanity check on fairness:** these values bracket PCD_v4's settings (dropout 0.25, wd 0.08,
noise 0.25, voxel-dropout 0.15) — the configuration this project already tuned across four
versions. ARM-F is being given the benefit of that prior work.

## 3. Equal budget — the binding table

| Quantity | ARM-B | ARM-F | Equal? |
|---|---|---|---|
| Candidate configurations | **4** | **4** | ✅ |
| Seeds per candidate | **2** | **2** | ✅ |
| Total training runs | **8** | **8** | ✅ |
| Max optimizer updates per run | identical (from base config) | identical | ✅ |
| Early-stopping policy | `patience=15, min_delta=0.002` | identical | ✅ |
| Selection metric | **NSD `val_r@1`** | identical | ✅ |
| Checkpoint-selection rule | best `val_r@1` | identical | ✅ |
| Batch schedule / order | identical (paired seeds) | identical | ✅ |
| Architecture parameters | 19 727 997 | 19 727 997 | ✅ |

ARM-A and ARM-C **do not tune** — A is the untuned floor and C is B's target-identity
control, which must inherit **B's selected `λ_neural`** so that B and C differ in *target
identity only* (protocol `25` §2). Selecting C's λ independently would introduce a second
difference and destroy the control.

## 4. Selection is blind to every endpoint that matters

Selection uses **NSD `val_r@1` only**. During tuning, **no readout whatsoever** of:

- ❌ NSD-Synthetic (shift 2) — the primary OOD endpoint
- ❌ SHARED1000 — sealed, touched once, at the very end
- ❌ NSD-Imagery — sealed (and now a floor anyway, `23` §0)
- ❌ reduced-data, noise, or LOSO shifts (4/5/6)
- ❌ held-out neural predictivity

> **The reason is exact, not decorative.** Every one of those is a Family-P member or a
> mechanism endpoint. Selecting on any of them would make the subsequent test circular:
> the arm would be chosen *for* the thing it is then claimed to be good at. This is rule 3
> and rule 14, and it is the single easiest way to destroy this program's validity.

## 5. Logging — every trial, including failures

Each of the 16 runs writes a registry row with: `arm`, `candidate_id` (B1–B4 / F1–F4),
`seed`, `lambda_neural` or the reg tuple, resolved config hash, **source-file hashes** (never
`git rev-parse` on a dirty tree — F-005), updates completed, wall-clock, peak memory, final
`val_r@1`, selected-epoch, and `status ∈ {completed, diverged, oom, killed}`.

**Failed and diverged trials are logged and reported** (rules 13, 15). A grid where ARM-F
"had a couple of runs crash" and quietly re-ran them is not an equal budget.

## 6. Pilot scope note

The **E-P1 screening pilot** (`27`) runs **one config per arm** (B: λ=0.5; F: F2) × 2 paired
seeds — it is a **futility screen**, not the tuning grid. **The full 8-run-per-arm grid is
required before any B-vs-F claim**, and E-P1 may not be used to select λ and then be reported
as evidence for the selected λ. E-P1 answers only "is there anything here at all?"

## 7. Compliance gate

`06_CLAIM_EVIDENCE_REGISTRY.csv` claim *"ARM-B outperforms tuned regularization"* stays
**FORBIDDEN** until all of §2–§5 are satisfied and every trial is in
`05_EXPERIMENT_REGISTRY.csv`. **The gate is procedural: it does not depend on the result.**
