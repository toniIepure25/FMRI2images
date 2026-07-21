# 24 — Gate M1.1: Corrected Estimand and Identifiability Framework

**Date:** 2026-07-17 · Simulation only. **No real data. Roy not reproduced.**
Code: `src/fmri2img/mindcompiler/state_transform_sim.py` · Tests:
`tests/mindcompiler/test_state_transform_sim.py` (13, passing) · CPU, NumPy.

**Status: `GATE_M1_S0_SYNTHETIC_IDENTIFIABILITY_CHECK_PASSED`.**

---

## 1. What was wrong with the Gate M1 simulation

The old `identity_confound.py` modelled **single-trial** perception→imagery coupling
(`imagery_trial = perception_trial @ W`). **The Roy design cannot observe that:** vision and
imagery were separate runs, paired **randomly within stimulus identity**. Single-trial
alignment is destroyed by that pairing. Modelling it as the positive world tested a quantity
the data cannot contain.

## 2. The corrected worlds (condition-level estimand)

| World | Generative rule | Meaning |
|---|---|---|
| **W0** | `μ^Y_k` arbitrary, unrelated to `μ^X_k` | condition-template null; no cross-content rule |
| **W1** | `μ^X_k = A f_k`, `μ^Y_k = B f_k` | vision/imagery related **only** through observable features `f_k` |
| **W2** | `μ^Y_k = T(μ^X_k) + η_k`, `η_k ∉ span(f)` | genuine condition-level transform, generalizes to unseen content |
| **W3** | `μ^Y_k = T(μ^X_k) + B f_k + u_k` | mixture |
| **W4** | single-trial `imagery = perception @ W` | non-primary; see §4 |

Perception and imagery repeats carry **independent** trial noise and are **randomly paired
within identity**, reproducing the Roy acquisition.

## 3. The estimand — unique neural contribution beyond features

**Your W1 point forces the definition.** In W1, `μ^Y_k = B f_k` and `μ^X_k = A f_k`, so a
condition-level linear map `T = B A⁺` **exists and generalizes to held-out content** — a
neural-source model *will* predict held-out imagery in W1, with **no state transformation
present**. Raw held-out prediction therefore cannot be the estimand.

**Primary estimand:** *unique neural contribution beyond stimulus features*, under
leave-one-identity-out — `R²(features + neural) − R²(features)`. It must be ≈0 in W0 and W1
and >0 in W2/W3.

**LOIO variance partition, mean over 12 seeds, n_id = 12:**

| World | R²(features) | R²(neural) | R²(both) | **unique neural** |
|---|---|---|---|---|
| W0 (null) | −0.254 | −0.551 | −0.569 | **−0.316** |
| **W1 (features only)** | 0.423 | 0.058 | 0.273 | **−0.150** |
| **W2 (transform)** | 0.429 | 0.712 | 0.709 | **+0.281** |
| W3 (mixture) | 0.465 | 0.434 | 0.540 | **+0.075** |

> **The estimand separates the worlds in the right direction.** W1 — where a neural-source
> model *does* generalize — correctly yields **no unique neural signal**, because features
> already explain the imagery templates. Only W2/W3 show positive unique neural contribution.
> This is the discriminator a real analysis must use; raw `vis2img` accuracy is not.

## 4. W4 correction — single-trial coupling does NOT simply "average away"

The prompt anticipated that single-trial coupling would vanish under random pairing. **The
simulation showed otherwise, and the correction strengthens the identifiability argument.**

A **linear** single-trial map `W` applied to trials with condition mean `μ^X_k` has
condition-mean image `μ^X_k @ Wᵀ` — because **averaging commutes with a linear operator**. So
the **condition-level** transform survives random pairing (measured unique neural ≈ 0.24,
W2-like). What is genuinely non-identifiable is the **trial-to-trial alignment** — the residual
orthogonal to condition means.

> **Consequence:** even a purely single-trial mechanism is read by this design as a
> condition-level transformation, and the two **cannot be separated**. The Roy design
> identifies *condition-level* structure only. This is a **stronger** limitation than "single
> trial averages away", and it is pinned by
> `test_single_trial_coupling_is_condition_level_indistinguishable_from_W2`.

## 5. Analytical identifiability argument

With `K` identities present in both train and test (repeat-level split) and a model of rank
`r` mapping `μ^X_k ↦ μ^Y_k`, the shortcut — memorizing the finite identity→template
association with no cross-content rule — exists when the training templates are **linearly
separable at the model's capacity**, i.e. roughly when `r ≳ K` or the ridge penalty is weak
relative to template separation. Concretely:

- If `r ≥ K` (or effective d.o.f. ≥ K), a linear map can interpolate all `K` training
  templates exactly regardless of any rule → repeat-level `r` is uninformative.
- If `r ≪ K` **and** regularization is strong, the map is forced to share structure across
  identities → repeat-level performance begins to reflect genuine cross-content regularity.

**This is a condition, not a universal claim:** not every linear model memorizes identity.
Whether NSD-Imagery (`K = 12`) sits in the memorizing regime depends on the selected rank and
ridge, which is exactly why **nested selection inside training identities** (never on the
held-out identity) is mandatory. Content hold-out sidesteps the issue entirely: a held-out
identity has no template to memorize.

## 6. Monte Carlo operating characteristics

Detector: `unique_neural > q95(pooled W0∪W1 null)`, per identity count.
**See `24b_monte_carlo_operating_characteristics.txt` for the generated table** (FPR against
W1, power against W2, across `n_id ∈ {6,12,24,64}` and a target-SNR sweep). The qualitative
result — null worlds below the transform worlds — holds across all tested `n_id`
(`test_separation_holds_across_identity_counts`). **Absolute power numbers are
configuration-specific and are NOT design-level sensitivity claims for real data** (see §7).

## 7. Claims retained, retracted, narrowed

**RETAINED:**
> Repeat-level evaluation alone cannot distinguish a reusable cross-content transformation
> from a condition-template mapping. *(Now with an analytical condition, §5.)*

**RETRACTED (my Gate M1 errors):**
- ~~"The identity-held-out test is valid at 12 identities."~~ →
  *Under the tested synthetic alternatives, complete identity hold-out correctly distinguishes
  a strong generalizable transformation from a condition-template null; **empirical sensitivity
  with 12 identities remains unknown.**"*
- The **0.447** sensitivity estimate, the **0.93** asymptote, **"attenuated by more than
  half"**, and any claim that **64 identities is generally sufficient** — all **configuration-
  specific artifacts of an arbitrary noise regime**, withdrawn as design-level statements.
- ~~"single-trial coupling averages away under random pairing"~~ → §4: the *condition-level*
  component survives; only trial alignment is lost.

**ADDED:** the estimand is **unique neural beyond features**, not raw held-out prediction —
because W1 generalizes without any transformation.

## 8. Gate levels

- ✅ `GATE_M1_S0_SYNTHETIC_IDENTIFIABILITY_CHECK_PASSED` — this document.
- ⬜ `GATE_M1_S1_REAL_DATA_REPRODUCTION`
- ⬜ `GATE_M1_S2_IDENTITY_BASELINE_EVALUATION`
- ⬜ `GATE_M1_S3_CONTENT_HELDOUT_EVALUATION`
- ⬜ `GATE_M1_S4_RICH_DATASET_CONFIRMATION` (Roy Dataset 2, 512 conditions)

**No Roy-level scientific verdict is permitted before S1–S3.**

## 9. NSD-Imagery reproduction readiness (S1 — NOT executed)

Required before download: file inventory + sizes from naturalscenesdataset.org; confirm
CC-BY-NC-ND access terms; map Roy's ROI definitions (nsdgeneral + visual ROIs already used by
this repo); reproduce denoised `vis2vis` inputs; write data/split manifests with hashes;
**preregister reproduction tolerance** (e.g. per-ROI median `vis2img` r within ±0.05 of Roy's
Fig 3E, or explain the discrepancy). **Do not download until context budget permits provenance
and integrity verification.**

## 10. What this can and cannot establish

**Can:** that the estimand and content-hold-out protocol correctly separate the four worlds on
known ground truth; that the discriminator is *unique neural beyond features*; that the Roy
design identifies condition-level (not single-trial) structure.

**Cannot:** that Roy's empirical result is artifactual (it is **not** presented as such), real
sensitivity at K=12, semantic independence, cross-subject universality, or any discovery.
