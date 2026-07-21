# 25 — Gate M1.2: Detector Calibration and the Boundary of E-M1

**Date:** 2026-07-17 · Simulation only. **No real data. Roy not reproduced.**
Code: `src/fmri2img/mindcompiler/null_family.py` · Base:
`state_transform_sim.py` · CPU, NumPy.

**Status: `GATE_M1_S0_FRAMEWORK_VALIDATED__CALIBRATION_PENDING`** (downgraded from
`..._CHECK_PASSED`).

---

## 1. Why the status was downgraded

The Gate M1.1 detector had two defects, both correctly identified:

1. **False-positive rate ~0.10 against W1**, short of a 0.05 confirmatory standard. It
   calibrated on a *pooled* W0+W1 null, which is not the least-favourable null.
2. **Power 1.00 at n_id=64 where the W2 effect mean was −0.002.** The detector "detected" a
   transformation because W2 was merely **less negative** than the null. That is not
   detection, and it must not count as one.

Both are fixed in `null_family.py`. **The status upgrades only when max per-null FPR ≤ 0.05
with a positive effect requirement, and the nested-selection tests pass.**

## 2. The canonical estimand name

**Do not write "unique neural contribution" unqualified.** Canonical:

> **Incremental source-state predictive value beyond the preregistered stimulus-feature
> battery**
> `ΔR²_source|F = R²(Y | F, X) − R²(Y | F)`, leave-one-identity-out.

**Interpretation, binding:**
- Positive ⇒ source-state neural measurements add predictive information **beyond the
  measured feature battery**.
- Positive does **NOT** prove a direct neural mechanism.
- Positive does **NOT** exclude unmeasured visual, semantic, attentional, behavioural, or
  subject-specific common causes.
- **Negative values are legitimate.** ΔR² is a **cross-validated predictive contrast**, not an
  information quantity, and a negative value means the source model generalizes worse — it is
  not "negative information".

The internal variable may remain `unique_neural` for compatibility; **all reports and registry
entries use the qualified name.**

## 3. The least-favourable null family

Every world below contains **no state transformation** — `μY` is never a function of `μX`
except through shared causes.

| World | Construction | Why it matters |
|---|---|---|
| **W0** | `μX`, `μY` arbitrary and unrelated | pure null |
| **W1a** | `μX = A f`, `μY = B f` | complete linear feature mediation |
| **W1b** | nonlinear `tanh` on both sides | mediation the linear feature model cannot fully capture |
| **W1c** ★ | `f = (f_obs, f_hidden)`; **only `f_obs` exposed** | **the decisive null** |
| **W1d** | observed features carry measurement error | attenuated feature model |
| **W1f** | shared unobserved condition-level attentional variable | task/nuisance confound |

### W1c is the boundary of what E-M1 can identify

In W1c both states are driven by latent features, but the analyst observes only a subset.
Source-state neural activity then carries information about the **hidden** features that the
feature battery lacks — producing **positive `ΔR²_source|F` with no state transformation
whatsoever**.

> **This limitation must be visible, not hidden.** No feature battery is exhaustive, so a
> positive real-data result is **always** compatible with W1c. This is precisely why
> NSD-Imagery can reach **Level B** and not **Level C** (§4).

## 4. The claim hierarchy — what each level licenses

| Level | Statement | Reachable on NSD-Imagery? |
|---|---|---|
| **A — Predictive generalization** | source-state data predict held-out target content | **Yes**, and **insufficient** — shared features may mediate entirely |
| **B — Incremental value over measured features** | `ΔR²_source|F > 0` | **Yes, at best.** Establishes information not captured by the *preregistered* battery. **Does not prove a neural transformation** |
| **C — State-transformation-specific** | a direct mechanism | **No.** Requires exhaustive controlled feature generation, factorial counterfactual stimuli, randomized state manipulation, or prospective prediction |

**Public NSD-Imagery can reach at most Level B. Level C requires MindStates prospective
acquisition.**

## 5. Calibration procedure

- **Threshold** = `max_j q₉₅(H_{0,j})` over the null family — the **least-favourable** null,
  not a pooled quantile — floored at `δ_min`.
- **Calibration and evaluation seeds are disjoint** (0…n_cal−1 vs 5000+).
- **Rejection requires BOTH** `ΔR² > threshold` **AND** `ΔR² > δ_min > 0`.
- **Criterion:** `max_j Pr_{H0j}(reject) ≤ 0.05` within Monte Carlo uncertainty.
- **Per-null FPR is reported, never only pooled.**

**Provisional synthetic `δ_min = 0.02`**, justified as roughly the smallest ΔR² that is
stably estimable at n_id = 12 in this configuration and clearly above the negative range the
null worlds occupy. **For real data, `δ_min` must be preregistered at subject level after
estimating reliability and noise ceilings, without examining the confirmatory result.**

### Results — the detector is valid but powerless, and that is the finding

Reduced run (15+15 seeds, 60 voxels, n_id=12; full table in
`25b_least_favourable_calibration.txt`). **Magnitudes are configuration-specific; the
structure is what transfers.**

| null | q95 | mean ΔR² | FPR |
|---|---|---|---|
| W0 | 0.106 | 0.012 | 0.00 |
| W1a | 0.091 | −0.012 | 0.00 |
| W1b | −0.011 | −0.097 | 0.00 |
| **W1c** | 0.371 | **+0.017** | 0.00 |
| W1d | 0.159 | 0.039 | 0.00 |
| **W1f** | **0.825** | **+0.470** | 0.00 |

**threshold 0.825 · max FPR 0.000 (criterion ≤0.05 MET) · POWER 0.00 · alt mean ΔR² 0.213**

**Two findings that transfer:**

1. **W1c behaves as predicted** — incomplete observed features give a **positive** mean ΔR²
   with **no transformation present**. Since no real battery is exhaustive, a positive
   real-data result is **always** compatible with W1c.
2. **W1f is the binding constraint, and worse than anticipated.** A single unobserved
   condition-level attentional/nuisance variable shared by both states produces mean
   **ΔR² = +0.470 — larger than the genuine transformation's +0.213.** Calibrating honestly
   against the least-favourable null sets the threshold **above** the alternative, and
   **power collapses to zero**.

> **Under least-favourable calibration the detector is valid (max FPR 0.000) but USELESS
> (power 0.00).** An honest E-M1 cannot separate a genuine condition-level transformation from
> a shared unobserved nuisance at n_id = 12 **unless that nuisance is bounded by design or
> measured**. This is a *design* finding, not a tuning problem: the attentional/nuisance
> confound must be **measured or excluded by design**, not assumed away. It is also the
> strongest argument yet that the flagship needs prospective acquisition.

## 6. Real-data feature battery — preregistered before any results

**Simple stimuli:** orientation · bar/cross structure · spatial occupancy · edge-energy maps ·
spatial-frequency summaries · retinotopic position.

**Naturalistic stimuli:** low-level image statistics · Gabor/edge features · colour statistics ·
independently fixed visual-model representations · object/category labels · scene labels ·
image embeddings · text embeddings.

**Rules:** a small number of **independently fixed** dimensions; **no tuning of a large
embedding space against imagery targets**; and **separate analyses for all / simple-only /
naturalistic-only** — these are not one homogeneous feature distribution and must not be pooled
as if they were.

## 7. Real-data conclusion language — fixed in advance

| | Conclusion |
|---|---|
| **A** | Repeat-level result reproduced; **no evidence of cross-content generalization** |
| **B** | **Condition-template baselines account** for the repeat-level prediction |
| **C** | Generalizes across content, but **explained by the measured feature battery** |
| **D** | **Incremental source-state value beyond the battery.** ⚠ **Do NOT translate into "a neural transformation mechanism has been proven"** — W1c remains compatible |
| **E** | **Inconclusive** — 12 contents insufficiently sensitive or feature-complete |

## 8. Stage separation (unchanged)

S0 synthetic calibration → **S1 exact real reproduction** → S2 identity-template baselines →
S3 content-held-out + incremental source → S4 rich-content confirmation (Roy Dataset 2).
**No Roy verdict before S1–S3.**

## 9. S1 status: **NOT STARTED**

No NSD-Imagery file was downloaded or inspected this session. Required before S1: file
inventory and sizes; licence confirmation (CC-BY-NC-ND); content-addressed download manifest;
Roy preprocessing and ROI mapping; denoised `vis2vis` inputs; **preregistered reproduction
tolerance**. Context budget did not permit provenance and integrity verification, and §13's
instruction not to download without it was followed.

## 10. Outstanding for the S0 upgrade

- Nested selection inside every outer fold (§8): feature-dim, source-dim and ridge tuned on
  **inner training identities only**, with a complexity-matched combined model.
- Conditional permutation testing (§9) verified to control W0/W1a and to **expose** the W1c
  limitation rather than hide it.
- Full recovery metrics (§10): FPR CIs, effect-size bias, interval coverage, model-selection
  and sign stability.
