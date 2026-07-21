# 26 — Gate M1.3: Statistical Correction and Task Separation

**Date:** 2026-07-17 · Simulation only. **No real data. Roy not reproduced.**
**Status: `GATE_M1_S0_FRAMEWORK_VALIDATED__OPERATING_CHARACTERISTICS_UNRESOLVED`**

---

## 1. My M1.2 FPR claim was statistically invalid — retracted

**I wrote:** *"max FPR 0.000 (criterion ≤ 0.05: MET)"* from **15 evaluation seeds per null**.

**Zero events in 15 trials does not establish FPR ≤ 0.05.** Clopper–Pearson 95% CIs:

| observed | 95% CI | compatible with ≤0.05? |
|---|---|---|
| **0/15** | **[0.0000, 0.2180]** | **NO** |
| 0/40 | [0.0000, 0.0881] | NO |
| 0/200 | [0.0000, 0.0183] | yes |
| 0/1000 | [0.0000, 0.0037] | yes |

The true FPR could be as high as **21.8%** and still produce my observation. **A minimum of
n ≥ 72 zero-event evaluations is required merely to bound FPR ≤ 0.05** (`ln 0.025 / ln 0.95`),
and §6's 1 000 seeds is the right target.

**Retracted:** "criterion met". **Every FPR must be reported with an exact binomial interval.**
Status downgraded accordingly.

## 2. Two tasks that I conflated

**Task P — predictive, Level B.** *Does source-state activity add held-out predictive value
beyond the preregistered measured feature battery?* Estimand `ΔR²_source|F`. **A predictive
claim.** Testable by conditional permutation that breaks the source association *given* the
measured battery. **Well-posed and answerable on NSD-Imagery.**

**Task C — latent-confounding sensitivity.** *How strong would an unmeasured shared
condition-level factor have to be to explain an observed increment?* **This is a
partial-identification / sensitivity problem, NOT a binary test.**

> **The M1.2 error:** I used an **unrestricted, arbitrarily strong** W1f to set the *primary
> rejection threshold for Task P*. That imports an unfalsifiable worst case into a predictive
> test and guarantees zero power — which is exactly what happened. **An unrestricted latent
> common cause makes causal interpretation unidentifiable by construction; it should not be
> allowed to destroy a predictive inference.**

Task C's output is a **robustness frontier** — the minimum latent strength needed to explain an
observed effect — **not** a threshold.

## 3. W1f reclassified

Renamed **`W1f_condition_locked_latent_common_cause`**: stable per content identity, shared
across source and target states, unobserved, **and unconstrained in the current simulation**.

**It is not "attention."** I labelled it that; that was wrong. Attention fluctuates at trial
and run level, and a *condition-locked* factor is a different object. Distinct worlds needed:

| world | nature | analogous to current impl? |
|---|---|---|
| **W1f-content** | stable latent content property — salience, memorability, personal meaning, unmeasured semantics | **YES — this is what I implemented** |
| W1f-trial | trial-level attention fluctuation across runs | no |
| W1f-session | run/session nuisance across conditions | no |
| W1f-subject | participant-specific factors shared across states | no |

## 4. The unmatched-comparison overreach — retracted

**I wrote:** *"W1f produces mean ΔR² +0.470, larger than the genuine transformation's +0.213"*
and concluded the nuisance is **stronger** than the transformation.

**Invalid.** Those were **one unmatched parameter setting**. W1f and W2 were not matched on
source reliability, target reliability, source–target predictive correlation, effective rank,
signal variance, or total explainable variance. Comparing their raw magnitudes says nothing
about which is intrinsically stronger — only that I happened to pick a strong nuisance.

**The legitimate claim, and the only one retained:**

> **Some latent-common-cause configurations are observationally indistinguishable from a
> condition-level transformation.**

That is enough to establish the identification limit. It does **not** license "the nuisance is
stronger", and I withdraw that.

## 5. What survives from M1.2 — unchanged and important

- **W1c (incomplete observed features) produces positive `ΔR²_source|F` with no transformation
  present.** Since no real feature battery is exhaustive, a positive real result is **always**
  W1c-compatible.
- **An unrestricted latent common cause makes direct causal interpretation observationally
  unidentifiable.**
- **NSD-Imagery reaches Level B at best**; Level C needs MindStates.

## 6. Nuisance-control inventory for NSD-Imagery (what can actually be controlled)

| proxy | status |
|---|---|
| stimulus identity; simple-vs-naturalistic family | **measured** |
| orientation, spatial occupancy, edge/SF summaries | **derivable** from stimuli |
| image/text embeddings; image complexity | **derivable** |
| run/session index; trial order; repeat number | **measured** (in the design) |
| VVIQ vividness | **measured** — subject level only, not per trial |
| salience, memorability | **inferable** via published models — external, imperfect |
| motion, physiological signals | **status unverified** |
| eye movements | **likely unavailable** (imagery with eyes closed/fixed) |
| imagery difficulty per trial | **unavailable** |

> **Do not call unavailable nuisances controlled.** Per-trial imagery difficulty and eye
> movements are the gaps most likely to carry a W1f-content-like factor, and they are exactly
> what cannot be measured retrospectively.

## 7. Corrected conclusion language

**Permitted after S3:** repeat-level prediction reproduced (or not) · identity-template
baseline sufficient (or not) · content-held-out generalization present (or not) · incremental
value beyond the **measured** battery present (or not) · result robust or sensitive **to
specified latent-confound strengths**.

**Forbidden:** direct neural mechanism proven · causal transformation established · all
stimulus mediation excluded · all attentional confounding excluded · **Roy result artifactual**
(unless the real reproduction and controls directly support that narrower claim).

## 8. Stage structure

**S0-P** predictive detector calibration (1 000 seeds, exact CIs) — **not done** ·
**S0-C** latent-confounding sensitivity frontier — **not done** ·
**S1** exact Roy reproduction — **not started** · S2 · S3 · S4.

**No transformation-mechanism verdict is available from S1–S3 at all.**

## 9. S1 status: NOT STARTED

§10 authorises starting S1 independently of the causal detector, and that is correct — exact
reproduction does not depend on Task C. It was not started this session: context budget was
consumed by the statistical corrections above, and downloading data without the budget to
verify checksums and provenance would violate the standing rule. **S1 is the next action and
is unblocked.**
