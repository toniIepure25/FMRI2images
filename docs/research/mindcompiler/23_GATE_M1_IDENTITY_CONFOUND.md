# 23 — Gate M1: The Identity Confound in vis2img Evaluation

**Date:** 2026-07-17 · Simulation only. **No real data was downloaded. Roy was not reproduced.**
Code: `src/fmri2img/mindcompiler/identity_confound.py` · CPU, NumPy, seconds to run.

---

## 1. The structural observation

With **12 stimulus identities** and 8–16 repeats each, any split over *repeats* **necessarily**
places every identity in train, validation and test. Roy et al.'s 4-fold cross-validated
protocol is a repeat-level split.

A model can therefore score well by learning *"this perception pattern belongs to stimulus k,
so emit stimulus k's average imagery pattern"* — **a lookup table containing no transformation
at all.**

## 2. Simulation result — the protocol cannot distinguish the two worlds

Two generative worlds, known ground truth:
- **`identity_template`** — imagery depends **only on which stimulus it was**. The perception
  *trial* is irrelevant. **No transformation exists.**
- **`true_transform`** — imagery is a genuine linear function of the single perception trial.

| World | Split | **vis2img r** | B-ID1 (identity template) r |
|---|---|---|---|
| **identity_template** *(no transformation)* | **repeat-level** | **0.394** | 0.603 |
| identity_template | identity-held-out | **−0.010** | undefined |
| true_transform | repeat-level | 0.662 | 0.543 |
| true_transform | identity-held-out | **0.447** | undefined |
| mixture | repeat-level | 0.551 | 0.496 |
| mixture | identity-held-out | 0.349 | undefined |

> **A world containing no neural transformation whatsoever yields vis2img r ≈ 0.39 under the
> repeat-level split** — inside Roy et al.'s reported range of r ≈ 0.3–0.5.
>
> Under identity-held-out evaluation the same world correctly collapses to **≈ 0**.

**Calibration caveat, stated plainly:** my noise regime is arbitrary and **not fitted to
NSD-Imagery**. The numerical coincidence between 0.394 and their 0.3–0.5 is **not** evidence
that their result *is* an artifact, and must not be reported as such. What the simulation
establishes is **structural**: the repeat-level protocol **cannot discriminate** the two
worlds, so a repeat-level r — whatever its value — is **not by itself evidence of a
transformation**.

## 3. The diagnostic works — the experiment is well-posed

The **vis2img vs B-ID1 comparison** separates the worlds, in the right direction:

- template world: vis2img **0.394 < 0.603** baseline → model *loses* to lookup
- transform world: vis2img **0.662 > 0.543** baseline → model *beats* lookup

**This is the primary diagnostic**, exactly as specified: *incremental prediction of vis2img
beyond the strongest identity-template baseline*. It is implemented and validated on
known ground truth.

## 4. Is 12 identities enough? — my prior expectation was WRONG

I expected to find that 12 contents make this unidentifiable. **The simulation says otherwise.**

| n_identities | held-out r (true_transform) | held-out r (identity_template) |
|---|---|---|
| **12** | **0.447** | −0.010 |
| 24 | 0.626 | −0.005 |
| 64 | 0.920 | 0.002 |
| 128 | 0.928 | 0.001 |
| 512 | 0.932 | 0.001 |

**At 12 identities the identity-held-out test is VALID** — correctly ≈ 0 under the null,
clearly positive under a strong alternative. What is compromised is **sensitivity and effect
size**: 0.447 against an asymptote of ~0.93, i.e. **attenuated by more than half**.

**Consequences:**
- A **positive** identity-held-out result at n = 12 is **interpretable and conservative**.
- A **null** is **ambiguous** — indistinguishable from a weak-but-real transformation.
- **Roy Dataset 2 (512 conditions) is where the effect size becomes estimable**, and it sits at
  the asymptote. It should be the **confirmatory** dataset; NSD-Imagery is **exploratory**.

## 5. Gate M1 decision: **NO VERDICT ISSUED**

All six permitted verdicts presuppose that the Roy reproduction ran. **It did not** — no data
was downloaded, no reproduction attempted, no real r computed. Issuing
`ROY_RESULT_REPLICATED__*` or `ROY_RESULT_NOT_REPRODUCED` would be fabrication.

`E_M1_UNIDENTIFIABLE_WITH_12_CONTENTS` was the verdict I expected to issue, and **§4 refutes
it**: the test is valid at 12, merely attenuated. Issuing it would have been wrong in the
direction of my own prior.

**What this session did establish, and it is substantive:**
1. The repeat-level protocol **cannot distinguish** a transformation from identity lookup.
2. Therefore **Roy's reported r cannot be read as evidence of a transformation without the
   identity-baseline comparison they did not run.** This is a genuine, checkable methodological
   gap in an actively-cited preprint.
3. The identity-held-out diagnostic **works**, is implemented, and is validated on ground truth.
4. **E-M1 is worth running**, and its design is now fixed in advance of seeing any real data.

## 6. Limits

Cannot establish: universal state transformation · semantic independence · cross-subject
universality · multi-state laws · path dependence · prospective prediction · A\*-level
discovery. **Nor whether Roy's actual result survives** — that needs the real data.
