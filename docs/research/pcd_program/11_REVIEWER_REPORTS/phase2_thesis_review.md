# Phase 2 — Adversarial Thesis Review

**Date:** 2026-07-16 · Five reviewer perspectives, run sequentially and scored independently.
**Disagreement is preserved, not averaged.**

---

## Reviewer 1 — Top ML venue (NeurIPS/ICLR/ICML)

**On A and B: reject without discussion.** B is MindEye2 + MindTuner adapters, and ZEBRA
already published zero-shot cross-subject with public weights. There is nothing here.

**On C:** the framing is the only interesting thing in this document. "Complexity hurts
imagery transfer" (NSD-Imagery, CVPR 2025) is a correlational finding; naming a mechanism and
testing it with one knob is a real contribution *if the control is airtight*.

**My objection, and it is the whole paper:** masked-ROI prediction is an auxiliary task.
Auxiliary tasks regularize. If λ_neural>0 helps, the null explanation — "you added a
regularizer" — is devastating and obvious. The matched-regularization arm cannot be an
afterthought; **it must be tuned to match training loss / effective capacity, not merely
"included"**, and if the authors cannot construct it convincingly I would reject.

**Score: weak accept, conditional on the control.** Without it, reject.

## Reviewer 2 — Computational neuroscience

**Grateful that the predictive-coding framing is gone.** The direction inversion (T6) would
have been fatal and the honesty in `01_TRUTH_AUDIT.md` is unusual.

**On C:** "identifiability" is doing heavy lifting and is being used loosely. Masked-ROI
prediction makes representations *predictive of held-out activity*; it does **not** make them
*identifiable* in the statistical sense (unique up to a known transformation). **Do not
smuggle a strong technical term in for a weak property.** Call it a neural-prediction
constraint and claim exactly that.

**Second objection:** predicting held-out ROI activity from other ROIs largely recovers
**functional connectivity structure**, much of which is stimulus-independent. The model may
improve at exploiting inter-ROI correlation without learning anything about visual
representation. **A stimulus-shuffled control is required** — predict held-out ROI activity
from other ROIs with stimulus identity destroyed. If the objective works equally well there,
it is connectivity, not representation.

**Score: weak accept** if the terminology is fixed and the shuffled control is added.

## Reviewer 3 — Neuroimaging methodology

**On C's data risk:** NSD-Imagery is small and low-SNR. That is the entire ballgame. A null
result would be **uninterpretable without a preregistered equivalence bound and a power
analysis grounded in NSD-Imagery's own reported effect sizes** — which the authors have not
read yet. Doing the power analysis *after* seeing the data is not acceptable.

**Also:** the compatibility audit is listed as blocking, which is correct, but it is deeper
than the plan admits — beta version, preprocessing parity, and imagery-condition definitions
all have to match, and NSD-Imagery's imagery trials are not simply "NSD with different
stimuli."

**Score: borderline.** The design is sound; the statistical realism is not yet demonstrated.

## Reviewer 4 — BCI

**C is the only candidate here with an application story**, and NSD-Imagery's own framing
supports it: imagery decoding is what BCI actually needs, since the signal is always
internally generated. A mechanism for the transfer gap is worth more to this community than
another point of perception R@1.

**Objection:** the authors' base model is at 16.4% R@1. **A transfer ratio computed on a weak
base model may not generalise to strong ones.** If INI improves the ratio for a model that
starts at 16% but MindEye2's ratio is already better, the finding is about *their* model, not
about decoding. **They must report the ratio for at least one strong external baseline** or
scope the claim explicitly to their capacity regime.

**Score: accept with the scoping requirement.**

## Reviewer 5 — Statistics

**n = 8, and NSD-Imagery may have fewer.** Every number in this program must be subject-level
with hierarchical bootstrap over subjects and stimuli. The plan says this; the plan also
proposes a "transfer ratio," which is **a ratio of two noisy estimates and therefore has ugly
variance behaviour and no clean CI**. Use a paired difference on a variance-stabilised scale,
or a hierarchical model with an interaction term, and **stop calling it a ratio**.

**On the metric fork (T11/T14):** three notions of "the result" currently coexist. Declaring
the primary metric *after* both are known is already contaminated (D-005). Declare on
methodological grounds and report all of them, permanently.

**Score: reject as specified; accept if the estimand is redefined.** The estimand is not yet
written down, and everything depends on it.

---

## Panel outcome

**Consensus:** A and B are rejected unanimously. C is the primary, unanimously.

**Preserved disagreements — none of these are resolved and all are entered as blocking:**

| # | Objection | Owner | Resolution required before |
|---|---|---|---|
| O-1 | Auxiliary task = regularizer confound; matched-regularization arm must be capacity-matched, not token | R1 | any positive claim |
| O-2 | "Identifiability" is a misused term; claim neural-prediction constraint only | R2 | any writeup |
| O-3 | Stimulus-shuffled control needed to rule out functional connectivity | R2 | interpreting the mechanism |
| O-4 | Preregistered equivalence bound + power analysis from NSD-Imagery's reported effects | R3 | any null claim |
| O-5 | Transfer effect may be specific to a weak (16.4%) base model; scope or add a strong baseline | R4 | the abstract |
| O-6 | "Transfer ratio" is a bad estimand; redefine as a paired difference / hierarchical interaction | R5 | the analysis plan |

**O-2, O-3, O-5 and O-6 change the design and are folded into `18`–`20` immediately.**
O-4 cannot be discharged until NSD-Imagery is obtained and read — **it is the top blocker.**
