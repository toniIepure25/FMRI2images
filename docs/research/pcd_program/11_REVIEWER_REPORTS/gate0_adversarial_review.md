# Gate 0 — Adversarial Review

**Date:** 2026-07-16 · **Reviewer role:** Agent F (adversarial), executed sequentially
(no true subagents spawned this session — Gate 0 was bounded and single-scope).
**Material reviewed:** `predictive_cortical_decoder.py`, `unified_model.py`,
`PCD_v4_8subject.yaml`, `pcd_neuroscience_analysis.py`, pod artifacts, `01_TRUTH_AUDIT.md`.

**Recommendation if submitted today: REJECT.** Not a close call.

---

## Fatal flaws

**F1 — The architecture is not what the paper says it is.**
The module claims Rao–Ballard predictive coding. It predicts lower→higher. Rao–Ballard
predicts higher→lower. Any reviewer who reads `forward()` finds this in one pass, and every
citation to Rao (1999), Friston (2005), and Clark (2013) in the docstring becomes evidence
of carelessness rather than grounding. The mechanism itself is fine; the framing is
indefensible.

**F2 — A headline contribution is computed from randomly-initialised weights.**
"Per-level uncertainty decomposition" is one of four stated novel contributions. Those heads
receive `grad is None` under every loss the training loop can construct. The paper would be
reporting a random linear projection. Worse: because it projects *trained* features, it
yields stable, reproducible, category-varying figures — it would have survived visual
inspection and made it into the paper. This is the finding that would end the project's
credibility post-publication.

**F3 — The result that motivated the current experiment is an artefact of measurement timing.**
v4 exists because v1 overfit at 82 pp and v4 "fixed" it to 25 pp. The 25 pp was read at
epoch 20. At epoch 110 the gap is 82 pp. Four versions, ~4× the compute, no gain. A reviewer
asking "what did v2–v4 buy?" gets no answer.

**F4 — Decoding accuracy cannot validate a biological theory.**
Even with F1 repaired, "validates predictive coding in the brain" is a category error. A
decoder's success is evidence about information present in the measurement. This is the
single most common failure mode in this literature and reviewers are primed for it.

## Serious concerns

**S1 — Novelty is entirely unassessed.** The literature matrix is empty. ROI-aware
transformers, hierarchical fMRI decoders, and predictive-coding networks are all dense,
active areas. The probability that "group ROIs into a hierarchy and feed cross-level
residuals to a transformer" is unpublished is low. No "first" claim is currently defensible.

**S2 — The number is not competitive and the comparison offered is invalid.**
16.4% R@1 on a 6 956 gallery. The repo's own frozen system reports 77.2% on SHARED1000.
These are not comparable — different gallery, protocol, split; PCD never ran on SHARED1000.
But a reviewer will ask the obvious question anyway, and the honest answer is "our novel
architecture is far behind our own non-novel baseline." There is no framing that hides this.

**S3 — n=8, single seed, single run.** Every reported PCD number comes from one resumed
run at one seed with no subject-level inference. Nothing here supports a confidence interval.

**S4 — `ablation_mode="random"` is a single hardcoded shuffle** (`random.Random(42)`).
The anatomical-specificity claim requires a degree-matched null *distribution*. One random
graph is an anecdote.

**S5 — The interpretation machinery measures the wrong quantity.**
`get_prediction_error_magnitudes` returns an L2 norm. The claim it feeds is about
*information*. Magnitude is not information. Separately, attention weights are presented as
"contribution" without any intervention validating them.

**S6 — 64% of input voxels bypass the hierarchy.** `nsdgeneral_other` (10 000 of ~15 500
voxels) is encoded directly into the aggregator with no prediction applied. Before any
hierarchy claim, show what fraction of the decoding signal flows through the hierarchy at
all. The current architecture makes it plausible the answer is "very little."

## What is genuinely good

- **The splits are clean and the sealed set is untouched.** Verified against NSD's own
  stimulus table, not against the project's own bookkeeping. This is the difference between
  a project with fixable problems and one with unfixable ones.
- **The failures were found by executable tests, not assertion.** Nine regression tests now
  make F1 and F2 unable to recur silently.
- **The code content is hash-reconciled across local and pod**, so the run is reproducible
  in practice despite its manifest being wrong.

## Rebuttal requirements — what would move this off REJECT

1. Strip the predictive-coding framing, or build a real top-down arm and test direction as a
   factor. Do not argue "predictive processing, broadly."
2. Delete the per-level uncertainty claim, or give the heads a proper scoring objective and
   demonstrate calibration. Note kappa is currently near-degenerate (CV ≈ 5%), so this is
   harder than it sounds.
3. Answer S6 first. If the hierarchy carries little signal, say so and publish that — it is
   a real finding and it costs one forward pass.
4. Parameter-matched flat control, multi-seed degree-matched random graphs, subject-level
   inference with hierarchical bootstrap.
5. Populate the literature matrix from inspected primary sources before any novelty claim.

## Reviewer's closing note (dissent preserved)

The orchestrator's plan halts v4 and pivots toward a falsifiable hierarchy test with an
honest negative prior. I agree with the halt. **I disagree with any implication that the
checkpoint is worthless** — it is the only artefact that can answer S6 and H1, and
discarding it would force a re-run to reach the same question. Halt the *training*; keep the
*weights*. This dissent is recorded in D-001 and accepted into D-004.
