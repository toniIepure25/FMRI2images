# 07 — Failure and Negative Result Ledger

Append-only. Nothing here is ever deleted. Negative results are results.

---

## F-001 — The v1→v4 regularization program produced no improvement

**Status:** VERIFIED · **Date:** 2026-07-16

| Version | Change | Outcome |
|---|---|---|
| v1 | d_model=768, 12 heads, 2 layers/level, dropout 0.1 | 167M params; ~17.97% val R@1 @ ep29; train 98.73% → **82 pp gap** *(UNVERIFIED — from handoff)* |
| v2 | d_model=512, 1 layer, bottleneck=128, dropout 0.4, noise 0.5, R-Drop from ep0 | 46.5M params; **0% R@1 after 30 epochs — could not learn.** Too many regularizers at maximum simultaneously |
| v3 | restored d_model=768, kept bottleneck=128 | **Hung at epoch 2.** Bottleneck in `ROIProjection` → 272 sequential CUDA launches in a Python loop |
| v4 | v1 architecture, regularization-only tuning | **16.36% val R@1 @ ep101; train 98.83% → 82 pp gap.** VERIFIED from `training_log.csv` |

**Net result of four versions and ~4× the compute: no measurable gain over v1.**

**The key negative finding:** v4's justification — "gap 25 pp vs v1's 82 pp, the
regularization IS working" — was measured at **epoch 20, before convergence**. By epoch 110
the gap was **82 pp, identical to v1**. Dropout 0.25, weight-decay 0.08, noise 0.25,
voxel-dropout 0.15, label-smoothing 0.1 and MixCo 0.2 **delayed** overfitting; they did not
reduce it.

**Lesson (generalizable, worth writing up):** a train/val gap read before convergence is not
a measure of overfitting — it is a measure of training progress. The v2→v3→v4 sequence was
steered for ~2 days by a statistic sampled at the wrong time.

**Why this matters for the thesis:** if regularization cannot close an 82 pp gap, the
capacity is likely in the wrong place. Per-subject ROI projections are reportedly ~57% of
params — see Risk 5 (subject memorization) and Candidate B.

---

## F-002 — Per-level kappa heads never trained (silent, entire project)

**Status:** VERIFIED · **Date:** 2026-07-16

All four `PerLevelKappaHeads` linear heads receive `grad is None` under every loss the
training loop can build, because `PCDModel.forward` returns only `(mu, kappa)` and
`level_kappas` is stashed in `_last_pcd_extras`, which **no loss reads** (only a smoke-test
script does). Proven by `tests/test_pcd_gradient_flow.py`.

**Invalidated by this bug:**
- Every "per-level uncertainty decomposition" claim.
- `scripts/analysis/pcd_neuroscience_analysis.py` wherever it reads `level_kappas`
  (`:263–274`, `:303–304`, `level_kappas_by_category`) — it analyses a **random linear
  projection of trained features**.

**Why it survived:** `level_kappas.grad_fn is not None` — the tensor *is* in the autograd
graph and looks connected. And a random projection of trained features is **not noise**: it
produces stable, reproducible, category-varying numbers. The analysis would have yielded
publishable-looking figures that mean nothing. This is the most dangerous class of bug in
the codebase and the reason gradient tests are now mandatory.

**Confirmed on the trained weights (2026-07-16, T13):** across 14 epochs of real training
(checkpoint epoch 98 → 112), `level_kappa_heads` moved by **4.3e-07** — 3 of 8 tensors
bit-identical — while `prediction_heads` moved 1.3e-02, `level_encoders` 3.3e-02 and
`vmf_decoder` 2.0e-01. A factor of 3×10⁴–5×10⁵. The 1e-7 residue is EMA float
accumulation, not gradient descent (AdamW skips `grad is None` parameters outright).
This is no longer an inference from a synthetic probe; it is measured on the artifact.

---

## F-006 — "Best" checkpoint is selected on val_loss, not the reported metric

**Status:** VERIFIED · **Date:** 2026-07-16

`checkpoint_best.pt` is saved at **epoch 98**. The best `val_r@1` in `training_log.csv` is at
**epoch 101**. `best_metric` is `None`; `val_loss` is a stored top-level key. So checkpoint
selection runs on **val_loss** while every report quotes **val R@1**.

Consequence: the checkpoint anyone loads and evaluates is **not** the best-R@1 checkpoint,
and "PCD_v4 achieves 16.36%" does not describe the artifact that would be shipped. Combined
with the 16.4%/3.3% metric fork (F-007/T11), the project currently has **three** different
notions of "the result": selection metric, image-level R@1, and trial-level R@1. All must be
declared explicitly before any reporting (D-005).

---

## F-007 — A 5× metric fork was reported silently

**Status:** VERIFIED · **Date:** 2026-07-16

`training_log.csv` carries `val_r@1` = **0.1636** (6 956-image gallery) and `val_r@1_trial`
= **0.0330** (19 236 trials). All prior PCD reporting used the larger number without
disclosing the aggregation. `average_repetitions: false` for training, yet evaluation
aggregates trials → images.

Separately, comparing either figure to the frozen system's 77.2% SHARED1000 result is
invalid: different gallery (1 000 vs 6 956), different protocol, different split, and PCD has
never been run on SHARED1000 at all.

---

## F-003 — Global kappa is near-degenerate

**Status:** VERIFIED · **Date:** 2026-07-16

Epoch 112: `kappa_mean=9.27, kappa_std=0.47, kappa_min=8.04, kappa_max=10.46` (CV ≈ 5%).
Concentration is essentially constant across stimuli → carries almost no per-sample
information. `kappa_reg.lambda_kappa: 0.05` (5× v1) applies steady downward pressure.

This reproduces the kappa-collapse failure mode already documented in
`docs/EXPERIMENT_CONTEXT.md` §13 and `lean/FMRIDecoding/Certificates.lean:36`
("`kappa_reg` applies steady downward pressure, trapping κ at 3–5"). **The project
diagnosed this before and reintroduced it.** Any uncertainty thesis on this checkpoint is
unsupported (Candidate C).

---

## F-004 — PCD_v4 run terminated at epoch 113, cause unknown

**Status:** UNKNOWN cause, VERIFIED occurrence · **Date:** 2026-07-16

`pcd_v4_resumed2.log` ends mid-epoch 113 at 09:38 with **no traceback, no OOM message, no
early-stop line**. Process absent, GPU idle. External kill / node pressure / OOM-killer all
consistent. Recorded as UNKNOWN — **not** as "completed". Do not cite epoch 113 as a
finishing point.

---

## F-005 — Run manifest records a commit whose code did not run

**Status:** VERIFIED · **Date:** 2026-07-16

`manifest.json` records `git_commit: b96affa`. The pod's tree is 4 commits behind local with
3 hand-copied modified files whose sha256 digests **match local HEAD `c65e834` exactly**.
So the executed code corresponds to `c65e834`, not `b96affa`.

Recoverable only because content hashes happened to match. Had the pod's copies drifted, the
run would have been **unattributable**. The manifest writer trusts `git rev-parse` on a dirty
tree — it must hash source files instead.
