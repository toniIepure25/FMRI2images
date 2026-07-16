# 02 — Research Charter

**Status:** DRAFT — thesis selection is a Gate 2 decision. Nothing here is locked.
**Last updated:** 2026-07-16

---

## Central research question (candidate)

> Does grouping visual-cortex ROIs into an anatomical hierarchy and encoding **cross-level
> residuals** — the component of each level not predictable from the level below — carry
> decoding-relevant information beyond a parameter-matched flat control?

This replaces the seed question ("does PCD validate predictive coding?"), which Gate 0
established is unanswerable as posed: the architecture does not implement predictive coding
(T6), and decoding accuracy could not validate a biological theory even if it did (rule 10/11).

## Candidate theses, assessed against Gate 0 evidence

### Candidate A — Mechanistic predictive-coding thesis
*A recurrent, top-down, precision-weighted cortical model is a superior and neurally
predictive inductive bias.*

- **Blocked as stated.** The implemented model predicts low→high (T6); it is not the
  claimed mechanism. A true top-down arm does not yet exist.
- Requires a **held-out neural-prediction objective** to make residuals identifiable
  (Risk 2). Without it, residuals are arbitrary task features whatever the direction.
- Feasible only after Gate 3. High compute, high risk.
- **Verdict:** not selectable now. Retain the *direction* question as a tested factor.

### Candidate B — Cross-subject functional-inference thesis
*Shared hierarchical prediction dynamics improve transfer to unseen subjects.*

- Untested. But per-subject ROI projections are reportedly ~96M of 167M params (57%,
  UNVERIFIED) — a large unrestricted per-subject capacity is a **memorization shortcut**
  (Risk 5), and the observed 82 pp overfit gap (T4) is consistent with exactly that.
- LOSO transfer has never been run. The 8-subject data supports it.
- **Verdict:** the most plausible *surviving* direction, but currently pure hypothesis.
  Note the diagnosis and the thesis are in tension: if per-subject projections memorize,
  cross-subject transfer is where that shows up — which makes B a real test, not a pivot.

### Candidate C — Uncertainty-aware cortical decoding thesis
*Hierarchical residuals enable calibrated selective decoding and conformal retrieval.*

- **Severely damaged by Gate 0.** Per-level kappa heads are untrained (T8) and global
  kappa is near-degenerate (mean 9.27, std 0.47, CV ≈ 5% — T9). There is currently no
  informative uncertainty signal to calibrate.
- Salvageable only by (i) giving kappa a proper scoring objective and (ii) removing/retuning
  `kappa_reg`, which is actively suppressing it — a documented failure mode
  (`EXPERIMENT_CONTEXT.md` §13).
- **Verdict:** not selectable on current evidence. The repo already has a separate
  conformal-prediction line (`docs/paper/calibrated_uncertainty_paper.md`) — PCD adds nothing to it today.

## Recommended primary thesis (provisional, pending Gate 2)

**A rigorous, parameter-matched, falsifiable test of whether anatomical hierarchy + cross-level
residual structure buys anything for fMRI visual decoding — reported honestly whichever way it falls.**

Rationale: it is the only question the existing code, data, and checkpoint can actually
answer; the ablation machinery already exists (`ablation_mode`: `full` / `no_errors` /
`reversed` / `random`); and it is cheap. The honest prior is **negative** — 64% of voxels
bypass the hierarchy entirely (T7), which predicts the hierarchy contributes little.
A well-powered negative result here is publishable and useful (rule 20; Outcome B).

**This is not a decision.** It is the leading option entering Gate 2.

## Non-goals

- Beating MindEye2 or the frozen 77.2% system. PCD is not competitive and this program is
  not a SOTA chase.
- Reconstruction. Retrieval is the endpoint until a thesis demands otherwise.
- Any claim about the human brain's implementation of predictive coding.
- Rescuing the v1→v4 line as a narrative of progress. It produced no measurable gain (T4).

## Claim boundaries (hard)

| Claim | Default status |
|---|---|
| "PCD validates predictive coding in the brain" | **FORBIDDEN** — rules 10, 11; contradicted by T6 |
| "Prediction errors carry more information than raw activations" | **FORBIDDEN** pending conditional-information analysis; residual magnitude ≠ information (rule 8) |
| "Per-level uncertainty decomposition" | **FORBIDDEN** — heads untrained (T8) |
| "Universal cross-subject processing strategy" | **FORBIDDEN** — mean-attention correlation over 8 subjects is not evidence (Risk 8) |
| "Anatomical hierarchy outperforms random grouping" | Permitted **only** vs degree-matched random graphs, multiple seeds, subject-level inference |
| "Results are consistent with hierarchical inference" | Permitted **only if** supported and heavily qualified |

## Success criteria

- **Outcome A (positive):** hierarchy beats parameter-matched flat + degree-matched random
  controls, on subject-level inference, multiple seeds, sealed SHARED1000, with clean provenance.
- **Outcome B (negative/mixed):** the above is rigorously tested and fails → publish the
  negative result plus the methodological findings (the gradient-flow bug class and the
  direction mislabel are both genuinely instructive).
- **Outcome C (termination):** documented inability to support any defensible claim.

**Gate 0 evidence currently favours Outcome B.** That is an acceptable, honest destination.

## Venue logic

Deferred to Gate 2, and determined by evidence, not ambition. Current read: a decoding-SOTA
ML venue is **not** the fit (PCD is not competitive). If the hierarchy question is answered
cleanly — either way — CCN or a methods/neuroimaging venue is the honest target.
