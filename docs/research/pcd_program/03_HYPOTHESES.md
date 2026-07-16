# 03 — Hypotheses

**Status:** DRAFT. Not preregistered until Gate 2 closes. Nothing here may be tested on
SHARED1000 before the analysis plan is frozen.
**Unit of inference is the SUBJECT (n=8), never the trial** (rule 9).

---

## H1 — The level-3 bypass dominates the fused representation

**Priority:** FIRST. Cheapest, and it gates everything else.

**Rationale:** `nsdgeneral_other` is allocated 10 000 of ~15 500 voxels (≈64%) and is
encoded **directly** into the aggregator with no prediction applied (T7). If the aggregator
routes most of its weight there, then the three "hierarchical" levels are decorative and
every downstream hierarchy claim is moot before it is tested.

- **H0:** aggregator attention weight on level 3 ≈ 0.25 (uniform over 4 levels).
- **H1:** level-3 weight is substantially higher, and ablating level 3 costs far more R@1
  than ablating levels 0–2 combined.
- **Estimand:** mean aggregator weight per level, and ΔR@1 under per-level lesion.
- **Unit:** subject (8 subjects, paired).
- **Endpoint:** val R@1 (both gallery definitions — D-005).
- **Analysis:** subject-level paired contrasts + hierarchical bootstrap over subjects and stimuli.
- **Cost:** one forward pass over val with the epoch-101 checkpoint. No training.
- **Kill criterion:** if level 3 carries >60% of aggregate weight **and** lesioning levels
  0–2 costs <2 pp R@1, the hierarchy is not doing meaningful work → the primary thesis
  (`02_RESEARCH_CHARTER.md`) is falsified at Gate 1 and the program pivots to Outcome B/C.
- **Permitted claim:** "in this model, most decoding signal is carried by the non-hierarchical
  residual ROI group."
- **Forbidden overclaim:** anything about the brain. This is a statement about *our decoder*.

**Caveat that must not be skipped:** attention weights are **not** explanation (rule 7).
The weights are the *hypothesis generator*; the **lesion** is the evidence.

---

## H2 — Anatomical hierarchy beats degree-matched random grouping

**Rationale:** `ablation_mode` already implements `full` / `no_errors` / `reversed` /
`random`. This is the project's actual falsifiable core.

- **H0:** anatomical grouping = random grouping (within CI), i.e. any benefit comes from
  having *a* grouped structure, not from *the anatomy*.
- **H1:** anatomical > random, consistently across subjects.
- **Unit:** subject. **Multiplicity family:** the 4 ablation arms, FDR-corrected.
- **Required controls:** `random` must be run with **multiple degree-matched seeds** — the
  current implementation hardcodes `random.Random(42)` (`predictive_cortical_decoder.py:402`),
  a **single fixed shuffle**. One random graph is an anecdote, not a null distribution.
  **This is an implementation defect to fix before H2 can be tested.**
- **Kill criterion:** if anatomical ≤ random across seeds → drop all anatomical-specificity
  claims (per the specification's kill rules).
- **Permitted claim:** "anatomical ROI grouping outperforms degree-matched random groupings."
- **Forbidden:** "the model recovers the brain's visual hierarchy."

---

## H3 — Cross-level residuals add conditional information beyond raw activity

**Rationale:** the `no_errors` arm tests this directly. But residual *magnitude* is not
information (rule 8), and `get_prediction_error_magnitudes` computes exactly a magnitude —
so the existing analysis cannot answer this.

- **Analysis (replaces magnitude):** cross-validated probes compared on matched folds —
  raw activity · predicted activity · residual · raw+residual · lower-level state ·
  lower-level state+residual — with **conditional variance partitioning**.
- **MI estimators are not permitted** until validated on simulations with known ground truth
  plus permutation nulls.
- **Identifiability caveat (Risk 2 — this is the crux):** residuals are optimized *only* for
  CLIP retrieval. They are therefore **arbitrary task-useful features** until shown to
  predict **held-out neural activity**. Absent that, H3 can only ever be a statement about
  our loss, not about the cortex.
- **Kill criterion:** no conditional information beyond raw → drop "richer prediction-error
  information" permanently.

---

## H4 — Direction: does high→low beat low→high?

**Rationale:** the implemented direction is low→high (T6). Canonical predictive coding is
high→low. Rather than assert either, **build both arms and test**.

- **Blocked on:** the top-down arm does not exist (D-003 defers it to Gate 3).
- **Critical constraint:** must be parameter- and FLOP-matched, same targets, same optimizer,
  same updates, same tuning budget.
- **Permitted claim if high→low wins:** "a top-down residual formulation is a better
  inductive bias for decoding." **Still not** "the brain does predictive coding."
- **Note:** fMRI at NSD's temporal resolution cannot resolve fast recurrent dynamics. A
  recurrent architecture result tests a **computational prior**, not measured temporal
  recurrence. This limitation is mandatory in any writeup.

---

## H5 — Per-subject projections memorize subject-specific shortcuts

**Rationale:** per-subject ROI projections are reportedly ~96M of 167M params (57%,
UNVERIFIED — must be re-derived). The 82 pp overfit gap that no amount of regularization
closed (F-001) is consistent with memorization located there rather than in the backbone.

- **Test:** LOSO transfer + few-shot adaptation curve to an unseen subject; compare
  unrestricted per-subject projections vs constrained shared adapters.
- **Endpoint:** zero-shot R@1, area under the adaptation curve, per-subject heterogeneity.
- **This is the strongest surviving positive direction** (Candidate B) and it explains F-001.

---

## Preregistration status

**None of H1–H5 is preregistered.** H1 is exploratory-but-cheap and runs first because it
can falsify the program at Gate 1 for the price of one forward pass. H2–H5 require the
frozen analysis plan (`Agent D`, Gate 2) before any confirmatory run.
