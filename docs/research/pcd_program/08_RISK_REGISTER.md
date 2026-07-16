# 08 — Risk Register

**Last updated:** 2026-07-16 · Severity: 🔴 critical · 🟠 high · 🟡 medium · 🟢 controlled

---

## 🟢 R-01 — Data leakage / test contamination — **CONTROLLED**

train ∩ val = 0; train ∩ SHARED1000 = 0; val ∩ SHARED1000 = 0; `split_by_image: true`
(T10, verified against NSD's own `nsd_stim_info_merged.csv`). Repeated presentations of an
image cannot cross the split.

**Residual exposure (not yet verified):** scaler/PCA/reliability fit on train only; the
16 384-entry contrastive queue excluding val; hyperparameters not selected on val.
`preprocessing.enabled: false` in v4 shrinks but does not close this. **Automate as tests.**

## 🔴 R-02 — Invalid uncertainty — **MATERIALIZED**

Per-level kappa heads receive zero gradient (T8/F-002); global kappa near-degenerate
(CV ≈ 5%, T9/F-003). Any per-level or calibration claim is FORBIDDEN. `kappa_reg=0.05` is
actively suppressing kappa — a failure mode this project already documented and reintroduced.

## 🔴 R-03 — Unsupported biological interpretation — **MATERIALIZED**

Code claims Rao–Ballard predictive coding and "validates predictive coding theory"; the
implementation predicts low→high (T6). Two defects: wrong direction, and decoding success
cannot validate a biological theory regardless (rules 10/11). Mitigation: D-002.

## 🔴 R-04 — Circular / invalid neuroscience analysis — **MATERIALIZED**

`pcd_neuroscience_analysis.py` reads untrained `level_kappas` (F-002) and computes residual
*magnitude* as if it were information (rule 8). It would emit stable, plausible, publishable-
looking figures that mean nothing. **Do not run it for evidence until rewritten.**
Deeper issue: representations optimized for CLIP may rediscover the target supervision rather
than brain organization — held-out neural prediction + noise ceilings are mandatory.

## 🟠 R-05 — Overfitting / capacity misallocation — **MATERIALIZED**

82 pp train/val gap that four rounds of regularization did not close (F-001). Suspected
location: per-subject ROI projections (~57% of params, UNVERIFIED). Ties directly to R-06.

## 🟠 R-06 — Subject-specific memorization

Large unrestricted per-subject projections can memorize subject shortcuts (Risk 5 in spec).
Untested — LOSO has never been run (H5). This is both the leading explanation for R-05 and
the leading surviving thesis (Candidate B).

## 🟠 R-07 — Shortcut learning via the level-3 bypass

`nsdgeneral_other` = ~64% of voxels, bypasses the hierarchy entirely into the aggregator
(T7). If it dominates, the hierarchy is decorative and the whole program is moot. **H1 tests
this first, for one forward pass.**

## 🟠 R-08 — Silent checkpoint corruption / architecture mismatch

`strict=False` was introduced for PCD checkpoint loading (`d99cf03`) to "handle buffer
mismatches". It equally hides a **silently half-initialised model**. A checkpoint that loads
without error under `strict=False` is *not* evidence it loaded correctly. Never dry-loaded.
Mitigation: D-004 — explicit key-set diff, elevated to a required Gate 1 test.

## 🟠 R-09 — Metric definition drift — **MATERIALIZED**

`val_r@1` = 16.4% (6 956 gallery) vs `val_r@1_trial` = 3.3% (19 236 trials) — a 5× fork
(T11). All prior reporting used the larger silently. Comparing either to the frozen system's
77.2% SHARED1000 is invalid (T12). Mitigation: D-005 — report both, always, with gallery size.

## 🟠 R-10 — Novelty overlap — **UNASSESSED**

`10_LITERATURE_MATRIX.csv` is **empty**. No search has been run. Every novelty claim,
including any "first", is currently unsupported. **Hard Gate 2 blocker.** Adjacent work
(MindEye2, Brain-Diffuser, ROI-aware transformers, predictive-coding nets, fMRI GNNs) is
dense — prior overlap is likely, not hypothetical.

## 🟠 R-11 — Statistical underpowering

n = 8 subjects. Asymptotic p-values are not definitive. Requires hierarchical bootstrap over
subjects *and* stimuli, subject-level paired contrasts, FDR correction, equivalence testing
for negative results. **Analysis pipelines must be validated on synthetic data with known
ground truth before touching real results.**

Specific defect: `ablation_mode="random"` hardcodes `random.Random(42)` — a **single fixed
shuffle**. One random graph is an anecdote; H2 needs a seed distribution.

## 🟡 R-12 — Provenance loss — **PARTIALLY MATERIALIZED**

Manifest records a commit whose code did not run (T2/F-005); recoverable only because
content hashes coincidentally matched. `PCD_v3/v4` configs untracked (T1). All checkpoints
and metrics single-homed on pod NFS with no hashes (T3). Mitigation: D-004 + commit configs.

## 🟡 R-13 — Compute exhaustion

Low right now: H100 idle, no runs active. D-001 prevented ~30 GPU-hours on a plateaued run
supporting no surviving claim. Keep value-of-information ordering: H1 costs one forward pass
and can falsify the program.

## 🔴 R-21 — Generic-regularization confound (objection O-1) — **MITIGATED, NOT ELIMINATED**

A masked-ROI objective is an auxiliary task, and auxiliary tasks regularize. A positive ARM-B
that is really "we added a regularizer" would be uninterpretable and rejected.

**Mitigation shipped this phase:** ARM-F (tuned regularization under the same budget), ARM-D
(any fixed target), ARM-E (generic non-anatomical reconstruction), exact parameter parity
across all eight arms, and config-diff validation. **Residual exposure:** ARM-F's tuning
budget must be genuinely equal — an under-tuned ARM-F is a straw man and a reviewer will say
so. Log every ARM-F trial in the manifest.

## 🟠 R-22 — Zero-shot invariant leak — **the contribution-forfeiting risk**

If any imagery sample touches training, model selection, λ selection, checkpoint selection,
early stopping, or representation design, our setting collapses into Spera et al. (2026) —
who fit on imagery, with alignment and augmentation, and do it well. **The entire distinction
is the seal.** Test-enforced (`test_no_arm_touches_imagery_or_the_sealed_set`), but the test
only checks configs; a leak via an analysis script would pass it.

## 🟠 R-15 — NSD-Synthetic semantic mismatch — **structural**

232 of 284 stimuli (82%) — spiral gratings, chromatic noise, pink noise, isolated words —
have essentially **no CLIP-semantic content**. A CLIP-retrieval decoder is near-chance on them
**by construction**. Reporting that as OOD failure would be misleading. Only 52 stimuli are
scene-derived. Mitigation: confirmatory endpoint restricted to the 52; the other 232 are
exploratory, plus neural predictivity (valid on all 284). See `26` §4–5.

## 🟠 R-16 — Slicing NSD-Imagery until something is significant

n = 4, 18 stimuli, 3 conditions. Stimulus type is a **preregistered fixed factor**, never an
exploratory slice. With n = 4 the exact sign-flip permutation distribution has 16 points, so
**p ≥ 1/16 = 0.0625 is the floor** — state the ceiling, do not report an asymptotic p that
pretends otherwise (`20` §8).

## 🟡 R-18 — Confirmatory synthetic set is small (52 stimuli)

Power comes from the **slope across OOD levels**, not from n per cell. Model it as a slope.

## 🟡 R-19 — NSD-Synthetic is one session (low SNR vs NSD-core's 40)

Never compare raw accuracies across datasets; normalise by noise ceiling.

## 🟡 R-20 — CLIP cache built for colour natural scenes

NSD-Synthetic stimuli are grayscale/synthetic. **Verify the target-embedding pipeline handles
them before any claim.** Untested.

## 🟡 R-23 — Pseudo-ROI / permutation single-seed anecdote

ARM-C and ARM-G each depend on one random draw. A single fixed shuffle is an anecdote — this
is PCD's `random.Random(42)` defect (`03` H2). Seed-sensitivity is tested; **multi-seed use is
mandatory in the protocol** and not yet enforced by a runner.

## 🟡 R-14 — Stale context steering decisions — **MATERIALIZED**

`docs/HANDOFF_CONTEXT.md` was ~90 epochs and every headline number out of date (T4), and its
instruction ("resume from epoch 23, still improving") would have burned ~30 GPU-hours on a
false premise. **Any handoff must be re-verified against the pod before it is acted on.**
