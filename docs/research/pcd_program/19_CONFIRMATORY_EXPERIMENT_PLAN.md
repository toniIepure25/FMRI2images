# 19 — Confirmatory Experiment Plan

**Date:** 2026-07-16 · Model: NCD (`18`) · Thesis: `17`

---

## 1. Protocol standardisation — a PRECONDITION, not a task

No comparison may run until all methods share, byte-for-byte:

| Item | Frozen value |
|---|---|
| Split | image-level, `split_by_image: true`, `exclude_shared1000: true`, seed 42 — **reuse the verified `split.json`** (T10), hash it |
| Gallery | **declared once, in writing, with size.** Report **both** image-level (6 956) and trial-level (19 236) R@1 — always, permanently (D-005) |
| Target cache | one CLIP cache, hashed |
| Preprocessing | identical; fit on train only (**test this**) |
| Checkpoint selection | **declared once.** Currently `val_loss` while reports quote R@1 (T14/F-006) — **must be fixed before any run** |
| Tuning budget | equal trials per arm, logged |
| Seeds | ≥3 for primary + strongest baseline + main ablations |

**The frozen 77.2% system and PCD_v4 may be compared only after re-evaluation under this
protocol** (T12). Until then they are not comparable to anything.

## 2. Evaluation regimes

| # | Regime | Priority | Notes |
|---|---|---|---|
| 1 | Within-subject perception decoding | P0 | sanity + capacity control |
| 2 | Shared 8-subject training | P0 | the training regime |
| 3 | **Perception → NSD-Imagery transfer** | **P0 — the thesis** | blocked on data (`14` §5) |
| 4 | Held-out ROI neural prediction | P0 | does the constraint predict out-of-sample? (kill criterion) |
| 5 | Sealed SHARED1000 | P1 | **once, at the end.** Never for selection |
| 6 | Reduced-data learning curves | P1 | fallback endpoint (`17` §5) |
| 7 | Reliability / noise robustness | P1 | fallback endpoint |
| 8 | LOSO transfer | P2 | **not a contribution** (ZEBRA/MindEye2 own it) — diagnostic only |
| 9 | Few-shot adaptation | P3 | descoped |
| 10 | External dataset | P3 | descoped |

## 3. Baselines

**Mandatory:**
- **Linear ridge** — *NSD-Imagery's own winner*; the baseline that must be beaten or conceded to
- Parameter-matched MLP
- Flat ROI transformer (= NCD with `λ_neural = 0`, no hierarchy)
- **NCD with `λ_neural = 0` + matched regularization** (O-1 — the decisive control)
- **NCD with shuffled neural target** (O-3)
- PCD_v4 checkpoint, re-evaluated under §1 protocol
- Frozen 77.2% system, re-evaluated under §1 protocol
- ≥1 strong external model, or an explicit capacity-regime scoping statement (O-5)

**Not run:** cross-subject alignment baselines (dead axis); recurrence-ablated arms (no
recurrence in v1); precision arms (no precision in v1).

## 4. Neuroscience analyses (only if the main effect survives)

Held-out neural predictivity per ROI, **noise-ceiling normalised** (Spearman–Brown split-half
on NSD's 3 repeats); conditional variance partitioning for `L_neural`'s contribution;
per-ROI category contrasts (faces/scenes/bodies/objects) — **possible only because ROIs are
unpooled** (`18` §3); cross-validated RDMs with noise ceilings; model-level lesions.

**Banned:** attention weights as explanation (rule 7); residual magnitude as information
(rule 8); MI estimators until validated on synthetic ground truth + permutation nulls.

## 5. Experiment matrix

| ID | Arm | Subjects | Seeds | Gate |
|---|---|---|---|---|
| E-00 | Protocol standardisation + leakage tests | — | — | **blocks all** |
| E-01 | H1 level-3 bypass diagnostic (PCD ep98 ckpt) | 8 | 1 | cheap, informative, **not a contribution** |
| E-02 | NCD tiny-overfit + gradient tests | 1 | 1 | Gate 3 |
| E-03 | NCD 1-subject pilot, λ ∈ {0, 0.5} | 1 | 2 | Gate 4 |
| E-04 | NCD 4-subject pilot, λ ∈ {0, 0.5} + matched-reg | 4 | 2 | **promotion gate** |
| E-05 | Confirmatory 8-subject, λ ∈ {0, 0.5}, +O-1, +O-3 | 8 | 3 | Gate 5 |
| E-06 | Imagery transfer eval | 8 | 3 | **the endpoint** |
| E-07 | Ablations (`18` §7) | 8 | 3 | Gate 5 |
| E-08 | Sealed SHARED1000 | 8 | 3 | **once, last** |

## 6. Promotion gates

- **E-02 → E-03:** all gradient + leakage tests pass; tiny-overfit reaches ~100% train R@1.
- **E-03 → E-04:** NCD `λ=0` within 3 pp of PCD_v4 under the standard protocol (i.e. the
  smaller model has not broken decoding), **and** `L_neural` beats the per-ROI mean baseline
  out-of-sample.
- **E-04 → E-05:** `λ>0` beats matched-reg on the §18 §6 estimand, 4 subjects, subject-level CI
  excluding 0. **Otherwise stop and report the negative** — do not escalate to 8 subjects
  hoping for significance.
