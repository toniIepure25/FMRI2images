# NeuroBridge-OT: Claim-Evidence Matrix

## Purpose

This document specifies the experimental evidence required to support each
publication claim. Claims require either 8-subject primary evidence or
4-subject legacy evidence (for V66a comparison only).

**Principle:** Publication-level claims MUST use 8-subject evidence unless
explicitly marked as preliminary/legacy.

---

## Claims and Required Evidence

### Claim 1: Learned ROI tokens outperform fixed ROI summaries in all-8 LOSO

| Field | Value |
|---|---|
| **Claim type** | Architecture ablation |
| **Evidence scope** | 8-subject primary |
| **Protocol** | Protocol D (8-fold LOSO strict zero-shot) |
| **Comparison** | `neurobridge_ot_8subj_loso` (learned) vs `neurobridge_ot_roi_summary_baseline` (fixed) |
| **Metric** | Mean R@1 across 8 LOSO folds |
| **Statistical test** | Paired t-test across 8 folds (p < 0.05), Cohen's d |
| **Status** | PENDING EXPERIMENT |

---

### Claim 2: OT alignment improves all-8 LOSO generalization

| Field | Value |
|---|---|
| **Claim type** | Architecture ablation |
| **Evidence scope** | 8-subject primary |
| **Protocol** | Protocol D |
| **Comparison** | Learned tokenizer + OT vs Learned tokenizer alone |
| **Metric** | Mean R@1 across 8 folds |
| **Statistical test** | Paired t-test, Cohen's d |
| **Status** | PENDING EXPERIMENT |

---

### Claim 3: Anatomical OT prior improves interpretability and/or performance

| Field | Value |
|---|---|
| **Claim type** | Architecture ablation |
| **Evidence scope** | 8-subject primary |
| **Protocol** | Protocol D |
| **Comparison** | OT + anatomical prior vs OT without prior |
| **Metric** | Mean R@1 + transport matrix interpretability analysis |
| **Statistical test** | Paired t-test for performance; qualitative for interpretability |
| **Status** | PENDING EXPERIMENT |

---

### Claim 4: 8-subject multi-subject training improves robustness over 4-subject training

| Field | Value |
|---|---|
| **Claim type** | Data scaling |
| **Evidence scope** | 8-subject primary + 4-subject legacy comparison |
| **Protocol** | Protocol C (8-subj) vs Legacy 4-subj config |
| **Comparison** | `neurobridge_ot_8subj_full` vs `neurobridge_ot_4subj_legacy_v66a_comparison` |
| **Metric** | Per-subject R@1 on SHARED1000, mean and std |
| **Note** | 4-subject comparison uses same architecture, only differs in training population |
| **Status** | PENDING EXPERIMENT |

---

### Claim 5: Few-shot adaptation improves low-data target-subject decoding across all 8 subjects

| Field | Value |
|---|---|
| **Claim type** | Transfer learning |
| **Evidence scope** | 8-subject primary (few-shot all-8) |
| **Protocol** | Protocol F |
| **Comparison** | Pretrained+adapted vs scratch, at each N |
| **Metric** | R@1 at N=10, 25, 50, 100, 250, 500, 1000 across all 8 targets |
| **Statistical test** | Per-N paired t-test across 8 subjects |
| **Status** | PENDING EXPERIMENT |

---

### Claim 6: Teacher distillation improves performance only where teacher artifacts exist

| Field | Value |
|---|---|
| **Claim type** | Knowledge transfer (teacher-coverage-limited) |
| **Evidence scope** | Subset — teacher-available subjects only |
| **Protocol** | Protocol C with w_teacher > 0, restricted to subj01/02/05/07 |
| **Comparison** | With teacher vs without teacher on 4 subjects with teacher artifacts |
| **Metric** | Per-subject R@1 on SHARED1000 |
| **Important note** | This claim is inherently teacher-coverage-limited. Cannot generalize to subj03/04/06/08 |
| **Status** | PENDING EXPERIMENT |

---

### Claim 7: vMF kappa predicts reliability across subjects

| Field | Value |
|---|---|
| **Claim type** | Calibration / uncertainty |
| **Evidence scope** | 8-subject primary |
| **Protocol** | Any protocol with vMF head enabled |
| **Analysis** | Correlation between kappa and retrieval accuracy per trial across all 8 subjects |
| **Metric** | Spearman correlation (kappa, correctness), calibration curves |
| **Status** | PENDING EXPERIMENT |

---

### Claim 8: NeuroBridge-OT improves over legacy Z1 all-8 baseline

| Field | Value |
|---|---|
| **Claim type** | Architecture comparison |
| **Evidence scope** | 8-subject primary |
| **Protocol** | Protocol B (subject-specific) or Protocol C (multi-subject) |
| **Comparison** | NeuroBridge-OT vs fixed ROI summary (Z1-equivalent) |
| **Metric** | R@1 on SHARED1000 across all 8 subjects |
| **Status** | PENDING EXPERIMENT |

---

## Evidence Classification Guide

| Scope | Subjects | When to use |
|---|---|---|
| 8-subject primary | All 8 | Publication claims, primary results |
| 4-subject legacy | subj01/02/05/07 | V66a comparison ONLY |
| Single-subject | Any 1 | Debugging, per-subject deep analysis |
| LOSO all-8 | 8 folds | Generalization claims |
| Few-shot all-8 | 8 targets | Transfer claims |
| Teacher-coverage-limited | subj01/02/05/07 | Teacher distillation claims |

Claims based solely on 4-subject evidence must be marked as **preliminary/legacy**
and should be superseded by 8-subject evidence before publication.
