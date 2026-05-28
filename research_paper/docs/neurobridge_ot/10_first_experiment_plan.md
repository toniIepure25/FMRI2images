# NeuroBridge-OT: First Experiment Plan (8-Subject Primary)

## Overview

This document defines the experiment execution sequence for NeuroBridge-OT
using all 8 NSD subjects. Experiments are ordered by dependency and cost,
starting with zero-cost audits and progressing to full training.

**Critical rule:** Do not run expensive training until all 8 subjects pass audit.

---

## Experiment 0 — All-8 Audit and Smoke Test

**Goal:** Verify all 8 subjects are usable. No GPU training required.

**Steps:**
1. Build index for all 8 subjects: `make index SUBJECT=subjXX`
2. Pre-extract fMRI features: `make preextract SUBJECT=subjXX`
3. Verify CLIP cache covers all subjects
4. Run full data audit

**Command:**
```bash
# On pod, after index/preextract:
python3 scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --check-rois --check-teachers --check-shared1000
```

**Smoke test (tiny dims, verify code paths):**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_smoke_test.yaml \
    --subjects subj01 --limit-batches 5 --seed 42
```

**Success criteria:**
- All 8 subjects have fMRI features, index, and ROI indices
- SHARED1000 count documented per subject
- Smoke test completes without error

**Estimated time:** ~30 min (index + preextract for 4 new subjects: ~4 hours)

---

## Experiment 1 — 8-Subject Fixed ROI Summary Baseline

**Goal:** Establish Z1-equivalent fixed ROI summary performance across all 8.

**Config:** `neurobridge_ot_roi_summary_baseline.yaml` (update to 8 subjects)

**Command:**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_roi_summary_baseline.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --seed 42
```

**Estimated time:** ~2-4 hours on H100

**Success criteria:** Per-subject SHARED1000 R@1 reported for all 8 subjects.

---

## Experiment 2 — 8-Subject Learned ROI Tokenizer (No OT)

**Goal:** Test learned tokenization versus fixed ROI summary.

**Config:** Modify `neurobridge_ot_no_ot.yaml` to use all 8 subjects.

**Command:**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_no_ot.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --seed 42
```

**Estimated time:** ~3-5 hours on H100

**Key comparison:** Experiment 2 R@1 vs Experiment 1 R@1 → measures tokenizer benefit.

---

## Experiment 3 — 8-Subject Learned ROI Tokenizer + OT (No Prior)

**Goal:** Test OT alignment benefit.

**Config:** `neurobridge_ot_8subj_full.yaml` with `use_anatomical_prior: false`

**Command:**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_full.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --seed 42
```

**Estimated time:** ~4-6 hours on H100

**Key comparison:** Experiment 3 R@1 vs Experiment 2 R@1 → measures OT benefit.

---

## Experiment 4 — 8-Subject OT + Anatomical Prior

**Goal:** Test whether anatomical prior improves alignment.

**Config:** Same as Experiment 3 but with `use_anatomical_prior: true`

**Command:**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_loso.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --seed 42
```

**Estimated time:** ~4-6 hours

**Key comparison:** Experiment 4 vs Experiment 3 → anatomical prior delta.

---

## Experiment 5 — 8-Subject Few-Shot Adaptation

**Goal:** Test data-efficiency transfer.

**Prerequisites:** Pretrained checkpoint from Experiment 3 or 4.

**Command:**
```bash
python3 scripts/neurobridge_ot/run_fewshot_adaptation.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_fewshot.yaml \
    --all-target-subjects \
    --few-shot-sizes 10 25 50 100 250 500 1000 \
    --seed 42
```

**Estimated time:** ~8-12 hours (8 targets × 7 sizes × 2 modes)

**Key output:** Data-efficiency curves for all 8 subjects.

---

## Deferred Experiments

### Teacher Distillation (after audit)

Teacher artifacts exist only for subj01 (V61a/V62a) and subj01/02/05/07 (V66a).
Teacher distillation experiments should be:
- Run separately on the 4 teacher-available subjects
- Clearly marked as "teacher-coverage-limited"
- NOT included in primary 8-subject claims

### Full LOSO (after Experiments 1-4)

8-fold LOSO requires 8 separate training runs (each on 7 subjects).
- Estimated time: ~30-40 hours total
- Only worthwhile after confirming architecture works in multi-subject setting

---

## Compute Budget Summary

| Experiment | GPU Hours (est.) | Priority |
|---|---|---|
| 0: Audit + smoke | <0.5 | IMMEDIATE |
| 1: Fixed ROI baseline | 2-4 | HIGH |
| 2: Learned tokenizer | 3-5 | HIGH |
| 3: + OT | 4-6 | HIGH |
| 4: + Anatomical prior | 4-6 | MEDIUM |
| 5: Few-shot | 8-12 | MEDIUM |
| Full LOSO (8 folds) | 30-40 | AFTER 1-4 |
| Teacher experiments | 4-8 | LOW (coverage-limited) |

**Total estimated for Experiments 0-5:** ~20-35 GPU hours on H100.

---

## 8-Subject Compute Profiling Results

*(To be filled after running the smoke profiling command)*

**Profiling command:**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_smoke_test.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --limit-batches 100 --seed 42
```

| Metric | Value |
|---|---|
| GPU type | NVIDIA H100 80GB |
| Batch size | 64 (eff. 512 with accum=8) |
| Peak memory | PENDING |
| Time per batch | PENDING |
| Est. time per epoch | PENDING |
| Est. full training (200 epochs) | PENDING |
| bf16 support | Expected: YES |
| Gradient checkpointing | Available, not default |
| Balanced subject sampling | YES |
| Dataloader errors | PENDING |
