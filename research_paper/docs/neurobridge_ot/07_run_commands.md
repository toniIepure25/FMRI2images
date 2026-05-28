# NeuroBridge-OT: Run Commands (8-Subject Primary)

## Prerequisites

```bash
# Ensure environment is set up
set -a && source .env && set +a
pip install -e ".[train,diffusion]"

# Build index and pre-extract for ALL 8 subjects
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    make index SUBJECT=$subj
    make preextract SUBJECT=$subj
done

# Build CLIP cache
make clip-cache

# Verify data availability (all 8 subjects)
python3 scripts/neurobridge_ot/audit_neurobridge_data.py \
    --check-rois --check-teachers --check-shared1000
```

## 1. Data Audit (all 8 subjects)

```bash
python3 scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --check-rois --check-teachers --check-shared1000 \
    --output experimental_results/neurobridge_ot_8subj_audit.json
```

## 2. Smoke Test (CPU, ~2 minutes)

```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_smoke_test.yaml \
    --subjects subj01 \
    --limit-batches 5 \
    --output-dir experimental_results/neurobridge_ot_smoke
```

## 3. 8-Subject Multi-Subject Training (~12-18 hours, H100)

```bash
nohup python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_full.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_8subj_full.log 2>&1 &
```

## 4. Evaluation

```bash
python3 scripts/neurobridge_ot/eval_neurobridge_ot.py \
    --checkpoint experimental_results/neurobridge_ot_8subj_full/checkpoints/best.pt \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --split shared1000 \
    --csls-k 3 \
    --output-dir experimental_results/neurobridge_ot_8subj_full/eval
```

## 5. 8-Fold LOSO Generalization (~48-64 hours)

```bash
# Strict zero-shot (no target info)
nohup python3 scripts/neurobridge_ot/run_loso_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_loso.yaml \
    --all-8-folds --mode strict \
    --output-dir experimental_results/neurobridge_ot_8subj_loso_strict \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_loso_strict.log 2>&1 &

# Unsupervised calibrated (target fingerprint only)
nohup python3 scripts/neurobridge_ot/run_loso_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_loso.yaml \
    --all-8-folds --mode unsupervised_calibrated \
    --output-dir experimental_results/neurobridge_ot_8subj_loso_calibrated \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_loso_calibrated.log 2>&1 &
```

## 6. Few-Shot Adaptation, All 8 Targets (~24-36 hours)

```bash
nohup python3 scripts/neurobridge_ot/run_fewshot_adaptation.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_fewshot.yaml \
    --all-target-subjects \
    --few-shot-sizes 10 25 50 100 250 500 1000 \
    --output-dir experimental_results/neurobridge_ot_8subj_fewshot \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_fewshot.log 2>&1 &
```

## 7. Ablation Suite (~160-240 hours)

```bash
# Full 8-subject ablation
nohup python3 scripts/neurobridge_ot/run_ablation_suite.py \
    --configs-dir configs/experiments/neurobridge_ot/ \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --output-dir experimental_results/neurobridge_ot_8subj_ablation \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_ablation.log 2>&1 &

# Quick ablation (limited batches for testing)
python3 scripts/neurobridge_ot/run_ablation_suite.py \
    --configs-dir configs/experiments/neurobridge_ot/ \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --limit-batches 100 --seed 42
```

## 8. Subject-Specific Replication (all 8)

```bash
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    nohup python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
        --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_subject_specific.yaml \
        --subjects $subj --gpu 0 --seed 42 \
        --output-dir experimental_results/neurobridge_ot_subject_specific/$subj \
        > logs/neurobridge_ot_subj_${subj}.log 2>&1 &
    wait  # Sequential, or remove wait for parallel on multi-GPU
done
```

## 9. Legacy 4-Subject Comparison (V66a baseline)

```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_4subj_legacy_v66a_comparison.yaml \
    --subjects subj01 subj02 subj05 subj07 \
    --gpu 0 --seed 42
```

## 10. Export Teacher Predictions (if checkpoints available)

```bash
# V66a only available for subj01/02/05/07
for subj in subj01 subj02 subj05 subj07; do
    python3 scripts/neurobridge_ot/export_teacher_predictions.py \
        --teacher-checkpoint experimental_results/V66a_roi_pretrain/$subj/checkpoints/best.pt \
        --teacher-name V66a --subject $subj \
        --output-dir experimental_results/neurobridge_ot/teachers
done
```

## 11. Summarize Results

```bash
python3 scripts/neurobridge_ot/summarize_neurobridge_results.py \
    --results-dir experimental_results/ \
    --output-dir experimental_results/neurobridge_ot_8subj_summary
```

---

## Recommended Execution Order

1. **Data audit** — verify all 8 subjects are ready.
2. **Smoke test** — validate implementation runs.
3. **8-subject multi-subject training** — primary baseline.
4. **Evaluation** — SHARED1000 metrics for all 8.
5. **Subject-specific replication** — per-subject upper bounds.
6. **8-fold LOSO strict** — generalization evidence.
7. **8-fold LOSO calibrated** — unsupervised calibration benefit.
8. **Few-shot adaptation** — data-efficiency curves.
9. **Ablation suite** — component contributions.
10. **Legacy 4-subj comparison** — V66a comparison (separate).
11. **Summarize** — generate paper tables.

## Compute Budget (H100 80GB, 8 subjects)

| Step | Time | Memory |
|---|---|---|
| Smoke test | 2 min | 2 GB |
| Single subject (×8) | 24-32 h total | 20 GB |
| Multi-subject (8 subj) | 12-18 h | 50-60 GB |
| LOSO strict (8 folds) | 48-64 h | 50 GB |
| LOSO calibrated (8 folds) | 48-64 h | 50 GB |
| Few-shot (8 targets × 7 sizes) | 24-36 h | 30 GB |
| Full ablation (6 configs) | 72-108 h | 50 GB |
| Legacy 4-subj | 8-12 h | 40 GB |
| **Total for all experiments** | **~240-340 h** | — |
