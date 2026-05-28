# NeuroBridge-OT: Run Commands

## Prerequisites

```bash
# Ensure environment is set up
set -a && source .env && set +a
pip install -e ".[train,diffusion]"

# Verify data availability
python scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj05 subj07 \
    --check-rois --check-teachers
```

## 1. Data Audit

```bash
python scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj05 subj07 \
    --check-rois \
    --check-teachers \
    --output experimental_results/neurobridge_ot_audit.json
```

## 2. Smoke Test (CPU, ~2 minutes)

```bash
python scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_smoke_test.yaml \
    --subjects subj01 \
    --limit-batches 5 \
    --output-dir experimental_results/neurobridge_ot_smoke
```

## 3. Full Multi-Subject Training (~8-12 hours, H100)

```bash
nohup python scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_full.yaml \
    --subjects subj01 subj02 subj05 subj07 \
    --gpu 0 \
    --seed 42 \
    --output-dir experimental_results/neurobridge_ot_full \
    > logs/neurobridge_ot_full.log 2>&1 &
```

## 4. Evaluation

```bash
python scripts/neurobridge_ot/eval_neurobridge_ot.py \
    --checkpoint experimental_results/neurobridge_ot_full/checkpoints/best.pt \
    --subjects subj01 subj02 subj05 subj07 \
    --split shared1000 \
    --csls-k 3 \
    --output-dir experimental_results/neurobridge_ot_full/eval
```

## 5. LOSO Generalization (~24-36 hours)

```bash
nohup python scripts/neurobridge_ot/run_loso_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_loso.yaml \
    --all-subjects subj01 subj02 subj05 subj07 \
    --output-dir experimental_results/neurobridge_ot_loso \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_loso.log 2>&1 &
```

## 6. Few-Shot Adaptation (~16-24 hours)

```bash
nohup python scripts/neurobridge_ot/run_fewshot_adaptation.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_fewshot.yaml \
    --source-subjects subj01 subj02 subj05 \
    --target-subject subj07 \
    --few-shot-sizes 10 25 50 100 250 500 1000 \
    --output-dir experimental_results/neurobridge_ot_fewshot \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_fewshot.log 2>&1 &
```

## 7. Ablation Suite (~120-160 hours)

```bash
# Run all ablations
nohup python scripts/neurobridge_ot/run_ablation_suite.py \
    --configs-dir configs/experiments/neurobridge_ot/ \
    --subjects subj01 subj02 subj05 subj07 \
    --output-dir experimental_results/neurobridge_ot_ablation \
    --gpu 0 --seed 42 \
    > logs/neurobridge_ot_ablation.log 2>&1 &

# Or run a single ablation
python scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_no_ot.yaml \
    --subjects subj01 subj02 subj05 subj07 \
    --gpu 0
```

## 8. Export Teacher Predictions (if checkpoints available)

```bash
# Export V61a predictions for subj01
python scripts/neurobridge_ot/export_teacher_predictions.py \
    --teacher-checkpoint experimental_results/V61a_finetune_difflr/subj01/checkpoints/best.pt \
    --teacher-name V61a \
    --subject subj01 \
    --output-dir experimental_results/neurobridge_ot/teachers

# Export V66a predictions for all subjects
for subj in subj01 subj02 subj05 subj07; do
    python scripts/neurobridge_ot/export_teacher_predictions.py \
        --teacher-checkpoint experimental_results/V66a_roi_pretrain/$subj/checkpoints/best.pt \
        --teacher-name V66a \
        --subject $subj \
        --output-dir experimental_results/neurobridge_ot/teachers
done
```

## 9. Summarize Results

```bash
python scripts/neurobridge_ot/summarize_neurobridge_results.py \
    --results-dir experimental_results/ \
    --output-dir experimental_results/neurobridge_ot_summary
```

## Recommended Execution Order

1. Data audit (verify prerequisites).
2. Smoke test (validate implementation).
3. Export teacher predictions (if available).
4. Full multi-subject training (baseline).
5. Evaluation on SHARED1000.
6. Key ablations (no_ot, roi_summary_baseline).
7. LOSO generalization.
8. Few-shot adaptation.
9. Remaining ablations.
10. Results summary.

## Compute Budget (H100 80GB)

| Step | Time | Memory |
|---|---|---|
| Smoke test | 2 min | 2 GB |
| Single subject | 3-4 h | 20 GB |
| Multi-subject (4) | 8-12 h | 40 GB |
| LOSO (4 folds) | 24-36 h | 40 GB |
| Few-shot (7 sizes) | 16-24 h | 20 GB |
| Full ablation (6 configs) | 48-72 h | 40 GB |
| **Total for all experiments** | **~100-150 h** | — |
