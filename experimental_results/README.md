# Experimental Results Analysis

This folder contains **evaluation metrics and analysis** for all completed training experiments. This is separate from training runs and focuses only on final results for comparison.

## Folder Structure

```
experimental_results/
├── exp001_baseline_ultimate/
│   ├── evaluation/
│   │   ├── clip_metrics.json          # CLIP embedding quality metrics
│   │   ├── probabilistic_metrics.json # KL divergence, uncertainty
│   │   ├── retrieval_results.json     # Top-k accuracy, rankings
│   │   └── summary_report.md          # Human-readable summary
│   ├── reconstructions/               # (Optional) Generated images
│   │   ├── gallery_top10.png
│   │   ├── gallery_worst10.png
│   │   └── samples/
│   ├── config.yaml                    # Training configuration used
│   ├── training_info.json             # Epochs, time, hardware info
│   └── notes.md                       # Observations, insights
│
├── exp002_higher_lr/
│   └── ... (same structure)
│
├── exp003_ablation_layer4/
│   └── ... (same structure)
│
└── comparison_reports/
    ├── baseline_vs_higher_lr.md
    ├── ablation_study_layers.md
    └── final_thesis_comparison.md
```

## Workflow: From Training to Analysis

### Phase 1: Training Completion ✅
- Train model to completion (e.g., 50 epochs)
- Checkpoint saved: `runs/DATE_TIME_NAME/checkpoints/best_model.pt`

### Phase 2: Evaluation (FIRST - MOST IMPORTANT) 🔍
**Run comprehensive evaluation** on the trained model:
```bash
python scripts/evaluate_ultimate_model.py \
    --checkpoint runs/YOUR_RUN/checkpoints/best_model.pt \
    --config experiments/YOUR_CONFIG.yaml \
    --output-dir experimental_results/exp001_baseline_ultimate/evaluation \
    --num-samples 1000
```

**Key Metrics for 2-Stage Probabilistic Model**:
1. **CLIP Embedding Quality**:
   - Cosine similarity (mean, median, std) - measures embedding alignment
   - Expected: >0.40 for good performance, >0.50 for excellent

2. **Retrieval Performance**:
   - Top-1 accuracy (correct image in top-1)
   - Top-5 accuracy (correct image in top-5)
   - Top-10 accuracy
   - Mean rank (lower = better)
   - Expected: Top-5 >40% indicates strong learning

3. **Probabilistic Metrics** (Novel for this model):
   - KL divergence (regularization quality)
   - Variance in predictions (uncertainty estimation)
   - Expected: KL ~0.1-0.3 for good balance

4. **Reconstruction Error**:
   - MSE/RMSE in embedding space
   - L2 distance (mean, std)

**Decision Point**: Based on these metrics, decide if:
- ✅ **Good** → Archive and maybe generate reconstructions
- ❌ **Poor** → Modify hyperparameters and train new experiment

### Phase 3: Image Reconstruction (OPTIONAL - Visual Validation) 🖼️
**Only if evaluation metrics are satisfactory**, generate image reconstructions:
```bash
python scripts/run_stage34_recon_eval.py \
    --checkpoint runs/YOUR_RUN/checkpoints/best_model.pt \
    --config experiments/YOUR_CONFIG.yaml \
    --output-dir experimental_results/exp001_baseline_ultimate/reconstructions \
    --num-images 50
```

**Why optional?**
- Image reconstruction is slow (requires Stable Diffusion inference)
- CLIP metrics already tell you embedding quality
- Use reconstructions for:
  - Visual validation of best model
  - Thesis figures/presentations
  - Comparing top experiments visually

### Phase 4: Archive and Document 📝
Copy configuration and add notes:
```bash
cp experiments/YOUR_CONFIG.yaml experimental_results/exp001_baseline_ultimate/config.yaml
```

Edit `notes.md` with observations:
- What hyperparameters were used
- Training stability observations
- Metric interpretation
- Ideas for next experiment

### Phase 5: Compare Experiments 📊
After running multiple experiments:
```bash
python scripts/compare_experimental_results.py \
    --experiments exp001_baseline exp002_higher_lr exp003_ablation \
    --output experimental_results/comparison_reports/baseline_comparison.md
```

## Naming Convention

**Format**: `expXXX_descriptive_name`

Examples:
- `exp001_baseline_ultimate` - Baseline with all 7 contributions
- `exp002_lr5e5` - Higher learning rate (5e-5)
- `exp003_lr1e6` - Lower learning rate (1e-6)
- `exp004_ablation_layer4` - Disable layer_4 output
- `exp005_ablation_layer8` - Disable layer_8 output
- `exp006_no_infonce` - Remove InfoNCE loss
- `exp007_kl_weight_01` - Higher KL weight (0.1 instead of 0.01)
- `exp008_more_blocks` - 8 blocks instead of 6
- `exp009_dropout_03` - Higher dropout (0.3 instead of 0.1)
- `exp010_final_thesis` - Best configuration for final thesis

## Quick Commands

### Evaluate completed training
```bash
cd ~/Bachelor_V2  # On JupyterHub cluster

# Evaluate
python scripts/evaluate_ultimate_model.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --output-dir experimental_results/exp001_baseline_ultimate/evaluation \
    --num-samples 1000
```

### Check results
```bash
# View metrics
cat experimental_results/exp001_baseline_ultimate/evaluation/eval_best_model.json

# List all experiments
ls -lh experimental_results/
```

### Compare multiple experiments
```bash
# After you have multiple experiments
python scripts/compare_experimental_results.py \
    --exp-dirs experimental_results/exp001_baseline_ultimate \
                experimental_results/exp002_higher_lr \
    --output experimental_results/comparison_reports/exp001_vs_exp002.md
```

## Metrics Priority for 2-Stage Probabilistic Model

### Tier 1: Essential (Check First)
1. **Cosine Similarity (mean)** - Overall embedding quality
2. **Top-5 Retrieval** - Practical retrieval performance
3. **KL Divergence** - Probabilistic component health

### Tier 2: Important (Context)
4. **Top-1 Retrieval** - Best-case performance
5. **Mean Rank** - Average performance
6. **L2 Distance** - Raw reconstruction error

### Tier 3: Diagnostic (Understanding)
7. **Cosine Similarity (std)** - Consistency across samples
8. **MSE/RMSE** - Alternative error metrics
9. **Median Rank** - Robustness to outliers

## Thesis Comparison Strategy

For your bachelor thesis, you'll want to show:

1. **Baseline Performance** (exp001)
   - All 7 novel contributions enabled
   - Standard hyperparameters

2. **Ablation Studies** (exp004-006)
   - Remove one contribution at a time
   - Shows importance of each component

3. **Hyperparameter Optimization** (exp002-003, exp007-009)
   - Learning rate sweep
   - KL weight tuning
   - Architecture depth

4. **Final Best Model** (exp010)
   - Best hyperparameters discovered
   - Use for final reconstructions and thesis figures

## Tips

1. **Evaluate immediately** after training finishes
2. **Don't generate reconstructions** for every experiment (slow)
3. **Focus on CLIP metrics** for comparing experiments
4. **Use reconstructions** only for best 2-3 models
5. **Document observations** in notes.md right away
6. **Compare systematically** using automated tools

---

*This folder is for **analysis only** - training runs stay in `runs/` directory*
