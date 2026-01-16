# Evaluation and Experiment Management Quick Guide

## Important: Remote Cluster Setup ☁️

**You are working on a JupyterHub remote cluster** - all training runs and checkpoints are on the cluster, not your local PC.

**Your Local PC**: Code repository, development, documentation  
**Remote Cluster**: Training, checkpoints, cache, large data

## Your Training Status

**Current Run**: `runs/20260115_172845_ultimate_novel_subj01` (on cluster)
- **Completed**: 33/50 epochs (66%)
- **Status**: Stopped, stable training with no NaN
- **Checkpoints**: epoch_31, epoch_32, epoch_33, best_model.pt (on cluster)

## Step 1: Evaluate Your Current Model 🔍

**This is the MOST IMPORTANT step** - run comprehensive evaluation BEFORE deciding to continue training or generate images.

### Complete Evaluation Workflow (Recommended)

**On your JupyterHub terminal**:

```bash
cd ~/Bachelor_V2  # Or wherever your code is on the cluster

# Run complete evaluation workflow - creates organized results structure
python scripts/evaluate_experiment.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --exp-name exp001_baseline_ultimate \
    --num-samples 1000
```

**This creates**:
```
experimental_results/exp001_baseline_ultimate/
├── evaluation/
│   ├── eval_results.json              # All metrics
│   ├── clip_metrics.json              # CLIP-specific (cosine, retrieval)
│   ├── probabilistic_metrics.json     # KL divergence, uncertainty
│   └── summary_report.md              # Human-readable summary + recommendations
├── config.yaml                        # Training config used
├── training_info.json                 # Training metadata
└── notes.md                           # Template for your observations
```

### Alternative: Basic Evaluation (No Structure)

If you just want quick metrics without organizing:

```bash
python scripts/evaluate_ultimate_model.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --output-dir outputs/eval/quick_test \
    --num-samples 1000
```

**What this evaluates (Most Important Metrics for 2-Stage Probabilistic Model)**:

### Tier 1: Essential Metrics
1. **Cosine Similarity** (mean ± std)
   - Measures CLIP embedding alignment
   - **Good**: >0.40 | **Excellent**: >0.50
   
2. **Top-5 Retrieval Accuracy**
   - Correct image in top-5 predictions
   - **Good**: >40% | **Excellent**: >50%
   
3. **KL Divergence** (Novel for probabilistic model)
   - Variational regularization quality
   - **Good**: 0.1-0.3 (balanced)

### Tier 2: Important Context
4. **Top-1 Retrieval** - Best-case performance
5. **Mean Rank** - Average ranking position
6. **L2 Distance** - Reconstruction error

### Interpretation

The evaluation will show you:
```
📊 SUMMARY:
   Cosine Similarity: 0.45 ± 0.12  ← Target >0.40
   Top-1 Retrieval:   15%           ← >10% is good
   Top-5 Retrieval:   45%           ← >40% is strong!
   Mean Rank:         8.5           ← Lower is better
```

**Decision**:
- ✅ **Metrics good** → Continue training to epoch 50 OR archive and start new experiment
- ⚠️ **Metrics moderate** → Review hyperparameters, maybe modify
- ❌ **Metrics poor** → Don't waste time on reconstructions, fix training first

## Step 2: Review Results and Decide 📊

**Check the summary report**:
```bash
# View the auto-generated summary
cat experimental_results/exp001_baseline_ultimate/evaluation/summary_report.md

# Or view in JupyterHub file browser
```

The report includes:
- ✅ Quick summary table with interpretations
- ✅ Detailed metrics breakdown
- ✅ **Automatic recommendation** (proceed/modify/retry)
- ✅ Configuration used
- ✅ Next steps

**Add your observations**:
```bash
# Edit the notes file
nano experimental_results/exp001_baseline_ultimate/notes.md
```

Document:
- Training stability observations
- Metric interpretation
- What worked well / what didn't
- Ideas for next experiment

## Step 3: (Optional) Generate Image Reconstructions 🖼️

**ONLY run this if evaluation metrics are satisfactory!**

Image reconstruction is slow (requires Stable Diffusion). Use it for:
- Visual validation of best models
- Thesis figures and presentations
- Comparing top 2-3 experiments visually

**On the JupyterHub cluster**:

```bash
cd ~/Bachelor_V2

python scripts/run_stage34_recon_eval.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --output-dir experimental_results/exp001_baseline_ultimate/reconstructions \
    --num-images 50
```

**Tip**: Don't generate reconstructions for every experiment. CLIP metrics tell you embedding quality - use reconstructions only for your best models.

## Step 4: Continue or Start New Experiment 🚀

### Option A: Continue Current Training

If evaluation shows model is learning well, continue to epoch 50:

**On the JupyterHub cluster**:

```bash
cd ~/Bachelor_V2

python scripts/archive_experiment.py \
    --run-dir runs/20260115_172845_ultimate_novel_subj01 \
    --exp-name exp001_ultimate_baseline \
    --description "Baseline: all 7 novel contributions, LR=1e-5, 370k voxels, preprocessed" \
    --notes "First stable training run with preprocessing. No NaN issues."
```

**What this does**:
- ✅ Copies checkpoints to `experiments_archive/exp001_ultimate_baseline/`
- ✅ Copies config file
- ✅ Generates professional README with all details
- ✅ Creates notes template for observations
- ✅ Organizes everything for thesis comparison.

**On the JupyterHub cluster**:

```bash
cd ~/Bachelor_V2

python scripts/train_ultimate_novel.py \
    --config experiments/ultimate_novel_subj01.yaml \
    --resume 
```bash
python scripts/train_ultimate_novel.py \
    --config experiments/ultimate_novel_subj01.yaml \
    --resume /bigdata/tonystark/runs/20260115_172845_ultimate_novel_subj01/checkpoints/epoch_33.pt
```

**Estimated time**: ~40 hours (17 more epochs)

**Disk space**: Auto-cleanup will keep only last 3 checkpoints (~10.

**On the JupyterHub cluster**:

```bash
cd ~/Bachelor_V2

# Use the existing reconstruction script
python scripts/run_stage34_recon_eval.py \
    --checkpoint 
```bash
# Use the existing reconstruction script
python scripts/run_stage34_recon_eval.py \
    --checkpoint /bigdata/tonystark/runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --output-dir outputs/recon/exp001_baseline \
    --num-images 50
```

**What this does**:
- ✅ Uses your trained encoder to predict CLIP embeddings
- ✅ Feeds embeddings to Stable Diffusion
- ✅ Generates image reconstructions
- ✅ Saves comparison galleries (fMRI→reconstruction vs ground truth)

## Experiment Comparison Workflow

For your thesis, you'll want to compare multiple experiments:

### Setup
```
experiments_archive/
├── exp001_ultimate_baseline/     ← Current run (33/50 epochs)
├── exp002_higher_lr/              ← Future: Try LR=5e-5
├── exp003_more_layers/            ← Future: Add more transformer blocks
├── exp004_ablation_layer4/        ← Future: Disable layer_4
└── comparison_report.md           ← Generated comparison
```

### After Running Multiple Experiments

```bash
# Compare two experiments
python scripts/compare_experiments.py \
    --experiments exp001_baseline exp002_higher_lr \
    --output experiments_archive/comparison_baseline_vs_higher_lr.md
**On the JupyterHub cluster**, check all your epoch checkpoints:

```bash
# Check checkpoints
ls -lh ~/Bachelor_V2/runs/20260115_172845_ultimate_novel_subj01/checkpoints/

# Check training logs
tail -n 50 ~/Bachelor_V2/runs/20260115_172845_ultimate_novel_subj01/training.log

# Check disk usage
du -sh ~/Bachelor_V2/runs/20260115_172845_ultimate_novel_subj01/

```bash
# If you have a metrics log file
tail -n 50 /bigdata/tonystark/runs/20260115_172845_ultimate_novel_subj01/training.log
```

## Decision Tree

```
33/50 epochs completed
    │
    ├─→ Run evaluation (Step 1)
    │       │
    │       ├─→ Metrics GOOD (cosine > 0.4, top5 > 40%)
    │       │       ├─→ Archive experiment (Step 2)
    │       │       └─→ Continue training to epoch 50 (Step 3)
    │       │
    │       └─→ Metrics POOR (cosine < 0.3, top5 < 20%)
    │               ├─→ Modify hyperparameters
    │               ├─→ Start new experiment (exp002_*)
    │               └─→ Compare with exp001
    │
    └─→ Generate reconstructions (Step 4)
            └─→ Visual quality assessment
```

## Tips

1. **Evaluation First**: Always evaluate before deciding to continue training
2. **Archive Often**: Archive experiments as soon as they're complete
3. **Document Chan (Run on Cluster)

```bash
# Quick status
du -sh ~/Bachelor_V2/runs/20260115_172845_ultimate_novel_subj01/

# View checkpoint info
python -c "import torch; ckpt = torch.load('runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt', map_location='cpu'); print(f'Epoch: {ckpt[\"epoch\"]}, Loss: {ckpt.get(\"metrics\", {}).get(\"loss\", \"N/A\")}')"

# Check GPU availability
nvidia-smi

# List all your runs
ls -lhtr ~/Bachelor_V2/runs/

# Find checkpoint files
find ~/Bachelor_V2/runs -name "*.pt" -type f
```

## Syncing Results to Local PC

To download evaluation results and visualizations:

```bash
# From your local PC, sync results
scp -r your-cluster:/path/to/outputs/eval/ ~/Desktop/'Bachelor V2'/outputs/
scp -r your-cluster:/path/to/experiments_archive/ ~/Desktop/'Bachelor V2'/
```

Or use JupyterHub's file browser to download result files.hon -c "import torch; ckpt = torch.load('/bigdata/tonystark/runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt', map_location='cpu'); print(f'Epoch: {ckpt[\"epoch\"]}, Loss: {ckpt[\"metrics\"][\"loss\"]:.4f}')"

# Check GPU availability
nvidia-smi
```

---

**Next Steps**:
1. ✅ Run evaluation (15 min)
2. ✅ Review metrics
3. ✅ Decide: continue or modify
4. ✅ Archive when complete
