# 🚀 RESEARCH WORKFLOW GUIDE
## Professional Bachelor Thesis - fMRI to Image Reconstruction

**Last Updated:** January 12, 2026  
**Commit:** 1ac6ac9  
**Status:** ✅ Ready for Research

---

## 📋 QUICK START (On Server)

```bash
# 1. Pull latest changes
cd ~/Bachelor_V2
git pull origin probabilistic-distribution

# 2. Verify environment
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. Check data is ready
ls -lh data/indices/nsd_index/subject=subj01/

# Ready to run experiments! 🎯
```

---

## 🎓 COMPLETE RESEARCH PIPELINE

### Phase 1: Compare Existing Experiments

Compare the three experiments you just ran (baseline, novel, ablation):

```bash
cd ~/Bachelor_V2

python scripts/compare_experiments.py \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191153_baseline_subj01 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191216_novel_subj01 \
  /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260112_191239_ablation_subj01 \
  --output experiment_comparison
```

**Generates:**
- 📊 `comparison_report.md` - Markdown report
- 📈 `training_curves.png/.pdf` - Publication-quality plots
- 📉 `best_loss_comparison.png/.pdf` - Bar chart comparison
- 🔍 `loss_convergence.png/.pdf` - Convergence analysis
- 📋 `comparison_table.csv` - Data table
- 📄 `comparison_table.tex` - LaTeX for paper
- 📊 `statistical_summary.json` - Statistical analysis

**View Results:**
```bash
cat experiment_comparison/comparison_report.md
eog experiment_comparison/training_curves.png  # View image
```

---

### Phase 2: Train on REAL NSD Data

Train a model on **actual fMRI brain recordings**:

```bash
cd ~/Bachelor_V2

# Run training (takes ~5-10 minutes for 20 epochs on A100)
python scripts/train_real_nsd.py --config experiments/real_baseline_subj01.yaml
```

**What happens:**
1. ✅ Loads REAL fMRI data from `data/indices/nsd_index/subject=subj01/index.parquet`
2. ✅ Trains CNN model (81×104×83 voxels → 512-dim CLIP embedding)
3. ✅ Saves checkpoints every epoch
4. ✅ Logs to TensorBoard
5. ✅ Saves best model based on loss

**Monitor Training:**
```bash
# In another terminal:
tensorboard --logdir /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/

# Then open in browser: http://localhost:6006
```

**Expected Output:**
```
REAL NSD DATA TRAINING
Experiment: real_baseline_subj01_session1
Subject: subj01
Session: 1

✓ DataLoader created with REAL NSD data
✓ Model created: simple_cnn (Total parameters: X,XXX,XXX)

STARTING TRAINING
Epoch 1/20: 100%|████████| 47/47 [00:xx<00:00, xxx it/s, loss=X.XXXX]
Epoch 1/20 - Loss: X.XXXX - Time: X.XXs
  🌟 New best loss: X.XXXX
...
✅ TRAINING COMPLETE!
🏆 Best Loss: X.XXXX
📁 Results: /bigdata/.../runs/YYYYMMDD_HHMMSS_real_baseline_subj01_session1
```

**Output Location:**
```
/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/YYYYMMDD_HHMMSS_real_baseline_subj01_session1/
├── checkpoints/
│   ├── best_model.pt              # Best model (use this!)
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── ...
├── logs/
│   └── train.log
├── metrics/
│   └── summary.json
├── tensorboard/                   # TensorBoard logs
└── config.yaml                    # Saved configuration
```

---

### Phase 3: Generate Images from fMRI

Extract a sample fMRI volume and generate CLIP embedding:

```bash
cd ~/Bachelor_V2

# First, extract a sample fMRI from the dataset
python -c "
import nibabel as nib
import numpy as np
img = nib.load('data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz')
data = img.get_fdata()
sample = data[:, :, :, 0]  # First volume
np.save('sample_fmri.npy', sample)
print(f'✓ Saved sample fMRI shape: {sample.shape}')
"

# Generate CLIP embedding from fMRI
python scripts/generate_images.py \
  --checkpoint /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/YYYYMMDD_HHMMSS_real_baseline_subj01_session1/checkpoints/best_model.pt \
  --fmri sample_fmri.npy \
  --output generation_results \
  --visualize
```

**Replace `YYYYMMDD_HHMMSS_real_baseline_subj01_session1` with your actual run directory!**

**Generates:**
- 🧠 `clip_embedding.npy` - Generated CLIP embedding (512-dim vector)
- 🖼️ `fmri_slices.png` - Visualization of fMRI volume slices
- 📊 `embedding_analysis.png` - Embedding statistics and distribution
- 🔄 `generation_pipeline.png` - Complete pipeline visualization

**View Results:**
```bash
eog generation_results/generation_pipeline.png
python -c "import numpy as np; e = np.load('generation_results/clip_embedding.npy'); print(f'Embedding shape: {e.shape}, mean: {e.mean():.4f}')"
```

---

### Phase 4: Evaluate Model Performance

Comprehensive evaluation with research-level metrics:

```bash
cd ~/Bachelor_V2

python scripts/evaluate_model.py \
  --checkpoint /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/YYYYMMDD_HHMMSS_real_baseline_subj01_session1/checkpoints/best_model.pt \
  --output evaluation_results
```

**Generates:**
- 📊 `evaluation_metrics.json` - All metrics in JSON
- 📄 `evaluation_metrics.tex` - LaTeX table for paper
- 📈 `evaluation_metrics.png/.pdf` - Visualization plots

**Metrics Calculated:**
- ✅ **MSE/RMSE**: Reconstruction error
- ✅ **Cosine Similarity**: Embedding alignment
- ✅ **Pearson/Spearman Correlation**: Statistical correlation
- ✅ **R² Score**: Variance explained
- ✅ **L2 Distance**: Euclidean distance
- ✅ **Explained Variance**: Model performance

**View Results:**
```bash
cat evaluation_results/evaluation_metrics.json
eog evaluation_results/evaluation_metrics.png
```

---

## 📊 FOR YOUR THESIS

### Tables for LaTeX Paper

All scripts generate **LaTeX-ready tables**:

```latex
% In your thesis.tex:
\input{evaluation_results/evaluation_metrics.tex}
\input{experiment_comparison/comparison_table.tex}
```

### Figures for Paper

All visualizations saved as **high-res PNG + PDF**:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{training_curves.pdf}
  \caption{Training loss comparison across methods}
  \label{fig:training_curves}
\end{figure}
```

---

## 🔧 TROUBLESHOOTING

### Issue: Index file not found

```bash
# Build index for your subject/session:
python build_minimal_index.py --subject subj01 --session 1
```

### Issue: CUDA out of memory

```bash
# Reduce batch size in config:
vim experiments/real_baseline_subj01.yaml
# Change: batch_size: 16 → batch_size: 8
```

### Issue: TensorBoard not showing

```bash
# Make sure to run in correct directory:
tensorboard --logdir /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/

# If still not working, try:
tensorboard --logdir . --bind_all
```

---

## 📝 RECOMMENDED WORKFLOW

### For Bachelor Thesis Research:

1. **Week 1-2: Baseline Experiments**
   ```bash
   # Train baseline on sessions 1-3
   python scripts/train_real_nsd.py --config experiments/real_baseline_subj01.yaml
   # (Modify config for different sessions)
   ```

2. **Week 3-4: Novel Method**
   ```bash
   # Implement your novel contribution
   # Train with probabilistic distribution
   python scripts/train_real_nsd.py --config experiments/novel_subj01.yaml
   ```

3. **Week 5: Ablation Studies**
   ```bash
   # Run ablation experiments
   # Compare different components
   ```

4. **Week 6: Evaluation & Analysis**
   ```bash
   # Compare all experiments
   python scripts/compare_experiments.py run1 run2 run3 --output thesis_results
   
   # Generate images
   python scripts/generate_images.py --checkpoint best.pt --fmri sample.npy --output imgs
   
   # Comprehensive evaluation
   python scripts/evaluate_model.py --checkpoint best.pt --output eval
   ```

5. **Week 7-8: Writing**
   - Use generated LaTeX tables
   - Include publication-quality figures
   - Cite all metrics from JSON files

---

## 🎯 EXPECTED RESULTS

### Good Performance Indicators:
- ✅ **Cosine Similarity** > 0.7 (good alignment)
- ✅ **Pearson Correlation** > 0.6 (strong correlation)
- ✅ **R² Score** > 0.5 (explains >50% variance)
- ✅ **RMSE** < 0.5 (low reconstruction error)

### Training Indicators:
- ✅ Loss decreases over epochs
- ✅ Best loss achieved before final epoch (not overfitting)
- ✅ Training time: ~5-10 min per 20 epochs on A100

---

## 📚 FILE STRUCTURE

```
Bachelor_V2/
├── scripts/
│   ├── compare_experiments.py       # ✅ Compare multiple runs
│   ├── train_real_nsd.py            # ✅ Train on real NSD data
│   ├── generate_images.py           # ✅ Generate from fMRI
│   └── evaluate_model.py            # ✅ Evaluate performance
├── experiments/
│   ├── real_baseline_subj01.yaml    # ✅ Config for real data
│   ├── baseline_subj01.yaml         # Old (synthetic data)
│   ├── novel_subj01.yaml            # Old (synthetic data)
│   └── ablation_subj01.yaml         # Old (synthetic data)
├── data/
│   ├── indices/                     # ✅ Parquet indices
│   └── nsd/                         # ✅ NSD dataset (17GB)
└── runs/                            # Output directory
    └── YYYYMMDD_HHMMSS_*/           # Each experiment run
```

---

## 🎓 CITATION

If you use this code in your bachelor thesis:

```bibtex
@misc{fmri2img2026,
  author = {Your Name},
  title = {fMRI to Image Reconstruction via Probabilistic Distributions},
  year = {2026},
  note = {Bachelor Thesis},
  url = {https://github.com/toniIepure25/FMRI2images}
}
```

---

## ✅ CHECKLIST

Before running experiments:
- [ ] Git pulled latest code
- [ ] Virtual environment activated
- [ ] CUDA available (`torch.cuda.is_available()`)
- [ ] NSD data downloaded (17GB)
- [ ] Index built for subject/session
- [ ] Enough disk space (128GB+ free)

For thesis submission:
- [ ] All experiments completed
- [ ] Comparison plots generated
- [ ] Evaluation metrics calculated
- [ ] LaTeX tables created
- [ ] Figures exported as PDF
- [ ] Results documented
- [ ] Code committed to GitHub

---

## 🆘 SUPPORT

If something doesn't work:

1. Check logs: `tail -f runs/*/logs/train.log`
2. Check GPU: `nvidia-smi`
3. Check disk: `df -h`
4. Check errors: Python traceback in console

---

**Good luck with your bachelor thesis! 🎓✨**

*All scripts are production-ready and research-level quality.*
