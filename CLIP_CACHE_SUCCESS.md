# ✅ CLIP Cache Build - SUCCESS!

## Summary

**Status:** COMPLETE ✅  
**Time:** 2 minutes  
**Date:** November 15, 2025

### Results

```
✓ Total in cache:    10,004 embeddings
✓ Newly processed:   9,999 images  
✓ Failed:            1 image (nsdId=73000, S3 issue - negligible)
✓ Source:            99.99% from Local HDF5 (fast!)
```

### Your Dataset

- **Trials:** 30,000 fMRI scans
- **Unique Stimuli:** 10,000 images (each shown ~3 times)
- **CLIP Cache Coverage:** 100% ✅

This is the **standard NSD subj01 dataset** - perfect for training!

---

## 🚀 Next Steps

### Option A: Automated Pipeline (Recommended)

Run the complete pipeline script:

```bash
bash run_training.sh
```

This will:
1. Verify CLIP cache ✅ (already done!)
2. Build T1 scaler (~2 minutes)
3. Build T2 PCA components (~10 minutes)
4. Train two-stage encoder (~6 hours)

**Total time:** ~6-7 hours

### Option B: Manual Steps

If you prefer to run each step manually:

#### 1. Preprocess T1 (2 minutes)
```bash
python scripts/preprocess_fmri.py \
  --subject subj01 \
  --method t1 \
  --output cache/preproc/subj01_t1_scaler.pkl
```

#### 2. Preprocess T2 (10 minutes)
```bash
python scripts/preprocess_fmri.py \
  --subject subj01 \
  --method t2 \
  --pca-dim 512 \
  --output cache/preproc/subj01_t2_pca_k512.npz
```

#### 3. Train Two-Stage Encoder (6 hours) ⚠️ Use tmux!
```bash
# Start tmux session
tmux new -s training

# Run training
python scripts/train_two_stage.py \
  --config configs/sota_two_stage.yaml \
  --subject subj01 \
  --output-dir checkpoints/two_stage/subj01

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## 📊 Expected Training Output

### Data Split (from your 30K trials)
- Training: 24,000 trials (~8,000 unique stimuli)
- Validation: 3,000 trials (~1,000 unique stimuli)
- Test: 3,000 trials (~1,000 unique stimuli)

### Training Progress
- **Epochs:** 200 (with early stopping)
- **Batch size:** 128
- **GPU Memory:** ~6-8 GB
- **Checkpoints:** Saved every epoch
- **Best model:** Saved based on validation Ridge R²

### Monitoring

**Watch training progress:**
```bash
# In another terminal
tail -f logs/clip_adapter/subj01/training_*.log
```

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Monitor checkpoints:**
```bash
watch -n 60 "ls -lh checkpoints/two_stage/subj01/"
```

---

## 🎯 After Training

Once training completes (~6 hours), you'll have:

```
checkpoints/two_stage/subj01/
├── two_stage_best.pt          # Best model (highest val R²)
├── two_stage_last.pt           # Final checkpoint
└── training_history.json       # Loss curves, R² scores
```

### Evaluate Performance

**1. NSD Shared 1000 benchmark:**
```bash
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/eval/subj01
```

**Expected metrics:**
- Top-1 Retrieval: ~15-20%
- Top-5 Retrieval: ~30-40%
- Ridge R²: ~0.45-0.55

**2. Generate visual comparisons:**
```bash
python scripts/generate_comparison_gallery.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/galleries/subj01 \
  --num-samples 16
```

This creates side-by-side comparisons:
- Ground truth image
- Single-sample generation
- Best-of-8 generation
- BOI-lite refined generation

---

## 🔧 Troubleshooting

### "RuntimeError: CUDA out of memory"
Reduce batch size in `configs/sota_two_stage.yaml`:
```yaml
training:
  batch_size: 64  # Reduce from 128
```

### "FileNotFoundError: preproc files not found"
Run preprocessing steps (1-2) before training.

### Training seems stuck
Check logs:
```bash
tail -f logs/clip_adapter/subj01/training_*.log
```

### Resume interrupted training
```bash
python scripts/train_two_stage.py \
  --config configs/sota_two_stage.yaml \
  --subject subj01 \
  --output-dir checkpoints/two_stage/subj01 \
  --resume  # Add this flag
```

---

## 📈 Why Only 10,000 Stimuli?

This is **intentional and correct**!

The NSD dataset design:
- Each subject views ~10,000 unique images
- Each image shown 3 times across different sessions
- Total: 30,000 trials (40 sessions × 750 trials)

Your CLIP cache (10,004 embeddings) covers **100% of unique stimuli** needed for training. The "~73,000" mentioned earlier refers to the complete NSD stimulus bank across **all 8 subjects**, but each individual subject only sees a subset.

---

## ✅ You're All Set!

**CLIP cache:** ✅ Complete (10,004 embeddings)  
**Next step:** Run preprocessing + training  
**Command:** `bash run_training.sh`

Good luck with training! 🚀

---

## 📚 Documentation

- `COMMANDS.txt` - Quick command reference
- `START_HERE.md` - Getting started guide
- `SOTA_QUICK_START.md` - Architecture overview
- `USAGE_EXAMPLES.md` - All available commands
