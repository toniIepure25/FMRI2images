# Checkpoint Resume Guide

## 🔍 Step 1: Find Your Checkpoint

**On the server, run:**

```bash
# List all training runs (sorted by date, newest first)
ls -lht /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/

# You should see something like:
# drwxr-xr-x 3 user group 4096 Jan 14 09:35 20260114_083000_ultimate_novel_subj01/
```

## 📂 Step 2: Check for Epoch 5 Checkpoint

```bash
# Replace YYYYMMDD_HHMMSS with your actual run directory name
ls -lh /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260114_*/checkpoints/

# You should see:
# epoch_05.pt  <-- This is what we need!
# best_model.pt
```

## 🚀 Step 3: Resume Training from Epoch 5

**IMPORTANT: First, kill the competing process (if it's yours):**

```bash
# Check what's using the GPU
nvidia-smi

# If you see your old training process:
kill 2348277  # Replace with the actual PID
```

**Then pull the latest code and resume:**

```bash
cd ~/Bachelor_V2
git pull origin probabilistic-distribution

# Resume from epoch 5 checkpoint (replace path with your actual checkpoint)
python scripts/train_ultimate_novel.py \
  --config experiments/ultimate_novel_subj01.yaml \
  --resume /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260114_083000_ultimate_novel_subj01/checkpoints/epoch_05.pt
```

## 📊 What the Resume Does

1. ✅ Loads model weights from epoch 5
2. ✅ Restores optimizer state (momentum, learning rate schedule)
3. ✅ Continues from epoch 6 → 50
4. ✅ Preserves best loss and metrics history
5. ✅ Saves to the SAME run directory

## 🆕 Starting Fresh (No Resume)

If checkpoint doesn't exist or you want to start over:

```bash
python scripts/train_ultimate_novel.py \
  --config experiments/ultimate_novel_subj01.yaml
```

## 🔧 Memory-Reduced Model

The config has been updated to use less memory:
- `latent_dim`: 768 → 512
- `n_blocks`: 6 → 4
- `head_hidden_dim`: 1024 → 768
- `enabled_layers`: Only 'final' (not layer_4, layer_8, layer_12)
- `predict_text_clip`: false (was true)

**Expected memory usage:** ~6-7GB (was ~10GB)

This should fit even with the competing process using 16GB.

## 📈 Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check training logs (in another terminal)
tail -f /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/*/logs/*.log
```

## ⚠️ If Still OOM

If you still get OOM errors after killing the other process:

1. Check if someone else is using the GPU:
   ```bash
   nvidia-smi
   ps -fp <PID>  # Check who owns the process
   ```

2. Reduce batch size further in config:
   ```yaml
   training:
     batch_size: 2  # Was 4
     gradient_accumulation_steps: 16  # Was 8 (keep effective batch at 32)
   ```

## ✅ Success Indicators

Training is working if you see:
- Epoch progress: `Epoch 6: 120batch [48:30, ...]`
- Decreasing losses: `recon=0.45, kl=0.26`
- No NaN values in losses
- Memory stable (~6-7GB for reduced model)
