# Quick Manual Setup (No Sudo Required)

> **For JupyterHub users without sudo access**

---

## ✅ You Already Have:
- ✓ Username configured: `md5_sd8f61177fd2312b9b32bd118ad1`
- ✓ `.env` file created

---

## 📦 Install MinIO Client (No Sudo)

Press **Ctrl+C** to cancel the password prompt, then run:

```bash
# Create bin directory
mkdir -p ~/bin

# Download MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc -O ~/bin/mc

# Make executable
chmod +x ~/bin/mc

# Add to PATH
export PATH="$HOME/bin:$PATH"

# Add to shell profile permanently
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc

# Verify installation
~/bin/mc --version
```

---

## 🔧 Configure MinIO

```bash
# Configure MinIO alias
~/bin/mc alias set uniminio https://ubbc1u.ro:9000 \
    cjChy0M8fUFIUWJ0SVd9 \
    TeIDlHx94IJS8ACQPNsmyFqo8IfnKbmzheKYkfxf \
    --api S3v4

# Test connection
~/bin/mc ls uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
```

**Expected**: Shows bucket contents (or empty if new)

---

## 🐍 Install Python Environment

```bash
# Activate environment setup
bash scripts/setup_env.sh

# Wait for installation (5-10 minutes)

# Activate
source activate_env.sh

# Verify
python3 --version
pip list | grep torch
```

---

## 📊 Build Caches (Streams from AWS S3)

```bash
# Make sure environment is activated
source activate_env.sh

# Build index (small file from S3)
python3 scripts/build_full_subj01_index.py

# Build CLIP cache (streams images from AWS S3, ~1-2 hours)
bash scripts/build_clip_for_training.sh

# Build target cache (streams from AWS S3, ~1-2 hours)
python3 scripts/build_target_clip_cache_robust.py \
    --cache_dir cache \
    --output cache/target_clip_cache_sd21.h5 \
    --batch_size 256 \
    --device cuda:0

# Build preprocessors (~2 minutes)
bash scripts/build_all_preprocessors.sh
```

---

## 📤 Upload to MinIO

```bash
# Upload CLIP caches (~5-10 min)
~/bin/mc cp --recursive \
    cache/clip_embeddings/ \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/clip_embeddings/

# Upload target cache
~/bin/mc cp cache/target_clip_cache_sd21.h5 \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/

# Upload preprocessors
~/bin/mc cp cache/*.pkl \
    uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/

# Verify upload
~/bin/mc ls -r uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/
```

---

## 🚀 Run Experiments

```bash
# Run with hybrid setup (streams from AWS S3)
python3 -m fmri2img.train \
    --config configs/experiments/a100_hybrid.yaml \
    --gpu 0

# Or all experiments in background
tmux new -s experiments
source activate_env.sh
bash scripts/run_all_experiments.sh 0
```

---

## 💡 Aliases for Convenience

Add to `~/.bashrc`:

```bash
# Add to end of file
alias mc='~/bin/mc'
alias activate='source ~/Bachelor_V2/activate_env.sh'
```

Then reload:
```bash
source ~/.bashrc
```

Now you can use:
```bash
mc ls uniminio/...
activate
```

---

## ✅ Summary

**No sudo needed!** Everything installs to your home directory:
- MinIO client: `~/bin/mc`
- Python venv: `~/Bachelor_V2/venv/`
- Data cache: `~/.cache/`

**Total time**: ~3 hours (mostly automated building caches)

---

**Next**: Press Ctrl+C to cancel password prompt, then run the commands above! 🚀
