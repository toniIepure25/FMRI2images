# Complete Command List - After Cloning

> **Copy and paste these commands one by one after cloning the repo**

---

## 📋 Complete Setup Commands

### **Step 1: Clone Repository**

**You're already here**: `/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1`

```bash
# Clone directly in your current directory
git clone https://github.com/toniIepure25/FMRI2images.git Bachelor_V2
cd Bachelor_V2
```

**OR if you want a different name:**
```bash
git clone https://github.com/toniIepure25/FMRI2images.git
cd FMRI2images
```

---

### **Step 2: Configure Environment**

```bash
# Copy configuration template
cp .env.jupyterhub .env

# Your username is: md5_sd8f61177fd2312b9b32bd118ad1
# Automatically set it:
sed -i 's/student01/md5_sd8f61177fd2312b9b32bd118ad1/g' .env
```

**Or edit manually:**
```bash
nano .env
# Find: USER=${USER:-student01}
# Change to: USER=${USER:-md5_sd8f61177fd2312b9b32bd118ad1}
# Save: Ctrl+X, Y, Enter
```

---

### **Step 3: Install Python Environment**

```bash
# This takes 5-10 minutes
bash scripts/setup_env.sh
```

**Wait for**: `✓ Setup complete!`

---

### **Step 4: Activate Environment**

```bash
source activate_env.sh
```

**You should see**: `(venv)` appear in your terminal prompt

---

### **Step 5: Verify Setup**

```bash
bash scripts/preflight.sh
```

**Expected**: `✓ All critical checks passed!`

---

### **Step 6: Run Smoke Test (No Data Needed!)**

```bash
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**Expected**: Training completes in 2-3 minutes

---

## 🎉 You're Done!

At this point:
- ✅ Environment installed
- ✅ Pipeline tested
- ✅ Everything works
- ✅ **NO 300GB download needed!**

---

## 📋 All Commands in One Block (Copy-Paste)

```bash
# ============================================
# COMPLETE SETUP - RUN AFTER CLONING
# ============================================

# 1. Clone repository
cd ~
git clone https://github.com/toniIepure25/FMRI2images.git Bachelor_V2
cd Bachelor_V2

# 2. Configure
cp .env.jupyterhub .env
echo ""
echo "⚠️  IMPORTANT: Edit .env file now!"
echo "Change: USER=\${USER:-student01}"
echo "To:     USER=\${USER:-YOUR_ACTUAL_USERNAME}"
echo ""
read -p "Press Enter to open nano editor..." 
nano .env

# 3. Install Python environment (5-10 minutes)
bash scripts/setup_env.sh

# 4. Activate environment
source activate_env.sh

# 5. Verify setup
bash scripts/preflight.sh

# 6. Run smoke test
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml

echo ""
echo "🎉 Setup complete! Pipeline is working!"
```

---

## 🔄 Every Time You Log In (Daily Use)

After the initial setup, each time you log in:

```bash
cd ~/Bachelor_V2
source activate_env.sh
```

That's it! Then run your experiments.

---

## 📊 What Each Command Does

| Command | What It Does | Time |
|---------|-------------|------|
| `git clone ...` | Downloads code | 1 min |
| `cp .env.jupyterhub .env` | Creates config | 1 sec |
| `nano .env` | Edit username | 1 min |
| `bash scripts/setup_env.sh` | Installs Python + packages | 5-10 min |
| `source activate_env.sh` | Activates environment | 1 sec |
| `bash scripts/preflight.sh` | Verifies setup | 10 sec |
| `bash scripts/run_experiment_simple.sh ...` | Runs test | 2-3 min |
| **TOTAL** | | **~10-15 minutes** |

---

## 🆘 If Something Goes Wrong

### **Problem: "git clone fails"**
```bash
# Check if you have access to the repo
# Or clone from a different location if you have it elsewhere
```

### **Problem: "setup_env.sh fails"**
```bash
# Check Python version
python --version  # Should be 3.11+

# Try force reinstall
bash scripts/setup_env.sh --force
```

### **Problem: "preflight.sh shows errors"**
```bash
# CUDA warnings are OK (if no GPU yet)
# Check what specific error it shows
```

### **Problem: "smoke test fails"**
```bash
# Check logs
cat runs/*/logs/train.log

# Or check what error message appears
```

### **Problem: "I don't know my username"**
```bash
# Run this to find it:
whoami
```

---

## ✅ Verification Checklist

After running all commands, verify:

- [ ] `(venv)` appears in terminal prompt
- [ ] `bash scripts/preflight.sh` passes
- [ ] Smoke test completes successfully
- [ ] You see output in `runs/` directory
- [ ] No error messages

**All checked?** You're ready! 🎉

---

## 🚀 Next Steps

After setup is complete:

1. **Read documentation:**
   ```bash
   cat SIMPLE_START.md
   cat MINIMAL_DATA_SETUP.md
   ```

2. **When ready for real experiments:**
   - Download subj01 data (40GB, not 300GB)
   - See `MINIMAL_DATA_SETUP.md` for details

3. **Run experiments:**
   ```bash
   source activate_env.sh
   bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml
   ```

---

## 💾 Alternative: Use Automated Script

Instead of running commands manually, use the automated script:

```bash
cd ~/Bachelor_V2
bash scripts/simple_setup.sh
```

This runs all steps interactively.

---

## 📝 Summary

**After cloning, run these 6 commands:**

1. `cp .env.jupyterhub .env`
2. `nano .env` (edit username)
3. `bash scripts/setup_env.sh`
4. `source activate_env.sh`
5. `bash scripts/preflight.sh`
6. `bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml`

**Total time**: 10-15 minutes  
**Data needed**: 0GB (smoke test uses no data)  
**Result**: Working pipeline! ✅

---

**You're ready to start your bachelor thesis work!** 🎓
