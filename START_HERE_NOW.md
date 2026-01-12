# ⚡ QUICK START - Do This Now!

> **Lost and don't know where to start? Follow these exact steps.**

---

## ✅ RIGHT NOW (Next 10 Minutes)

### **Step 1: Send This Email**

**To**: Your IT technician / System administrator  
**Subject**: MinIO Access Request for Bachelor Thesis  

```
Hi,

I need access to MinIO Object Store for my bachelor thesis project.
I'm at the MinIO web interface but it shows "There are no Access Keys yet".

Could you please create access keys for me? I need:
1. Access Key ID
2. Secret Access Key
3. Bucket name (if NSD dataset is available on MinIO)

Thank you!
```

✅ **SEND THIS NOW** - Don't wait, it might take a day to get response.

---

### **Step 2: Continue Setup (Don't Wait!)**

While waiting for MinIO credentials, set up the project AND RUN IT:

```bash
# Open terminal and run these commands:

# 1. Go to project
cd ~/Desktop/"Bachelor V2"

# 2. Copy config template
cp .env.jupyterhub .env

# 3. Edit your username
nano .env
# Change line: USER=${USER:-student01}
# To:          USER=${USER:-YOUR_ACTUAL_USERNAME}
# Press: Ctrl+X, then Y, then Enter

# 4. Install everything (takes 5-10 minutes)
bash scripts/setup_env.sh

# 5. Wait for installation to complete
# You'll see: "✓ Setup complete!"

# 6. Activate environment
source activate_env.sh

# 7. Check everything works
bash scripts/preflight.sh

# 8. RUN SMOKE TEST (works WITHOUT any data download!)
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

**Expected result**: Pipeline runs successfully! ✅

**IMPORTANT**: Smoke test works WITHOUT downloading 300GB of NSD data!

---

## ⏸️ PAUSE HERE

You've done the important stuff! Now you're waiting for:
- ⏳ Python installation to finish (5-10 min)
- ⏳ Technician to respond (hours to days)

**What to do while waiting:**
- ☕ Get coffee
- 📖 Read `COMPLETE_SETUP_GUIDE.md` (optional)
- 💻 Explore your experiment configs in `configs/experiments/`

---

## 🔄 WHEN TECHNICIAN RESPONDS

### **Option A: They Give You Credentials** ✅

They send you something like:
```
Endpoint: https://ubbc1u.ro:9000
Access Key: your_key_here
Secret Key: your_secret_here
Bucket: nsd-data
```

**Do this:**

```bash
# 1. Copy MinIO config
cd ~/Desktop/"Bachelor V2"
cp .env.minio .env

# 2. Edit with credentials
nano .env

# Update these lines:
# AWS_ENDPOINT_URL=https://ubbc1u.ro:9000  (use their URL)
# AWS_ACCESS_KEY_ID=your_key_here          (use their key)
# AWS_SECRET_ACCESS_KEY=your_secret_here   (use their secret)
# MINIO_BUCKET=nsd-data                    (use their bucket)
# USER=YOUR_USERNAME                        (your username)

# Save: Ctrl+X, Y, Enter

# 3. Test it works
source activate_env.sh
python scripts/test_minio.py

# 4. If test passes, you're done! Run experiment:
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

---

### **Option B: They Say "No MinIO" or No Response** ❌

**That's fine! You have TWO options:**

#### **Option B1: Download ONLY What You Need (RECOMMENDED)** ⭐

You DON'T need 300GB! For bachelor thesis:

📖 **Read**: `MINIMAL_DATA_SETUP.md`

**Summary:**
1. Download ONLY subj01 (~40GB, not 300GB!)
2. Download ONLY first 10,000 images (~10GB)
3. Total: 40GB (NOT 300GB)
4. Perfectly fine for bachelor thesis!

#### **Option B2: Work Without ANY Data Download** 🚀

You can start TODAY without downloading anything:

```bash
# Smoke test works WITHOUT NSD data!
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

Then download subj01 later when you need real results.

---

## 📊 Decision Flowchart

```
START HERE
    ↓
Did you send email to technician?
    ↓
   NO → Send it now! (see Step 1 above)
   YES → Continue ↓
    ↓
Did you run scripts/setup_env.sh?
    ↓
   NO → Run it now! (see Step 2 above)
   YES → Continue ↓
    ↓
Did technician respond?
    ↓
   NO → Keep waiting, read COMPLETE_SETUP_GUIDE.md
    │
   YES → Got credentials?
           ↓
          YES → Configure MinIO (Option A above)
           │
          NO → Download NSD manually (Option B above)
```

---

## 🎯 Today's Goal

By end of today, you should have:
- ✅ Sent email to technician
- ✅ Installed Python environment
- ✅ Verified setup with `preflight.sh`
- ✅ Read `COMPLETE_SETUP_GUIDE.md` (at least first half)

**Tomorrow's goal** depends on technician:
- Got MinIO? → Configure and test
- No MinIO? → Start downloading NSD dataset

---

## 🆘 Stuck? Check This

### **Problem: "setup_env.sh fails"**
```bash
# Run:
bash scripts/setup_env.sh --force

# If still fails, check:
python --version  # Should be 3.11+
pip --version     # Should exist
```

### **Problem: "preflight.sh shows errors"**
```bash
# Read the error message, it tells you what's wrong
# Common: "CUDA not found" → OK for CPU-only setup
# Common: "No space left" → Clean up disk
```

### **Problem: "I don't know my username"**
```bash
# Run this:
whoami
# That's your username!
```

### **Problem: "technician hasn't responded in 3 days"**
```bash
# Just use local setup - don't wait forever
# Follow Option B above (download NSD manually)
```

---

## 📚 What File Should I Read Next?

**Right now** (in order):
1. ✅ This file (you're here!)
2. 📖 `COMPLETE_SETUP_GUIDE.md` - Your main guide
3. 📖 `QUICK_REFERENCE.md` - Keep this handy

**If you get MinIO access:**
4. 📖 `MINIO_STEP_BY_STEP.md` - Beginner MinIO guide
5. 📖 `MINIO_SETUP_GUIDE.md` - Detailed MinIO config

**If you don't get MinIO:**
4. 📖 `DATA_REQUIREMENTS.md` - How to download NSD

---

## ✅ Quick Self-Check

Answer these:

1. **Did I send the technician email?**
   - [ ] Yes → Good!
   - [ ] No → Do it now (takes 2 minutes)

2. **Did I run `setup_env.sh`?**
   - [ ] Yes, it completed → Good!
   - [ ] Yes, it failed → Check errors, ask for help
   - [ ] No → Run it now

3. **Does `source activate_env.sh` work?**
   - [ ] Yes, I see `(venv)` in terminal → Good!
   - [ ] No → Re-run setup_env.sh

4. **Did `preflight.sh` pass?**
   - [ ] Yes, all checks passed → Good!
   - [ ] Some checks failed → Check which ones, might be OK
   - [ ] Didn't run it yet → Run it now

**All checked?** You're on track! ✅

---

## 🎉 Summary

**What you need to do:**

1. ✉️ **Email technician** (2 min) - Do it NOW
2. ⚙️ **Install environment** (10 min) - Run `setup_env.sh`
3. ⏳ **Wait for response** (hours/days) - Read docs meanwhile
4. 🔄 **Configure MinIO OR download NSD** - Depends on response
5. 🚀 **Run experiments** - You're ready!

**You're NOT stuck - you're just at step 3 (waiting).** Keep moving forward! 💪

---

## 💬 Still Confused?

**Answer this one question**: 

**"Did you already email the technician about MinIO access?"**

- **YES, waiting for response** → Perfect! Continue with Step 2 (setup_env.sh) and read COMPLETE_SETUP_GUIDE.md
- **NO, haven't emailed yet** → Send the email NOW (template at top), then do Step 2
- **YES, they said NO** → Use local setup, follow DATA_REQUIREMENTS.md to download NSD
- **YES, they gave me credentials** → Follow "Option A" above to configure MinIO

**You now have a clear action!** 🎯

---

**Next file to read**: `COMPLETE_SETUP_GUIDE.md`

**Don't read the MinIO guides until you have credentials!**
