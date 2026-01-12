# MinIO Setup - Step by Step (For Beginners)

> **You're looking at MinIO and don't know what to do. Start here!**

---

## 🎯 What You Need to Do (Simple Version)

**Goal**: Get access to the NSD dataset on MinIO instead of downloading 300GB

**Time needed**: 15-30 minutes (mostly waiting for technician)

---

## 📝 Step 1: Talk to Your Technician

You need to ask them to **create access keys** for you. Here's exactly what to say:

### **Message to Send:**

```
Hi [Technician Name],

I need access to the MinIO Object Store for my bachelor thesis. 
Could you please create access keys for me?

I need:
1. Access Key ID
2. Secret Access Key  
3. The bucket name where the NSD dataset is stored (if available)
4. The MinIO endpoint URL

The MinIO interface shows "There are no Access Keys yet" so I think 
you need to create one for my user account.

Thank you!
```

---

## 📸 What You're Seeing Now

Your screenshots show:
1. **First image**: "Access Keys" page - **"There are no Access Keys yet"**
2. **Second image**: "Buckets" page - Loading, no buckets visible yet

**This is normal!** You need the administrator to:
- Create access keys for you
- OR give you the existing keys
- Maybe create a bucket for NSD data (if not done yet)

---

## ⏳ While You Wait for Technician

### **Option A: Continue with Local Setup (Recommended)**

Don't block your progress! Set up the project with local configuration:

```bash
# 1. Navigate to project
cd ~/Desktop/"Bachelor V2"

# 2. Use the JupyterHub config (local paths)
cp .env.jupyterhub .env

# 3. Edit your username
nano .env
# Change: USER=${USER:-student01}
# To:     USER=${USER:-your_actual_username}
# Save: Ctrl+X, Y, Enter

# 4. Set up Python environment
bash scripts/setup_env.sh

# 5. Activate environment
source activate_env.sh

# 6. Verify setup works
bash scripts/preflight.sh
```

**This sets everything up so you're ready to go - with or without MinIO!**

### **Option B: Explore MinIO Documentation**

While waiting, you can read:
- `MINIO_QUICK_ANSWER.md` - Quick overview
- `MINIO_SETUP_GUIDE.md` - Detailed guide (for later)

But **don't worry about them now** - come back after you get credentials.

---

## ✅ Once Technician Gives You Credentials

They should give you something like:

```
MinIO Endpoint: https://ubbc1u.ro:9000
Access Key ID: minioadmin_tony
Secret Key: SomeSecretPassword123
Bucket: nsd-data
```

### **Then Do This:**

```bash
# 1. Copy MinIO config template
cd ~/Desktop/"Bachelor V2"
cp .env.minio .env

# 2. Edit with your credentials
nano .env

# 3. Find these lines and update:
AWS_ENDPOINT_URL=https://ubbc1u.ro:9000          # Use what they gave you
AWS_ACCESS_KEY_ID=minioadmin_tony                # Use what they gave you
AWS_SECRET_ACCESS_KEY=SomeSecretPassword123      # Use what they gave you
MINIO_BUCKET=nsd-data                            # Use what they gave you
USER=your_username                                # Your actual username

# 4. Save: Ctrl+X, Y, Enter

# 5. Test connection
source activate_env.sh
python scripts/test_minio.py

# 6. If test passes, verify data
python scripts/verify_dataset.py --allow-s3-only

# 7. Run first experiment
bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
```

---

## 🤔 What If Technician Says "No MinIO Access" or "NSD Not on MinIO"?

**That's fine!** Use local setup instead:

### **Plan B: Local Setup (Classic Approach)**

```bash
# 1. Use local configuration
cp .env.jupyterhub .env
nano .env  # Edit USER=your_username

# 2. Follow COMPLETE_SETUP_GUIDE.md
# It has step-by-step instructions for downloading NSD dataset manually

# 3. Main difference:
#    - With MinIO: No download needed (5 min setup)
#    - Without MinIO: Download 300GB dataset (2-8 hours)
```

**Both approaches work perfectly!** MinIO just saves you the download time.

---

## 🎓 Understanding MinIO (Simple Explanation)

### **What is MinIO?**
Think of it like **Dropbox or Google Drive, but for your university**.

- Instead of everyone downloading 300GB of NSD data
- One copy sits on the MinIO server
- Everyone accesses it over the network
- Your code caches what it needs locally (~20-50GB)

### **Why Use It?**
✅ No 300GB download  
✅ Saves disk space  
✅ Shares data with other students  
✅ Faster setup (5 min vs 8 hours)  

### **Why Not Use It?**
❌ Needs network access  
❌ Slightly slower first run (caches locally after)  
❌ Requires IT setup  

---

## 🚦 Decision Tree

```
Do you have MinIO credentials?
│
├─ NO → Ask technician
│       │
│       ├─ They give you credentials → Follow "Once Technician Gives You Credentials" above
│       │
│       └─ They say "No MinIO" → Use local setup (Plan B above)
│
└─ YES → Follow "Once Technician Gives You Credentials" above
```

---

## 🆘 Common Questions

### **Q: Is MinIO required for this project?**
**A:** No! It's optional. Your code works with:
- Local disk (download NSD dataset)
- MinIO/S3 (stream from server)
- Hybrid (mix of both)

### **Q: What if I can't get MinIO access?**
**A:** No problem! Use the local setup:
1. Follow `COMPLETE_SETUP_GUIDE.md`
2. Download NSD dataset manually
3. Everything works the same way

### **Q: Can I switch later?**
**A:** Yes! You can:
- Start with MinIO, switch to local later
- Start with local, switch to MinIO later
- Use both (hybrid approach)

### **Q: The technician created a bucket but no access keys. What now?**
**A:** You still need access keys to connect. Ask them to:
1. Click "Create access key" button (in your first screenshot)
2. Generate a new key pair
3. Give you both the Access Key ID and Secret Key

### **Q: I see the MinIO interface but I'm not an administrator**
**A:** That's fine! You're logged in as a user. You need:
- Administrator to create access keys for you
- OR administrator to give you existing keys

### **Q: How do I know if NSD data is already on MinIO?**
**A:** After you get credentials, run:
```bash
python scripts/test_minio.py
```
It will list all buckets and check for NSD data structure.

---

## 🎯 What to Do RIGHT NOW

### **Immediate Action (Next 5 Minutes):**

1. **Send message to technician** (copy the template above)
2. **Continue with local setup** while waiting:
   ```bash
   cd ~/Desktop/"Bachelor V2"
   cp .env.jupyterhub .env
   nano .env  # Set your username
   bash scripts/setup_env.sh
   source activate_env.sh
   bash scripts/preflight.sh
   ```

3. **Read** `COMPLETE_SETUP_GUIDE.md` - your main guide

### **After Technician Responds:**

- **If they give credentials**: Follow "Once Technician Gives You Credentials" section
- **If they say no MinIO**: Continue with local setup (download NSD dataset)
- **If they ask what you need**: Show them the message template

---

## 📚 Which Guide Should I Read?

Right now, you're confused because there are many guides. Here's the priority:

### **Read NOW (before anything else):**
1. ✅ **This file** (`MINIO_STEP_BY_STEP.md`) - You're reading it!
2. ✅ **COMPLETE_SETUP_GUIDE.md** - Main setup guide (works with or without MinIO)

### **Read AFTER you get MinIO credentials:**
3. `MINIO_QUICK_ANSWER.md` - Quick MinIO overview
4. `MINIO_SETUP_GUIDE.md` - Detailed MinIO configuration

### **Reference (keep handy):**
5. `QUICK_REFERENCE.md` - Command cheat sheet
6. `DATA_REQUIREMENTS.md` - Data/model requirements

---

## ✅ Summary

**You're at a decision point:**

```
┌─────────────────────────────────────┐
│     Try to Get MinIO Access         │
│  (Ask technician - might save time) │
└──────────────┬──────────────────────┘
               │
               ├─ Got credentials? 
               │  └─→ Use MinIO (5 min setup)
               │
               └─ No credentials?
                  └─→ Use local setup (8 hour download)
```

**Both paths lead to success! MinIO just saves download time.**

---

## 🚀 Next Steps (Clear Action Plan)

**Step 1**: Send email to technician (NOW)  
**Step 2**: Continue with local setup while waiting (DON'T BLOCK)  
**Step 3**: When technician replies:
- Got keys? → Configure MinIO
- No keys? → Download NSD dataset

**You cannot fail!** Both approaches work. 🎉

---

## 💬 Need Help?

If you're still confused, answer these questions:

1. **Did you send the message to the technician?** 
   - [ ] Yes → Wait for response, continue with local setup
   - [ ] No → Send it now (use template above)

2. **Did you set up the Python environment?**
   - [ ] Yes → Good! Run `bash scripts/preflight.sh`
   - [ ] No → Run the "While You Wait" commands above

3. **Do you have NSD dataset downloaded locally?**
   - [ ] Yes → You can skip MinIO entirely!
   - [ ] No → Either wait for MinIO or start downloading

**Whatever your answers, you have a clear next step!** 🎯

---

**TL;DR**: 
1. Ask technician for MinIO access keys
2. Meanwhile, set up Python environment (local config)
3. When you get keys, configure MinIO
4. If no keys, download NSD dataset
5. Either way, you'll be running experiments soon!

**You've got this!** 💪
