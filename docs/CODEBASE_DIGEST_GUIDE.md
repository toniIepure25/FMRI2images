# Codebase Digest Guide

**Purpose**: Guide for generating optimized `codebase.md` for ChatGPT using `npx ai-digest`

---

## 🎯 What's Included in codebase.md

The `.aidigestignore` file has been optimized to include ONLY the essential code for understanding the production pipeline.

### ✅ Included Files (Perfect for ChatGPT)

#### 1. **Core Pipeline Script**
```
scripts/run_production.sh (main orchestrator - 874 lines)
```

#### 2. **Production Scripts** (9 scripts used by run_production.sh)
```
scripts/build_full_index.py          - Step 1: Build NSD index
scripts/build_clip_cache.py          - Step 2: Build 512-D CLIP cache
scripts/train_mlp.py                 - Step 3: Train MLP encoder
scripts/build_target_clip_cache_robust.py - Step 4: Build 1024-D target cache
scripts/build_target_clip_cache.py   - Step 4: Fallback builder
scripts/train_clip_adapter.py        - Step 5: Train adapter (512→1024)
scripts/decode_diffusion.py          - Step 6: Generate images
scripts/eval_reconstruction.py       - Step 7: Evaluate results
scripts/compare_evals.py             - Step 8: Compare metrics
```

#### 3. **Configuration Files**
```
configs/production_optimal.yaml      - Main config (k=100 PCA)
configs/production_improved.yaml     - Alternative config
```

#### 4. **Source Code** (src/fmri2img/)
```
src/fmri2img/__init__.py
src/fmri2img/data/                   - Dataset loaders, preprocessing
src/fmri2img/models/                 - MLP, adapter, encoders
src/fmri2img/eval/                   - Evaluation metrics
src/fmri2img/io/                     - I/O utilities
src/fmri2img/utils/                  - Helper functions
```

#### 5. **Essential Documentation**
```
README.md                                    - Project overview
docs/ARCHITECTURE_IMPROVEMENTS.md            - Improvement ideas (NEW)
docs/IMPROVEMENTS_READY.md                   - Ready-to-run guide (NEW)
docs/OPTIMAL_CONFIGURATION_GUIDE.md          - Configuration details
docs/MLP_IMPLEMENTATION.md                   - Architecture details
docs/DIFFUSION_DECODER.md                    - Diffusion pipeline
docs/PREPROCESSING_IMPLEMENTATION.md         - Preprocessing details
docs/RIDGE_BASELINE.md                       - Baseline comparison
docs/GETTING_STARTED_DIFFUSION.md            - Setup guide
```

#### 6. **Setup Files**
```
pyproject.toml                       - Package configuration
requirements.txt                     - Python dependencies
Makefile                             - Build commands
```

---

## ❌ Excluded Files (Not Relevant)

### Data & Outputs
- All data files: `*.npy`, `*.parquet`, `*.hdf5`, `*.csv`, etc.
- All outputs: `outputs/`, `checkpoints/`, `reports/`, `recon/`
- All caches: `cache/`, `clip_cache/`, etc.

### Test & Experimental Scripts
- `scripts/test_*.py` - Test scripts
- `scripts/verify_*.py` - Verification scripts
- `scripts/check_*.py` - Checking scripts
- `scripts/ablate_*.py` - Ablation experiments
- `scripts/nsd_*.py` - Old NSD-specific scripts
- `scripts/plot_*.py` - Plotting utilities
- `scripts/summarize_*.py` - Summary utilities

### Non-Essential Configs & Docs
- `configs/clip.yaml`, `configs/data.yaml`, `configs/logging.yaml`
- `docs/ABLATION_SUMMARY.md` - Historical ablation results
- `docs/DIFFUSION_SUMMARY.md` - Old diffusion notes
- `docs/SKLEARN_DEPENDENCY_FIX.md` - Technical fix doc
- `CLEANUP_SUMMARY.md` - Old cleanup notes

### IDE, Logs, Temp Files
- `.vscode/`, `.idea/`, `logs/`, `tmp/`
- `*.log`, `*.tmp`, `*.bak`
- `.DS_Store`, `Thumbs.db`

---

## 🚀 Usage

### Generate codebase.md
```bash
cd "/home/tonystark/Desktop/Bachelor V2"
npx ai-digest
```

### Expected Output
```
✅ codebase.md generated (~50-100KB)
   - 1 main pipeline script (run_production.sh)
   - 9 production Python scripts
   - 2 configuration files
   - Full src/fmri2img/ package
   - 9 essential documentation files
   - Setup/config files
```

### File Size Estimate
- **Total**: ~50-100KB (perfect for ChatGPT context window)
- **Scripts**: ~20-30KB (10 files)
- **Source code**: ~15-25KB (src/fmri2img/)
- **Configs**: ~5KB (2 YAML files)
- **Docs**: ~15-30KB (9 markdown files)
- **Setup**: ~5KB (requirements, pyproject.toml)

---

## 💡 What ChatGPT Can Understand

With this `codebase.md`, ChatGPT will understand:

### 1. **Complete Pipeline Architecture**
- All 8 steps of run_production.sh
- Dependencies between steps
- How data flows through the pipeline
- Configuration loading and validation

### 2. **Code Implementation Details**
- MLP encoder architecture (100-D → [3072, 1536] → 512-D)
- CLIP adapter architecture (512-D → 1024-D)
- Preprocessing pipeline (reliability masking, PCA, scaling)
- Diffusion generation (SD-2.1, 250 steps, guidance 7.5)
- Evaluation metrics (cosine similarity, retrieval)

### 3. **Current Configuration**
- PCA k=100 (was k=3 - critical improvement!)
- MLP hidden dims: [3072, 1536]
- Training: batch=128, epochs=100, patience=20
- Diffusion: steps=250, guidance=7.5, scheduler=dpm
- Expected performance: 0.80-0.85 cosine

### 4. **Improvement Ideas**
- Multi-scale processing [k=10, 50, 100]
- Transformer encoder architecture
- ResNet-style encoder
- Variational encoder
- Attention-based feature selection

### 5. **How to Run & Troubleshoot**
- Full retraining command
- Expected timeline (2.5-3.5 hours)
- What to monitor during training
- Common issues and solutions

---

## 📊 ChatGPT Context Quality

### Perfect For:
✅ Understanding complete pipeline logic  
✅ Debugging script errors  
✅ Suggesting architecture improvements  
✅ Explaining configuration choices  
✅ Reviewing code quality  
✅ Planning new features  
✅ Understanding data flow  
✅ Comparing with literature  

### Not Included (By Design):
❌ Actual data (too large)  
❌ Model weights (binary files)  
❌ Generated images (outputs)  
❌ Experimental code (test scripts)  
❌ Historical docs (old notes)  

---

## 🔄 Keeping codebase.md Updated

### When to Regenerate:
1. After modifying any production script
2. After updating configs (production_optimal.yaml)
3. After changing src/fmri2img/ code
4. After adding new essential documentation
5. Before starting a new ChatGPT session

### Quick Update:
```bash
# After making changes to code
npx ai-digest

# Upload new codebase.md to ChatGPT
# Now ChatGPT has latest code context!
```

---

## 🎯 Example ChatGPT Prompts

With the generated `codebase.md`, you can ask ChatGPT:

### Architecture Questions:
```
"Looking at codebase.md, explain how the MLP encoder processes 
100-D PCA features and why we chose [3072, 1536] hidden dims."

"Analyze the adapter training in train_clip_adapter.py - is the 
architecture optimal for mapping 512-D to 1024-D?"

"Review the preprocessing pipeline - why is PCA k=100 so critical 
compared to k=3?"
```

### Debugging:
```
"I'm getting OOM errors during MLP training. Based on the code in 
train_mlp.py, what batch size should I use?"

"The adapter test cosine is 0.85 but images still look poor. Looking 
at decode_diffusion.py, what could be wrong?"
```

### Improvements:
```
"Based on the code in src/fmri2img/models/, how would you implement 
a Transformer encoder to replace the MLP?"

"I want to add multi-scale processing [k=10, 50, 100]. Show me how 
to modify the preprocessing and MLP code."
```

### Comparison:
```
"Compare my pipeline in codebase.md with the Brain-Diffusion paper. 
What are the key differences?"

"Is my configuration in production_optimal.yaml state-of-the-art 
for brain decoding?"
```

---

## 📝 Tips for ChatGPT Sessions

### 1. Start with Context
```
"I'm uploading codebase.md which contains my brain decoding pipeline. 
Please review it and summarize the architecture."
```

### 2. Reference Specific Files
```
"Looking at scripts/train_mlp.py in codebase.md, the loss function 
uses cosine + MSE + triplet. Is this optimal?"
```

### 3. Ask for Implementation
```
"Based on the existing code structure in src/fmri2img/models/, 
implement a Transformer encoder class."
```

### 4. Request Analysis
```
"Analyze the complete pipeline in codebase.md and identify the 
top 3 bottlenecks for quality improvement."
```

---

## ✅ Verification Checklist

After running `npx ai-digest`, verify:

- [ ] `codebase.md` file exists
- [ ] File size is 50-100KB (not too large, not too small)
- [ ] Contains run_production.sh
- [ ] Contains all 9 production scripts
- [ ] Contains production_optimal.yaml
- [ ] Contains src/fmri2img/ package code
- [ ] Contains essential docs (9 files)
- [ ] Does NOT contain any data files
- [ ] Does NOT contain any outputs/checkpoints
- [ ] Does NOT contain test scripts

### Quick Check:
```bash
# Check codebase.md exists and size
ls -lh codebase.md

# Count included files (should be ~30-50 files)
grep -c "^# File:" codebase.md

# Verify no data files included
grep "\.npy\|\.parquet\|\.hdf5" codebase.md && echo "⚠️ Data files found!" || echo "✅ No data files"
```

---

## 🎉 Result

You now have a perfectly curated `codebase.md` that:
- Contains ALL essential pipeline code
- Excludes ALL irrelevant data/outputs
- Is perfectly sized for ChatGPT's context window
- Enables ChatGPT to fully understand your pipeline
- Can be updated easily as code changes

**Generate it**: `npx ai-digest`  
**Upload to ChatGPT**: Attach `codebase.md`  
**Start asking**: ChatGPT now understands your complete pipeline!

---

**Questions?** See `.aidigestignore` for detailed inclusion/exclusion rules.
