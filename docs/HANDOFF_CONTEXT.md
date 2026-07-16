# PCD Training Handoff — Complete Context for Continuation

**Date**: July 15, 2026
**Project**: fMRI-to-Image Neural Decoding (FMRI2images)
**Active Branch**: `feature/predictive-cortical-decoder`
**Goal**: Train a novel "Predictive Cortical Decoder" (PCD) architecture on all 8 NSD subjects, targeting a separate architecture paper with neuroscientific insights.

---

## 1. WHAT WE ARE BUILDING

### The Big Picture
We have two papers in progress:
1. **Conformal Prediction Paper** — calibrated uncertainty for neural decoding (mostly complete, needs stronger base model)
2. **Architecture Paper (THIS ONE)** — "Predictive Cortical Decoding": a novel brain-inspired architecture that models the ventral visual stream as a hierarchical prediction-error system

### The PCD Architecture
Instead of treating all brain regions equally (like MindEye/MindEye2), PCD groups the 17 NSD ROIs into 4 hierarchical levels mirroring the ventral visual stream:

- **Level 0 (Early Visual)**: V1v, V1d, V2v, V2d — processes raw visual tokens
- **Level 1 (Mid Visual)**: V3v, V3d, V3A, V3B, V4 — receives prediction errors from Level 0
- **Level 2 (Category-Selective)**: FFA1, FFA2, PPA, EBA, OFA, OPA, RSC — receives prediction errors from Level 1
- **Level 3 (Residual)**: nsdgeneral_other — processes directly (no predictions from above)

Each level has a Transformer encoder. Levels 0→1 and 1→2 have prediction heads that predict the next level's representations. Only the **prediction errors** (actual minus predicted) are fed to the next level. This tests the neuroscience hypothesis that prediction errors along the ventral stream carry more information than raw activations.

The aggregated multi-level output goes through a vMF decoder to produce (mu, kappa) on the CLIP hypersphere.

### Key Novel Contributions
- Hierarchical prediction-error processing (tests predictive coding theory)
- Per-level uncertainty decomposition (which visual hierarchy levels are most/least confident)
- 8-subject joint training with per-subject ROI projections + shared predictive backbone
- Ablation framework to validate hierarchy (full vs no-errors vs reversed vs random)

---

## 2. WHERE THINGS STAND RIGHT NOW

### Training Status: NEEDS RESTART
The PCD_v4 training was running on the H100 pod and completed **24 epochs** with promising metrics before the process was killed (it got stuck on an expensive evaluation step).

**Last training metrics (PCD_v4, epoch 23)**:
- Val R@1 = 5.99%, CSLS R@1 = 9.86%
- Train R@1 at epoch 20 = 30.86%
- Train/val gap ≈ 25% (MUCH better than v1's 82% gap)
- Still improving when killed — was getting new bests every few epochs

**Checkpoint saved**: `experimental_results/PCD_v4_8subject/subj01/checkpoint_last.pt` (epoch 23, from July 15 15:22)

### The Immediate Task
1. **Kill any stale processes** on the pod
2. **Fix the config** to disable the expensive evaluation that causes hangs (see Section 5)
3. **Resume training from the epoch 23 checkpoint** — the model was still improving
4. Let it train for the full 200 epochs (ETA ~30 hours from restart)

---

## 3. INFRASTRUCTURE

### Local Machine (Windows)
- **Workspace**: `D:\ComputaCenter\FMRI2images`
- **Git branch**: `feature/predictive-cortical-decoder`
- **Shell**: PowerShell (use `;` instead of `&&`, no heredocs)

### Remote H100 Pod (Kubernetes)
- **Pod name**: `orchestraiq-jupyter-54644cff87-gz6n2` (check if still this name)
- **kubeconfig**: `C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml`
- **kubectl pattern**: `kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" exec POD_NAME -- bash -c "COMMAND"`
- **kubectl cp pattern**: `kubectl --kubeconfig "..." cp LOCAL_RELATIVE_PATH POD_NAME:/home/jovyan/work/FMRI2images/REMOTE_PATH`
  - IMPORTANT: Use relative paths from workspace root, not absolute Windows paths
- **GPU**: NVIDIA H100 80GB HBM3, CUDA 12.8
- **RAM**: ~1 TB
- **Storage**: `/home/jovyan/work/` (NFS-backed, persistent)
- **Python**: 3.13 (conda base)
- **Repo on pod**: `/home/jovyan/work/FMRI2images/` (same git repo, `feature/predictive-cortical-decoder` branch)

### Key Environment Setup on Pod
```bash
cd /home/jovyan/work/FMRI2images
set -a && source .env && set +a
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
```

### Launching Training on Pod
```bash
nohup python3 -u scripts/training/train_unified.py \
  --config configs/experiments/PCD_v4_8subject.yaml \
  --gpu 0 \
  --resume experimental_results/PCD_v4_8subject/subj01/checkpoint_last.pt \
  > pcd_v4_resumed.log 2>&1 &
```

---

## 4. EXPERIMENT HISTORY (What Was Tried)

### PCD_v1 (configs/experiments/PCD_v1_8subject.yaml)
- **Architecture**: d_model=768, 12 heads, 2 layers/level, dropout=0.1
- **Result**: 167M params. Reached 18% val R@1 but **catastrophic overfitting** (train 99.8%, gap 82%). Stagnated at epoch 29.
- **Diagnosis**: Per-subject ROI projections alone had 96M params (57% of model), memorizing subject-specific patterns. Dropout=0.1 too weak.

### PCD_v2 (configs/experiments/PCD_v2_8subject.yaml)
- **Changes**: d_model=512, 1 layer/level, bottleneck_dim=128, dropout=0.4, noise=0.5, R-Drop from epoch 0
- **Result**: 46.5M params. **Model couldn't learn at all** (0% R@1 after 30 epochs). Early-stopped.
- **Diagnosis**: Too many regularization knobs turned to maximum simultaneously.

### PCD_v3 (configs/experiments/PCD_v3_8subject.yaml)
- **Changes**: Restored d_model=768, layers=2, kept bottleneck=128, moderate reg
- **Result**: **Hung at epoch 2**. The bottleneck in ROIProjection (two sequential linear layers per ROI × 8 subjects × 17 ROIs = 272 sequential CUDA kernel launches in a Python loop) caused hangs.
- **Lesson**: Bottleneck in ROIProjection causes training hangs. Do NOT use bottleneck_dim.

### PCD_v4 (configs/experiments/PCD_v4_8subject.yaml) ← CURRENT
- **Changes**: v1 architecture EXACTLY (no bottleneck), only regularization tuning:
  - dropout: 0.1 → 0.25
  - weight_decay: 0.05 → 0.08
  - label_smoothing: 0.05 → 0.1
  - fmri_noise_std: 0.1 → 0.25
  - voxel_dropout: 0.1 → 0.15
  - kappa_reg: 0.01 → 0.05
  - MixCo alpha: 0.15 → 0.2
- **Result after 24 epochs**: 
  - Val R@1 = 5.99% (epoch 23), improving
  - Train R@1 = 30.86% (epoch 20) — **overfitting gap ~25%** (vs v1's 82%)
  - The regularization IS working but metrics are lower than v1 overall
  - Still in early training; model was improving when killed

### Additional Bug Fixes Applied During This Process
1. **`@property decoder` fix**: PCDModel needs `decoder` and `encoder` properties in `unified_model.py` to proxy calls to `self.pcd.vmf_decoder` and `self.pcd`. Without this, training crashes with `AttributeError: 'PCDModel' object has no attribute 'decoder'`.
2. **Scatter-based `_tokenize_multi`**: The original in-place tensor mutation (`tokens[mask] = ...`) caused CUDA deadlocks. Replaced with cat+sort reassembly.
3. **Level index buffers**: `_level_roi_indices` registered as persistent buffers instead of creating tensors dynamically in forward pass.
4. **Checkpoint loading**: `strict=False` for PCD models in `load_checkpoint()` to handle buffer mismatches.

---

## 5. THE CRITICAL FIX NEEDED BEFORE RESUMING

The config file `configs/experiments/PCD_v4_8subject.yaml` needs these evaluation settings changed:

```yaml
# CURRENT (causes 3-4 hour hangs every ~10 epochs):
evaluation:
  mc_tta_samples: 8          # 8 MC forward passes at eval → expensive
  compute_probabilistic: true  # 64 MC samples × 19K trials → hangs on CPU

# FIX TO:
evaluation:
  mc_tta_samples: 0           # disable MC-TTA during training
  compute_probabilistic: false  # disable probabilistic metrics during training
```

The `compute_probabilistic: true` combined with `mc_samples_prob_metrics: 64` in the inference section triggers ~1.2M forward passes that run on CPU instead of GPU, taking 3-4 hours. This causes the training to appear stuck after certain epochs.

**The local file (Windows) has already been updated with this fix. The pod still has the old version.** You need to:
1. Copy the fixed config to the pod
2. Resume training from the checkpoint

---

## 6. KEY FILES

### Architecture Code
- `src/fmri2img/models/predictive_cortical_decoder.py` — Core PCD module (PredictiveCorticalDecoder, ROIProjection, LevelTransformerEncoder, PredictionHead, HierarchicalAggregator, PerLevelKappaHeads)
- `src/fmri2img/models/unified_model.py` — PCDModel wrapper class, integrates PCD into the UnifiedModel factory. Contains the critical `@property decoder` and `@property encoder` proxies.
- `src/fmri2img/models/vmf_decoder.py` — VonMisesFisherDecoder used by PCD

### Training
- `scripts/training/train_unified.py` — Main training script (handles PCD via model.architecture_type == "pcd")
- `configs/experiments/PCD_v4_8subject.yaml` — Current config (NEEDS the eval fix above)
- `configs/experiments/PCD_v1_8subject.yaml` — Original config (for reference)

### Analysis & Paper (created, awaiting training completion)
- `scripts/analysis/pcd_neuroscience_analysis.py` — Neuroscience analysis script (per-level info, category errors, cross-subject consistency)
- `scripts/analysis/run_pcd_ablations.py` — Ablation study generator
- `scripts/orchestration/run_pcd_finetune.sh` — Phase 2 fine-tuning orchestration
- `scripts/orchestration/run_pcd_ablations.sh` — Ablation orchestration
- `docs/paper/pcd_paper.md` — Initial paper draft

### Checkpoints
- `experimental_results/PCD_v4_8subject/subj01/checkpoint_last.pt` — Last checkpoint (epoch 23, 2.68 GB)
- `experimental_results/PCD_v1_8subject/subj01/checkpoint_last.pt` — v1 checkpoint (epoch 12)

---

## 7. REMAINING PLAN (in order)

### Phase 1: Complete 8-Subject Joint Pretraining ← IMMEDIATE
1. Fix the config (disable compute_probabilistic and mc_tta_samples)
2. Copy fixed config + code to pod
3. Resume from epoch 23 checkpoint
4. Train to 200 epochs or early stopping (patience=30)
5. Expected: val R@1 should reach 15-25% range based on trajectory

### Phase 2: Per-Subject Fine-Tuning
- Config: `configs/experiments/PCD_v1_finetune.yaml` (already created)
- Script: `scripts/orchestration/run_pcd_finetune.sh`
- Fine-tune the pretrained model separately for each of the 8 subjects
- Lower LR, fewer epochs, frozen lower levels

### Phase 3: Ablation Experiments
- Script: `scripts/analysis/run_pcd_ablations.py`
- Test: full PCD vs no-errors vs reversed hierarchy vs random hierarchy vs flat Transformer vs MLP baseline
- This validates the neuroscience hypothesis (prediction errors > raw activations)

### Phase 4: Neuroscience Analysis
- Script: `scripts/analysis/pcd_neuroscience_analysis.py`
- Extract: per-level information contribution, category-conditional error maps, cross-subject consistency, error-kappa correlation
- These generate the key figures for the paper

### Phase 5: Paper Draft
- Draft: `docs/paper/pcd_paper.md` (initial version exists)
- Update with actual results from Phases 1-4

---

## 8. KNOWN GOTCHAS

1. **PowerShell on Windows**: Use `;` not `&&` for command chaining. No heredoc support.
2. **kubectl cp paths**: Use relative paths from workspace root, not absolute Windows paths.
3. **Auto-review blocks**: kubectl commands that mutate the pod (cp, exec with kill/nohup) get blocked by auto-review. You'll need to retry with `request_smart_mode_approval=true`.
4. **Pod name may change**: If the pod was restarted, the name `orchestraiq-jupyter-54644cff87-gz6n2` may be different. Check with `kubectl get pods`.
5. **tqdm must be disabled**: `TQDM_DISABLE=1` prevents massive NFS log file writes that can kill the process.
6. **PYTHONUNBUFFERED=1**: Required for real-time log output.
7. **OMP_NUM_THREADS=4**: Prevents CPU thread oversubscription.
8. **Bottleneck in ROIProjection**: Do NOT use `bottleneck_dim` — it causes training hangs. The bottleneck code exists in the codebase but should not be activated via config.
9. **PCDModel.decoder property**: Essential for training loop compatibility. If unified_model.py is ever reverted to git, this fix is lost and training crashes immediately.
10. **Code on pod vs git**: The pod has locally-copied files that may differ from git HEAD. Always copy the latest from Windows after any code changes.
11. **Two files must always be in sync on pod**: `src/fmri2img/models/predictive_cortical_decoder.py` AND `src/fmri2img/models/unified_model.py` — they have interdependent changes (bottleneck param, decoder property).

---

## 9. METRICS REFERENCE

### PCD_v4 Training Trajectory (24 epochs completed)
| Epoch | Val R@1 | CSLS R@1 | Train R@1 | MedR | Status |
|---|---|---|---|---|---|
| 5 | 0.17% | 0.23% | — | 827 | warmup |
| 10 | 2.03% | 3.42% | 11.33% | 88 | learning |
| 14 | 5.09% | 8.63% | — | 37 | best so far |
| 20 | 5.09% | 9.04% | 30.86% | 38 | plateau? |
| 23 | 5.99% | 9.86% | — | 32 | new best |

### Comparison: v1 at same epochs
| Epoch | v1 Val R@1 | v1 Train R@1 | v1 Gap |
|---|---|---|---|
| 10 | 4.57% | — | — |
| 15 | 12.64% | 93.36% | 81% |
| 20 | 15.49% | — | — |
| 29 | 17.97% | 98.73% | 81% |

v4 has lower val R@1 (6% vs 18%) but MUCH lower overfitting (25% gap vs 82%). With more training and the model still improving, v4 should reach a reasonable level.

### SOTA Benchmarks (for context)
| Method | PixCorr | SSIM | Alex(2) | Alex(5) |
|---|---|---|---|---|
| MindEye | 0.309 | 0.323 | 0.947 | 0.978 |
| MindEye2 | 0.320 | 0.341 | 0.960 | 0.983 |
| Brain Diffuser | 0.254 | 0.356 | 0.942 | 0.962 |

Our best single-subject model (non-PCD): 77.2% R@1 on SHARED1000.

---

## 10. PROJECT RULES (from .cursor/rules/)

- Never fabricate experimental numbers — all metrics must trace to source files
- Mathematical notation uses LaTeX: `\mu`, `\kappa`, `\mathcal{L}`
- Every module needs type hints, docstring, and test
- All experiments specified by YAML config — no magic numbers
- Library code goes in `src/fmri2img/`, CLI code in `scripts/`
- Use `logging`, not `print`
- Loss math runs in float32 even under AMP
- `pathlib.Path` for filesystem, never hardcoded paths
- Cite as Author et al., Year

---

## 11. QUICK START COMMANDS

### Check pod status
```bash
kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" get pods
```

### Check if training is running
```bash
kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" exec POD_NAME -- bash -c "ps aux | grep train_unified | grep -v grep"
```

### Check training progress
```bash
kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" exec POD_NAME -- bash -c "grep -E 'Epoch [0-9]+/|R@1=.*R@5|New best|Train retrieval' /home/jovyan/work/FMRI2images/LOG_FILE.log | tail -30"
```

### Copy file to pod
```bash
kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" cp RELATIVE_PATH POD_NAME:/home/jovyan/work/FMRI2images/REMOTE_PATH
```

### Start training
```bash
kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" exec POD_NAME -- bash -c "cd /home/jovyan/work/FMRI2images && set -a && source .env && set +a && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && nohup python3 -u scripts/training/train_unified.py --config configs/experiments/PCD_v4_8subject.yaml --gpu 0 --resume experimental_results/PCD_v4_8subject/subj01/checkpoint_last.pt > pcd_v4_resumed.log 2>&1 & echo STARTED"
```

### Kill training
```bash
kubectl --kubeconfig "C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml" exec POD_NAME -- bash -c "pkill -9 -f train_unified"
```
