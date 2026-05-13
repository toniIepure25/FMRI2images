# Live Artifact Audit — Cortex2Canvas Demo

**Date:** 2026-05-12
**Repository:** /home/tonystark/Desktop/Bachelor V2
**Kubernetes pod:** orchestraiq-jupyter-54644cff87-nxd9x (namespace: runai-romania-dev)

---

## Selected Best Model/Artifact Strategy

**Strategy:** V62a 768-D MLP, single-model live inference

**Reason:**
- Backend only supports single-model inference (no fusion, no MC-TTA)
- Local and remote CLIP galleries are both 768-D (no 197K-D gallery available)
- V62a is compatible with the existing gallery and backend
- V61a 197K-D would require: 197K-D gallery, token-level CLIP cache, dual-head model support, MC-TTA wrapper

**Leaderboard evidence:**
| Model | CSLS R@1 | Counterspace | Notes |
|-------|----------|-------------|-------|
| V61a MC-TTA(16) k=10 | 80.8% | 197K-D | Not compatible with backend |
| V61a MC-TTA(16) k=3 | 83.1% | 197K-D | Not compatible with backend |
| V61a deterministic | 79.1% | 197K-D | Not compatible with backend |
| V62a | 48.3% | 768-D | **Currently deployed** |
| Best fusion V61a+V62a | 84.6% | cross-space | Not compatible (needs fusion scorer) |

---

## Local Backend Capability

| Feature | Supported? |
|---------|-----------|
| V62a 768-D MLP | Yes |
| V61a 197K-D dual-head | No (needs 197K-D gallery + token-targets support) |
| MC-TTA | No (backend only does single deterministic forward pass) |
| Fusion V61a + V62a | No (single model only) |
| CSLS k=3 override | No (hardcoded to k=20, csls_k=10) |
| 10k gallery | Yes |
| Live reconstruction | No (reconstruction_available always False) |
| z-score stats | Optional (not currently loaded; pre-extracted features are pre-normalized) |
| Live SSE streaming | Yes (/api/infer-stream/{trial_idx}) |
| Trial index | Yes (/api/trials) |
| nsdId → trial lookup | Yes (/api/nsd-id-to-trial/{nsd_id}) |

---

## Remote Discovery

| Artifact | Remote Path | Size | Leaderboard Evidence | Copied? |
|----------|------------|------|---------------------|---------|
| V61a checkpoint_best.pt | `/home/jovyan/work/FMRI2images/experimental_results/V61a_finetune_difflr/subj01/checkpoint_best.pt` | 21 GB | CSLS R@1=79.1%, MC-TTA=80.8% | No (too large; for reference) |
| V61a checkpoint_last.pt | same dir | ~21 GB | — | No |
| V61a config.yaml | same dir | <1 KB | dual_head, 197K-D, softclip | No |
| V61a MC-TTA metrics | same dir/metrics/shared1000_mctta16_metrics.json | <1 KB | k10=80.8%, k3=83.1% | No |
| V62a checkpoint_best.pt | `/home/jovyan/work/FMRI2images/experimental_results/V62a_cls_retrieval_768d/subj01/checkpoint_best.pt` | 8.3 GB | CSLS R@1=48.3% | No (local V62a_model_only.pt is equivalent) |
| CLIP gallery (768-D) | `/home/jovyan/work/FMRI2images/outputs/clip_cache/clip.parquet` | 30 MB | 10,000 images | Already exists locally |
| Token CLIP cache | `/home/jovyan/work/FMRI2images/outputs/clip_cache/tokens_ViT-L-14_projected.h5` | 2.1 GB | 257×768 per image | Not needed for V62a |

---

## Local Installation

| Artifact | Local Path | Exists? | Size | Purpose |
|----------|-----------|---------|------|---------|
| V62a_model_only.pt | `demo_thesis/local_backend_data/checkpoints/V62a_model_only.pt` | Yes | 2.1 GB | MLP encoder, 768-D, ~554M params, VMF decoder |
| fmri_features.npy | `demo_thesis/local_backend_data/subj01/fmri_features.npy` | Yes | 1.8 GB | 30,000 × 15,724 voxels |
| index.parquet | `demo_thesis/local_backend_data/subj01/index.parquet` | Yes | 851 KB | 30,000 trial rows, subject/session/nsdId |
| clip.parquet | `demo_thesis/local_backend_data/clip/clip.parquet` | Yes | 30 MB | 10,000 × 768-D CLIP embeddings |

---

## Health Check Result

```
Status:            inference_ready
Device:            cuda (NVIDIA GeForce RTX 3080 Laptop GPU)
Model Loaded:      true
Features Loaded:   true (30000 × 15724)
Gallery Loaded:    true (10000 × 768-D)
Trial Index:       true (30000 rows)
Z-Score Stats:     false (pre-extracted features are pre-normalized)
Inference:         true
Live Retrieval:    true
Reconstruction:    false
Missing:           []
Errors:            {}
Model Meta:
  Encoder:         mlp
  Hidden:          [8192, 8192, 4096, 2048]
  Input Dim:       15724
  Embedding Dim:   768
  Kappa Mode:      softplus
  Model Type:      vmf
```

## Live Inference Result

```
Endpoint:           /api/infer/0 (trial 0)
Top-1 Rank:         1
Top-1 nsdId:        62175
Top-1 CSLS:         0.2961
Kappa:              27.06
Delta:              None (V62a MLP has no per-ROI delta)
Gallery Size:       10,000
Latency:            <1s total
Mode:               live (model + features + gallery all loaded)
```

---

## Missing for Full Live Pipeline

1. **V61a 197K-D encoder support** — Backend creates model via `create_model()` from fmri2img, which can technically build dual_head models, but:
   - Needs 197K-D gallery (token-level embeddings)
   - Needs token cache h5 file
   - Needs config updates for dual_head encoder type

2. **MC-TTA inference wrapper** — Backend does single forward pass. MC-TTA requires multiple passes with dropout enabled and averaging.

3. **Fusion scorer V61a + V62a** — Two models + cross-space gallery + z-score normalization of scores. Significant backend refactor.

4. **CSLS k=3 override** — Hardcoded to k=20, csls_k=10 in `_csls_retrieval()`. Would need env var support.

5. **197K-D gallery embeddings** — Not available locally or remotely. Would need to be generated from 197K-D CLIP model.

6. **Z-score stats** — Not loaded (pre-extracted features are pre-normalized). Could add for completeness but not needed for current setup.

7. **Reconstruction model/assets** — Stable Diffusion backend not configured. `reconstruction_available` always False.

8. **Stable Diffusion backend** — Not implemented in demo_thesis/backend.

9. **Live image reconstruction** — Not supported.

10. **PYTHONPATH fix** — Backend needs `PYTHONPATH=../src` to find fmri2img package. Should add to .env or backend startup script.

---

## Recommended Next Steps

**Priority 1 — Immediate (done):**
- ✅ V62a single-model live retrieval: Working
- ✅ Trial lookup: Working
- ✅ Gallery search with CSLS: Working
- ✅ Health endpoint: Working

**Priority 2 — Near-term:**
- Fix PYTHONPATH for production backend startup (add to env or startup script)
- Consider downloading V61a 197K-D gallery if available from another source
- Add CSLS k env override (minor code change)

**Priority 3 — Future:**
- Implement MC-TTA support in backend
- Implement V61a + V62a fusion scoring
- Add Stable Diffusion reconstruction
- UI provenance updates for live mode

---

## Commands to Start Backend/Frontend

```bash
# Backend
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
PYTHONPATH=../src python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
npm run dev  # http://localhost:3000
```

## Verification Commands

```bash
# Typecheck
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis && npm run typecheck

# Build
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis && npm run build

# Artifact verification
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis && python3 backend/verify_artifacts.py

# Health check
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool

# Smoke test (when server is running)
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis && python3 backend/smoke_test_backend.py
```
