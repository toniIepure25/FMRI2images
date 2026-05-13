# Full Live Pipeline Audit — Cortex2Canvas Demo v4.0

**Date:** 2026-05-12
**Backend version:** 4.0 (multi-model, MC-TTA, fusion, provenance-audited)

---

## 1. Executive Summary

| Category | Status | Details |
|----------|--------|---------|
| **Model forward** | LIVE_LOCAL | V62a MLP (554M params, CUDA) runs live per request on RTX 3080 Laptop |
| **vMF projection / κ** | LIVE_LOCAL | Kappa extracted from live model output |
| **CSLS gallery search** | LIVE_LOCAL | 10,000 × 768-D gallery cosine + CSLS per request |
| **Top-K** | LIVE_LOCAL | Ranked output from live CSLS |
| **fMRI data** | CACHED_LOCAL | Pre-extracted from fmri_features.npy (30k trials × 15,724 voxels) |
| **Preprocessing** | CACHED_LOCAL | Features pre-normalized; z-score stats not loaded |
| **ROI masking** | CACHED_LOCAL | Features pre-masked to nsdgeneral ROI |
| **Reconstruction** | UNAVAILABLE | No local Stable Diffusion backend |
| **Trial metrics (PixCorr/SSIM)** | UNAVAILABLE | Only available from cached JSON replay |
| **V61a / 197K-D** | UNAVAILABLE | V61a checkpoint exists remotely (21 GB) but 197K-D gallery not available |
| **MC-TTA** | CODE_READY | Implemented in backend, needs V61a checkpoint + 197K gallery |
| **Fusion V61a+V62a** | CODE_READY | Implemented in backend, needs V61a + both galleries |
| **Backend fallback** | Verified | fusion → v62_single fallback works, reports reason |

---

## 2. Pipeline Step Truth Table

| UI Step | Backend Source | Live? | Source | Provenance Label |
|---------|---------------|-------|--------|-----------------|
| Load fMRI betas | `fmri_features.npy[trial_idx]` (indexed live) | CACHED_LOCAL | Pre-extracted .npy | CACHED_LOCAL |
| Preprocessing | `_apply_zscore()` or skip | CACHED_LOCAL | Pre-normalized features | CACHED_LOCAL |
| ROI masking | N/A (pre-masked) | CACHED_LOCAL | Pre-masked in .npy | CACHED_LOCAL |
| MLP encoder | `_model_forward()` → model(x) | LIVE_LOCAL | PyTorch on CUDA | LIVE_LOCAL |
| vMF projection | Kappa from model output | LIVE_LOCAL | Same forward pass | LIVE_LOCAL |
| CSLS gallery search | `_csls_retrieval()` over 10K gallery | LIVE_LOCAL | NumPy on CPU | LIVE_LOCAL |
| Top-K hypotheses | Sorted CSLS output | LIVE_LOCAL | Same as CSLS | LIVE_LOCAL |
| Reconstruction | None | UNAVAILABLE | No SD backend | UNAVAILABLE |
| κ parameter | From model output | LIVE_LOCAL | VMF decoder head | LIVE_LOCAL |
| δ parameter | Model dependent | UNAVAILABLE | V62a MLP has no delta | UNAVAILABLE |
| PixCorr / SSIM | From cached JSON | CACHED_LOCAL | Replay data only | CACHED_LOCAL (or UNAVAILABLE) |

---

## 3. Backend Modes Tested

| Mode | Status | Fallback? | Reason |
|------|--------|-----------|--------|
| `v62_single` | Working | No | All artifacts present (checkpoint + fmri + gallery) |
| `v61_single` | Not tested | Would fallback | V61a checkpoint + 197K gallery missing locally |
| `v61_mctta` | Not tested | Would fallback | Same as above |
| `fusion_v61_v62` | Fallback | Yes → v62_single | V61a model missing |
| Backend-down | Working | Replay mode | Frontend detects backend unreachable |

---

## 4. Health Check (v62_single mode)

```
Status:            inference_ready
Device:            cuda (RTX 3080 Laptop)
Backend mode:      v62_single
Effective mode:    v62_single
Fallback:          false
V62a loaded:       true (MLP encoder, 768-D, softplus kappa)
V61a loaded:       false
Gallery 768:       true (10000 × 768-D)
Gallery 197K:      false
Features:          true (30000 × 15724)
Trial index:       true (30000 rows)
Z-score stats:     false
Live retrieval:    true
MC-TTA available:  false
Fusion available:  false
Reconstruction:    false (unavailable)
Missing:           []
```

---

## 5. Inference Result (trial 0, v62_single)

```
Rank:             #1
nsdId:            62175
CSLS:             0.2961
Kappa:            27.06
Delta:            None
Timings:
  load_betas:     0.03ms
  preprocessing:  0.00ms
  model_forward:  117ms (CUDA)
  gallery_search: 2.1ms (CPU NumPy)
  Total:          ~120ms
Provenance:
  load_betas:     CACHED_LOCAL
  preprocessing:  CACHED_LOCAL
  roi_mask:       CACHED_LOCAL
  model_forward:  LIVE_LOCAL
  vmf_projection: LIVE_LOCAL
  retrieval:      LIVE_LOCAL
  top_k:          LIVE_LOCAL
  reconstruction: UNAVAILABLE
```

---

## 6. Commands to Reproduce

```bash
# Backend (v62_single):
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
PYTHONPATH=../src C2C_BACKEND_MODE=v62_single python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend:
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
npm run dev

# Health check:
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool

# Inference:
curl -s http://127.0.0.1:8000/api/infer/0 | python3 -m json.tool

# Artifact verification:
python3 backend/verify_artifacts.py

# Smoke test:
python3 backend/smoke_test_backend.py
```

---

## 7. Remaining Gaps

| Gap | Impact | Resolution |
|-----|--------|-----------|
| **197K-D gallery missing** | Blocks V61a, MC-TTA, fusion live modes | Need to build 197K-D CLIP embeddings from token cache or regenerate |
| **V61a checkpoint too large (21 GB)** | Local memory/disk constraint | Strip optimizer state (→ ~5 GB model-only) or download from remote |
| **No SDK backend** | Reconstruction always UNAVAILABLE | Add Stable Diffusion 2.1 server or accept as permanent limitation |
| **Z-score stats missing** | Preprocessing returns CACHED_LOCAL | Existing features are pre-normalized; not critical |
| **Frontend provenance labels** | Labels say "Cached"/"Replay" not "CACHED_LOCAL" | Minor UI update to add backend-provided provenance to Phase 2 panels |

---

## 8. Files Changed

| File | Change |
|------|--------|
| `demo_thesis/backend/main.py` | Full rewrite — v4.0 multi-model, MC-TTA, fusion, provenance per step, fallback logic |
| `demo_thesis/src/lib/api.ts` | Extended BackendHealth type to match new health response fields |
| `demo_thesis/reports/live_artifact_audit_20260512.md` | Previous audit |
| `demo_thesis/reports/full_live_pipeline_audit_20260512.md` | This report |

## 9. Verification Results

- `npm run typecheck` — **PASSES**
- `npm run build` — **PASSES**
- `python3 backend/verify_artifacts.py` — **INFERENCE READY**
- `/api/health` — **inference_ready**, live_retrieval_available=true
- `/api/infer/0` — **Working**, rank #1, kappa=27.06
- Fallback fusion_v61_v62 → v62_single — **Working**, reports fallback reason
