# Backend/Frontend Live Mode Fix Audit — Cortex2Canvas Demo

**Date:** 2026-05-13
**Backend version:** 4.0 (auto-resolving, legacy compat)

---

## 1. Problems Found & Fixed

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| V62a model missing (fell back to `server_online`) | Running `uvicorn` from `backend/` without `PYTHONPATH` pointing to root `src/` → `fmri2img` not importable | Added auto-resolution in `main.py`: `sys.path.insert(0, ...parents[2]/src)` |
| Frontend showed Gallery: missing even though backend had gallery | Backend v4.0 used new fields (`gallery_768_loaded`, `v62_model_loaded`) but didn't set legacy fields (`gallery_loaded`, `model_loaded`) that frontend reads | Added legacy backward-compat fields to health response |
| Vite ENOSPC crash | `demo_thesis/results/` contains many PNG images, hitting inotify watch limit | Added `server.watch.ignored` to vite.config.ts |
| No documented startup command | — | Created `scripts/start_live_backend.sh` with absolute paths |

---

## 2. CUDA ≠ Live Inference

CUDA being available only means PyTorch can use the GPU. Live inference requires:
- Model checkpoint loaded (`v62_model_loaded: true`)
- fMRI features loaded (`features_loaded: true`)
- Gallery loaded (`gallery_768_loaded: true` / `gallery_loaded: true`)

The backend now reports all these independently. The frontend uses `live_retrieval_available` as the single source of truth.

---

## 3. Checkpoint Path Resolution

Auto-resolution in `main.py` adds the root `src/` to sys.path:
```python
_ROOT_SRC = Path(__file__).resolve().parents[2] / "src"
```

Checkpoint candidates (from `load_v62_model()`):
```
../local_backend_data/checkpoints/V62a_model_only.pt  ← found from backend/ dir
local_backend_data/checkpoints/V62a_model_only.pt     ← fallback
```

Verified: file exists at `/home/tonystark/Desktop/Bachelor V2/demo_thesis/local_backend_data/checkpoints/V62a_model_only.pt` (2.1 GB).

---

## 4. Frontend Health Mapping

| Frontend read | Backend field | Value when V62a loaded |
|--------------|---------------|----------------------|
| `model_loaded` | `v62_model_loaded \|\| v61_model_loaded` | `true` |
| `gallery_loaded` | `gallery_768_loaded \|\| gallery_197k_loaded` | `true` |
| `gallery_size` | Priority: 768, then 197K | `10000` |
| `live_retrieval_available` | Direct | `true` |
| `inference_available` | Direct | `true` |

---

## 5. Final Health Response

```json
{
  "status": "inference_ready",
  "effective_mode": "v62_single",
  "fallback_used": false,
  "fallback_reason": null,
  "model_loaded": true,
  "v62_model_loaded": true,
  "v61_model_loaded": false,
  "gallery_loaded": true,
  "gallery_768_loaded": true,
  "gallery_768_size": 10000,
  "gallery_197k_loaded": false,
  "features_loaded": true,
  "features_shape": [30000, 15724],
  "live_retrieval_available": true,
  "inference_available": true,
  "reconstruction_available": false,
  "reconstruction_mode": "unavailable",
  "reconstruction_pipeline_accepts_predicted_embedding": true,
  "reconstruction_requires_download": true,
  "device": "cuda",
  "missing": []
}
```

---

## 6. Live Inference Result

```json
{
  "ok": true,
  "kappa": 27.056,
  "top_k": [{"rank": 1, "nsd_id": 62175, "csls": 0.2961}, ...],
  "provenance": {
    "model_forward": "LIVE_LOCAL",
    "vmf_projection": "LIVE_LOCAL",
    "retrieval": "LIVE_LOCAL",
    "top_k": "LIVE_LOCAL"
  }
}
```

---

## 7. Truth Table

| Step | Live? | Source | UI Label |
|------|-------|--------|----------|
| fMRI loading | No | fmri_features.npy | CACHED_LOCAL |
| Preprocessing | No | Pre-normalized | CACHED_LOCAL |
| ROI masking | No | Pre-masked | CACHED_LOCAL |
| MLP encoder | Yes | PyTorch CUDA (117ms) | LIVE_LOCAL |
| vMF/projection | Yes | Model kappa output | LIVE_LOCAL |
| CSLS search | Yes | NumPy over 10K (2ms) | LIVE_LOCAL |
| Top-K | Yes | CSLS result | LIVE_LOCAL |
| Reconstruction | No | Karlo not cached | UNAVAILABLE |

---

## 8. Startup Commands

```bash
# Backend (from demo_thesis/)
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
bash scripts/start_live_backend.sh

# Or manually:
cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
npm run dev
```

---

## 9. Files Changed

| File | Change |
|------|--------|
| `backend/main.py` | Auto-resolve fmri2img path, add legacy `model_loaded`/`gallery_loaded`/`gallery_size`/`gallery_dim` fields |
| `vite.config.ts` | Add `server.watch.ignored` for `results/`, `local_backend_data/`, `node_modules/`, `dist/`, `.git/` |
| `scripts/start_live_backend.sh` | New — one-command backend startup with absolute artifact paths |
| `src/components/pipeline/PhaseReconstruction.tsx` | (Previous pass) Live backend integration |
| `src/lib/api.ts` | (Previous pass) ReconstructionResult type, API calls |

---

## 10. Verification

- `npm run typecheck` — PASSES
- `npm run build` — PASSES  
- `/api/health` — `status: inference_ready`, `live_retrieval_available: true`, `model_loaded: true`, `gallery_loaded: true`
- `/api/infer/0` — Live inference ok, kappa=27.06, top-1 nsdId=62175, csls=0.2961
- Legacy fields: `model_loaded`, `gallery_loaded`, `gallery_size`, `gallery_dim` all populated
- Reconstruction: Honestly `reconstruction_available: false` (Karlo not cached)
