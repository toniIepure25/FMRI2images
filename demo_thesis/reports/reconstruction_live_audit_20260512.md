# Reconstruction Live Audit — Cortex2Canvas Demo

**Date:** 2026-05-12
**Backend version:** 4.0 (with recon module)

---

## 1. Hardware Check

| Resource | Status | Details |
|----------|--------|---------|
| GPU | Available | NVIDIA GeForce RTX 3080 Laptop GPU |
| VRAM | 15 GB | Sufficient for SD float16 (~4 GB needed) |
| CUDA | Yes | PyTorch 2.11.0+cu128 (system), 2.8.0+cu128 (venv) |
| diffusers | 0.35.2 | Installed in venv |
| transformers | 4.57.1 | Installed in venv |
| accelerate | 1.11.0 | Installed in venv |
| xformers | No | Not installed |
| SD model cached | No | Needs download (~5 GB) |
| Cached reconstructions | None | No per-case images in local_backend_data/reconstructions/cached/ |
| Composite panels | Yes | reconstruction_results/ (20 panels in best/medium/hard/) |
| Disk space | 143 GB free | Sufficient for model download |

---

## 2. Reconstruction Pipeline Analysis

| Source | Path | Type | Dimensions | Compatible? | Notes |
|--------|------|------|-----------|-------------|-------|
| `decode_diffusion.py` | scripts/reconstruction/ | SD 2.1 + CLIP adapter | 512→1024→SD | Needs adapter trained for 768-D | References Takagi & Nishimoto (2023), MindEye2 |
| `diffusion_utils.py` | src/fmri2img/generation/ | SD 2.1, prompt_embeds | 512-D | No (SD expects 1024-D) | Code exists but would fail on dimension mismatch |
| `StableDiffusionImageVariationPipeline` | diffusers 0.35.2 | CLIP embedding → image | 768-D ViT-L/14 | **Yes** | Designed for this use case; lambdalabs/sd-image-variations-diffusers |
| `reconstruction_results/` | root | Composite PNG panels | — | N/A | Ground-truth + retrieved + reconstruction comparison panels |
| `local_backend_data/reconstructions/` | demo_thesis/ | Empty directories | — | N/A | Created by recon.py; live dir and cached dir exist but are empty |

---

## 3. Reconstruction Mode Decision

| Mode | Status | Reason |
|------|--------|--------|
| LIVE_LOCAL | **Code-ready, model not cached** | `recon.py` can load `StableDiffusionImageVariationPipeline` with 768-D V62a embeddings. Requires ~5 GB model download. Set `C2C_RECON_ALLOW_DOWNLOAD=true` |
| CACHED_LOCAL | **Unavailable** | No per-case reconstruction assets cached. `reconstruction_results/` panels are composite and not mapped to demo cases. |
| RETRIEVAL_CONDITIONED_VARIATION | **Not implemented** | Would use SD img2img from retrieved image. Not direct brain reconstruction — would be honestly labeled. |
| UNAVAILABLE | **Current state** | No model cached, no cached assets. Backend reports this honestly. |

---

## 4. Backend Implementation

### Files created/modified:
- `demo_thesis/backend/recon.py` — Standalone reconstruction module (NEW)
- `demo_thesis/backend/main.py` — Integrated recon endpoints and health fields (MODIFIED)

### Health endpoint (new recon fields):
```json
{
  "reconstruction_available": false,
  "reconstruction_mode": "disabled",
  "reconstruction_live_local_available": false,
  "reconstruction_cached_local_available": false,
  "reconstruction_model_loaded": false,
  "reconstruction_model_cached": false,
  "reconstruction_device": "cpu",
  "reconstruction_dtype": "float16",
  "reconstruction_requires_download": true,
  "reconstruction_allow_download": false,
  "reconstruction_model_id": "lambdalabs/sd-image-variations-diffusers",
  "reconstruction_supported_modes": ["live", "cached", "disabled"]
}
```

### New endpoints:
- `GET /api/reconstruct/{trial_idx}` — Runs retrieval, gets CLIP embedding, generates or retrieves reconstruction
- `GET /api/recon-assets/{filename}` — Serves generated or cached reconstruction images

### Env vars:
- `C2C_RECON_MODE=auto|live|cached|disabled`
- `C2C_RECON_MODEL_ID=lambdalabs/sd-image-variations-diffusers`
- `C2C_RECON_ALLOW_DOWNLOAD=false` (default — must be explicitly set to download model)
- `C2C_RECON_DTYPE=float16`
- `C2C_RECON_HEIGHT=512`, `C2C_RECON_WIDTH=512`
- `C2C_RECON_STEPS=50`, `C2C_RECON_GUIDANCE=3.0`, `C2C_RECON_SEED=42`

---

## 5. Pipeline Truth Table

| Step | Live? | Local? | Cached? | Backend field | UI label | Verified |
|------|-------|--------|---------|---------------|----------|----------|
| Load fMRI betas | No | Yes | Yes | CACHED_LOCAL | Cached local | ✓ |
| Preprocessing | No | Yes | Yes | CACHED_LOCAL | Cached local (pre-normalized) | ✓ |
| ROI masking | No | Yes | Yes | CACHED_LOCAL | Cached local (pre-masked) | ✓ |
| MLP encoder | Yes | Yes | No | LIVE_LOCAL | Live local | ✓ (CUDA, 117ms) |
| vMF projection | Yes | Yes | No | LIVE_LOCAL | Live local | ✓ |
| CSLS gallery search | Yes | Yes | No | LIVE_LOCAL | Live local | ✓ (2ms) |
| Top-K | Yes | Yes | No | LIVE_LOCAL | Live local | ✓ |
| Reconstruction | No | — | No | UNAVAILABLE | Unavailable | ✓ (no cached model) |

---

## 6. How to Enable Live Reconstruction

```bash
# 1. Allow model download (one-time, ~5 GB)
export C2C_RECON_ALLOW_DOWNLOAD=true

# 2. Start backend with reconstruction
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
PYTHONPATH=../src:backend C2C_BACKEND_MODE=v62_single C2C_RECON_MODE=live C2C_RECON_ALLOW_DOWNLOAD=true \
  python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. Test
curl -s http://127.0.0.1:8000/api/reconstruct/0 | python3 -m json.tool
```

The first request will download the model (~5 GB, ~5-10 minutes on fast connection). Subsequent requests will use the cached model.

---

## 7. Remaining Gaps

| Gap | Impact | Resolution |
|-----|--------|-----------|
| SD model not cached | Blocks live reconstruction | Run with `C2C_RECON_ALLOW_DOWNLOAD=true` once |
| No cached reconstruction assets | No per-case cached images | Could generate offline with `decode_diffusion.py` |
| xformers not installed | Slightly slower generation | Optional — `pip install xformers` |
| Venv vs system Python | Backend uses system Python (2.11), diffusers only in venv (2.8) | Would need to install diffusers in system Python or run backend from venv |
| 197K-D gallery missing | Blocks V61a/MC-TTA/fusion | Separate effort |

---

## 8. Verification Results

- `npm run typecheck` — **PASSES**
- `npm run build` — **PASSES**
- `/api/health` — Correctly reports `reconstruction_available: false`, `reconstruction_mode: "disabled"`, `reconstruction_requires_download: true`
- `/api/reconstruct/0` — Returns `{"ok": false, "mode": "UNAVAILABLE", ...}` with clear error when model not cached
