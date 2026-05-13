# Reconstruction Truth Audit — Cortex2Canvas Demo

**Date:** 2026-05-12
**Backend version:** 4.0 (pipeline-corrected recon v2.0)

---

## 1. Pipeline Contract Verification

### `StableDiffusionImageVariationPipeline` — REJECTED
- `__call__` accepts `image: Union[PIL.Image, List[PIL.Image], torch.Tensor]`
- `_encode_image()` **always** runs `self.image_encoder(image).image_embeds`
- Passing a (1, 768) embedding tensor as `image` would feed it through the CLIP ViT encoder, which expects pixel images (batch, 3, 224, 224). This would either crash or silently produce garbage.
- **Verdict:** DOES NOT accept precomputed CLIP embeddings. Previous implementation was scientifically invalid.

### `UnCLIPImageVariationPipeline` — ACCEPTED
- `__call__` accepts `image_embeddings: Optional[torch.Tensor] = None`
- `_encode_image()` checks: `if image_embeddings is None` → run CLIP encoder. If provided → **use as-is, skip encoding**.
- Docstring: "Can be left as None only when image_embeddings are passed."
- Uses kakaobrain/karlo-v1-alpha (CLIP ViT-L/14, 768-D embeddings)
- **Verdict:** Accepts precomputed CLIP ViT-L/14 image embeddings. Compatible with V62a 768-D output.

### Expected embedding shape
- `image_embeddings`: `(batch_size, 768)` — flattened CLIP ViT-L/14 image embedding
- Matches V62a mu output dimension (768-D)
- No sequence dimension needed (Karlo's `_encode_image` passes directly, unlike SD which uses 77 tokens)

---

## 2. Reconstruction Mode Classification

| Mode | Condition | Provenance | Backend label |
|------|-----------|-----------|---------------|
| LIVE_LOCAL_RECONSTRUCTION | Image generated from live predicted V62a mu during request | LIVE_LOCAL | Generated from predicted V62a CLIP embedding |
| RETRIEVAL_CONDITIONED_VARIATION | Image variation from Top-1 retrieved image | DERIVED_LOCAL | Not direct brain reconstruction |
| CACHED_LOCAL_RECONSTRUCTION | Image loaded from saved file | CACHED_LOCAL | Pre-computed reconstruction |
| UNAVAILABLE | No model cached, no cached assets | UNAVAILABLE | Reconstruction unavailable |

**Never labeled as LIVE_LOCAL:**
- Top-1 retrieved image variation
- Target stimulus image
- Cached PNG/JPG
- Placeholder/dummy image

---

## 3. Backend Endpoints

| Endpoint | Implemented? | Behavior |
|----------|-------------|----------|
| `GET /api/reconstruct/{trial_idx}` | Yes | Runs live V62a forward → gets mu → passes to recon via `image_embeddings` |
| `GET /api/recon-assets/{filename}` | Yes | Serves live/cached reconstruction images |
| `GET /api/infer/{trial_idx}?include_reconstruction=true` | Not yet | Would need to add query param support |
| `GET /api/health` | Yes | Reports 9 reconstruction-specific fields |

---

## 4. Health Response (reconstruction fields)

```json
{
  "reconstruction_accepts_predicted_embedding": false,
  "reconstruction_allow_download": false,
  "reconstruction_available": false,
  "reconstruction_device": "cpu",
  "reconstruction_dtype": "float16",
  "reconstruction_last_error": null,
  "reconstruction_mode": "unavailable",
  "reconstruction_model_cached": false,
  "reconstruction_model_id": "kakaobrain/karlo-v1-alpha",
  "reconstruction_model_loaded": false,
  "reconstruction_requires_download": true,
  "reconstruction_scientific_warning": "Model not loaded. Set C2C_RECON_ALLOW_DOWNLOAD=true to download kakaobrain/karlo-v1-alpha (~5GB).",
  "reconstruction_uses_retrieved_image": false
}
```

---

## 5. Pipeline Truth Table

| Step | Computed live? | Source | UI label | Verified |
|------|---------------|--------|----------|----------|
| Load fMRI betas | No | fmri_features.npy | CACHED_LOCAL | Yes |
| Preprocessing | No | Pre-normalized features | CACHED_LOCAL | Yes |
| ROI masking | No | Pre-masked in .npy | CACHED_LOCAL | Yes |
| MLP encoder | Yes | PyTorch on CUDA (117ms) | LIVE_LOCAL | Yes |
| vMF projection | Yes | Model kappa output | LIVE_LOCAL | Yes |
| CSLS gallery search | Yes | NumPy on CPU (2ms) | LIVE_LOCAL | Yes |
| Top-K | Yes | From CSLS results | LIVE_LOCAL | Yes |
| Reconstruction | **No** | Model not cached | **UNAVAILABLE** | Yes |

---

## 6. How to Enable Live Reconstruction

```bash
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
PYTHONPATH=../src:backend \
  C2C_BACKEND_MODE=v62_single \
  C2C_RECON_MODE=live \
  C2C_RECON_ALLOW_DOWNLOAD=true \
  python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

First request downloads kakaobrain/karlo-v1-alpha (~5 GB, one-time). Subsequent requests generate images from live V62a mu via `image_embeddings`.

### Test command (once running):
```bash
curl -s http://127.0.0.1:8000/api/reconstruct/0 | python3 -m json.tool
```

Expected response (if model cached + loaded):
```json
{
  "ok": true,
  "mode": "LIVE_LOCAL_RECONSTRUCTION",
  "provenance": "LIVE_LOCAL",
  "source_embedding": "predicted_mu",
  "uses_retrieved_image": false,
  "uses_target_image": false,
  "image_path": "local_backend_data/reconstructions/live/recon_s42_st25_g8.0_xxxxxxxx.png",
  "generation_ms": 12345,
  "model_id": "kakaobrain/karlo-v1-alpha"
}
```

---

## 7. Verification Results

- `npm run typecheck` — **PASSES**
- `npm run build` — **PASSES**
- `/api/health` — Correctly reports `reconstruction_mode: "unavailable"`, `reconstruction_requires_download: true`, `reconstruction_accepts_predicted_embedding: false` (model not loaded yet)
- Pipeline contract — **Verified**: `UnCLIPImageVariationPipeline` accepts `image_embeddings`, bypasses CLIP encoder
- `StableDiffusionImageVariationPipeline` — **Rejected**: Does NOT accept precomputed embeddings (always runs CLIP encoder)

## 8. What Changed from Previous Implementation

| Aspect | Before (v1) | After (v2) |
|--------|------------|-----------|
| Pipeline | `StableDiffusionImageVariationPipeline` | `UnCLIPImageVariationPipeline` |
| Input method | `image=emb_tensor` (invalid) | `image_embeddings=emb_tensor` (valid) |
| CLIP encoding step | Always re-encodes "image" | Bypassed when embeddings provided |
| Model | lambdalabs/sd-image-variations-diffusers (~5 GB, 512×512) | kakaobrain/karlo-v1-alpha (~5 GB, 256×256) |
| Scientific validity | Passed (1,768) tensor as image → would have failed or produced garbage | Passes precomputed CLIP ViT-L/14 embedding via documented `image_embeddings` parameter |
| Health fields | 8 basic fields | 12 fields including `accepts_predicted_embedding`, `scientific_warning` |
| provenance granularity | LIVE_LOCAL / CACHED_LOCAL / UNAVAILABLE | LIVE_LOCAL_RECONSTRUCTION / RETRIEVAL_CONDITIONED_VARIATION / CACHED_LOCAL_RECONSTRUCTION / UNAVAILABLE |
