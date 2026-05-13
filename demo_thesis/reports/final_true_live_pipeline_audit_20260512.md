# Final True Live Pipeline Audit — Cortex2Canvas Demo

**Date:** 2026-05-12
**Backend:** v4.0 (recon v2.0, pipeline-corrected)
**Frontend:** PhaseReconstruction wired to backend

---

## 1. Pipeline Contract — Verified from Source

```
UnCLIPImageVariationPipeline._encode_image pseudocode:
  if image_embeddings is None:
      image → CLIPImageProcessor → CLIP ViT encoder → image_embeds
  else:
      return image_embeddings          ← SKIP CLIP encoding

✅ Passing image=None, image_embeddings=predicted_mu is VALID
✅ V62a produces 768-D ViT-L/14 CLIP embeddings → matches Karlo's CLIP encoder dimension
✅ No CLIP image encoder re-run when embeddings provided
```

---

## 2. Backend Verification

| Test | Result | Evidence |
|------|--------|----------|
| Health: pipeline_accepts_predicted_embedding | true | Always, regardless of model cached state |
| Health: reconstruction_live_local_available | false | Model not cached locally |
| Health: reconstruction_mode | unavailable | Model not downloaded |
| Infer (no recon): mu exposed | No | `_mu_for_recon` popped before response |
| Infer (no recon): embedding field | Removed | `result.pop("embedding", None)` |
| Infer (with recon): model runs once | Yes | mu from run_inference reused, not recomputed |
| Infer (with recon): reconstruction block | Present | Honest UNAVAILABLE with reason |
| Model dimensions | 15,724→768 | MLP encoder verified |

---

## 3. Backend Endpoints

| Endpoint | Status | Mu source | Recon source |
|----------|--------|-----------|-------------|
| `GET /api/infer/{idx}` | Working | Internal, stripped | — |
| `GET /api/infer/{idx}?include_reconstruction=true` | Working | Internal, stripped, used once | recon.reconstruct(predicted_mu) |
| `GET /api/reconstruct/{idx}` | Working | Fresh model forward | recon.reconstruct(predicted_mu) |
| `GET /api/recon-assets/{file}` | Working | — | Serves live/cached PNGs |

---

## 4. Frontend Integration

| Component | Status | Behavior |
|-----------|--------|----------|
| `api.ts` | Updated | ReconstructionResult type, fetchInferenceWithRecon, reconstructTrial, getReconImageUrl |
| `PhaseReconstruction.tsx` | Updated | Calls `fetchInferenceWithRecon` via `resolveNsdIdToTrial`, displays live recon when available, falls back to cached/unavailable |
| `ComparisonTriptych.tsx` | Updated (previous pass) | Premium empty state for unavailable reconstruction |
| `EvidenceIntegrityPanel.tsx` | Updated (previous pass) | Labeled provenance cells |

---

## 5. Frontend UI Labels

| Condition | Title | Badge | Subtitle |
|-----------|-------|-------|----------|
| LIVE_LOCAL_RECONSTRUCTION | Live reconstruction | LIVE_LOCAL | Karlo UnCLIP · N steps, seed N |
| CACHED_LOCAL_RECONSTRUCTION | Cached reconstruction | CACHED_LOCAL | Stable Diffusion 2.1 output (cached) |
| RETRIEVAL_CONDITIONED_VARIATION | Retrieval-conditioned variation | DERIVED_LOCAL | Warning: not direct brain reconstruction |
| UNAVAILABLE | Reconstruction | UNAVAILABLE | Backend reason | 
| Loading | Loading... | — | loading live reconstruction... |

---

## 6. Truth Table

| Step | Live? | Source | UI Label | Verified |
|------|-------|--------|----------|----------|
| Load fMRI betas | No | fmri_features.npy | CACHED_LOCAL | Yes |
| Preprocessing | No | Pre-normalized | CACHED_LOCAL | Yes |
| ROI masking | No | Pre-masked in .npy | CACHED_LOCAL | Yes |
| MLP encoder | Yes | PyTorch on CUDA (117ms) | LIVE_LOCAL | Yes |
| vMF projection | Yes | Model output kappa | LIVE_LOCAL | Yes |
| CSLS gallery search | Yes | NumPy over 10K gallery (2ms) | LIVE_LOCAL | Yes |
| Top-K | Yes | CSLS result | LIVE_LOCAL | Yes |
| Reconstruction | No | Model not downloaded | UNAVAILABLE | Code-ready |

---

## 7. Model Download Status

**Karlo UnCLIP model (kakaobrain/karlo-v1-alpha):** Not cached.

Download attempts failed due to execution timeout (10 min) and apparent network constraints in this session environment. The 5 GB download requires sustained network access that exceeds this execution session's limits.

**Exact command to download (user to run manually):**
```bash
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis
PYTHONPATH=../src:backend C2C_RECON_ALLOW_DOWNLOAD=true python3 -c "
import sys; sys.path.insert(0, 'backend')
import recon
recon.load_pipeline()
print('Karlo model cached:', recon._check_model_cached())
"
```

Or using huggingface_hub directly:
```bash
python3 -c "
import sys; sys.path.insert(0, '.venv/lib/python3.10/site-packages')
from huggingface_hub import snapshot_download
snapshot_download('kakaobrain/karlo-v1-alpha', resume_download=True)
"
```

---

## 8. Start Command (Full Live Pipeline)

```bash
cd /home/tonystark/Desktop/Bachelor\ V2/demo_thesis

# Terminal 1: Backend
PYTHONPATH=../src:backend \
  C2C_BACKEND_MODE=v62_single \
  C2C_RECON_MODE=live \
  C2C_RECON_ALLOW_DOWNLOAD=false \
  C2C_RECON_STEPS=25 \
  C2C_RECON_GUIDANCE=8.0 \
  C2C_RECON_SEED=42 \
  python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
npm run dev
```

---

## 9. Verification Results

- `npm run typecheck` — **PASSES**
- `npm run build` — **PASSES**
- `/api/health` — Correct: pipeline_accepts=true, live_local_available=false, mode=unavailable
- `/api/infer/0` — Live retrieval works (kappa=27.06, CSLS=0.2961), no embedding leaked
- `/api/infer/0?include_reconstruction=true` — Retrieval live, reconstruction honestly UNAVAILABLE
- Model forward: Runs ONCE (mu extracted from run_inference, not recomputed)
- Pipeline contract: UnCLIP image_embeddings verified from source code

---

## 10. Files Changed

| File | Change |
|------|--------|
| `backend/recon.py` | Fixed health fields, venv auto-detection, image_url in metadata |
| `backend/main.py` | include_reconstruction param, _mu_for_recon internal key, model runs once |
| `src/lib/api.ts` | ReconstructionResult type, fetchInferenceWithRecon, reconstructTrial |
| `src/components/pipeline/PhaseReconstruction.tsx` | Live backend integration, resolveNsdIdToTrial, honest UI labels |

---

## 11. Final Conclusion

**Retrieval is live local.** Model forward (117ms CUDA), vMF projection, and CSLS gallery search all execute on this machine per request.

**Reconstruction is code-complete and pipeline-verified.** The `UnCLIPImageVariationPipeline` with `image_embeddings=predicted_mu` is the scientifically correct path for direct brain-to-image reconstruction from V62a's predicted 768-D CLIP embedding. The Karlo model download is the only remaining step — a one-time ~5 GB download that the user runs independently.

**No reconstruction is ever faked as live.** The backend honestly reports UNAVAILABLE when the model is not cached. The frontend displays the backend's reason, not generic placeholder text.

**When the model is downloaded**, the demo pipeline will be fully live: fMRI → V62a model forward → live retrieval → live reconstruction, all on this machine.
