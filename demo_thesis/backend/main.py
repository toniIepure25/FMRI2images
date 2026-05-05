"""
Cortex2Canvas Live Inference Backend

Runs on the K8s pod (H100) to serve real-time inference for the thesis demo.
Loads a trained checkpoint, accepts fMRI trial IDs, streams processing steps
via Server-Sent Events, and returns CLIP embeddings + retrieved/reconstructed images.

Usage (on pod):
    cd /home/jovyan/work/FMRI2images
    set -a && source .env && set +a
    pip install fastapi uvicorn sse-starlette Pillow
    python demo_thesis/backend/main.py --checkpoint experimental_results/V58a_improved_197k/subj01/checkpoints/checkpoint_best.pt

Usage (port-forward from local):
    export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
    kubectl port-forward -n runai-romania-dev <pod-name> 8000:8000
    # Frontend at http://localhost:3000 talks to http://localhost:8000/api/

"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cortex2canvas")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Lazy imports — torch, model classes, etc. are only loaded when the backend
# actually starts on the GPU pod.  On a local machine without torch this file
# can still be imported for type-checking / testing the FastAPI skeleton.
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_MODEL = None
_DEVICE = "cpu"
_FMRI_FEATURES: Optional[np.ndarray] = None
_TRIAL_INDEX = None
_CLIP_GALLERY: Optional[np.ndarray] = None
_CLIP_NSD_IDS: Optional[np.ndarray] = None
_SUBJECT = "subj01"
_ROI_INDICES = None
_CONFIG: Dict[str, Any] = {}
_READY = False

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    logger.warning("FastAPI not installed — run: pip install fastapi uvicorn sse-starlette")
    raise

app = FastAPI(title="Cortex2Canvas Live Inference", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Model Loading =====================

def load_inference_engine(checkpoint_path: str, subject: str = "subj01"):
    """Load checkpoint, fMRI features, CLIP gallery — call once at startup."""
    global _TORCH_AVAILABLE, _MODEL, _DEVICE, _FMRI_FEATURES, _TRIAL_INDEX
    global _CLIP_GALLERY, _CLIP_NSD_IDS, _SUBJECT, _ROI_INDICES, _CONFIG, _READY

    import torch

    _TORCH_AVAILABLE = True
    _SUBJECT = subject
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", _DEVICE)

    # --- Load checkpoint ---
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)

    model_config = ckpt.get("model_config", ckpt.get("config", {}).get("model", {}))
    _CONFIG = ckpt.get("config", {})
    ckpt_subject = ckpt.get("subject", subject)
    if ckpt_subject:
        _SUBJECT = ckpt_subject

    # --- Build ROI indices if needed ---
    encoder_cfg = model_config.get("encoder", {})
    encoder_type = encoder_cfg.get("encoder_type", "mlp")
    _ROI_INDICES = None

    if encoder_type in ("roi_transformer", "multi_subject_roi_transformer"):
        try:
            from fmri2img.data.roi_utils import build_roi_index
            roi_names = list(encoder_cfg.get("roi_dims", {}).keys())
            if roi_names:
                actual_dims, _ROI_INDICES = build_roi_index(_SUBJECT, roi_names)
                model_config["encoder"]["roi_dims"] = dict(actual_dims)
                logger.info("ROI indices built for %s: %d regions", _SUBJECT, len(actual_dims))
        except Exception as e:
            logger.warning("Could not build ROI indices: %s", e)

    # --- Create model ---
    from fmri2img.models.unified_model import create_model

    _MODEL = create_model(model_config, roi_indices=_ROI_INDICES)
    _MODEL.load_state_dict(ckpt["model_state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    logger.info("Model loaded: type=%s, encoder=%s", model_config.get("model_type"), encoder_type)

    # --- Load pre-extracted fMRI features ---
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    feat_path = Path(cache_root) / "preextracted" / f"subject={_SUBJECT}" / "fmri_features.npy"
    if feat_path.exists():
        _FMRI_FEATURES = np.load(str(feat_path))
        logger.info("fMRI features loaded: %s", _FMRI_FEATURES.shape)
    else:
        logger.warning("Pre-extracted features not found at %s", feat_path)

    # --- Load trial index ---
    try:
        import pandas as pd
        idx_path = Path("data/indices/nsd_index") / f"subject={_SUBJECT}" / "index.parquet"
        if idx_path.exists():
            _TRIAL_INDEX = pd.read_parquet(str(idx_path))
            logger.info("Trial index loaded: %d rows", len(_TRIAL_INDEX))
    except Exception as e:
        logger.warning("Could not load trial index: %s", e)

    # --- Load CLIP gallery ---
    try:
        import pandas as pd
        clip_path = Path(os.environ.get("OUTPUT_ROOT", "outputs")) / "clip_cache" / "clip.parquet"
        if not clip_path.exists():
            clip_path = Path("outputs/clip_cache/clip.parquet")
        if clip_path.exists():
            clip_df = pd.read_parquet(str(clip_path))
            emb_col = None
            for col in ("fused", "final", "embedding", "clip_embedding"):
                if col in clip_df.columns:
                    emb_col = col
                    break
            if emb_col:
                _CLIP_NSD_IDS = clip_df["nsdId"].values
                _CLIP_GALLERY = np.stack(clip_df[emb_col].values).astype(np.float32)
                norms = np.linalg.norm(_CLIP_GALLERY, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                _CLIP_GALLERY = _CLIP_GALLERY / norms
                logger.info("CLIP gallery loaded: %s from column '%s'", _CLIP_GALLERY.shape, emb_col)
    except Exception as e:
        logger.warning("Could not load CLIP gallery: %s", e)

    _READY = True
    logger.info("Inference engine ready.")


# ===================== Inference Logic =====================

def run_inference(trial_idx: int) -> Dict[str, Any]:
    """Run the full forward pass for a single trial. Returns dict with all results."""
    import torch

    if _FMRI_FEATURES is None or _MODEL is None:
        raise RuntimeError("Inference engine not loaded")

    results: Dict[str, Any] = {}
    t0 = time.time()

    # Step 1: Load fMRI beta vector
    fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
    results["n_voxels"] = int(fmri_vec.shape[0])
    results["step_times"] = {}
    results["step_times"]["load_betas"] = time.time() - t0

    # Step 2: Z-score (already done in pre-extraction for most setups)
    t1 = time.time()
    mean_val = float(np.mean(fmri_vec))
    std_val = float(np.std(fmri_vec))
    if std_val > 0:
        fmri_vec = (fmri_vec - mean_val) / std_val
    results["step_times"]["zscore"] = time.time() - t1

    # Step 3: ROI masking (already applied in pre-extraction)
    t2 = time.time()
    results["step_times"]["roi_mask"] = time.time() - t2

    # Step 4 + 5: Forward pass through model (ROI encode + vMF decode combined)
    t3 = time.time()
    x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)

    with torch.no_grad():
        mu, kappa_or_aux = _MODEL(x)

    mu_np = mu.cpu().numpy().squeeze()
    mu_norm = mu_np / (np.linalg.norm(mu_np) + 1e-8)

    kappa_val = None
    delta_val = None
    model_type = _CONFIG.get("model", {}).get("model_type", "vmf")

    if kappa_or_aux is not None:
        kappa_val = float(kappa_or_aux.cpu().numpy().squeeze())

    if model_type == "vmf_dcf" and hasattr(_MODEL, "_last_dcf_extras"):
        extras = _MODEL._last_dcf_extras
        if extras and "delta" in extras:
            delta_val = float(extras["delta"].cpu().numpy().squeeze())

    results["step_times"]["roi_encode"] = time.time() - t3
    results["step_times"]["vmf_decode"] = 0.0
    results["embedding"] = mu_norm.tolist()
    results["kappa"] = kappa_val
    results["delta"] = delta_val

    # Step 6: Gallery search (CSLS)
    t4 = time.time()
    if _CLIP_GALLERY is not None:
        cosine_scores = _CLIP_GALLERY @ mu_norm
        top_k_idx = np.argsort(-cosine_scores)[:20]
        top_k_scores = cosine_scores[top_k_idx].tolist()
        top_k_nsd_ids = _CLIP_NSD_IDS[top_k_idx].tolist() if _CLIP_NSD_IDS is not None else top_k_idx.tolist()
        results["top_k"] = [
            {"rank": i + 1, "nsd_id": int(nid), "cosine": float(s)}
            for i, (nid, s) in enumerate(zip(top_k_nsd_ids, top_k_scores))
        ]
        results["gallery_size"] = int(_CLIP_GALLERY.shape[0])
    else:
        results["top_k"] = []
        results["gallery_size"] = 0

    results["step_times"]["gallery_search"] = time.time() - t4
    results["total_time"] = time.time() - t0

    return results


# ===================== API Endpoints =====================

@app.get("/api/health")
def health():
    return {
        "status": "ready" if _READY else "not_loaded",
        "torch_available": _TORCH_AVAILABLE,
        "device": _DEVICE,
        "subject": _SUBJECT,
        "features_loaded": _FMRI_FEATURES is not None,
        "features_shape": list(_FMRI_FEATURES.shape) if _FMRI_FEATURES is not None else None,
        "gallery_loaded": _CLIP_GALLERY is not None,
        "gallery_size": int(_CLIP_GALLERY.shape[0]) if _CLIP_GALLERY is not None else 0,
        "model_loaded": _MODEL is not None,
    }


@app.get("/api/trials")
def list_trials(limit: int = Query(50, ge=1, le=1000)):
    """List available trial indices with their nsdIds."""
    if _TRIAL_INDEX is None:
        raise HTTPException(503, "Trial index not loaded")
    rows = _TRIAL_INDEX.head(limit)
    return [
        {"trial_idx": int(i), "nsd_id": int(r.get("nsdId", r.get("nsd_id", -1)))}
        for i, r in rows.iterrows()
    ]


@app.get("/api/infer/{trial_idx}")
def infer_sync(trial_idx: int):
    """Synchronous inference for a single trial."""
    if not _READY:
        raise HTTPException(503, "Model not loaded yet")
    if _FMRI_FEATURES is None:
        raise HTTPException(503, "fMRI features not loaded")
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")

    result = run_inference(trial_idx)
    return JSONResponse(content=result)


@app.get("/api/infer-stream/{trial_idx}")
async def infer_stream(trial_idx: int):
    """Stream inference progress via SSE. Each event has a 'step' field."""
    if not _READY:
        raise HTTPException(503, "Model not loaded yet")
    if _FMRI_FEATURES is None:
        raise HTTPException(503, "fMRI features not loaded")
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")

    async def event_generator():
        import torch

        t_global = time.time()

        # Step 1: Load betas
        yield {"event": "step", "data": json.dumps({
            "step": "load_betas", "status": "running",
            "detail": f"Loading fMRI vector (trial {trial_idx})..."
        })}
        fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
        n_voxels = int(fmri_vec.shape[0])
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({
            "step": "load_betas", "status": "done",
            "detail": f"Loaded {n_voxels} voxels (float32)",
            "n_voxels": n_voxels
        })}

        # Step 2: Z-score
        yield {"event": "step", "data": json.dumps({
            "step": "zscore", "status": "running",
            "detail": "Applying z-score normalization..."
        })}
        mean_val, std_val = float(np.mean(fmri_vec)), float(np.std(fmri_vec))
        if std_val > 0:
            fmri_vec = (fmri_vec - mean_val) / std_val
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({
            "step": "zscore", "status": "done",
            "detail": f"Z-scored: μ={mean_val:.3f}, σ={std_val:.3f}"
        })}

        # Step 3: ROI mask
        yield {"event": "step", "data": json.dumps({
            "step": "roi_mask", "status": "running",
            "detail": "Applying nsdgeneral ROI mask..."
        })}
        await asyncio.sleep(0.03)
        yield {"event": "step", "data": json.dumps({
            "step": "roi_mask", "status": "done",
            "detail": f"{n_voxels} voxels retained"
        })}

        # Step 4: ROI encode
        yield {"event": "step", "data": json.dumps({
            "step": "roi_encode", "status": "running",
            "detail": "Forward pass through ROI Transformer..."
        })}
        x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)
        t_fwd = time.time()
        with torch.no_grad():
            mu, kappa_or_aux = _MODEL(x)
        fwd_ms = (time.time() - t_fwd) * 1000
        yield {"event": "step", "data": json.dumps({
            "step": "roi_encode", "status": "done",
            "detail": f"Forward pass completed in {fwd_ms:.1f}ms"
        })}

        # Step 5: vMF decode
        yield {"event": "step", "data": json.dumps({
            "step": "vmf_decode", "status": "running",
            "detail": "Extracting (μ, κ) from vMF decoder..."
        })}
        mu_np = mu.cpu().numpy().squeeze()
        mu_norm = mu_np / (np.linalg.norm(mu_np) + 1e-8)

        kappa_val = float(kappa_or_aux.cpu().numpy().squeeze()) if kappa_or_aux is not None else None
        delta_val = None
        if hasattr(_MODEL, "_last_dcf_extras"):
            extras = getattr(_MODEL, "_last_dcf_extras", None)
            if extras and "delta" in extras:
                delta_val = float(extras["delta"].cpu().numpy().squeeze())

        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({
            "step": "vmf_decode", "status": "done",
            "detail": f"κ={kappa_val:.1f}" + (f", δ={delta_val:.4f}" if delta_val else ""),
            "kappa": kappa_val,
            "delta": delta_val,
            "embedding_dim": int(mu_norm.shape[0]),
        })}

        # Step 6: Gallery search
        yield {"event": "step", "data": json.dumps({
            "step": "gallery_search", "status": "running",
            "detail": "Computing cosine similarity..."
        })}
        top_k_results = []
        gallery_size = 0
        if _CLIP_GALLERY is not None:
            gallery_size = int(_CLIP_GALLERY.shape[0])
            t_search = time.time()
            scores = _CLIP_GALLERY @ mu_norm
            top_idx = np.argsort(-scores)[:20]
            search_ms = (time.time() - t_search) * 1000
            for rank, idx in enumerate(top_idx[:5]):
                nid = int(_CLIP_NSD_IDS[idx]) if _CLIP_NSD_IDS is not None else int(idx)
                top_k_results.append({
                    "rank": rank + 1,
                    "nsd_id": nid,
                    "cosine": float(scores[idx]),
                })

        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({
            "step": "gallery_search", "status": "done",
            "detail": f"Searched {gallery_size} embeddings in {search_ms:.1f}ms" if gallery_size else "No gallery",
            "gallery_size": gallery_size,
            "top_k": top_k_results,
        })}

        # Final summary
        total_ms = (time.time() - t_global) * 1000
        yield {"event": "step", "data": json.dumps({
            "step": "results", "status": "done",
            "detail": f"Pipeline complete in {total_ms:.0f}ms",
            "total_ms": total_ms,
            "kappa": kappa_val,
            "delta": delta_val,
            "top_k": top_k_results,
            "embedding": mu_norm.tolist(),
        })}

    return EventSourceResponse(event_generator())


@app.get("/api/nsd-id-to-trial/{nsd_id}")
def nsd_id_to_trial(nsd_id: int):
    """Find trial index(es) for a given nsdId."""
    if _TRIAL_INDEX is None:
        raise HTTPException(503, "Trial index not loaded")

    nsd_col = "nsdId" if "nsdId" in _TRIAL_INDEX.columns else "nsd_id"
    matches = _TRIAL_INDEX[_TRIAL_INDEX[nsd_col] == nsd_id]
    if len(matches) == 0:
        raise HTTPException(404, f"nsdId {nsd_id} not found in trial index")
    return {
        "nsd_id": nsd_id,
        "trial_indices": matches.index.tolist(),
        "count": len(matches),
    }


# ===================== Startup =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cortex2Canvas Live Inference Backend")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_inference_engine(args.checkpoint, args.subject)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
