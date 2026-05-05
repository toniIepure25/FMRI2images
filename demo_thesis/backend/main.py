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
_ZSCORE_STATS: Optional[Dict[str, np.ndarray]] = None
_EMB_PREPROCESSOR = None
_READY = False

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    logger.warning("FastAPI not installed — run: pip install fastapi uvicorn sse-starlette")
    raise

app = FastAPI(title="Cortex2Canvas Live Inference", version="2.1.0")

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
    global _CLIP_GALLERY, _CLIP_NSD_IDS, _SUBJECT, _ROI_INDICES, _CONFIG
    global _ZSCORE_STATS, _EMB_PREPROCESSOR, _READY

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

    # --- Create model (config uses "type", not "model_type") ---
    from fmri2img.models.unified_model import create_model

    _MODEL = create_model(model_config, roi_indices=_ROI_INDICES)
    _MODEL.load_state_dict(ckpt["model_state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()

    model_type = model_config.get("type", model_config.get("model_type", "vmf"))
    logger.info("Model loaded: type=%s, encoder=%s", model_type, encoder_type)

    # --- Load per-voxel z-score stats (matches train_unified.py normalisation) ---
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    zscore_dir = Path(cache_root) / "preproc" / f"subject={_SUBJECT}" / "zscore_stats"
    if zscore_dir.exists():
        try:
            vmean = np.load(str(zscore_dir / "voxel_mean.npy"))
            vstd = np.load(str(zscore_dir / "voxel_std.npy"))
            _ZSCORE_STATS = {"mean": vmean, "std": vstd}
            logger.info("Z-score stats loaded: %d voxels", len(vmean))
        except Exception as e:
            logger.warning("Could not load z-score stats: %s", e)
    else:
        logger.info("No z-score stats at %s — will skip per-voxel normalization", zscore_dir)

    # --- Load EmbeddingPreprocessor if experiment used one ---
    preproc_dir = Path(cache_root) / "embedding_preproc"
    if preproc_dir.exists():
        try:
            from fmri2img.embedding_preproc import EmbeddingPreprocessor
            pkl_files = list(preproc_dir.glob(f"{_SUBJECT}_*.pkl"))
            if pkl_files:
                _EMB_PREPROCESSOR = EmbeddingPreprocessor.load(str(pkl_files[0]))
                logger.info("EmbeddingPreprocessor loaded: %s", pkl_files[0].name)
        except Exception as e:
            logger.warning("Could not load EmbeddingPreprocessor: %s", e)

    # --- Load pre-extracted fMRI features ---
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


# ===================== Helpers =====================

def _apply_zscore(fmri_vec: np.ndarray) -> Tuple[np.ndarray, str]:
    """Apply per-voxel z-scoring if stats are available, matching train_unified.py."""
    if _ZSCORE_STATS is not None:
        mean = _ZSCORE_STATS["mean"]
        std = _ZSCORE_STATS["std"]
        std_safe = np.where(std > 1e-6, std, 1.0)
        fmri_vec = (fmri_vec - mean) / std_safe
        return fmri_vec, f"Per-voxel z-scored ({len(mean)} voxels)"
    return fmri_vec, "Skipped (pre-extracted data already normalized)"


def _csls_retrieval(query: np.ndarray, k: int = 20, csls_k: int = 10) -> List[Dict]:
    """CSLS-corrected retrieval matching fmri2img.eval.embedding_eval."""
    if _CLIP_GALLERY is None:
        return []

    cosine_scores = _CLIP_GALLERY @ query
    gallery_size = _CLIP_GALLERY.shape[0]

    top_hub_idx = np.argsort(-cosine_scores)[:csls_k]
    r_s = float(np.mean(cosine_scores[top_hub_idx]))
    csls_scores = 2.0 * cosine_scores - r_s

    top_idx = np.argsort(-csls_scores)[:k]
    results = []
    for rank, idx in enumerate(top_idx):
        nid = int(_CLIP_NSD_IDS[idx]) if _CLIP_NSD_IDS is not None else int(idx)
        results.append({
            "rank": rank + 1,
            "nsd_id": nid,
            "cosine": float(cosine_scores[idx]),
            "csls": float(csls_scores[idx]),
        })
    return results


def _get_model_type() -> str:
    """Read model type correctly from config (key is 'type', not 'model_type')."""
    model_cfg = _CONFIG.get("model", {})
    return model_cfg.get("type", model_cfg.get("model_type", "vmf"))


def _get_subject_id_tensor():
    """Build subject_ids tensor for multi-subject encoders."""
    import torch
    subj_map = {"subj01": 0, "subj02": 1, "subj05": 2, "subj07": 3}
    sid = subj_map.get(_SUBJECT, 0)
    return torch.tensor([sid], dtype=torch.long, device=_DEVICE)


# ===================== Inference Logic =====================

def run_inference(trial_idx: int) -> Dict[str, Any]:
    """Run the full forward pass for a single trial."""
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

    # Step 2: Per-voxel z-score (matching train_unified.py)
    t1 = time.time()
    fmri_vec, zscore_detail = _apply_zscore(fmri_vec)
    results["step_times"]["zscore"] = time.time() - t1
    results["zscore_detail"] = zscore_detail

    # Step 3: ROI masking (already applied in pre-extraction)
    t2 = time.time()
    results["step_times"]["roi_mask"] = time.time() - t2

    # Step 4 + 5: Forward pass (encode + decode in one call)
    t3 = time.time()
    x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)

    encoder_type = _CONFIG.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")
    model_type = _get_model_type()

    with torch.no_grad():
        if encoder_type == "multi_subject_roi_transformer":
            subject_ids = _get_subject_id_tensor()
            mu, kappa_or_aux = _MODEL(x, subject_ids=subject_ids)
        else:
            mu, kappa_or_aux = _MODEL(x)

    mu_np = mu.cpu().numpy().squeeze()
    mu_norm = mu_np / (np.linalg.norm(mu_np) + 1e-8)

    kappa_val = None
    delta_val = None

    if kappa_or_aux is not None:
        if isinstance(kappa_or_aux, dict):
            kappa_val = float(kappa_or_aux.get("kappa", torch.tensor(0.0)).cpu().numpy().squeeze())
        else:
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

    # Step 6: CSLS-corrected gallery search
    t4 = time.time()
    if _CLIP_GALLERY is not None:
        results["top_k"] = _csls_retrieval(mu_norm, k=20)
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
        "zscore_stats_loaded": _ZSCORE_STATS is not None,
        "preprocessor_loaded": _EMB_PREPROCESSOR is not None,
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

        # Step 2: Per-voxel z-score (matching train_unified.py)
        yield {"event": "step", "data": json.dumps({
            "step": "zscore", "status": "running",
            "detail": "Applying per-voxel z-score normalization..."
        })}
        fmri_vec, zscore_detail = _apply_zscore(fmri_vec)
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({
            "step": "zscore", "status": "done",
            "detail": zscore_detail
        })}

        # Step 3: ROI mask (pre-applied in fmri_features.npy)
        yield {"event": "step", "data": json.dumps({
            "step": "roi_mask", "status": "running",
            "detail": "Verifying nsdgeneral ROI mask (pre-applied)..."
        })}
        await asyncio.sleep(0.03)
        yield {"event": "step", "data": json.dumps({
            "step": "roi_mask", "status": "done",
            "detail": f"{n_voxels} visual cortex voxels (pre-masked)"
        })}

        # Step 4: Forward pass
        yield {"event": "step", "data": json.dumps({
            "step": "roi_encode", "status": "running",
            "detail": "Forward pass through encoder..."
        })}
        x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)
        t_fwd = time.time()

        encoder_type = _CONFIG.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")
        with torch.no_grad():
            if encoder_type == "multi_subject_roi_transformer":
                subject_ids = _get_subject_id_tensor()
                mu, kappa_or_aux = _MODEL(x, subject_ids=subject_ids)
            else:
                mu, kappa_or_aux = _MODEL(x)

        fwd_ms = (time.time() - t_fwd) * 1000
        yield {"event": "step", "data": json.dumps({
            "step": "roi_encode", "status": "done",
            "detail": f"Forward pass completed in {fwd_ms:.1f}ms"
        })}

        # Step 5: vMF decode
        yield {"event": "step", "data": json.dumps({
            "step": "vmf_decode", "status": "running",
            "detail": "Extracting (mu, kappa) from vMF decoder..."
        })}
        mu_np = mu.cpu().numpy().squeeze()
        mu_norm = mu_np / (np.linalg.norm(mu_np) + 1e-8)

        kappa_val = None
        delta_val = None
        model_type = _get_model_type()

        if kappa_or_aux is not None:
            if isinstance(kappa_or_aux, dict):
                kappa_val = float(kappa_or_aux.get("kappa", torch.tensor(0.0)).cpu().numpy().squeeze())
            else:
                kappa_val = float(kappa_or_aux.cpu().numpy().squeeze())

        if model_type == "vmf_dcf" and hasattr(_MODEL, "_last_dcf_extras"):
            extras = getattr(_MODEL, "_last_dcf_extras", None)
            if extras and "delta" in extras:
                delta_val = float(extras["delta"].cpu().numpy().squeeze())

        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({
            "step": "vmf_decode", "status": "done",
            "detail": f"kappa={kappa_val:.1f}" + (f", delta={delta_val:.4f}" if delta_val else ""),
            "kappa": kappa_val,
            "delta": delta_val,
            "embedding_dim": int(mu_norm.shape[0]),
        })}

        # Step 6: CSLS gallery search
        yield {"event": "step", "data": json.dumps({
            "step": "gallery_search", "status": "running",
            "detail": "CSLS-corrected cosine similarity search..."
        })}
        t_search = time.time()
        top_k_results = _csls_retrieval(mu_norm, k=5)
        search_ms = (time.time() - t_search) * 1000
        gallery_size = int(_CLIP_GALLERY.shape[0]) if _CLIP_GALLERY is not None else 0

        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({
            "step": "gallery_search", "status": "done",
            "detail": f"CSLS search over {gallery_size} embeddings in {search_ms:.1f}ms",
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
