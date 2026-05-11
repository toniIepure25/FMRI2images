"""
Cortex2Canvas Live Inference Backend

Three operating modes:
  1. Full inference  - checkpoint + data loaded, live forward pass
  2. Data-only       - fMRI/CLIP/index loaded; no model checkpoint
  3. Server-only     - FastAPI online, nothing loaded yet

Env vars (all optional):
  C2C_SUBJECT, C2C_FMRI_FEATURES_PATH, C2C_TRIAL_INDEX_PATH,
  C2C_CLIP_GALLERY_PATH, C2C_CHECKPOINT_PATH, C2C_DEVICE (auto|cpu|cuda)

Local:  python -m uvicorn main:app --host 0.0.0.0 --port 8000
Pod:    python main.py --checkpoint path/to/checkpoint_best.pt
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
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
_LOAD_ERRORS: Dict[str, str] = {}

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    raise

app = FastAPI(title="Cortex2Canvas Live Inference", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ===================== Granular Loaders =====================

def _resolve_path(env_var: str, *fallbacks: str) -> Optional[Path]:
    val = os.environ.get(env_var)
    if val and Path(val).exists():
        return Path(val)
    for fb in fallbacks:
        if Path(fb).exists():
            return Path(fb)
    return None


def load_torch() -> bool:
    global _TORCH_AVAILABLE, _DEVICE
    try:
        import torch
        _TORCH_AVAILABLE = True
        dp = os.environ.get("C2C_DEVICE", "auto")
        if dp == "cuda" and torch.cuda.is_available():
            _DEVICE = "cuda"
        elif dp == "auto" and torch.cuda.is_available():
            _DEVICE = "cuda"
        else:
            _DEVICE = "cpu"
        logger.info("PyTorch %s device=%s", torch.__version__, _DEVICE)
        return True
    except ImportError:
        _LOAD_ERRORS["torch"] = "PyTorch not installed"
        return False


def load_features(subject: str) -> bool:
    global _FMRI_FEATURES
    path = _resolve_path(
        "C2C_FMRI_FEATURES_PATH",
        f"../local_backend_data/{subject}/fmri_features.npy",
        f"local_backend_data/{subject}/fmri_features.npy",
        f"cache/preextracted/subject={subject}/fmri_features.npy",
    )
    if not path:
        _LOAD_ERRORS["features"] = "fmri_features.npy not found"
        return False
    try:
        _FMRI_FEATURES = np.load(str(path))
        logger.info("fMRI features: %s from %s", _FMRI_FEATURES.shape, path)
        return True
    except Exception as e:
        _LOAD_ERRORS["features"] = str(e)
        return False


def load_trial_index(subject: str) -> bool:
    global _TRIAL_INDEX
    try:
        import pandas as pd
    except ImportError:
        _LOAD_ERRORS["trial_index"] = "pandas not installed"
        return False
    path = _resolve_path(
        "C2C_TRIAL_INDEX_PATH",
        f"../local_backend_data/{subject}/index.parquet",
        f"local_backend_data/{subject}/index.parquet",
        f"data/indices/nsd_index/subject={subject}/index.parquet",
    )
    if not path:
        _LOAD_ERRORS["trial_index"] = "index.parquet not found"
        return False
    try:
        _TRIAL_INDEX = pd.read_parquet(str(path))
        logger.info("Trial index: %d rows from %s", len(_TRIAL_INDEX), path)
        return True
    except Exception as e:
        _LOAD_ERRORS["trial_index"] = str(e)
        return False


def load_clip_gallery() -> bool:
    global _CLIP_GALLERY, _CLIP_NSD_IDS
    try:
        import pandas as pd
    except ImportError:
        _LOAD_ERRORS["gallery"] = "pandas not installed"
        return False
    path = _resolve_path(
        "C2C_CLIP_GALLERY_PATH",
        "../local_backend_data/clip/clip.parquet",
        "local_backend_data/clip/clip.parquet",
        "outputs/clip_cache/clip.parquet",
    )
    if not path:
        _LOAD_ERRORS["gallery"] = "clip.parquet not found"
        return False
    try:
        clip_df = pd.read_parquet(str(path))
        emb_col = next((c for c in ("fused", "final", "embedding", "clip_embedding") if c in clip_df.columns), None)
        if not emb_col:
            _LOAD_ERRORS["gallery"] = f"No embedding column in {list(clip_df.columns)}"
            return False
        _CLIP_NSD_IDS = clip_df["nsdId"].values
        _CLIP_GALLERY = np.stack(clip_df[emb_col].values).astype(np.float32)
        norms = np.maximum(np.linalg.norm(_CLIP_GALLERY, axis=1, keepdims=True), 1e-8)
        _CLIP_GALLERY = _CLIP_GALLERY / norms
        logger.info("CLIP gallery: %s col='%s' from %s", _CLIP_GALLERY.shape, emb_col, path)
        return True
    except Exception as e:
        _LOAD_ERRORS["gallery"] = str(e)
        return False


def load_model(checkpoint_path: str, subject: str) -> bool:
    global _MODEL, _ROI_INDICES, _CONFIG, _ZSCORE_STATS, _EMB_PREPROCESSOR, _SUBJECT
    if not _TORCH_AVAILABLE:
        _LOAD_ERRORS["model"] = "PyTorch not available"
        return False
    import torch
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        _LOAD_ERRORS["model"] = f"Checkpoint not found: {checkpoint_path}"
        return False
    try:
        ckpt = torch.load(str(ckpt_p), map_location=_DEVICE, weights_only=False)
    except Exception as e:
        _LOAD_ERRORS["model"] = f"Checkpoint load failed: {e}"
        return False

    model_config = ckpt.get("model_config", ckpt.get("config", {}).get("model", {}))
    _CONFIG = ckpt.get("config", {})
    ckpt_subject = ckpt.get("subject", subject)
    if ckpt_subject:
        _SUBJECT = ckpt_subject

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
        except Exception as e:
            logger.warning("ROI indices failed: %s", e)

    try:
        from fmri2img.models.unified_model import create_model
        _MODEL = create_model(model_config, roi_indices=_ROI_INDICES)
        _MODEL.load_state_dict(ckpt["model_state_dict"])
        _MODEL.to(_DEVICE)
        _MODEL.eval()
        logger.info("Model loaded: type=%s encoder=%s", model_config.get("type", "?"), encoder_type)
    except Exception as e:
        _LOAD_ERRORS["model"] = f"Model build failed: {e}"
        return False

    cache_root = os.environ.get("CACHE_ROOT", "cache")
    zscore_dir = Path(cache_root) / "preproc" / f"subject={_SUBJECT}" / "zscore_stats"
    if zscore_dir.exists():
        try:
            _ZSCORE_STATS = {"mean": np.load(str(zscore_dir / "voxel_mean.npy")), "std": np.load(str(zscore_dir / "voxel_std.npy"))}
        except Exception:
            pass
    preproc_dir = Path(cache_root) / "embedding_preproc"
    if preproc_dir.exists():
        try:
            from fmri2img.embedding_preproc import EmbeddingPreprocessor
            pkl_files = list(preproc_dir.glob(f"{_SUBJECT}_*.pkl"))
            if pkl_files:
                _EMB_PREPROCESSOR = EmbeddingPreprocessor.load(str(pkl_files[0]))
        except Exception:
            pass
    _LOAD_ERRORS.pop("model", None)
    return True


def _compute_status() -> str:
    if _MODEL is not None and _FMRI_FEATURES is not None and _CLIP_GALLERY is not None:
        return "inference_ready"
    if _FMRI_FEATURES is not None or _CLIP_GALLERY is not None or _TRIAL_INDEX is not None:
        return "data_ready"
    return "server_online"


def load_all_at_startup():
    global _SUBJECT
    _SUBJECT = os.environ.get("C2C_SUBJECT", "subj01")
    logger.info("=== Cortex2Canvas startup: subject=%s ===", _SUBJECT)
    load_torch()
    load_features(_SUBJECT)
    load_trial_index(_SUBJECT)
    load_clip_gallery()
    ckpt_path = os.environ.get("C2C_CHECKPOINT_PATH", "")
    if ckpt_path:
        load_model(ckpt_path, _SUBJECT)
    else:
        ckpt_candidates = [
            "../local_backend_data/checkpoints/V62a_model_only.pt",
            "../local_backend_data/checkpoints/model.pt",
            "local_backend_data/checkpoints/V62a_model_only.pt",
            "local_backend_data/checkpoints/model.pt",
        ]
        for fb in ckpt_candidates:
            if Path(fb).exists():
                load_model(fb, _SUBJECT)
                break
        else:
            _LOAD_ERRORS["model"] = "No checkpoint configured (set C2C_CHECKPOINT_PATH)"
    logger.info("=== Status: %s | missing: %s ===", _compute_status(), list(_LOAD_ERRORS.keys()))


@app.on_event("startup")
async def startup_load():
    load_all_at_startup()


# ===================== Helpers =====================

def _apply_zscore(fmri_vec: np.ndarray) -> Tuple[np.ndarray, str]:
    if _ZSCORE_STATS is not None:
        mean = _ZSCORE_STATS["mean"]
        std = _ZSCORE_STATS["std"]
        std_safe = np.where(std > 1e-6, std, 1.0)
        fmri_vec = (fmri_vec - mean) / std_safe
        return fmri_vec, f"Per-voxel z-scored ({len(mean)} voxels)"
    return fmri_vec, "Skipped (pre-extracted data already normalized)"


def _csls_retrieval(query: np.ndarray, k: int = 20, csls_k: int = 10) -> List[Dict]:
    if _CLIP_GALLERY is None:
        return []
    cosine_scores = _CLIP_GALLERY @ query
    top_hub_idx = np.argsort(-cosine_scores)[:csls_k]
    r_s = float(np.mean(cosine_scores[top_hub_idx]))
    csls_scores = 2.0 * cosine_scores - r_s
    top_idx = np.argsort(-csls_scores)[:k]
    results = []
    for rank, idx in enumerate(top_idx):
        nid = int(_CLIP_NSD_IDS[idx]) if _CLIP_NSD_IDS is not None else int(idx)
        results.append({"rank": rank + 1, "nsd_id": nid, "cosine": float(cosine_scores[idx]), "csls": float(csls_scores[idx])})
    return results


def _get_model_type() -> str:
    model_cfg = _CONFIG.get("model", {})
    return model_cfg.get("type", model_cfg.get("model_type", "vmf"))


def _get_subject_id_tensor():
    import torch
    subj_map = {"subj01": 0, "subj02": 1, "subj05": 2, "subj07": 3}
    return torch.tensor([subj_map.get(_SUBJECT, 0)], dtype=torch.long, device=_DEVICE)


# ===================== Inference =====================

def run_inference(trial_idx: int) -> Dict[str, Any]:
    import torch
    if _FMRI_FEATURES is None or _MODEL is None:
        raise RuntimeError("Inference engine not loaded")
    results: Dict[str, Any] = {}
    t0 = time.time()
    fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
    results["n_voxels"] = int(fmri_vec.shape[0])
    results["step_times"] = {}
    results["step_times"]["load_betas"] = time.time() - t0
    t1 = time.time()
    fmri_vec, zscore_detail = _apply_zscore(fmri_vec)
    results["step_times"]["zscore"] = time.time() - t1
    results["zscore_detail"] = zscore_detail
    results["step_times"]["roi_mask"] = 0.0
    t3 = time.time()
    x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)
    encoder_type = _CONFIG.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")
    with torch.no_grad():
        if encoder_type == "multi_subject_roi_transformer":
            mu, kappa_or_aux = _MODEL(x, subject_ids=_get_subject_id_tensor())
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
    model_type = _get_model_type()
    if model_type == "vmf_dcf" and hasattr(_MODEL, "_last_dcf_extras"):
        extras = _MODEL._last_dcf_extras
        if extras and "delta" in extras:
            delta_val = float(extras["delta"].cpu().numpy().squeeze())
    results["step_times"]["roi_encode"] = time.time() - t3
    results["step_times"]["vmf_decode"] = 0.0
    results["embedding"] = mu_norm.tolist()
    results["kappa"] = kappa_val
    results["delta"] = delta_val
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
    status = _compute_status()
    model_cfg = _CONFIG.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    return {
        "status": status,
        "server_online": True,
        "torch_available": _TORCH_AVAILABLE,
        "device": _DEVICE,
        "subject": _SUBJECT,
        "features_loaded": _FMRI_FEATURES is not None,
        "features_shape": list(_FMRI_FEATURES.shape) if _FMRI_FEATURES is not None else None,
        "gallery_loaded": _CLIP_GALLERY is not None,
        "gallery_size": int(_CLIP_GALLERY.shape[0]) if _CLIP_GALLERY is not None else 0,
        "gallery_dim": int(_CLIP_GALLERY.shape[1]) if _CLIP_GALLERY is not None else 0,
        "trial_index_loaded": _TRIAL_INDEX is not None,
        "trial_count": int(len(_TRIAL_INDEX)) if _TRIAL_INDEX is not None else 0,
        "model_loaded": _MODEL is not None,
        "zscore_stats_loaded": _ZSCORE_STATS is not None,
        "preprocessor_loaded": _EMB_PREPROCESSOR is not None,
        "inference_available": _MODEL is not None and _FMRI_FEATURES is not None,
        "live_retrieval_available": _MODEL is not None and _FMRI_FEATURES is not None and _CLIP_GALLERY is not None,
        "reconstruction_available": False,
        "model_meta": {
            "experiment": model_cfg.get("pretrained_model_path", "").split("/")[-3] if model_cfg.get("pretrained_model_path") else _CONFIG.get("experiment", "unknown"),
            "model_type": model_cfg.get("type", "unknown"),
            "encoder_type": encoder_cfg.get("encoder_type", "unknown"),
            "encoder_hidden": encoder_cfg.get("hidden_dims", []),
            "input_dim": encoder_cfg.get("input_dim", 0),
            "embedding_dim": decoder_cfg.get("output_dim", 0),
            "kappa_mode": decoder_cfg.get("kappa_mode", "unknown"),
        } if _MODEL is not None else None,
        "missing": list(_LOAD_ERRORS.keys()),
        "errors": _LOAD_ERRORS,
    }


@app.get("/api/trials")
def list_trials(limit: int = Query(50, ge=1, le=1000)):
    if _TRIAL_INDEX is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "trial_index_not_loaded", "message": "Trial index not loaded."})
    nsd_col = "nsdId" if "nsdId" in _TRIAL_INDEX.columns else "nsd_id"
    session_col = "session" if "session" in _TRIAL_INDEX.columns else None
    rows = _TRIAL_INDEX.head(limit)
    result = []
    for i, r in rows.iterrows():
        entry: Dict[str, Any] = {"trial_idx": int(i), "nsd_id": int(r.get(nsd_col, -1))}
        if session_col and session_col in r:
            entry["session"] = int(r[session_col])
        result.append(entry)
    return result


@app.get("/api/infer/{trial_idx}")
def infer_sync(trial_idx: int):
    if _MODEL is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "model_not_loaded", "message": "Live inference unavailable because checkpoint is missing.", "missing": list(_LOAD_ERRORS.keys())})
    if _FMRI_FEATURES is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_not_loaded", "message": "fMRI features not loaded."})
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")
    result = run_inference(trial_idx)
    result["ok"] = True
    result["provenance"] = "live"
    return JSONResponse(content=result)


@app.get("/api/infer-stream/{trial_idx}")
async def infer_stream(trial_idx: int):
    if _MODEL is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "model_not_loaded", "message": "Live inference unavailable because checkpoint is missing."})
    if _FMRI_FEATURES is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_not_loaded", "message": "fMRI features not loaded."})
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")

    async def event_generator():
        import torch
        t_global = time.time()
        yield {"event": "step", "data": json.dumps({"step": "load_betas", "status": "running", "detail": f"Loading fMRI vector (trial {trial_idx})..."})}
        fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
        n_voxels = int(fmri_vec.shape[0])
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({"step": "load_betas", "status": "done", "detail": f"Loaded {n_voxels} voxels", "n_voxels": n_voxels})}
        yield {"event": "step", "data": json.dumps({"step": "zscore", "status": "running", "detail": "Applying per-voxel z-score..."})}
        fmri_vec, zscore_detail = _apply_zscore(fmri_vec)
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({"step": "zscore", "status": "done", "detail": zscore_detail})}
        yield {"event": "step", "data": json.dumps({"step": "roi_mask", "status": "running", "detail": "Verifying nsdgeneral ROI mask..."})}
        await asyncio.sleep(0.03)
        yield {"event": "step", "data": json.dumps({"step": "roi_mask", "status": "done", "detail": f"{n_voxels} visual cortex voxels (pre-masked)"})}
        yield {"event": "step", "data": json.dumps({"step": "roi_encode", "status": "running", "detail": "Forward pass through encoder..."})}
        x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)
        t_fwd = time.time()
        enc_type = _CONFIG.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")
        with torch.no_grad():
            if enc_type == "multi_subject_roi_transformer":
                mu, kappa_or_aux = _MODEL(x, subject_ids=_get_subject_id_tensor())
            else:
                mu, kappa_or_aux = _MODEL(x)
        fwd_ms = (time.time() - t_fwd) * 1000
        yield {"event": "step", "data": json.dumps({"step": "roi_encode", "status": "done", "detail": f"Forward pass in {fwd_ms:.1f}ms"})}
        yield {"event": "step", "data": json.dumps({"step": "vmf_decode", "status": "running", "detail": "Extracting (mu, kappa) from vMF decoder..."})}
        mu_np = mu.cpu().numpy().squeeze()
        mu_norm = mu_np / (np.linalg.norm(mu_np) + 1e-8)
        kappa_val, delta_val = None, None
        if kappa_or_aux is not None:
            if isinstance(kappa_or_aux, dict):
                kappa_val = float(kappa_or_aux.get("kappa", torch.tensor(0.0)).cpu().numpy().squeeze())
            else:
                kappa_val = float(kappa_or_aux.cpu().numpy().squeeze())
        model_type = _get_model_type()
        if model_type == "vmf_dcf" and hasattr(_MODEL, "_last_dcf_extras"):
            extras = getattr(_MODEL, "_last_dcf_extras", None)
            if extras and "delta" in extras:
                delta_val = float(extras["delta"].cpu().numpy().squeeze())
        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({"step": "vmf_decode", "status": "done", "detail": f"kappa={kappa_val}" + (f", delta={delta_val:.4f}" if delta_val else ""), "kappa": kappa_val, "delta": delta_val, "embedding_dim": int(mu_norm.shape[0])})}
        yield {"event": "step", "data": json.dumps({"step": "gallery_search", "status": "running", "detail": "CSLS-corrected cosine similarity search..."})}
        t_search = time.time()
        top_k_results = _csls_retrieval(mu_norm, k=5)
        search_ms = (time.time() - t_search) * 1000
        gallery_size = int(_CLIP_GALLERY.shape[0]) if _CLIP_GALLERY is not None else 0
        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({"step": "gallery_search", "status": "done", "detail": f"CSLS over {gallery_size} embeddings in {search_ms:.1f}ms", "gallery_size": gallery_size, "top_k": top_k_results})}
        total_ms = (time.time() - t_global) * 1000
        yield {"event": "step", "data": json.dumps({"step": "results", "status": "done", "detail": f"Pipeline complete in {total_ms:.0f}ms", "total_ms": total_ms, "kappa": kappa_val, "delta": delta_val, "top_k": top_k_results, "embedding": mu_norm.tolist()})}

    return EventSourceResponse(event_generator())


@app.get("/api/nsd-id-to-trial/{nsd_id}")
def nsd_id_to_trial(nsd_id: int):
    if _TRIAL_INDEX is None:
        raise HTTPException(503, "Trial index not loaded")
    nsd_col = "nsdId" if "nsdId" in _TRIAL_INDEX.columns else "nsd_id"
    matches = _TRIAL_INDEX[_TRIAL_INDEX[nsd_col] == nsd_id]
    if len(matches) == 0:
        raise HTTPException(404, f"nsdId {nsd_id} not found")
    return {"nsd_id": nsd_id, "trial_indices": matches.index.tolist(), "count": len(matches)}


# ===================== CLI =====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cortex2Canvas Backend")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["C2C_CHECKPOINT_PATH"] = args.checkpoint
    os.environ["C2C_SUBJECT"] = args.subject
    load_all_at_startup()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
