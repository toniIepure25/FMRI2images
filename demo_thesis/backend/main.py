"""
Cortex2Canvas Live Inference Backend — v4.0 (multi-model, honesty-audited)

Backend modes (C2C_BACKEND_MODE env var):
  v62_single        — V62a MLP (768-D), single forward pass     [default]
  v61_single        — V61a dual-head (197K-D), single forward pass
  v61_mctta         — V61a with MC-TTA dropout averaging
  fusion_v61_v62    — V61a MC-TTA + V62a, z-score fusion, w=0.80

Artifact env vars:
  C2C_V62_CHECKPOINT_PATH, C2C_V61_CHECKPOINT_PATH,
  C2C_CLIP_GALLERY_768_PATH, C2C_CLIP_GALLERY_197K_PATH,
  C2C_TOKEN_CACHE_PATH, C2C_CSLS_K, C2C_MCTTA_SAMPLES,
  C2C_FUSION_WEIGHT_V61, C2C_DEVICE

Every inference response includes provenance per step.
Reconstruction is never faked — only reported as available if truly live.
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

# Auto-resolve fmri2img package path (root src/ directory)
_ROOT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_ROOT_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT_SRC))

logger = logging.getLogger("cortex2canvas")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

# ── Globals ──
_TORCH_AVAILABLE = False
_DEVICE = "cpu"

# V62a (768-D MLP)
_V62_MODEL = None
_V62_CONFIG: Dict[str, Any] = {}

# V61a (197K-D dual-head)
_V61_MODEL = None
_V61_CONFIG: Dict[str, Any] = {}

# Shared data
_FMRI_FEATURES: Optional[np.ndarray] = None
_TRIAL_INDEX = None
_CLIP_GALLERY_768: Optional[np.ndarray] = None
_CLIP_NSD_IDS_768: Optional[np.ndarray] = None
_CLIP_GALLERY_197K: Optional[np.ndarray] = None
_CLIP_NSD_IDS_197K: Optional[np.ndarray] = None
_ZSCORE_STATS: Optional[Dict[str, np.ndarray]] = None
_SUBJECT = "subj01"
_ROI_INDICES = None
_LOAD_ERRORS: Dict[str, str] = {}

# Backend mode
_BACKEND_MODE = "v62_single"
_EFFECTIVE_MODE = "v62_single"
_FALLBACK_USED = False
_FALLBACK_REASON: Optional[str] = None

# MC-TTA / Fusion params
_MCTTA_SAMPLES = 16
_CSLS_K = 10
_FUSION_WEIGHT_V61 = 0.80
_FUSION_METHOD = "zscore"

# ── FastAPI setup ──
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    raise

app = FastAPI(title="Cortex2Canvas Live Inference", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Static file serving for reconstruction outputs
_recon_live_dir = Path("local_backend_data/reconstructions/live")
_recon_live_dir.mkdir(parents=True, exist_ok=True)
_recon_cached_dir = Path("local_backend_data/reconstructions/cached")
_recon_cached_dir.mkdir(parents=True, exist_ok=True)

# Import recon module (lazy, only used when endpoints are called)
_RECON = None


def _get_recon():
    global _RECON
    if _RECON is None:
        try:
            import recon as r
            _RECON = r
        except ImportError:
            logger.warning("recon module not available")
            _RECON = False
    return _RECON


# =========== Helpers ===========

def _resolve_path(env_var: str, *fallbacks: str) -> Optional[Path]:
    val = os.environ.get(env_var)
    if val and Path(val).exists():
        return Path(val)
    for fb in fallbacks:
        if Path(fb).exists():
            return Path(fb)
    return None


def _read_config_from_env() -> None:
    global _BACKEND_MODE, _CSLS_K, _MCTTA_SAMPLES, _FUSION_WEIGHT_V61, _FUSION_METHOD
    _BACKEND_MODE = os.environ.get("C2C_BACKEND_MODE", "v62_single")
    _CSLS_K = int(os.environ.get("C2C_CSLS_K", "10"))
    _MCTTA_SAMPLES = int(os.environ.get("C2C_MCTTA_SAMPLES", "16"))
    _FUSION_WEIGHT_V61 = float(os.environ.get("C2C_FUSION_WEIGHT_V61", "0.80"))
    _FUSION_METHOD = os.environ.get("C2C_FUSION_METHOD", "zscore")


# =========== Loaders ===========

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
    path = _resolve_path("C2C_FMRI_FEATURES_PATH",
        f"../local_backend_data/{subject}/fmri_features.npy",
        f"local_backend_data/{subject}/fmri_features.npy")
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
    path = _resolve_path("C2C_TRIAL_INDEX_PATH",
        f"../local_backend_data/{subject}/index.parquet",
        f"local_backend_data/{subject}/index.parquet")
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


def _load_gallery(path_var: str, *fallbacks: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    path = _resolve_path(path_var, *fallbacks)
    if not path:
        return None, None, False
    try:
        import pandas as pd
        df = pd.read_parquet(str(path))
        emb_col = next((c for c in ("fused", "final", "embedding", "clip_embedding") if c in df.columns), None)
        if not emb_col:
            _LOAD_ERRORS["gallery"] = f"No embedding column in {list(df.columns)}"
            return None, None, False
        ids = df["nsdId"].values
        gallery = np.stack(df[emb_col].values).astype(np.float32)
        norms = np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8)
        gallery = gallery / norms
        logger.info("Gallery %s: %s col='%s' from %s", path_var, gallery.shape, emb_col, path)
        return gallery, ids, True
    except Exception as e:
        _LOAD_ERRORS[f"gallery_{path_var}"] = str(e)
        return None, None, False


def load_clip_gallery_768() -> bool:
    global _CLIP_GALLERY_768, _CLIP_NSD_IDS_768
    g, ids, ok = _load_gallery("C2C_CLIP_GALLERY_768_PATH",
        "../local_backend_data/clip/clip.parquet",
        "local_backend_data/clip/clip.parquet")
    _CLIP_GALLERY_768, _CLIP_NSD_IDS_768 = g, ids
    if not ok:
        _LOAD_ERRORS["gallery_768"] = _LOAD_ERRORS.pop(f"gallery_C2C_CLIP_GALLERY_768_PATH", "clip.parquet not found")
    return ok


def load_clip_gallery_197k() -> bool:
    global _CLIP_GALLERY_197K, _CLIP_NSD_IDS_197K
    g, ids, ok = _load_gallery("C2C_CLIP_GALLERY_197K_PATH")
    _CLIP_GALLERY_197K, _CLIP_NSD_IDS_197K = g, ids
    if not ok and "C2C_CLIP_GALLERY_197K_PATH" in _LOAD_ERRORS:
        pass  # Keep original error
    return ok


def _load_one_model(checkpoint_path: str, label: str) -> Tuple[Any, Dict[str, Any], bool]:
    if not _TORCH_AVAILABLE:
        return None, {}, False
    import torch
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        _LOAD_ERRORS[f"{label}_checkpoint"] = f"Not found: {checkpoint_path}"
        return None, {}, False
    try:
        ckpt = torch.load(str(ckpt_p), map_location=_DEVICE, weights_only=False)
    except Exception as e:
        _LOAD_ERRORS[f"{label}_checkpoint"] = f"Load failed: {e}"
        return None, {}, False

    model_config = ckpt.get("model_config", ckpt.get("config", {}).get("model", {}))
    config = ckpt.get("config", {})

    roi_indices = None
    encoder_cfg = model_config.get("encoder", {})
    encoder_type = encoder_cfg.get("encoder_type", "mlp")
    if encoder_type in ("roi_transformer", "multi_subject_roi_transformer"):
        try:
            from fmri2img.data.roi_utils import build_roi_index
            roi_names = list(encoder_cfg.get("roi_dims", {}).keys())
            if roi_names:
                actual_dims, roi_indices = build_roi_index(_SUBJECT, roi_names)
                model_config["encoder"]["roi_dims"] = dict(actual_dims)
        except Exception as e:
            logger.warning("[%s] ROI indices failed: %s", label, e)

    try:
        from fmri2img.models.unified_model import create_model
        model = create_model(model_config, roi_indices=roi_indices)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(_DEVICE)
        model.eval()
        logger.info("[%s] Model loaded: type=%s encoder=%s", label, model_config.get("type", "?"), encoder_type)
        return model, config, True
    except Exception as e:
        _LOAD_ERRORS[f"{label}_model"] = f"Build failed: {e}"
        return None, {}, False


def load_v62_model() -> bool:
    global _V62_MODEL, _V62_CONFIG, _ROI_INDICES, _SUBJECT
    path = os.environ.get("C2C_V62_CHECKPOINT_PATH", "")
    candidates = [
        path,
        "../local_backend_data/checkpoints/V62a_model_only.pt",
        "local_backend_data/checkpoints/V62a_model_only.pt",
    ]
    for fp in candidates:
        if fp and Path(fp).exists():
            model, cfg, ok = _load_one_model(fp, "v62")
            if ok:
                _V62_MODEL, _V62_CONFIG = model, cfg
                ckpt_subject = cfg.get("subject", _SUBJECT)
                if ckpt_subject:
                    _SUBJECT = ckpt_subject
                _LOAD_ERRORS.pop("v62_checkpoint", None)
                _LOAD_ERRORS.pop("v62_model", None)
                return True
    _LOAD_ERRORS["v62_model"] = "V62a checkpoint not found"
    return False


def load_v61_model() -> bool:
    global _V61_MODEL, _V61_CONFIG
    path = os.environ.get("C2C_V61_CHECKPOINT_PATH", "")
    if path and Path(path).exists():
        model, cfg, ok = _load_one_model(path, "v61")
        if ok:
            _V61_MODEL, _V61_CONFIG = model, cfg
            _LOAD_ERRORS.pop("v61_checkpoint", None)
            _LOAD_ERRORS.pop("v61_model", None)
            return True
    _LOAD_ERRORS["v61_model"] = "V61a checkpoint not found (set C2C_V61_CHECKPOINT_PATH)"
    return False


# =========== Z-score / preprocessing ===========

def _load_zscore_stats():
    global _ZSCORE_STATS
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    zscore_dir = Path(cache_root) / "preproc" / f"subject={_SUBJECT}" / "zscore_stats"
    if zscore_dir.exists():
        try:
            _ZSCORE_STATS = {
                "mean": np.load(str(zscore_dir / "voxel_mean.npy")),
                "std": np.load(str(zscore_dir / "voxel_std.npy")),
            }
        except Exception:
            pass


def _apply_zscore(fmri_vec: np.ndarray) -> Tuple[np.ndarray, str]:
    if _ZSCORE_STATS is not None:
        mean = _ZSCORE_STATS["mean"]
        std = _ZSCORE_STATS["std"]
        std_safe = np.where(std > 1e-6, std, 1.0)
        return (fmri_vec - mean) / std_safe, f"Per-voxel z-scored ({len(mean)} voxels)"
    return fmri_vec, "Skipped (pre-extracted / pre-normalized features)"


# =========== CSLS retrieval ===========

def _csls_retrieval(query: np.ndarray, gallery: np.ndarray, ids: np.ndarray,
                     k: int, csls_k: int) -> List[Dict]:
    if gallery is None or gallery.shape[0] == 0:
        return []
    cosine_scores = gallery @ query
    top_hub_idx = np.argsort(-cosine_scores)[:min(csls_k, len(cosine_scores))]
    r_s = float(np.mean(cosine_scores[top_hub_idx]))
    csls_scores = 2.0 * cosine_scores - r_s
    top_idx = np.argsort(-csls_scores)[:k]
    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            "rank": rank + 1,
            "nsd_id": int(ids[idx]),
            "cosine": float(cosine_scores[idx]),
            "csls": float(csls_scores[idx]),
        })
    return results


# =========== Model forward ===========

def _model_forward(model, x: "torch.Tensor", config: Dict[str, Any], encoder_type: str = "mlp",
                   enable_dropout: bool = False) -> Dict[str, Any]:
    import torch
    with torch.no_grad():
        if enable_dropout:
            # Enable dropout for MC-TTA
            def _enable_dropout(m):
                if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                    m.train()
            model.apply(_enable_dropout)

        if encoder_type == "multi_subject_roi_transformer":
            subj_map = {"subj01": 0, "subj02": 1, "subj05": 2, "subj07": 3}
            sids = torch.tensor([subj_map.get(_SUBJECT, 0)], dtype=torch.long, device=_DEVICE)
            mu, kappa_or_aux = model(x, subject_ids=sids)
        else:
            mu, kappa_or_aux = model(x)

        if enable_dropout:
            model.eval()

    mu_np = mu.cpu().numpy().squeeze()
    mu_norm = mu_np / (np.linalg.norm(mu_np) + 1e-8)

    kappa_val = None
    delta_val = None
    if kappa_or_aux is not None:
        if isinstance(kappa_or_aux, dict):
            kappa_val = float(kappa_or_aux.get("kappa", torch.tensor(0.0)).cpu().numpy().squeeze())
        else:
            kappa_val = float(kappa_or_aux.cpu().numpy().squeeze())

    mtype = config.get("model", {}).get("type", "vmf")
    if mtype == "vmf_dcf" and hasattr(model, "_last_dcf_extras"):
        extras = model._last_dcf_extras
        if extras and "delta" in extras:
            delta_val = float(extras["delta"].cpu().numpy().squeeze())

    return {"mu": mu_norm, "kappa": kappa_val, "delta": delta_val, "dim": int(mu_norm.shape[0])}


def _forward_mctta(model, x, config, encoder_type: str, n_samples: int = 16) -> Dict[str, Any]:
    """MC-TTA: Multiple stochastic forward passes with dropout enabled, average mu, then re-normalize."""
    import torch
    mus = []
    kappas = []
    dim = None
    for _ in range(n_samples):
        result = _model_forward(model, x, config, encoder_type, enable_dropout=True)
        mus.append(result["mu"])
        if result["kappa"] is not None:
            kappas.append(result["kappa"])
        if dim is None:
            dim = result["dim"]
    mu_avg = np.mean(np.stack(mus), axis=0)
    mu_norm = mu_avg / (np.linalg.norm(mu_avg) + 1e-8)
    kappa_avg = float(np.mean(kappas)) if kappas else None
    return {"mu": mu_norm, "kappa": kappa_avg, "delta": None, "dim": dim or 768}


def _get_encoder_type(config: Dict[str, Any]) -> str:
    return config.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")


# =========== Inference pipeline ===========

def run_inference(trial_idx: int) -> Dict[str, Any]:
    import torch
    if _FMRI_FEATURES is None:
        raise RuntimeError("Inference engine not loaded")

    results: Dict[str, Any] = {"mode": _EFFECTIVE_MODE, "fallback_used": _FALLBACK_USED}
    if _FALLBACK_REASON:
        results["fallback_reason"] = _FALLBACK_REASON

    t0 = time.time()
    fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
    n_voxels = int(fmri_vec.shape[0])
    results["n_voxels"] = n_voxels
    results["step_times_ms"] = {"load_betas": (time.time() - t0) * 1000}

    # Build provenance
    prov: Dict[str, str] = {
        "load_betas": "CACHED_LOCAL",
        "preprocessing": "CACHED_LOCAL" if _ZSCORE_STATS is None else "LIVE_LOCAL",
        "roi_mask": "CACHED_LOCAL",
        "model_forward": "LIVE_LOCAL",
        "vmf_projection": "LIVE_LOCAL",
        "retrieval": "LIVE_LOCAL",
        "top_k": "LIVE_LOCAL",
        "reconstruction": "UNAVAILABLE",
    }

    t_z = time.time()
    fmri_vec, zscore_detail = _apply_zscore(fmri_vec)
    results["step_times_ms"]["preprocessing"] = (time.time() - t_z) * 1000
    results["zscore_detail"] = zscore_detail
    results["step_times_ms"]["roi_mask"] = 0.0

    x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)

    # ── Mode dispatch ──
    if _EFFECTIVE_MODE == "fusion_v61_v62" and _V61_MODEL is not None and _V62_MODEL is not None:
        t_fwd = time.time()
        enc61 = _get_encoder_type(_V61_CONFIG)
        enc62 = _get_encoder_type(_V62_CONFIG)

        # V61a: MC-TTA over 197K-D
        v61_result = _forward_mctta(_V61_MODEL, x, _V61_CONFIG, enc61, _MCTTA_SAMPLES)
        # V62a: single deterministic forward over 768-D
        v62_result = _model_forward(_V62_MODEL, x, _V62_CONFIG, enc62, enable_dropout=False)

        results["step_times_ms"]["model_forward"] = (time.time() - t_fwd) * 1000
        results["step_times_ms"]["vmf_projection"] = 0.0
        results["kappa"] = v62_result["kappa"]
        results["delta"] = v62_result["delta"]
        results["v61_mu_dim"] = v61_result["dim"]
        results["v62_mu_dim"] = v62_result["dim"]
        prov["mctta"] = "LIVE_LOCAL"
        prov["fusion"] = "LIVE_LOCAL"

        t_search = time.time()
        if _CLIP_GALLERY_197K is not None and _CLIP_GALLERY_768 is not None and \
           _CLIP_NSD_IDS_197K is not None and _CLIP_NSD_IDS_768 is not None:
            # Run both retrievals
            v61_top = _csls_retrieval(v61_result["mu"], _CLIP_GALLERY_197K, _CLIP_NSD_IDS_197K, k=50, csls_k=3)
            v62_top = _csls_retrieval(v62_result["mu"], _CLIP_GALLERY_768, _CLIP_NSD_IDS_768, k=50, csls_k=10)
            results["gallery_size_197k"] = int(_CLIP_GALLERY_197K.shape[0])
            results["gallery_size_768"] = int(_CLIP_GALLERY_768.shape[0])

            # Build score dicts by nsdId
            v61_scores = {r["nsd_id"]: r["csls"] for r in v61_top}
            v62_scores = {r["nsd_id"]: r["csls"] for r in v62_top}
            all_ids = sorted(set(list(v61_scores.keys()) + list(v62_scores.keys())))

            # Z-score normalize
            v61_vals = np.array([v61_scores.get(nid, -999) for nid in all_ids])
            v62_vals = np.array([v62_scores.get(nid, -999) for nid in all_ids])
            v61_z = (v61_vals - np.mean(v61_vals)) / (np.std(v61_vals) + 1e-8)
            v62_z = (v62_vals - np.mean(v62_vals)) / (np.std(v62_vals) + 1e-8)
            fused = _FUSION_WEIGHT_V61 * v61_z + (1 - _FUSION_WEIGHT_V61) * v62_z
            top_idx = np.argsort(-fused)[:_CSLS_K]

            top_k = []
            for rank, idx in enumerate(top_idx):
                nid = all_ids[idx]
                top_k.append({
                    "rank": rank + 1,
                    "nsd_id": nid,
                    "fused_score": float(fused[idx]),
                    "v61_csls": float(v61_scores.get(nid, -999)),
                    "v62_csls": float(v62_scores.get(nid, -999)),
                    "csls": float(fused[idx]),
                })
            results["top_k"] = top_k
            results["fusion"] = {
                "method": _FUSION_METHOD,
                "weight_v61": _FUSION_WEIGHT_V61,
                "candidate_universe": "union_10k",
                "computed_live": True,
            }
        else:
            results["top_k"] = []
            results["missed_fusion"] = "One or both galleries missing"
        results["step_times_ms"]["gallery_search"] = (time.time() - t_search) * 1000

    elif _EFFECTIVE_MODE == "v61_mctta" and _V61_MODEL is not None:
        t_fwd = time.time()
        enc_type = _get_encoder_type(_V61_CONFIG)
        result = _forward_mctta(_V61_MODEL, x, _V61_CONFIG, enc_type, _MCTTA_SAMPLES)
        results["step_times_ms"]["model_forward"] = (time.time() - t_fwd) * 1000
        results["step_times_ms"]["vmf_projection"] = 0.0
        results["kappa"] = result["kappa"]
        results["delta"] = result["delta"]
        results["embedding_dim"] = result["dim"]
        results["mctta"] = {"enabled": True, "samples": _MCTTA_SAMPLES, "aggregation": "embedding_mean", "computed_live": True}
        prov["mctta"] = "LIVE_LOCAL"

        t_search = time.time()
        g, ids = (_CLIP_GALLERY_197K, _CLIP_NSD_IDS_197K)
        if g is not None and ids is not None:
            results["top_k"] = _csls_retrieval(result["mu"], g, ids, k=_CSLS_K, csls_k=3)
            results["gallery_size"] = int(g.shape[0])
        else:
            results["top_k"] = []
            results["gallery_size"] = 0
        results["step_times_ms"]["gallery_search"] = (time.time() - t_search) * 1000

    elif _EFFECTIVE_MODE == "v61_single" and _V61_MODEL is not None:
        t_fwd = time.time()
        enc_type = _get_encoder_type(_V61_CONFIG)
        result = _model_forward(_V61_MODEL, x, _V61_CONFIG, enc_type, enable_dropout=False)
        results["step_times_ms"]["model_forward"] = (time.time() - t_fwd) * 1000
        results["step_times_ms"]["vmf_projection"] = 0.0
        results["kappa"] = result["kappa"]
        results["delta"] = result["delta"]
        results["embedding_dim"] = result["dim"]

        t_search = time.time()
        g, ids = (_CLIP_GALLERY_197K, _CLIP_NSD_IDS_197K)
        if g is not None and ids is not None:
            results["top_k"] = _csls_retrieval(result["mu"], g, ids, k=_CSLS_K, csls_k=3)
            results["gallery_size"] = int(g.shape[0])
        else:
            results["top_k"] = []
            results["gallery_size"] = 0
        results["step_times_ms"]["gallery_search"] = (time.time() - t_search) * 1000

    else:
        # v62_single (default / fallback)
        t_fwd = time.time()
        result = _model_forward(_V62_MODEL, x, _V62_CONFIG, "mlp", enable_dropout=False)
        results["step_times_ms"]["model_forward"] = (time.time() - t_fwd) * 1000
        results["step_times_ms"]["vmf_projection"] = 0.0
        results["kappa"] = result["kappa"]
        results["delta"] = result["delta"]
        results["embedding_dim"] = result["dim"]
        results["_mu_for_recon"] = result["mu"].tolist()

        t_search = time.time()
        g, ids = (_CLIP_GALLERY_768, _CLIP_NSD_IDS_768)
        if g is not None and ids is not None:
            results["top_k"] = _csls_retrieval(result["mu"], g, ids, k=_CSLS_K, csls_k=_CSLS_K)
            results["gallery_size"] = int(g.shape[0])
        else:
            results["top_k"] = []
            results["gallery_size"] = 0
        results["step_times_ms"]["gallery_search"] = (time.time() - t_search) * 1000

    total_ms = (time.time() - t0) * 1000
    results["total_ms"] = total_ms
    results["provenance"] = prov
    return results


# =========== Status / mode resolution ===========

def _compute_status() -> str:
    model_loaded = _V62_MODEL is not None or _V61_MODEL is not None
    if model_loaded and _FMRI_FEATURES is not None and (_CLIP_GALLERY_768 is not None or _CLIP_GALLERY_197K is not None):
        return "inference_ready"
    if _FMRI_FEATURES is not None or _CLIP_GALLERY_768 is not None or _TRIAL_INDEX is not None:
        return "data_ready"
    return "server_online"


def _resolve_effective_mode() -> str:
    global _EFFECTIVE_MODE, _FALLBACK_USED, _FALLBACK_REASON
    target = _BACKEND_MODE

    needs_v61 = target in ("v61_single", "v61_mctta", "fusion_v61_v62")
    needs_v62 = target in ("v62_single", "fusion_v61_v62")
    needs_197k = target in ("v61_single", "v61_mctta", "fusion_v61_v62")

    if needs_v62 and _V62_MODEL is None:
        _EFFECTIVE_MODE = "server_online"
        _FALLBACK_USED = True
        _FALLBACK_REASON = "V62a model missing; falling back to data-only"
        return _EFFECTIVE_MODE

    if needs_v61 and _V61_MODEL is None:
        if _V62_MODEL is not None:
            _EFFECTIVE_MODE = "v62_single"
            _FALLBACK_USED = True
            _FALLBACK_REASON = "V61a model missing; falling back to V62a single"
            return _EFFECTIVE_MODE
        _EFFECTIVE_MODE = "server_online"
        _FALLBACK_USED = True
        _FALLBACK_REASON = "Requested mode requires V61a which is missing"
        return _EFFECTIVE_MODE

    if needs_197k and _CLIP_GALLERY_197K is None:
        if _V62_MODEL is not None:
            _EFFECTIVE_MODE = "v62_single"
            _FALLBACK_USED = True
            _FALLBACK_REASON = "197K-D gallery missing; falling back to V62a single"
            return _EFFECTIVE_MODE
        _EFFECTIVE_MODE = "data_ready"
        _FALLBACK_USED = True
        _FALLBACK_REASON = "197K-D gallery missing for requested mode"
        return _EFFECTIVE_MODE

    if target == "fusion_v61_v62" and (_CLIP_GALLERY_768 is None or _CLIP_GALLERY_197K is None):
        _FALLBACK_USED = True
        if _V62_MODEL is not None and _CLIP_GALLERY_768 is not None:
            _EFFECTIVE_MODE = "v61_mctta" if _V61_MODEL is not None and _CLIP_GALLERY_197K is not None else "v62_single"
            _FALLBACK_REASON = "Fusion requires both galleries; one missing"
        return _EFFECTIVE_MODE

    _EFFECTIVE_MODE = target
    _FALLBACK_USED = False
    _FALLBACK_REASON = None
    return _EFFECTIVE_MODE


# =========== Startup ===========

def load_all_at_startup():
    global _SUBJECT
    _SUBJECT = os.environ.get("C2C_SUBJECT", "subj01")
    _read_config_from_env()
    logger.info("=== Cortex2Canvas v4.0 startup: subject=%s mode=%s ===", _SUBJECT, _BACKEND_MODE)
    load_torch()
    load_features(_SUBJECT)
    load_trial_index(_SUBJECT)
    load_clip_gallery_768()
    load_clip_gallery_197k()
    _load_zscore_stats()

    if _BACKEND_MODE in ("v62_single", "fusion_v61_v62"):
        load_v62_model()
    if _BACKEND_MODE in ("v61_single", "v61_mctta", "fusion_v61_v62"):
        load_v61_model()

    _resolve_effective_mode()
    logger.info("=== Effective mode: %s | fallback=%s | reason=%s | status=%s ===",
                _EFFECTIVE_MODE, _FALLBACK_USED, _FALLBACK_REASON, _compute_status())
    logger.info("=== Missing: %s ===", list(_LOAD_ERRORS.keys()))


@app.on_event("startup")
async def startup_load():
    load_all_at_startup()


# =========== API ===========

@app.get("/api/health")
def health():
    v62_cfg = _V62_CONFIG.get("model", {})
    v62_enc = v62_cfg.get("encoder", {})
    v62_dec = v62_cfg.get("decoder", {})

    # Get reconstruction status
    recon_status: Dict[str, Any] = {}
    r = _get_recon()
    if r:
        try:
            recon_status = r.get_status()
        except Exception:
            recon_status = {"reconstruction_available": False, "reconstruction_mode": "error"}
    else:
        recon_status = {"reconstruction_available": False, "reconstruction_mode": "disabled"}

    base = {
        "status": _compute_status(),
        "server_online": True,
        "torch_available": _TORCH_AVAILABLE,
        "device": _DEVICE,
        "subject": _SUBJECT,
        "backend_mode": _BACKEND_MODE,
        "effective_mode": _EFFECTIVE_MODE,
        "fallback_used": _FALLBACK_USED,
        "fallback_reason": _FALLBACK_REASON,
        # Models
        "v62_model_loaded": _V62_MODEL is not None,
        "v61_model_loaded": _V61_MODEL is not None,
        "active_models": [m for m, loaded in [("v62", _V62_MODEL is not None), ("v61", _V61_MODEL is not None)] if loaded],
        # Legacy backward-compat fields
        "model_loaded": _V62_MODEL is not None or _V61_MODEL is not None,
        "gallery_loaded": _CLIP_GALLERY_768 is not None or _CLIP_GALLERY_197K is not None,
        "gallery_size": int((_CLIP_GALLERY_768 if _CLIP_GALLERY_768 is not None else _CLIP_GALLERY_197K).shape[0]) if (_CLIP_GALLERY_768 is not None or _CLIP_GALLERY_197K is not None) else 0,
        "gallery_dim": int((_CLIP_GALLERY_768 if _CLIP_GALLERY_768 is not None else _CLIP_GALLERY_197K).shape[1]) if (_CLIP_GALLERY_768 is not None or _CLIP_GALLERY_197K is not None) else 0,
        # Data sources
        "features_loaded": _FMRI_FEATURES is not None,
        "features_shape": list(_FMRI_FEATURES.shape) if _FMRI_FEATURES is not None else None,
        "gallery_768_loaded": _CLIP_GALLERY_768 is not None,
        "gallery_768_size": int(_CLIP_GALLERY_768.shape[0]) if _CLIP_GALLERY_768 is not None else 0,
        "gallery_768_dim": int(_CLIP_GALLERY_768.shape[1]) if _CLIP_GALLERY_768 is not None else 0,
        "gallery_197k_loaded": _CLIP_GALLERY_197K is not None,
        "gallery_197k_size": int(_CLIP_GALLERY_197K.shape[0]) if _CLIP_GALLERY_197K is not None else 0,
        "gallery_197k_dim": int(_CLIP_GALLERY_197K.shape[1]) if _CLIP_GALLERY_197K is not None else 0,
        "trial_index_loaded": _TRIAL_INDEX is not None,
        "trial_count": int(len(_TRIAL_INDEX)) if _TRIAL_INDEX is not None else 0,
        "zscore_stats_loaded": _ZSCORE_STATS is not None,
        # Capabilities
        "inference_available": (_V62_MODEL is not None or _V61_MODEL is not None) and _FMRI_FEATURES is not None,
        "live_retrieval_available": (_V62_MODEL is not None or _V61_MODEL is not None) and _FMRI_FEATURES is not None and (_CLIP_GALLERY_768 is not None or _CLIP_GALLERY_197K is not None),
        "mctta_available": _V61_MODEL is not None and _EFFECTIVE_MODE in ("v61_mctta", "fusion_v61_v62"),
        "fusion_available": _V61_MODEL is not None and _V62_MODEL is not None and _CLIP_GALLERY_197K is not None and _CLIP_GALLERY_768 is not None and _EFFECTIVE_MODE == "fusion_v61_v62",
        **recon_status,
        # Config
        "csls_k": _CSLS_K,
        "mctta_samples": _MCTTA_SAMPLES if _V61_MODEL is not None else None,
        "fusion_weight_v61": _FUSION_WEIGHT_V61 if _EFFECTIVE_MODE == "fusion_v61_v62" else None,
        "fusion_method": _FUSION_METHOD if _EFFECTIVE_MODE == "fusion_v61_v62" else None,
        # Model metadata (V62a primary)
        "model_meta": {
            "experiment": _V62_CONFIG.get("experiment", "unknown"),
            "model_type": v62_cfg.get("type", "unknown"),
            "encoder_type": v62_enc.get("encoder_type", "unknown"),
            "encoder_hidden": v62_enc.get("hidden_dims", []),
            "input_dim": v62_enc.get("input_dim", 0),
            "embedding_dim": v62_dec.get("output_dim", 0),
            "kappa_mode": v62_dec.get("kappa_mode", "unknown"),
        } if _V62_MODEL is not None else None,
        "missing": list(_LOAD_ERRORS.keys()),
        "errors": _LOAD_ERRORS,
    }
    return base


@app.get("/api/infer/{trial_idx}")
def infer_sync(trial_idx: int, include_reconstruction: bool = Query(False)):
    if _FMRI_FEATURES is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_not_loaded", "message": "fMRI features not loaded."})
    if _V62_MODEL is None and _V61_MODEL is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "no_model", "message": "No model loaded.", "fallback": "Try with different C2C_BACKEND_MODE"})
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")

    result = run_inference(trial_idx)
    result["ok"] = True
    # Get mu from inference result (computed once during run_inference)
    mu_for_recon = result.pop("_mu_for_recon", None)

    # Optionally run reconstruction from the live predicted mu
    if include_reconstruction and mu_for_recon is not None:
        import numpy as np
        nsd_id = None
        if _TRIAL_INDEX is not None:
            nsd_col = "nsdId" if "nsdId" in _TRIAL_INDEX.columns else "nsd_id"
            if trial_idx < len(_TRIAL_INDEX):
                nsd_id = int(_TRIAL_INDEX.iloc[trial_idx][nsd_col])

        r = _get_recon()
        if r:
            recon_result = r.reconstruct(trial_idx, np.array(mu_for_recon), nsd_id)
            recon_result.pop("image_path", None)
            result["reconstruction"] = recon_result
        else:
            result["reconstruction"] = {"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
                                        "reason": "Reconstruction module not available"}
    elif include_reconstruction:
        result["reconstruction"] = {"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
                                    "reason": "Could not extract predicted embedding from inference"}

    # Remove full embedding from default response
    result.pop("embedding", None)
    return JSONResponse(content=result)


@app.get("/api/infer-stream/{trial_idx}")
async def infer_stream(trial_idx: int):
    if _FMRI_FEATURES is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_not_loaded"})
    if _V62_MODEL is None and _V61_MODEL is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "no_model"})
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")

    async def event_generator():
        import torch
        t_global = time.time()

        yield {"event": "step", "data": json.dumps({"step": "load_betas", "status": "running", "provenance": "CACHED_LOCAL"})}
        fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
        n_voxels = int(fmri_vec.shape[0])
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({"step": "load_betas", "status": "done", "n_voxels": n_voxels, "provenance": "CACHED_LOCAL"})}

        yield {"event": "step", "data": json.dumps({"step": "zscore", "status": "running", "provenance": "LIVE_LOCAL" if _ZSCORE_STATS is not None else "CACHED_LOCAL"})}
        fmri_vec, zscore_detail = _apply_zscore(fmri_vec)
        await asyncio.sleep(0.05)
        yield {"event": "step", "data": json.dumps({"step": "zscore", "status": "done", "detail": zscore_detail, "provenance": "LIVE_LOCAL" if _ZSCORE_STATS is not None else "CACHED_LOCAL"})}

        yield {"event": "step", "data": json.dumps({"step": "roi_mask", "status": "running", "provenance": "CACHED_LOCAL"})}
        await asyncio.sleep(0.03)
        yield {"event": "step", "data": json.dumps({"step": "roi_mask", "status": "done", "detail": f"{n_voxels} visual cortex voxels (pre-masked)", "provenance": "CACHED_LOCAL"})}

        yield {"event": "step", "data": json.dumps({"step": "roi_encode", "status": "running", "provenance": "LIVE_LOCAL"})}
        x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)
        t_fwd = time.time()
        # Use V62a for stream mode by default
        result = _model_forward(_V62_MODEL, x, _V62_CONFIG, "mlp", enable_dropout=False)
        fwd_ms = (time.time() - t_fwd) * 1000
        yield {"event": "step", "data": json.dumps({"step": "roi_encode", "status": "done", "detail": f"Forward pass in {fwd_ms:.1f}ms", "provenance": "LIVE_LOCAL"})}

        yield {"event": "step", "data": json.dumps({"step": "vmf_decode", "status": "running", "provenance": "LIVE_LOCAL"})}
        mu_norm = result["mu"]
        kappa_val = result["kappa"]
        delta_val = result["delta"]
        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({"step": "vmf_decode", "status": "done", "kappa": kappa_val, "delta": delta_val, "embedding_dim": result["dim"], "provenance": "LIVE_LOCAL"})}

        yield {"event": "step", "data": json.dumps({"step": "gallery_search", "status": "running", "provenance": "LIVE_LOCAL"})}
        t_search = time.time()
        g, ids = (_CLIP_GALLERY_768, _CLIP_NSD_IDS_768)
        if g is not None and ids is not None:
            top_k_results = _csls_retrieval(mu_norm, g, ids, k=5, csls_k=_CSLS_K)
            gallery_size = int(g.shape[0])
        else:
            top_k_results, gallery_size = [], 0
        search_ms = (time.time() - t_search) * 1000
        await asyncio.sleep(0.02)
        yield {"event": "step", "data": json.dumps({"step": "gallery_search", "status": "done", "detail": f"CSLS over {gallery_size} in {search_ms:.1f}ms", "gallery_size": gallery_size, "top_k": top_k_results, "provenance": "LIVE_LOCAL"})}

        total_ms = (time.time() - t_global) * 1000
        yield {"event": "step", "data": json.dumps({"step": "results", "status": "done", "detail": f"Pipeline complete in {total_ms:.0f}ms", "total_ms": total_ms, "kappa": kappa_val, "delta": delta_val, "top_k": top_k_results, "provenance": "LIVE_LOCAL"})}

    return EventSourceResponse(event_generator())


@app.get("/api/trials")
def list_trials(limit: int = Query(50, ge=1, le=1000)):
    if _TRIAL_INDEX is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "trial_index_not_loaded"})
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


@app.get("/api/nsd-id-to-trial/{nsd_id}")
def nsd_id_to_trial(nsd_id: int):
    if _TRIAL_INDEX is None:
        raise HTTPException(503, "Trial index not loaded")
    nsd_col = "nsdId" if "nsdId" in _TRIAL_INDEX.columns else "nsd_id"
    matches = _TRIAL_INDEX[_TRIAL_INDEX[nsd_col] == nsd_id]
    if len(matches) == 0:
        raise HTTPException(404, f"nsdId {nsd_id} not found")
    return {"nsd_id": nsd_id, "trial_indices": matches.index.tolist(), "count": len(matches)}


# =========== Reconstruction ===========

@app.get("/api/reconstruct/{trial_idx}")
def reconstruct(trial_idx: int):
    """Generate or retrieve a reconstruction image for a trial."""
    if _FMRI_FEATURES is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_not_loaded"})
    if _V62_MODEL is None and _V61_MODEL is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "no_model"})
    if trial_idx < 0 or trial_idx >= _FMRI_FEATURES.shape[0]:
        raise HTTPException(400, f"trial_idx must be 0..{_FMRI_FEATURES.shape[0] - 1}")

    r = _get_recon()
    if not r:
        return JSONResponse(content={"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
                                     "reason": "recon module not available", "image_path": None})

    # Run retrieval first to get the CLIP embedding
    import torch
    fmri_vec = _FMRI_FEATURES[trial_idx].astype(np.float32)
    fmri_vec, _ = _apply_zscore(fmri_vec)
    x = torch.from_numpy(fmri_vec).unsqueeze(0).float().to(_DEVICE)

    result = _model_forward(_V62_MODEL, x, _V62_CONFIG, "mlp", enable_dropout=False)
    mu_norm = result["mu"]

    # Get nsd_id for cached fallback
    nsd_id = None
    if _TRIAL_INDEX is not None:
        nsd_col = "nsdId" if "nsdId" in _TRIAL_INDEX.columns else "nsd_id"
        if trial_idx < len(_TRIAL_INDEX):
            nsd_id = int(_TRIAL_INDEX.iloc[trial_idx][nsd_col])

    # Run reconstruction
    recon_result = r.reconstruct(trial_idx, mu_norm, nsd_id)
    recon_result.pop("image_path", None)
    return JSONResponse(content=recon_result)


@app.get("/api/recon-assets/{filename:path}")
def serve_recon_asset(filename: str):
    """Serve cached or live reconstruction images."""
    # Try live dir first
    live_path = _recon_live_dir / filename
    if live_path.exists():
        return FileResponse(str(live_path))
    # Try cached dir
    cached_path = _recon_cached_dir / filename
    if cached_path.exists():
        return FileResponse(str(cached_path))
    raise HTTPException(404, f"Reconstruction asset not found: {filename}")


# =========== CLI ===========

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cortex2Canvas Backend v4.0")
    parser.add_argument("--checkpoint", type=str, help="V62a checkpoint path (legacy)")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.checkpoint:
        os.environ["C2C_V62_CHECKPOINT_PATH"] = args.checkpoint
    os.environ["C2C_SUBJECT"] = args.subject
    load_all_at_startup()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
