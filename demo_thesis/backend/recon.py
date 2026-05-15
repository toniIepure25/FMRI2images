"""
Cortex2Canvas Local Reconstruction Module — v2.0 (pipeline-corrected)

Provides honest, provenance-tracked reconstruction from live predicted CLIP image embeddings.

Modes:
  LIVE_LOCAL_RECONSTRUCTION     — Image generated from live predicted V62a mu embedding
  RETRIEVAL_CONDITIONED_VARIATION — Image variation from Top-1 retrieved image (if no direct path)
  CACHED_LOCAL_RECONSTRUCTION   — Image loaded from pre-computed assets
  UNAVAILABLE                   — No reconstruction available

Pipeline: UnCLIPImageVariationPipeline (kakaobrain/karlo-v1-alpha)
  - Accepts precomputed CLIP ViT-L/14 image embeddings via `image_embeddings` parameter
  - Does NOT run the CLIP image encoder on raw pixels when embeddings are provided
  - 768-D ViT-L/14 embeddings match V62a output dimension
  - ~4 GB VRAM in float16, ~5 GB download
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Auto-detect venv's site-packages if available (needed for diffusers + transformers)
_VENV_SITE = Path(__file__).resolve().parents[2] / ".venv" / "lib" / "python3.10" / "site-packages"
if _VENV_SITE.exists():
    import sys as _sys
    if str(_VENV_SITE) not in _sys.path:
        _sys.path.insert(0, str(_VENV_SITE))
    # Force venv's transformers to be found first
    for p in list(_sys.path):
        if 'site-packages' in p and '.venv' not in p:
            _sys.path.remove(p)
            _sys.path.append(p)  # Move system packages to end

logger = logging.getLogger("cortex2canvas.recon")

# ── Config ──
RECON_MODE = os.environ.get("C2C_RECON_MODE", "auto")
RECON_MODEL_ID = os.environ.get("C2C_RECON_MODEL_ID", "kakaobrain/karlo-v1-alpha")
RECON_CACHE_DIR = os.environ.get("C2C_RECON_CACHE_DIR", "local_backend_data/reconstructions/cached")
RECON_OUTPUT_DIR = os.environ.get("C2C_RECON_OUTPUT_DIR", "local_backend_data/reconstructions/live")
RECON_DTYPE = os.environ.get("C2C_RECON_DTYPE", "float16")
RECON_HEIGHT = int(os.environ.get("C2C_RECON_HEIGHT", "256"))       # Karlo max is 256
RECON_WIDTH = int(os.environ.get("C2C_RECON_WIDTH", "256"))
RECON_STEPS = int(os.environ.get("C2C_RECON_STEPS", "25"))
RECON_GUIDANCE = float(os.environ.get("C2C_RECON_GUIDANCE", "8.0"))
RECON_SEED = int(os.environ.get("C2C_RECON_SEED", "42"))
RECON_ALLOW_DOWNLOAD = os.environ.get("C2C_RECON_ALLOW_DOWNLOAD", "false").lower() == "true"

# ── State ──
_PIPE = None
_PIPE_LOADED = False
_MODEL_CACHED = False
_LAST_ERROR: Optional[str] = None
_DEVICE = "cpu"
_HAS_VALID_PATH = False  # True if direct embedding → image path is confirmed valid


def _check_model_cached() -> bool:
    """Check if the reconstruction model is available in HF cache."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    model_slug = RECON_MODEL_ID.replace("/", "--")
    candidates = [
        Path(hf_home) / f"models--{model_slug}",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.safetensors")):
            return True
        if candidate.exists() and any(candidate.rglob("diffusion_pytorch_model*")):
            return True
    return False


def get_status() -> Dict[str, Any]:
    """Return full reconstruction status for health endpoint."""
    global _PIPE_LOADED, _MODEL_CACHED, _LAST_ERROR, _DEVICE, _HAS_VALID_PATH

    _MODEL_CACHED = _check_model_cached()
    live_available = _PIPE_LOADED and _MODEL_CACHED
    cached_available = Path(RECON_CACHE_DIR).exists() and any(Path(RECON_CACHE_DIR).glob("*.png"))

    effective_mode = RECON_MODE
    if effective_mode == "auto":
        if live_available:
            effective_mode = "live_local_reconstruction"
        elif cached_available:
            effective_mode = "cached_local_reconstruction"
        elif RECON_ALLOW_DOWNLOAD:
            effective_mode = "live_pending_download"
        else:
            effective_mode = "unavailable"
    elif effective_mode == "live":
        if live_available:
            effective_mode = "live_local_reconstruction"
        elif RECON_ALLOW_DOWNLOAD:
            effective_mode = "live_pending_download"
        else:
            effective_mode = "unavailable"

    return {
        "reconstruction_available": effective_mode != "unavailable",
        "reconstruction_mode": effective_mode,
        "reconstruction_pipeline_accepts_predicted_embedding": True,
        "reconstruction_live_local_available": _PIPE_LOADED and _MODEL_CACHED,
        "reconstruction_uses_retrieved_image": False,
        "reconstruction_model_loaded": _PIPE_LOADED,
        "reconstruction_model_cached": _MODEL_CACHED,
        "reconstruction_device": _DEVICE,
        "reconstruction_dtype": RECON_DTYPE,
        "reconstruction_requires_download": not _MODEL_CACHED,
        "reconstruction_allow_download": RECON_ALLOW_DOWNLOAD,
        "reconstruction_last_error": _LAST_ERROR,
        "reconstruction_model_id": RECON_MODEL_ID,
        "reconstruction_scientific_warning": (
            None if _MODEL_CACHED
            else "Model not downloaded. Set C2C_RECON_ALLOW_DOWNLOAD=true to download kakaobrain/karlo-v1-alpha (~5GB)."
        ),
    }


def load_pipeline() -> Tuple[bool, Optional[str]]:
    """Load UnCLIPImageVariationPipeline. Returns (success, error_message)."""
    global _PIPE, _PIPE_LOADED, _LAST_ERROR, _DEVICE, _HAS_VALID_PATH

    if _PIPE_LOADED:
        return True, None

    if RECON_MODE == "disabled":
        _LAST_ERROR = "Reconstruction disabled via C2C_RECON_MODE=disabled"
        return False, _LAST_ERROR

    try:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _LAST_ERROR = "PyTorch not available"
        return False, _LAST_ERROR

    if not _check_model_cached():
        msg = f"Model {RECON_MODEL_ID} not cached locally (~5 GB download needed)"
        if not RECON_ALLOW_DOWNLOAD:
            _LAST_ERROR = msg + ". Set C2C_RECON_ALLOW_DOWNLOAD=true to download."
            logger.warning("load_pipeline: %s", _LAST_ERROR)
            return False, _LAST_ERROR
        logger.info("load_pipeline: downloading %s (RECON_ALLOW_DOWNLOAD=true)...", RECON_MODEL_ID)

    try:
        import torch
        from diffusers import UnCLIPImageVariationPipeline
        torch_dtype = torch.float32 if RECON_DTYPE == "float32" else torch.float16
        logger.info("load_pipeline: Loading UnCLIPImageVariationPipeline (dtype=%s, device=%s)...", RECON_DTYPE, _DEVICE)
        t0 = time.time()

        _PIPE = UnCLIPImageVariationPipeline.from_pretrained(
            RECON_MODEL_ID,
            torch_dtype=torch_dtype,
        )
        _PIPE = _PIPE.to(_DEVICE)

        try:
            _PIPE.enable_attention_slicing()
        except Exception:
            pass

        _PIPE_LOADED = True
        _HAS_VALID_PATH = True
        _LAST_ERROR = None
        logger.info("Pipeline loaded in %.1fs on %s (accepts precomputed CLIP image embeddings)", time.time() - t0, _DEVICE)
        return True, None

    except Exception as e:
        _LAST_ERROR = str(e)
        logger.error("Pipeline load failed: %s", e)
        _PIPE_LOADED = False
        _PIPE = None
        _HAS_VALID_PATH = False
        return False, _LAST_ERROR


def generate_from_embedding(
    clip_embedding: np.ndarray,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a reconstruction image directly from a live predicted CLIP image embedding.

    Uses UnCLIPImageVariationPipeline with image_embeddings parameter
    (bypasses the CLIP image encoder entirely — uses predicted mu directly).

    Args:
        clip_embedding: Normalized 768-D CLIP ViT-L/14 image embedding (1D array)
        seed, steps, guidance: Override defaults

    Returns:
        (local_path, metadata_dict) — path is None if generation failed
    """
    import torch
    from PIL import Image

    if not _PIPE_LOADED:
        success, err = load_pipeline()
        if not success:
            return None, {"error": err, "mode": "UNAVAILABLE"}

    s = seed if seed is not None else RECON_SEED
    st = steps if steps is not None else RECON_STEPS
    g = guidance if guidance is not None else RECON_GUIDANCE

    # Prepare embedding: normalize → (1, 768) → tensor
    emb = clip_embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    emb_tensor = torch.from_numpy(emb).unsqueeze(0).to(_DEVICE)

    if RECON_DTYPE == "float16":
        emb_tensor = emb_tensor.half()

    generator = torch.Generator(device=_DEVICE).manual_seed(s)

    t0 = time.time()
    try:
        with torch.no_grad():
            output = _PIPE(
                image=None,                              # No image — use precomputed embedding
                image_embeddings=emb_tensor,             # Live predicted V62a mu
                num_images_per_prompt=1,
                decoder_num_inference_steps=st,
                decoder_guidance_scale=g,
                generator=generator,
            )
        gen_ms = (time.time() - t0) * 1000
        image: Image.Image = output.images[0]

        # Save to output dir
        out_dir = Path(RECON_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_hash = hashlib.md5(emb.tobytes()).hexdigest()[:12]
        filename = f"recon_s{s}_st{st}_g{g:.1f}_{emb_hash}.png"
        save_path = out_dir / filename
        image.save(str(save_path))

        metadata = {
            "ok": True,
            "mode": "LIVE_LOCAL_RECONSTRUCTION",
            "provenance": "LIVE_LOCAL",
            "image_url": f"/api/recon-assets/{filename}",
            "local_path": str(save_path),
            "source_embedding": "predicted_mu",
            "uses_retrieved_image": False,
            "uses_target_image": False,
            "model_id": RECON_MODEL_ID,
            "device": _DEVICE,
            "dtype": RECON_DTYPE,
            "seed": s,
            "steps": st,
            "guidance": g,
            "generation_ms": gen_ms,
            "local_path": str(save_path),
            "embedding_dim": int(emb.shape[0]),
        }
        logger.info("Live reconstruction generated in %.0fms → %s", gen_ms, filename)
        return str(save_path), metadata

    except Exception as e:
        logger.error("Reconstruction generation failed: %s", e)
        _LAST_ERROR = str(e)
        return None, {"ok": False, "error": str(e), "mode": "UNAVAILABLE"}


def generate_variation_from_retrieved(
    retrieved_image_path: str,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    RETRIEVAL_CONDITIONED_VARIATION — generate an image variation from the Top-1 retrieved image.

    This is NOT direct brain reconstruction. It conditions on the retrieved image,
    not on the predicted brain embedding. Honest labeling required.
    """
    import torch
    from PIL import Image

    if not _PIPE_LOADED:
        success, err = load_pipeline()
        if not success:
            return None, {"ok": False, "error": err, "mode": "UNAVAILABLE"}

    s = seed if seed is not None else RECON_SEED
    st = steps if steps is not None else RECON_STEPS
    g = guidance if guidance is not None else RECON_GUIDANCE

    try:
        pil_image = Image.open(retrieved_image_path).convert("RGB")
    except Exception as e:
        return None, {"ok": False, "error": f"Cannot open retrieved image: {e}", "mode": "UNAVAILABLE"}

    generator = torch.Generator(device=_DEVICE).manual_seed(s)

    t0 = time.time()
    try:
        with torch.no_grad():
            output = _PIPE(
                image=pil_image,
                num_images_per_prompt=1,
                decoder_num_inference_steps=st,
                decoder_guidance_scale=g,
                generator=generator,
            )
        gen_ms = (time.time() - t0) * 1000
        image = output.images[0]

        out_dir = Path(RECON_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"var_s{s}_st{st}_g{g:.1f}_{hashlib.md5(retrieved_image_path.encode()).hexdigest()[:8]}.png"
        save_path = out_dir / filename
        image.save(str(save_path))

        metadata = {
            "ok": True,
            "mode": "RETRIEVAL_CONDITIONED_VARIATION",
            "provenance": "DERIVED_LOCAL",
            "source_embedding": "top1_retrieved_image",
            "uses_retrieved_image": True,
            "uses_target_image": False,
            "warning": "Generated from retrieved image, not directly from predicted brain embedding.",
            "generation_ms": gen_ms,
            "local_path": str(save_path),
        }
        return str(save_path), metadata

    except Exception as e:
        _LAST_ERROR = str(e)
        return None, {"ok": False, "error": str(e), "mode": "UNAVAILABLE"}


def get_cached_reconstruction(trial_idx: int, nsd_id: Optional[int] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """Try to load a cached reconstruction for a given trial."""
    cache_dir = Path(RECON_CACHE_DIR)
    if not cache_dir.exists():
        return None, {"ok": False, "mode": "UNAVAILABLE", "error": "No cache directory"}

    for pattern in [f"trial_{trial_idx:06d}_*.png", f"trial_{trial_idx}_*.png",
                    f"nsd_{nsd_id}_*.png" if nsd_id else None, "*.png"]:
        if pattern is None:
            continue
        matches = sorted(cache_dir.glob(pattern))
        if matches:
            path = str(matches[0])
            return path, {"ok": True, "mode": "CACHED_LOCAL_RECONSTRUCTION", "provenance": "CACHED_LOCAL",
                         "image_url": f"/api/recon-assets/{matches[0].name}",
                         "source_embedding": "cached_asset", "uses_retrieved_image": False,
                         "local_path": path}
    return None, {"ok": False, "mode": "UNAVAILABLE", "error": "No cached reconstruction for this trial"}


def reconstruct(trial_idx: int, clip_embedding: np.ndarray,
                nsd_id: Optional[int] = None,
                retrieved_image_path: Optional[str] = None) -> Dict[str, Any]:
    """Main entry point. ALWAYS attempts live generation when RECON_MODE=live."""
    logger.info("Reconstruction requested: mode=%s allow_download=%s cached=%s",
                RECON_MODE, RECON_ALLOW_DOWNLOAD, _MODEL_CACHED)

    # Mode 1: Live — always attempt generate_from_embedding when RECON_MODE=live
    # or RECON_ALLOW_DOWNLOAD=true. generate_from_embedding() internally calls
    # load_pipeline() which handles download if needed.
    if RECON_MODE in ("live", "auto") and RECON_ALLOW_DOWNLOAD:
        logger.info("Calling generate_from_embedding() (will call load_pipeline if needed)...")
        path, meta = generate_from_embedding(clip_embedding)
        if path:
            return {**meta, "image_path": path}
        logger.warning("Live generation failed: %s", meta.get("error", "unknown"))
        # Live failed — fall through to cached
        status = get_status()
        if status.get("reconstruction_cached_local_available"):
            cpath, cmeta = get_cached_reconstruction(trial_idx, nsd_id)
            if cpath:
                return {"ok": True, **cmeta, "image_path": cpath,
                        "fallback_used": True,
                        "fallback_reason": f"Live generation failed: {meta.get('error')}"}
        return {"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
                "reason": meta.get("error", "Generation failed"),
                "last_error": _LAST_ERROR, "image_path": None}

    # If pipeline already loaded, use it
    if _PIPE_LOADED and RECON_MODE != "disabled":
        path, meta = generate_from_embedding(clip_embedding)
        if path:
            return {**meta, "image_path": path}
        return {"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
                "reason": meta.get("error", "Generation failed"), "image_path": None}

    # Mode 2: Cached only
    status = get_status()
    if status["reconstruction_mode"] == "cached_local_reconstruction":
        cpath, cmeta = get_cached_reconstruction(trial_idx, nsd_id)
        if cpath:
            return {"ok": True, **cmeta, "image_path": cpath}
        return {"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
                "reason": "No cached reconstruction for this trial", "image_path": None}

    # Mode 3: Unavailable
    reason = status.get("reconstruction_scientific_warning", "Reconstruction unavailable")
    if RECON_MODE == "live" and not RECON_ALLOW_DOWNLOAD:
        reason = "Set C2C_RECON_ALLOW_DOWNLOAD=true to enable Karlo model download."
    return {"ok": False, "mode": "UNAVAILABLE", "provenance": "UNAVAILABLE",
            "reason": reason, "image_path": None, "last_error": _LAST_ERROR}


# Initialize
_MODEL_CACHED = _check_model_cached()


def warmup() -> Dict[str, Any]:
    """Explicitly load pipeline and return status. Used by /api/reconstruction/warmup."""
    global _MODEL_CACHED
    _MODEL_CACHED = _check_model_cached()
    success, err = load_pipeline()
    return {
        "ok": success,
        "model_cached": _MODEL_CACHED,
        "model_loaded": _PIPE_LOADED,
        "allow_download": RECON_ALLOW_DOWNLOAD,
        "requires_download": not _MODEL_CACHED,
        "device": _DEVICE,
        "recon_mode": RECON_MODE,
        "error": err if not success else None,
    }
