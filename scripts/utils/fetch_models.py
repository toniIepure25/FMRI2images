#!/usr/bin/env python3
"""Fetch required Hugging Face models into caches (offline-friendly).

This script is intentionally conservative:
- If models are already cached, it will not re-download them.
- If HF_TOKEN is required for a gated model, it will print an actionable error.

It supports the minimal set we need for the pipeline:
- Diffusion model (default: DIFFUSION_MODEL_ID)

Exit codes:
  0: success
  2: missing deps / auth / download failed
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _p(msg: str) -> None:
    print(msg)


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        _p(f"ERROR: missing dependency '{pkg}': {e}")
        _p("Fix: run make setup (and ensure diffusion extras are installed)")
        raise


def main() -> int:
    # Import lazily so preflight can run without them.
    try:
        _require("diffusers")
        _require("transformers")
        _require("huggingface_hub")
    except Exception:
        return 2

    from huggingface_hub import snapshot_download

    model_id = os.environ.get("DIFFUSION_MODEL_ID", "sd2-community/stable-diffusion-2-1")
    revision = os.environ.get("DIFFUSION_MODEL_REV") or None
    token = os.environ.get("HF_TOKEN") or None

    # Default HF caches under CACHE_ROOT if unset (don't override user settings).
    cache_root = os.environ.get("CACHE_ROOT")
    if cache_root:
        cache_root_path = Path(cache_root).expanduser().resolve()
        os.environ.setdefault("HF_HOME", str(cache_root_path / "hf"))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_root_path / "hf" / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root_path / "hf" / "transformers"))
        os.environ.setdefault("DIFFUSERS_CACHE", str(cache_root_path / "hf" / "diffusers"))

    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    cache_dir_path: Optional[Path] = Path(cache_dir).expanduser() if cache_dir else None

    _p(f"models: ensuring cached model: {model_id}")
    if revision:
        _p(f"models: revision={revision}")
    if cache_dir_path:
        _p(f"models: cache_dir={cache_dir_path}")

    # First try offline-only: if already cached, avoid any network.
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir_path) if cache_dir_path else None,
            token=token,
            local_files_only=True,
        )
        _p("models: OK (already cached)")
        return 0
    except Exception:
        pass

    # Otherwise attempt a download/resume.
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir_path) if cache_dir_path else None,
            token=token,
            local_files_only=False,
            resume_download=True,
        )
    except Exception as e:
        msg = str(e)
        _p(f"ERROR: failed to download model '{model_id}': {msg}")
        if ("401" in msg or "403" in msg) and token is None:
            _p("This looks like a gated/private model. Set HF_TOKEN in your .env and retry.")
        else:
            _p("If you're offline, ensure the model is already present in the HF cache.")
        return 2

    # Lightweight sanity: confirm some model files exist in the snapshot.
    try:
        from huggingface_hub import snapshot_download as _sd

        local_dir = _sd(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir_path) if cache_dir_path else None,
            token=token,
            local_files_only=True,
        )
        p = Path(local_dir)
        has_any = any((p / n).exists() for n in ["model_index.json", "scheduler", "unet", "vae", "text_encoder"])
        if not has_any:
            _p(f"WARNING: model snapshot downloaded but expected components not found under: {p}")
        else:
            _p(f"models: snapshot present at {p}")
    except Exception:
        # Best-effort
        pass

    _p("models: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
