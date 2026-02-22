#!/usr/bin/env python3
"""Preflight checks for running FMRI2images on a fresh machine.

Design goals:
- Clear PASS/FAIL output with actionable fixes
- Best-effort (never crash due to missing optional tooling)
- No external deps beyond what's already commonly installed in this repo

Typical usage:
    python scripts/preflight.py

Exit codes:
    0: PASS
    2: FAIL (action required)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str
    fix: Optional[str] = None


def _human_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024**3):.1f} GB"


def _getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(v: Optional[str]) -> Optional[Path]:
    if not v:
        return None
    return Path(v).expanduser().resolve()


def _try_import(module: str) -> Optional[str]:
    try:
        __import__(module)
        return None
    except Exception as e:
        return str(e)


def check_paths_contract() -> List[CheckResult]:
    """Print/validate the key paths contract used by the repo.

    Contract:
      - NSD_DATA_ROOT: where the user mounts the NSD dataset
      - CACHE_ROOT: caches (preproc, clip cache, hf caches)
      - OUTPUT_ROOT: all generated outputs (runs, reports, paper artifacts)
      - CHECKPOINT_ROOT: trained checkpoints
    """
    keys = [
        "NSD_DATA_ROOT",
        "CACHE_ROOT",
        "OUTPUT_ROOT",
        "CHECKPOINT_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
    ]

    results: List[CheckResult] = []
    for k in keys:
        p = _resolve_path(os.environ.get(k))
        results.append(CheckResult(name=f"path:{k}", ok=True, details=str(p) if p else "<unset>"))

    # Warn if HF caches are unset (recommended under CACHE_ROOT)
    cache_root = _resolve_path(_getenv("CACHE_ROOT"))
    hf_home = _resolve_path(os.environ.get("HF_HOME"))
    hub_cache = _resolve_path(os.environ.get("HF_HUB_CACHE"))
    if cache_root and (hf_home is None and hub_cache is None):
        results.append(
            CheckResult(
                name="hf_cache_recommended",
                ok=True,
                details="HF cache env vars are unset",
                fix="Recommended: set HF_HOME/HF_HUB_CACHE under CACHE_ROOT to make downloads reproducible on pods.",
            )
        )

    return results


def check_python() -> CheckResult:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    fix = None if ok else "Install Python >= 3.10 (recommended: 3.10/3.11)."
    return CheckResult(
        name="python",
        ok=ok,
        details=f"Python {v.major}.{v.minor}.{v.micro}",
        fix=fix,
    )


def check_repo_layout() -> CheckResult:
    required = [REPO_ROOT / "pyproject.toml", REPO_ROOT / "src" / "fmri2img"]
    missing = [str(p) for p in required if not p.exists()]
    ok = len(missing) == 0
    fix = None
    if not ok:
        fix = "You're not in a valid repo checkout. Re-clone and run from repo root."
    return CheckResult(
        name="repo_layout",
        ok=ok,
        details="OK" if ok else f"Missing: {missing}",
        fix=fix,
    )


def check_env_vars() -> List[CheckResult]:
    # Required for the new workflow.
    required = [
        "NSD_DATA_ROOT",
        "OUTPUT_ROOT",
        "CACHE_ROOT",
        "CHECKPOINT_ROOT",
    ]

    results: List[CheckResult] = []
    for k in required:
        v = os.environ.get(k)
        ok = v is not None and v != ""
        fix = None
        if not ok:
            fix = (
                f"Set {k} in .env (cp .env.example .env) and source it, or export it in your shell."
            )
        results.append(CheckResult(name=f"env:{k}", ok=ok, details=str(v) if v else "<unset>", fix=fix))

    # Optional but recommended
    optional = [
        "DIFFUSION_ENABLED",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "DEVICE",
        "CUDA_VISIBLE_DEVICES",
    ]
    for k in optional:
        v = os.environ.get(k)
        ok = True
        details = str(v) if v else "<unset>"
        fix = None
        if v is None or v == "":
            if k in {"HF_HOME", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE", "HF_HUB_CACHE"}:
                fix = "Recommended: set HF caches to keep model downloads under CACHE_ROOT."
        results.append(CheckResult(name=f"env(opt):{k}", ok=ok, details=details, fix=fix))

    return results


def check_dirs() -> List[CheckResult]:
    results: List[CheckResult] = []
    for k in ["OUTPUT_ROOT", "CACHE_ROOT", "CHECKPOINT_ROOT"]:
        v = _getenv(k)
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
            ok = True
            details = f"{p} (exists/writable)"
            fix = None
        except Exception as e:
            ok = False
            details = f"{p} (cannot create/write: {e})"
            fix = f"Fix permissions or choose a different path for {k}."
        results.append(CheckResult(name=f"dir:{k}", ok=ok, details=details, fix=fix))
    return results


def check_disk() -> CheckResult:
    min_free_gb = float(_getenv("PREFLIGHT_MIN_FREE_GB", "80"))
    # Check free space on CACHE_ROOT mount if defined, else repo.
    base = _getenv("CACHE_ROOT") or str(REPO_ROOT)
    path = Path(base).expanduser().resolve()
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_free_gb
    fix = None
    if not ok:
        fix = (
            f"Free up disk space on {path} or lower PREFLIGHT_MIN_FREE_GB (currently {min_free_gb})."
        )
    return CheckResult(
        name="disk_free",
        ok=ok,
        details=f"path={path} free={_human_gb(usage.free)} required>={min_free_gb:.0f} GB",
        fix=fix,
    )


def check_torch_cuda() -> List[CheckResult]:
    results: List[CheckResult] = []
    try:
        import torch  # type: ignore

        results.append(CheckResult("torch_import", True, f"torch={torch.__version__}"))

        device = _getenv("DEVICE", "cuda")
        if device == "cpu":
            results.append(CheckResult("cuda_required", True, "DEVICE=cpu (CUDA not required)"))
            return results

        cuda_ok = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
        if not cuda_ok:
            results.append(
                CheckResult(
                    "cuda_available",
                    False,
                    "torch.cuda.is_available() is False",
                    fix="Install a CUDA-enabled PyTorch build and ensure NVIDIA driver is present.",
                )
            )
            return results

        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_mem = props.total_memory
        results.append(CheckResult("cuda_available", True, f"GPU[{idx}]={name}"))
        results.append(CheckResult("gpu_memory", True, f"total={_human_gb(total_mem)}"))
        results.append(CheckResult("cuda_version", True, f"torch_cuda={torch.version.cuda}"))

    except ModuleNotFoundError:
        results.append(
            CheckResult(
                "torch_import",
                False,
                "torch not installed",
                fix="Run make setup (or install deps) to install torch/diffusion extras.",
            )
        )
    except Exception as e:
        results.append(CheckResult("torch_cuda", False, f"Unexpected error: {e}", fix="Inspect stacktrace."))

    return results


def check_imports() -> List[CheckResult]:
    results: List[CheckResult] = []
    try:
        import fmri2img  # noqa: F401

        results.append(CheckResult("import_fmri2img", True, "import ok"))
    except Exception as e:
        results.append(
            CheckResult(
                "import_fmri2img",
                False,
                f"import failed: {e}",
                fix="Run make setup (pip install -e .) in the active environment.",
            )
        )
    return results


def check_deps() -> List[CheckResult]:
    """Check essential Python deps for the pipeline."""
    results: List[CheckResult] = []

    # Core deps for indexing/caching & HDF5 stimuli
    for module, why in [
        ("numpy", "core"),
        ("pandas", "core"),
        ("pyarrow", "parquet index/cache"),
        ("h5py", "HDF5 stimuli"),
    ]:
        err = _try_import(module)
        ok = err is None
        fix = None
        if not ok:
            fix = f"Install '{module}' (required for {why}). Run make setup in the correct env."
        results.append(CheckResult(name=f"dep:{module}", ok=ok, details="import ok" if ok else err, fix=fix))

    diffusion_enabled = _truthy(os.environ.get("DIFFUSION_ENABLED"))
    if diffusion_enabled:
        for module in ["diffusers", "transformers", "huggingface_hub", "accelerate"]:
            err = _try_import(module)
            ok = err is None
            fix = None
            if not ok:
                fix = (
                    "Diffusion is enabled (DIFFUSION_ENABLED=1). "
                    f"Install '{module}' (pip install -e '.[diffusion]' or equivalent)."
                )
            results.append(
                CheckResult(name=f"dep(diffusion):{module}", ok=ok, details="import ok" if ok else err, fix=fix)
            )
    else:
        results.append(
            CheckResult(
                name="dep(diffusion):skipped",
                ok=True,
                details="DIFFUSION_ENABLED is not truthy; skipping diffusers checks",
            )
        )

    return results


def print_report(results: List[CheckResult]) -> int:
    width = 88
    def line(s: str) -> None:
        print(s[:width])

    line("=" * width)
    line("FMRI2images preflight")
    line(f"repo={REPO_ROOT}")
    line(f"platform={platform.platform()}")
    line("=" * width)

    failures = [r for r in results if not r.ok]

    for r in results:
        status = "PASS" if r.ok else "FAIL"
        line(f"[{status}] {r.name}: {r.details}")
        if (not r.ok) and r.fix:
            line(f"       fix: {r.fix}")
        elif r.ok and r.fix:
            # Optional recommendation
            line(f"       note: {r.fix}")

    line("=" * width)
    if failures:
        line(f"FAIL: {len(failures)} check(s) failed")
        return 2
    line("PASS: all required checks passed")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for FMRI2images")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Override PREFLIGHT_MIN_FREE_GB for this run.",
    )
    args = parser.parse_args(argv)

    # Allow CLI override for disk threshold
    if args.min_free_gb is not None:
        os.environ["PREFLIGHT_MIN_FREE_GB"] = str(args.min_free_gb)

    results: List[CheckResult] = []
    results.append(check_python())
    results.append(check_repo_layout())
    results.extend(check_env_vars())
    results.extend(check_paths_contract())
    results.extend(check_dirs())
    results.append(check_disk())
    results.extend(check_deps())
    results.extend(check_torch_cuda())
    results.extend(check_imports())
    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
