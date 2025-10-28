#!/usr/bin/env python3
"""
Check HuggingFace cache status for a model.

Quickly shows if a model is cached, file count, size, and cache path.

Usage:
    python scripts/check_hf_cache.py
    python scripts/check_hf_cache.py --model-id runwayml/stable-diffusion-v1-5
    python scripts/check_hf_cache.py --hf-home /custom/cache
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def get_cache_root(hf_home: Optional[str] = None) -> Path:
    """Determine HuggingFace cache root directory."""
    if hf_home or os.getenv("HF_HOME"):
        return Path(hf_home or os.getenv("HF_HOME")).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface"


def get_dir_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def check_cache_status(model_id: str, hf_home: Optional[str] = None) -> int:
    """
    Check if model is cached and print status.
    
    Args:
        model_id: HuggingFace model ID
        hf_home: Override HF_HOME
        
    Returns:
        0 if cached, 1 if not cached
    """
    cache_root = get_cache_root(hf_home)
    
    print("=" * 80)
    print("HUGGINGFACE CACHE STATUS")
    print("=" * 80)
    print(f"Model ID: {model_id}")
    print(f"Cache root: {cache_root}")
    print("")
    
    # Check if cache root exists
    if not cache_root.exists():
        print("❌ Cache root does not exist")
        print("")
        print("The model is NOT cached.")
        print("")
        print("To download:")
        print(f"  python scripts/download_sd_model.py --model-id {model_id}")
        return 1
    
    # Check for model-specific cache directory
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_path = cache_root / "hub" / model_cache_name
    
    if not model_cache_path.exists():
        print("❌ Model not found in cache")
        print("")
        print(f"Expected path: {model_cache_path}")
        print("")
        print("To download:")
        print(f"  python scripts/download_sd_model.py --model-id {model_id}")
        return 1
    
    # Model is cached - compute stats
    try:
        total_size = get_dir_size(model_cache_path)
        file_count = sum(1 for _ in model_cache_path.rglob("*") if _.is_file())
        
        print("✅ Model IS cached")
        print("")
        print(f"Cache path: {model_cache_path}")
        print(f"Files: {file_count}")
        print(f"Total size: {format_size(total_size)}")
        print("")
        print("You can use this model in decode_diffusion.py immediately.")
        
        return 0
        
    except Exception as e:
        print(f"⚠️  Cache exists but couldn't read stats: {e}")
        print("")
        print(f"Cache path: {model_cache_path}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Check HuggingFace cache status for a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check default model
  python scripts/check_hf_cache.py
  
  # Check specific model
  python scripts/check_hf_cache.py --model-id runwayml/stable-diffusion-v1-5
  
  # Check with custom HF_HOME
  python scripts/check_hf_cache.py --hf-home /custom/cache
        """
    )
    
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-2-1",
        help="HuggingFace model ID (default: stabilityai/stable-diffusion-2-1)"
    )
    
    parser.add_argument(
        "--hf-home",
        help="Override HF_HOME environment variable"
    )
    
    args = parser.parse_args()
    
    return check_cache_status(
        model_id=args.model_id,
        hf_home=args.hf_home
    )


if __name__ == "__main__":
    sys.exit(main())
