#!/usr/bin/env python3
"""
Pre-download Stable Diffusion model to HuggingFace cache.

Idempotent: exits with code 0 if model is already cached.
Shows Rich progress bar with speed and ETA during download.
Resumes interrupted downloads automatically.

Usage:
    python scripts/download_sd_model.py
    python scripts/download_sd_model.py --model-id runwayml/stable-diffusion-v1-5
    python scripts/download_sd_model.py --cache-dir /path/to/cache
    python scripts/download_sd_model.py --no-progress  # Quiet mode
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_cache_root(cache_dir: Optional[str] = None, hf_home: Optional[str] = None) -> Path:
    """
    Determine the HuggingFace cache root directory.
    
    Priority: --cache-dir > HF_HOME env > default (~/.cache/huggingface)
    """
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
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


def check_if_cached(model_id: str, cache_root: Path) -> bool:
    """
    Check if model is already cached by examining the cache directory.
    
    Returns:
        True if model directory exists in cache, False otherwise
    """
    # HuggingFace stores models in: cache_root/hub/models--{org}--{name}/
    model_slug = model_id.replace("/", "--")
    model_cache_path = cache_root / "hub" / f"models--{model_slug}"
    
    # Check if the model directory exists and has content
    if not model_cache_path.exists():
        return False
    
    # Check if there are any snapshot directories (indicates downloaded model)
    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False
    
    # Check if there's at least one snapshot with files
    try:
        snapshot_dirs = list(snapshots_dir.iterdir())
        if not snapshot_dirs:
            return False
        
        # Check if the first snapshot has files
        for snapshot in snapshot_dirs:
            if snapshot.is_dir():
                files = list(snapshot.iterdir())
                if files:
                    return True
        return False
    except (OSError, PermissionError):
        return False


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    hf_home: Optional[str] = None,
    no_progress: bool = False
) -> int:
    """
    Download Stable Diffusion model with Rich progress bar.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Override cache directory
        hf_home: Override HF_HOME environment variable
        no_progress: Disable progress bar (quiet mode)
        
    Returns:
        0 if successful or already cached, non-zero on error
    """
    # Determine cache root
    cache_root = get_cache_root(cache_dir, hf_home)
    
    logger.info("=" * 80)
    logger.info("STABLE DIFFUSION MODEL DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Cache root: {cache_root}")
    logger.info("")
    
    # Check if already cached
    if check_if_cached(model_id, cache_root):
        # Model already exists - compute size and report
        model_cache_path = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
        if model_cache_path.exists():
            total_size = get_dir_size(model_cache_path)
            file_count = sum(1 for _ in model_cache_path.rglob("*") if _.is_file())
            
            logger.info("✅ Model already cached (no download needed)")
            logger.info(f"   Path: {model_cache_path}")
            logger.info(f"   Files: {file_count}")
            logger.info(f"   Size: {format_size(total_size)}")
            logger.info("")
            logger.info("You can now run decode_diffusion.py immediately.")
            return 0
        else:
            logger.info("✅ Model appears cached")
            return 0
    
    # Model not cached - proceed with download
    logger.info("Model not found in cache - starting download...")
    logger.info("Size: ~5 GB (varies by model)")
    logger.info("This may take 10-30 minutes depending on your connection.")
    logger.info("")
    
    # Check dependencies
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("ERROR: huggingface_hub not installed")
        logger.error("Install with: pip install huggingface_hub")
        return 1
    
    # Setup Rich progress (optional)
    progress_bar = None
    if not no_progress:
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
            
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            )
        except ImportError:
            logger.warning("Rich not installed - progress bar disabled")
            logger.warning("Install with: pip install rich")
    
    try:
        # Download with resumable downloads
        if progress_bar:
            with progress_bar:
                task = progress_bar.add_task(f"Downloading {model_id}", total=None)
                
                cache_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_root,
                    resume_download=True,
                    local_dir_use_symlinks=False,
                    repo_type="model"
                )
        else:
            logger.info("Downloading (this may take a while)...")
            cache_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_root,
                resume_download=True,
                local_dir_use_symlinks=False,
                repo_type="model"
            )
        
        # Success - compute final size
        cache_path = Path(cache_path)
        total_size = get_dir_size(cache_path)
        file_count = sum(1 for _ in cache_path.rglob("*") if _.is_file())
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Model cached at: {cache_path}")
        logger.info(f"Total files: {file_count}")
        logger.info(f"Total size: {format_size(total_size)}")
        logger.info("")
        logger.info("You can now run decode_diffusion.py without waiting:")
        logger.info(f"  python scripts/decode_diffusion.py --model-id {model_id} [args]")
        logger.info("")
        
        return 0
        
    except OSError as e:
        logger.error("")
        logger.error("ERROR: Disk space or permissions issue")
        logger.error(f"Details: {e}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error(f"  1. Check free space: df -h {cache_root}")
        logger.error(f"  2. Check permissions: ls -la {cache_root.parent}")
        logger.error(f"  3. Try different cache: --cache-dir /path/with/space")
        return 1
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            logger.error("")
            logger.error("ERROR: Network connection issue")
            logger.error(f"Details: {e}")
            logger.error("")
            logger.error("Troubleshooting:")
            logger.error("  1. Check internet: ping huggingface.co")
            logger.error("  2. Try again (download will resume)")
            logger.error("  3. Check firewall/proxy settings")
        else:
            logger.error("")
            logger.error(f"ERROR: Download failed: {e}")
            logger.error("")
            logger.error("The download can be resumed - just run the command again.")
        
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download Stable Diffusion model to HuggingFace cache (idempotent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model
  python scripts/download_sd_model.py
  
  # Download specific model
  python scripts/download_sd_model.py --model-id runwayml/stable-diffusion-v1-5
  
  # Use custom cache directory
  python scripts/download_sd_model.py --cache-dir /mnt/large-disk/cache
  
  # Quiet mode (no progress bar)
  python scripts/download_sd_model.py --no-progress
        """
    )
    
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-2-1",
        help="HuggingFace model ID (default: stabilityai/stable-diffusion-2-1)"
    )
    
    parser.add_argument(
        "--cache-dir",
        help="Override cache directory (default: ~/.cache/huggingface)"
    )
    
    parser.add_argument(
        "--hf-home",
        help="Override HF_HOME environment variable"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (quiet mode)"
    )
    
    args = parser.parse_args()
    
    return download_model(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        hf_home=args.hf_home,
        no_progress=args.no_progress
    )


if __name__ == "__main__":
    sys.exit(main())
