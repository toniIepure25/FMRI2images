#!/usr/bin/env python3
"""
Download all beta files for a subject to local cache.

This script downloads all session beta files from S3 and places them in the
project's cache directory (cache/s3_cache/) with the exact SHA256 hash filenames
that NIfTILoader expects, so training can use cached files immediately.

Usage:
    python scripts/download_all_betas.py --subject subj01
    
This will download ~20GB of beta files (40 sessions) to cache/s3_cache/
"""

import argparse
import hashlib
import logging
from pathlib import Path
import s3fs
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_cache_filename(s3_url: str) -> str:
    """Generate cache filename from S3 URL (same algorithm as NIfTILoader)."""
    cache_key = hashlib.sha256(s3_url.encode()).hexdigest()
    return f"{cache_key}.nii.gz"


def download_betas(subject: str, cache_dir: Path, force: bool = False):
    """Download all beta files for a subject to cache directory."""
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 filesystem (anonymous access)
    s3 = s3fs.S3FileSystem(anon=True)
    
    # NSD beta file location on S3
    base_path = f"natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    
    logger.info("="*80)
    logger.info(f"DOWNLOADING BETA FILES FOR {subject.upper()}")
    logger.info("="*80)
    logger.info(f"Source: s3://{base_path}")
    logger.info(f"Cache directory: {cache_dir.absolute()}")
    logger.info("")
    
    # Find all session files
    pattern = f"{base_path}/betas_session*.nii.gz"
    logger.info(f"Searching for beta files: {pattern}")
    
    files = s3.glob(pattern)
    logger.info(f"Found {len(files)} beta files")
    
    if len(files) == 0:
        logger.error("❌ No beta files found! Check subject name")
        return
    
    logger.info("")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for s3_path in tqdm(files, desc="Downloading beta files"):
        # Generate full S3 URL (what the index uses)
        s3_url = f"s3://{s3_path}"
        
        # Generate cache filename (same as NIfTILoader)
        cache_file = cache_dir / get_cache_filename(s3_url)
        
        # Extract session name for logging
        session_name = Path(s3_path).name
        
        # Check if already cached
        if cache_file.exists() and not force:
            size_mb = cache_file.stat().st_size / (1024**2)
            tqdm.write(f"  ✓ {session_name} ({size_mb:.0f} MB) - already cached")
            skipped += 1
            continue
        
        # Download to cache
        try:
            tqdm.write(f"  ⬇ Downloading {session_name}...")
            with s3.open(s3_path, 'rb') as s3_file:
                with open(cache_file, 'wb') as local_file:
                    # Copy in 1MB chunks
                    while True:
                        chunk = s3_file.read(1024 * 1024)
                        if not chunk:
                            break
                        local_file.write(chunk)
            
            size_mb = cache_file.stat().st_size / (1024**2)
            tqdm.write(f"  ✅ {session_name} ({size_mb:.0f} MB) - downloaded")
            downloaded += 1
            
        except Exception as e:
            tqdm.write(f"  ❌ Failed: {session_name} - {e}")
            failed += 1
            if cache_file.exists():
                cache_file.unlink()  # Remove partial file
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in cache_dir.glob("*.nii.gz") if f.is_file())
    total_gb = total_size / (1024**3)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Downloaded: {downloaded} files")
    logger.info(f"⏭️  Skipped (already cached): {skipped} files")
    logger.info(f"❌ Failed: {failed} files")
    logger.info(f"📁 Total beta files in cache: {downloaded + skipped}/{len(files)}")
    logger.info(f"💾 Total disk space: {total_gb:.2f} GB")
    logger.info("="*80)
    
    if downloaded + skipped == len(files):
        logger.info("✅ All beta files are now cached! Training will be fast.")
    else:
        logger.warning(f"⚠️  Only {downloaded + skipped}/{len(files)} files cached. Some may still download from S3.")


def main():
    parser = argparse.ArgumentParser(
        description="Download all NSD beta files to local cache for fast training"
    )
    parser.add_argument(
        "--subject",
        default="subj01",
        help="Subject ID (default: subj01)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/s3_cache"),
        help="Cache directory (default: cache/s3_cache)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download existing files"
    )
    
    args = parser.parse_args()
    
    logger.info("")
    logger.info("NSD Beta Files Downloader")
    logger.info("=" * 80)
    logger.info(f"This will download ~20GB of fMRI beta files for {args.subject}")
    logger.info(f"Files will be cached in: {args.cache_dir.absolute()}")
    logger.info("=" * 80)
    logger.info("")
    
    download_betas(args.subject, args.cache_dir, args.force)


if __name__ == "__main__":
    main()
