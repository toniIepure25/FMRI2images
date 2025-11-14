#!/usr/bin/env python3
"""
Download NSD stimuli HDF5 file locally for fast image loading.

This downloads the 40GB nsd_stimuli.hdf5 file from S3 to local storage,
which dramatically speeds up CLIP cache building and eliminates HTTP fallback.
"""

import argparse
import logging
from pathlib import Path
import s3fs
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_stimuli_hdf5(output_dir: Path):
    """Download NSD stimuli HDF5 file from S3."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "nsd_stimuli.hdf5"
    
    # S3 path
    s3_path = "natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    s3_url = f"s3://{s3_path}"
    
    logger.info("="*80)
    logger.info("NSD STIMULI HDF5 DOWNLOADER")
    logger.info("="*80)
    logger.info(f"Source: {s3_url}")
    logger.info(f"Destination: {output_file.absolute()}")
    logger.info("")
    
    # Check if already exists
    if output_file.exists():
        size_gb = output_file.stat().st_size / (1024**3)
        logger.info(f"⚠️  File already exists ({size_gb:.1f} GB)")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping download.")
            return
        output_file.unlink()
    
    # Initialize S3
    logger.info("Connecting to S3...")
    s3 = s3fs.S3FileSystem(anon=True)
    
    # Get file size
    try:
        file_info = s3.info(s3_path)
        total_size = file_info['size']
        size_gb = total_size / (1024**3)
        logger.info(f"File size: {size_gb:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not get file size: {e}")
        total_size = None
    
    logger.info("")
    logger.info("Starting download...")
    logger.info("This will take 30-60 minutes depending on your connection.")
    logger.info("")
    
    try:
        # Download with progress bar
        with s3.open(s3_path, 'rb') as s3_file:
            with open(output_file, 'wb') as local_file:
                # Read in 10MB chunks
                chunk_size = 10 * 1024 * 1024
                
                if total_size:
                    # Progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        while True:
                            chunk = s3_file.read(chunk_size)
                            if not chunk:
                                break
                            local_file.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # No progress bar
                    logger.info("Downloading (no progress available)...")
                    while True:
                        chunk = s3_file.read(chunk_size)
                        if not chunk:
                            break
                        local_file.write(chunk)
        
        # Verify
        final_size = output_file.stat().st_size
        final_gb = final_size / (1024**3)
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ DOWNLOAD COMPLETE!")
        logger.info("="*80)
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {final_gb:.2f} GB")
        logger.info("")
        logger.info("To use this file, set the NSD_HDF5 environment variable:")
        logger.info(f"export NSD_HDF5={output_file.absolute()}")
        logger.info("")
        logger.info("Or add to your ~/.bashrc:")
        logger.info(f"echo 'export NSD_HDF5={output_file.absolute()}' >> ~/.bashrc")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_file.exists():
            output_file.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download NSD stimuli HDF5 file for fast local access"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/nsd_hdf5"),
        help="Output directory (default: cache/nsd_hdf5)"
    )
    
    args = parser.parse_args()
    
    download_stimuli_hdf5(args.output_dir)


if __name__ == "__main__":
    main()
