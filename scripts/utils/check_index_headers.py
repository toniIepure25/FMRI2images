#!/usr/bin/env python3
"""
Header bounds check for NSD canonical index.

Validates that all beta_index values are within the bounds of their
corresponding beta files by checking NIfTI headers (no data loading).
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Set

# Silence nibabel qfac warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem
from fmri2img.data.nsd_index_reader import read_subject_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_index_headers(index_path: str, max_files: int = None) -> bool:
    """
    Check that all beta_index values are within bounds of their beta files.
    
    Args:
        index_path: Path to index file or root directory
        max_files: Limit number of unique beta files to check (for testing)
        
    Returns:
        True if all indices are valid, False otherwise
    """
    # Read index (handle both file and directory paths)
    try:
        if index_path.endswith('.parquet') and 'subject=' in index_path:
            df = pd.read_parquet(index_path)
        elif index_path.endswith('.parquet'):
            df = pd.read_parquet(index_path)
        else:
            # Try to find any subject partition
            test_subjects = ['subj01', 'subj02', 'subj03']
            df = None
            for subj in test_subjects:
                try:
                    df = read_subject_index(index_path, subj)
                    logger.info(f"Found index for {subj}")
                    break
                except:
                    continue
            
            if df is None:
                raise FileNotFoundError("No valid index found")
                
    except Exception as e:
        logger.error(f"Failed to read index from {index_path}: {e}")
        return False
    
    logger.info(f"Loaded index with {len(df)} trials")
    
    # Get unique beta files and their max indices
    file_max_indices: Dict[str, int] = {}
    for _, row in df.iterrows():
        beta_path = row['beta_path']
        beta_index = int(row['beta_index'])
        
        if beta_path in file_max_indices:
            file_max_indices[beta_path] = max(file_max_indices[beta_path], beta_index)
        else:
            file_max_indices[beta_path] = beta_index
    
    unique_files = list(file_max_indices.keys())
    if max_files:
        unique_files = unique_files[:max_files]
        logger.info(f"Limiting check to {len(unique_files)} files")
    
    logger.info(f"Checking bounds for {len(unique_files)} unique beta files")
    
    # Initialize S3 loader
    s3_fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(s3_fs)
    
    errors = []
    
    for i, beta_path in enumerate(unique_files):
        try:
            # Get header info (no data loading)
            header_info = nifti_loader.get_header(beta_path)
            shape = header_info['shape']
            
            if len(shape) < 4:
                logger.warning(f"File {beta_path} has shape {shape} (not 4D)")
                continue
                
            max_trial_index = shape[3] - 1  # 0-based indexing
            required_max = file_max_indices[beta_path]
            
            if required_max > max_trial_index:
                error_msg = f"File {beta_path}: max beta_index={required_max} exceeds bounds (0-{max_trial_index})"
                errors.append(error_msg)
                logger.error(error_msg)
            else:
                logger.debug(f"✓ {beta_path}: indices 0-{required_max} within bounds (0-{max_trial_index})")
                
            if (i + 1) % 10 == 0:
                logger.info(f"Checked {i + 1}/{len(unique_files)} files...")
                
        except Exception as e:
            error_msg = f"Failed to check {beta_path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    if errors:
        logger.error(f"Found {len(errors)} bound violations:")
        for error in errors:
            logger.error(f"  {error}")
        return False
    else:
        logger.info("✅ All beta_index values are within bounds!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Check NSD index beta_index bounds")
    parser.add_argument("index_path", help="Path to index file or root directory")
    parser.add_argument("--max-files", type=int, help="Limit number of files to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = check_index_headers(args.index_path, args.max_files)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()