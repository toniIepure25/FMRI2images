#!/usr/bin/env python3
"""
DEPRECATED: Use `python -m fmri2img.data.nsd_index_builder` instead.

This script redirects to the unified API for backward compatibility.
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.warning("⚠️  DEPRECATED: scripts/nsd_build_index_s3.py")
    logger.warning("   Use: python -m fmri2img.data.nsd_index_builder")
    logger.warning("   Redirecting to unified API...")
    
    parser = argparse.ArgumentParser(description="Build canonical NSD index (DEPRECATED)")
    parser.add_argument("--subjects", nargs="+", default=["subj01"], 
                       help="Subjects to process (e.g., subj01 subj02)")
    parser.add_argument("--out-root", default="data/indices/nsd_index",
                       help="Output root path (local or S3)")
    
    args = parser.parse_args()
    
    # Convert to new unified API call
    cmd = [
        sys.executable, "-m", "fmri2img.data.nsd_index_builder",
        "--subjects"] + args.subjects
    
    # Map old out-root to new output path
    if args.out_root != "data/indices/nsd_index":
        output_path = Path(args.out_root) / "unified_index.parquet"
        cmd.extend(["--output-path", str(output_path)])
    
    cmd.extend(["--output-format", "parquet"])
    
    logger.info(f"Redirecting to: {' '.join(cmd)}")
    
    try:
        # Execute the new unified command
        result = subprocess.run(cmd, check=True)
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Unified API call failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())