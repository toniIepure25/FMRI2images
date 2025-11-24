#!/usr/bin/env python3
"""
Quick test to see if preprocessing data loading works
"""
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.data.nsd_index_reader import read_subject_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing index loading...")
    
    try:
        df = read_subject_index("data/indices/nsd_index", "subj01")
        logger.info(f"✓ Loaded index: {len(df)} rows")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required = ["beta_file", "volume_index"]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"⚠️  Missing columns: {missing}")
            logger.info(f"  Available columns: {list(df.columns)}")
            
            # Check for alternatives
            if "beta_path" in df.columns:
                logger.info("  Found 'beta_path' (alternative to 'beta_file')")
            if "beta_index" in df.columns:
                logger.info("  Found 'beta_index' (alternative to 'volume_index')")
        else:
            logger.info("✓ All required columns present")
            
        # Show a sample row
        logger.info("\nSample row:")
        first_row = df.iloc[0]
        for col in df.columns:
            logger.info(f"  {col}: {first_row[col]}")
            
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
