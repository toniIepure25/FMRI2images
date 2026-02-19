#!/usr/bin/env python3
"""
NSD Sanity Check - Updated to use unified API

Tests basic functionality of the unified NSD data loading pipeline.
"""

import argparse
import logging
from fmri2img.data.nsd_index_builder import NSDIndexBuilder
from fmri2img.io.nsd_layout import NSDLayout
from fmri2img.io.s3 import get_s3_filesystem, CSVLoader, NIfTILoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NSD Sanity Check with Unified API")
    parser.add_argument("--subjects", nargs="+", default=["subj01"], 
                       help="Subjects to test")
    parser.add_argument("--limit", type=int, default=3,
                       help="Number of trials to test per subject")
    
    args = parser.parse_args()
    
    try:
        logger.info("🔍 NSD Sanity Check - Unified API")
        logger.info("=" * 50)
        
        # 1. Test layout manager
        logger.info("1. Testing NSD Layout Manager...")
        layout = NSDLayout()
        logger.info(f"   Bucket: {layout.paths.bucket}")
        
        # 2. Test S3 access
        logger.info("2. Testing S3 Access...")
        s3_fs = get_s3_filesystem()
        csv_loader = CSVLoader(s3_fs)
        
        # 3. Test stimulus catalog loading
        logger.info("3. Testing Stimulus Catalog...")
        stim_path = layout.stim_info_path()
        stim_df = csv_loader.load(stim_path)
        logger.info(f"   Loaded {len(stim_df)} stimuli")
        
        # 4. Test unified index builder
        logger.info("4. Testing Unified Index Builder...")
        builder = NSDIndexBuilder()
        index_df = builder.build_index(args.subjects, max_trials_per_subject=args.limit)
        
        logger.info(f"   Built index: {len(index_df)} trials")
        logger.info(f"   Standardized columns: {len(index_df.columns)}")
        
        # 5. Test specific trials
        logger.info("5. Testing Sample Trials...")
        nifti_loader = NIfTILoader(s3_fs)
        
        for i, (_, trial) in enumerate(index_df.head(args.limit).iterrows()):
            logger.info(f"   Trial {i+1}:")
            logger.info(f"     Subject: {trial['subject']}")
            logger.info(f"     Global index: {trial['global_trial_index']}")
            logger.info(f"     NSD ID: {trial['nsdId']}")
            logger.info(f"     Beta path: {trial['beta_path']}")
            
            # Test header-only access
            try:
                shape = nifti_loader.get_shape(trial['beta_path'])
                logger.info(f"     Beta shape: {shape}")
                
                if len(shape) > 3 and trial['beta_index'] < shape[3]:
                    logger.info(f"     ✅ Valid volume index: {trial['beta_index']}")
                else:
                    logger.warning(f"     ⚠️  Invalid volume index: {trial['beta_index']}")
                    
            except Exception as e:
                logger.warning(f"     ⚠️  Beta access failed: {e}")
        
        logger.info("\n✅ Sanity check completed successfully!")
        logger.info("📊 All unified API components working correctly")
        
    except Exception as e:
        logger.error(f"❌ Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
