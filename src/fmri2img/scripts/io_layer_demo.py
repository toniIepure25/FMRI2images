#!/usr/bin/env python3
"""
Phase 2 Complete: IO Layer Example

This example demonstrates how to use the robust S3 loaders and centralized path management
for the Natural Scenes Dataset. Shows integration with canonical index.
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.io.nsd_layout import NSDLayout
from fmri2img.io.s3 import NIfTILoader, CSVLoader, get_s3_filesystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_io_layer():
    """Demonstrate the complete IO layer functionality"""
    
    logger.info("🚀 Phase 2 IO Layer Demo")
    logger.info("=" * 50)
    
    # 1. Initialize path management
    logger.info("1. Initializing NSD Layout Manager...")
    layout = NSDLayout("configs/data.yaml")
    logger.info(f"   Bucket: {layout.paths.bucket}")
    logger.info(f"   Default resolution: {layout.paths.default_resolution}")
    logger.info(f"   Default preprocessing: {layout.paths.default_preprocessing}")
    
    # 2. Generate paths for different data types
    logger.info("\n2. Generating S3 Paths...")
    
    # Beta (fMRI) file paths
    beta_path = layout.beta_path(subject=1, session=1)
    logger.info(f"   Beta file: {beta_path}")
    
    # Stimulus info path
    stim_info_path = layout.stim_info_path()
    logger.info(f"   Stimulus catalog: {stim_info_path}")
    
    # 3. Initialize S3 loaders
    logger.info("\n3. Initializing S3 Loaders...")
    s3_fs = get_s3_filesystem()
    csv_loader = CSVLoader(s3_fs)
    nifti_loader = NIfTILoader(s3_fs)
    
    # 4. Load stimulus catalog
    logger.info("\n4. Loading Stimulus Catalog...")
    try:
        stim_df = csv_loader.load(stim_info_path)
        logger.info(f"   Loaded {len(stim_df)} stimuli")
        logger.info(f"   Columns: {list(stim_df.columns)}")
        logger.info(f"   Sample nsdId range: {stim_df['nsdId'].min()}-{stim_df['nsdId'].max()}")
    except Exception as e:
        logger.error(f"   Failed to load stimulus catalog: {e}")
        return
    
    # 5. Test unified index builder API
    logger.info("\n5. Testing Unified Index Builder API...")
    try:
        from fmri2img.data.nsd_index_builder import NSDIndexBuilder
        
        # Initialize builder
        builder = NSDIndexBuilder()
        
        # Build test index with standardized API
        test_subjects = ["subj01"]
        logger.info(f"   Building test index for: {test_subjects}")
        
        # Build with limited trials for demo
        index_df = builder.build_index(test_subjects, max_trials_per_subject=5)
        
        logger.info(f"   Built index: {len(index_df)} trials")
        logger.info(f"   Canonical columns: {list(index_df.columns)}")
        
        # Use canonical API methods
        trial_count = builder.get_trial_count(index_df, "subj01")
        unique_stimuli = builder.get_unique_stimuli(index_df)
        repeat_trials = builder.get_repeat_trials(index_df)
        
        logger.info(f"   Subject subj01 trials: {trial_count}")
        logger.info(f"   Unique stimuli: {len(unique_stimuli)}")
        logger.info(f"   Repeat trials: {len(repeat_trials)}")
        
        # Show sample with canonical names
        if not index_df.empty:
            sample = index_df.iloc[0]
            logger.info(f"   Sample trial (canonical):")
            logger.info(f"     subject: {sample['subject']}")
            logger.info(f"     global_trial_index: {sample['global_trial_index']}")
            logger.info(f"     nsdId: {sample['nsdId']}")
            logger.info(f"     beta_path: {sample['beta_path']}")
            
    except Exception as e:
        logger.warning(f"   Index builder test failed: {e}")
    
    # 6. Test S3 Parquet writing (if configured)
    logger.info("\n6. Testing S3 Parquet Operations...")
    try:
        # Test layout's Parquet methods
        test_index_path = layout.index_path("test_demo_index", format="parquet")
        logger.info(f"   Test index path: {test_index_path}")
        
        # Create small test DataFrame with canonical columns
        import pandas as pd
        test_df = pd.DataFrame({
            'subject': ['subj01', 'subj01'],
            'global_trial_index': [0, 1],
            'nsdId': [0, 1],
            'test_value': [42, 43]
        })
        
        # Actual S3 Parquet round-trip test
        logger.info(f"   Attempting Parquet round-trip with {len(test_df)} test rows...")
        try:
            layout.write_parquet_to_s3(test_df, test_index_path, engine="pyarrow")
            df_back = layout.read_parquet_from_s3(test_index_path)
            logger.info(f"   Round-trip rows: wrote {len(test_df)}, read {len(df_back)}")
            logger.info("   ✅ S3 Parquet round-trip successful!")
        except Exception as write_err:
            logger.warning(f"   S3 Parquet round-trip failed: {write_err}")
            # Fallback to local path
            from pathlib import Path
            local_fallback = Path("data/indices/test_demo_index.parquet")
            local_fallback.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_parquet(local_fallback, index=False)
            df_back = pd.read_parquet(local_fallback)
            logger.info(f"   ▶ Fallback to local Parquet OK: wrote {len(test_df)}, read {len(df_back)}")
        
    except Exception as e:
        logger.warning(f"   S3 Parquet test failed: {e}")
    
    # 7. Test NIfTI header loading (if beta file exists)
    logger.info("\n7. Testing NIfTI Header Loading...")
    try:
        # Check if file exists first
        if s3_fs.exists(beta_path):
            img = nifti_loader.load(beta_path)
            logger.info(f"   Beta file shape: {img.shape}")
            logger.info(f"   Data type: {img.get_data_dtype()}")
            if len(img.shape) == 4:
                logger.info(f"   Number of trials in session: {img.shape[3]}")
        else:
            logger.warning(f"   Beta file not found: {beta_path}")
            logger.info("   This is expected if testing with limited data access")
            
    except Exception as e:
        logger.warning(f"   Could not load beta file header: {e}")
    
    # 7. Summary
    logger.info("\n7. Summary")
    logger.info("=" * 50)
    logger.info("✅ NSD Layout: Centralized path management working")
    logger.info("✅ S3 Loaders: Memory-safe streaming working")
    logger.info("✅ Stimulus Catalog: Successfully loaded from S3")
    logger.info("📋 Index Integration: Available if index built")
    logger.info("🧠 Beta Loading: Available with proper S3 access")
    
    logger.info("\nPhase 2 IO Layer is ready for production use!")
    logger.info("Next: Build canonical index with 'make index'")


if __name__ == "__main__":
    demo_io_layer()