#!/usr/bin/env python3
"""
Phase 2 Complete: IO Layer Example

This example demonstrates how to use the robust S3 loaders and centralized path management
for the Natural Scenes Dataset. This replaces naive file handling with production-ready
memory-safe loaders.

Key Features Demonstrated:
- Centralized path management with NSDLayout
- Memory-safe S3 data loading
- Proper error handling and caching
- Integration with canonical index from Phase 1
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.io.nsd_layout import NSDLayout
from fmri2img.io.s3 import NIfTILoader, CSVLoader, get_s3_filesystem
from fmri2img.data.nsd_index import NSDIndex

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
    
    # Alternative preprocessing
    beta_path_alt = layout.beta_path(
        subject=2, session=5, 
        preprocessing="betas_fithrf"
    )
    logger.info(f"   Alt preprocessing: {beta_path_alt}")
    
    # Stimulus files
    stim_path = layout.stim_hdf5_path()
    logger.info(f"   Stimuli HDF5: {stim_path}")
    
    stim_info_path = layout.stim_info_path()
    logger.info(f"   Stimulus metadata: {stim_info_path}")
    
    # COCO fallback
    coco_url = layout.coco_http_url(391895, 'train2017')
    logger.info(f"   COCO fallback: {coco_url}")
    
    # 3. Demonstrate S3 file system operations
    logger.info("\n3. S3 Filesystem Operations...")
    s3_fs = get_s3_filesystem()
    
    # Check file existence
    fsspec_beta_path = beta_path.replace("s3://", "")
    exists = s3_fs.exists(fsspec_beta_path)
    logger.info(f"   Beta file exists: {exists}")
    
    if exists:
        file_info = s3_fs.info(fsspec_beta_path)
        size_mb = file_info.get('size', 0) / (1024**2)
        logger.info(f"   File size: {size_mb:.1f} MB")
    
    # 4. Load stimulus metadata with CSV loader
    logger.info("\n4. Loading Stimulus Metadata...")
    csv_loader = CSVLoader()
    
    try:
        # Load first 1000 rows for demo
        stim_df = csv_loader.load(stim_info_path, nrows=1000)
        logger.info(f"   Loaded metadata: {stim_df.shape}")
        logger.info(f"   Columns: {list(stim_df.columns)[:5]}...")
        
        # Show statistics
        unique_coco = stim_df['cocoId'].nunique()
        logger.info(f"   Unique COCO images: {unique_coco}")
        
        subject_cols = [col for col in stim_df.columns if col.startswith('subject')]
        logger.info(f"   Subject columns: {len(subject_cols)}")
        
    except Exception as e:
        logger.error(f"   Failed to load CSV: {e}")
    
    # 5. Demonstrate NIfTI loading (if file exists)
    logger.info("\n5. NIfTI Loading Demo...")
    
    if exists:
        nifti_loader = NIfTILoader()
        try:
            # Get header info without loading full data
            header_info = nifti_loader.get_header(beta_path)
            logger.info(f"   NIfTI shape: {header_info['shape']}")
            logger.info(f"   Data type: {header_info['dtype']}")
            logger.info(f"   Voxel size: {header_info['voxel_size'][:3]}")
            
            # Could load full data like this (but would be large):
            # img = nifti_loader.load(beta_path)
            # data = img.get_fdata()
            
        except Exception as e:
            logger.error(f"   NIfTI loading failed: {e}")
    else:
        logger.info("   Skipping NIfTI demo - file not available")
    
    # 6. Integration with canonical index
    logger.info("\n6. Integration with Canonical Index...")
    
    try:
        # Load the index we built in Phase 1
        index_path = "data/indices/test_nsd_index.parquet"
        if Path(index_path).exists():
            nsd_index = NSDIndex(index_path)
            
            # Get trial information
            trial_info = nsd_index.get_subject(1).head(1)  # Get first trial for subject 1
            if not trial_info.empty:
                trial = trial_info.iloc[0]
                logger.info(f"   Trial example: {trial['global_trial_id']}")
                logger.info(f"   NSD ID: {trial['nsd_id']}")
                logger.info(f"   COCO ID: {trial['coco_id']}")
                
                # The beta file path is already in the index!
                beta_from_index = trial['beta_file']
                logger.info(f"   Beta from index: {beta_from_index}")
                
                # Generate full S3 URL using layout
                full_beta_url = f"s3://{layout.paths.bucket}/{beta_from_index}"
                logger.info(f"   Full S3 URL: {full_beta_url}")
                
        else:
            logger.info("   Index not found - run Phase 1 test first")
            
    except Exception as e:
        logger.error(f"   Index integration failed: {e}")
    
    # 7. Path validation and utilities
    logger.info("\n7. Path Validation...")
    
    # Test subject/session validation
    valid_cases = [
        (1, 1), (8, 30), (3, 15)
    ]
    invalid_cases = [
        (99, 1), (1, 999), (-1, 5)
    ]
    
    for subject, session in valid_cases:
        is_valid = layout.validate_subject_session(subject, session)
        logger.info(f"   Subject {subject}, Session {session}: {'✓' if is_valid else '✗'}")
    
    for subject, session in invalid_cases:
        is_valid = layout.validate_subject_session(subject, session)
        logger.info(f"   Subject {subject}, Session {session}: {'✓' if is_valid else '✗'}")
    
    # Available options
    resolutions = layout.get_available_resolutions()
    logger.info(f"   Available resolutions: {resolutions}")
    
    pipelines = layout.get_available_preprocessing()
    logger.info(f"   Available preprocessing: {pipelines}")
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Phase 2 Complete!")
    logger.info("\nKey Achievements:")
    logger.info("✅ Centralized path management with NSDLayout")
    logger.info("✅ Memory-safe S3 loaders for NIfTI, HDF5, CSV")
    logger.info("✅ Robust error handling and caching")
    logger.info("✅ Integration with Phase 1 canonical index")
    logger.info("✅ COCO dataset fallback support")
    logger.info("✅ Production-ready IO layer")
    
    return True

if __name__ == "__main__":
    demo_io_layer()