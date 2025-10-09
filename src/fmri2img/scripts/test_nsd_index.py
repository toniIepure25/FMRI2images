#!/usr/bin/env python3
"""
Test script for the canonical NSD index builder
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmri2img.data.nsd_index_builder import NSDIndexBuilder
from fmri2img.data.nsd_index import NSDIndex
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_index_builder():
    """Test the index builder with a small dataset"""
    logger.info("Testing NSD Index Builder")
    
    try:
        # Initialize builder
        builder = NSDIndexBuilder("configs/data.yaml")
        
        # Build index for one subject, limited sessions
        logger.info("Building test index...")
        index_df = builder.build_full_index(
            subjects=[1], 
            sessions=[1, 2]  # Just first 2 sessions for testing
        )
        
        if index_df.empty:
            logger.error("Failed to build index")
            return False
        
        # Validate index
        logger.info("Validating index...")
        validation_results = builder.validate_index(index_df)
        
        # Save test index
        output_path = builder.save_index(index_df, "test_nsd_index.parquet")
        
        logger.info("Index builder test completed successfully!")
        logger.info(f"Created index with {len(index_df)} trials")
        
        return True, output_path
        
    except Exception as e:
        logger.error(f"Index builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_index_interface(index_path):
    """Test the index interface"""
    logger.info("Testing NSD Index Interface")
    
    try:
        # Load index
        nsd_idx = NSDIndex(index_path)
        
        # Test basic properties
        logger.info(f"Subjects: {nsd_idx.subjects}")
        logger.info(f"Sessions: {nsd_idx.sessions}")
        
        # Test summary
        summary = nsd_idx.summary
        logger.info("Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Test querying
        if nsd_idx.subjects:
            subject = nsd_idx.subjects[0]
            
            # Get subject data
            subject_data = nsd_idx.get_subject(subject)
            logger.info(f"Subject {subject}: {len(subject_data)} trials")
            
            # Get session data
            if not subject_data.empty:
                session = subject_data['session'].iloc[0]
                session_data = nsd_idx.get_session(subject, session)
                logger.info(f"Session {session}: {len(session_data)} trials")
                
                # Get specific trial
                if not session_data.empty:
                    trial = nsd_idx.get_trial(subject, session, 0)
                    if trial is not None:
                        logger.info(f"Trial example: {trial['global_trial_id']}")
                        logger.info(f"  NSD ID: {trial['nsd_id']}")
                        logger.info(f"  Beta file: {trial['beta_file']}")
        
        # Test train/val split
        logger.info("Testing train/val split...")
        train_df, val_df = nsd_idx.create_train_val_split(val_fraction=0.3)
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")
        
        # Test clean trials
        clean_df = nsd_idx.get_clean_trials()
        logger.info(f"Clean trials: {len(clean_df)}")
        
        logger.info("Index interface test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Index interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("Starting NSD Index tests...")
    
    # Test 1: Index builder
    success, index_path = test_index_builder()
    if not success:
        logger.error("Index builder test failed")
        return 1
    
    # Test 2: Index interface
    success = test_index_interface(index_path)
    if not success:
        logger.error("Index interface test failed")
        return 1
    
    logger.info("All tests passed!")
    logger.info(f"Test index saved to: {index_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())