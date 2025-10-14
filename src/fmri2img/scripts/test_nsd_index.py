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
    """Test the index builder with unified API"""
    logger.info("Testing NSD Index Builder")
    
    try:
        # Initialize builder
        builder = NSDIndexBuilder()
        
        # Build index for one subject with limited trials
        logger.info("Building test index...")
        index_df = builder.build_index(
            subjects=["subj01"],
            max_trials_per_subject=5  # Limited for testing
        )
        
        if index_df.empty:
            logger.error("Failed to build index")
            raise AssertionError("Failed to build index")
        
        # Validate index
        logger.info("Validating index...")
        builder.validate_index(index_df)
        
        logger.info("✅ Index builder test completed successfully")
                
        logger.info(f"Built index with {len(index_df)} trials")
        
        # Test canonical columns
        required_columns = [
            'subject', 'global_trial_index', 'nsdId', 
            'beta_path', 'beta_index'
        ]
        
        for col in required_columns:
            assert col in index_df.columns, f"Missing column: {col}"
        assert len(index_df) > 0, "Index should have trials"
        
        # Test that beta_path contains full S3 URLs
        assert index_df["beta_path"].str.startswith("s3://").all(), "beta_path must be full S3 URL"
        logger.info("✅ All beta_path entries are full S3 URLs")
        
    except Exception as e:
        logger.error(f"Index builder test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_index_interface():
    """Test the canonical index builder interface"""
    logger.info("Testing NSD Index Builder Canonical API")
    
    try:
        # Build fresh index for interface testing
        builder = NSDIndexBuilder()
        index_df = builder.build_index(["subj01"], max_trials_per_subject=3)
        
        # Test canonical column names
        required_columns = [
            'subject', 'global_trial_index', 'nsdId', 'beta_path'
        ]
        for col in required_columns:
            assert col in index_df.columns, f"Missing canonical column: {col}"
        
        # Test that beta_path contains full S3 URLs
        assert index_df["beta_path"].str.startswith("s3://").all(), "beta_path must be full S3 URL"
        logger.info("✅ All beta_path entries are full S3 URLs in interface test")
        
        # Test canonical API methods
        trial_count = builder.get_trial_count(index_df, "subj01")
        assert trial_count == 3, f"Expected 3 trials, got {trial_count}"
        
        unique_stimuli = builder.get_unique_stimuli(index_df)
        assert len(unique_stimuli) > 0, "Should have unique stimuli"
        
        repeat_trials = builder.get_repeat_trials(index_df)
        # With only 3 trials, likely no repeats
        
        # Test session filtering
        session_trials = builder.get_session_trials(index_df, "subj01", 1)
        assert len(session_trials) == 3, "All test trials should be in session 1"
        
        logger.info("✅ All canonical API tests passed")
        
    except Exception as e:
        logger.error(f"Index interface test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main test function"""
    logger.info("Starting NSD Index tests...")
    
    try:
        # Test 1: Index builder  
        test_index_builder()
        
        # Test 2: Index interface
        test_index_interface()
        
        logger.info("All tests passed!")
        return 0
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())