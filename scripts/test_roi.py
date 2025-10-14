#!/usr/bin/env python3
"""
Test ROI Pooling Functionality
==============================

Simple test to verify ROI pooling works with mock data.
"""

import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fmri2img.data.roi import ROIPooler, ROIDef
    ROI_AVAILABLE = True
except ImportError:
    ROI_AVAILABLE = False


def create_mock_beta_volume(shape=(64, 64, 32), save_path=None):
    """Create a mock beta volume NIfTI file."""
    data = np.random.randn(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    
    if save_path:
        nib.save(img, save_path)
        logger.info(f"Saved mock beta volume to {save_path}")
    
    return img


def create_mock_roi_masks(shape=(64, 64, 32), roi_dir=None, n_rois=3):
    """Create mock ROI mask files."""
    if roi_dir is None:
        roi_dir = Path(tempfile.mkdtemp())
    
    roi_files = []
    
    for i in range(n_rois):
        # Create a random ROI mask
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Random blob ROI
        center = []
        for s in shape:
            low = min(5, s//4)
            high = max(low + 1, s - s//4)
            center.append(np.random.randint(low, high))
        radius = np.random.randint(2, max(3, min(shape)//8))
        
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    dist = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)**0.5
                    if dist <= radius:
                        mask[x, y, z] = 1
        
        # Save mask
        roi_file = roi_dir / f"roi_{i:02d}_test.nii.gz"
        mask_img = nib.Nifti1Image(mask, affine=np.eye(4))
        nib.save(mask_img, roi_file)
        roi_files.append(roi_file)
        
        logger.info(f"Created ROI {i}: {mask.sum()} voxels at {roi_file}")
    
    return roi_files


def test_roi_pooler():
    """Test ROI pooler functionality."""
    if not ROI_AVAILABLE:
        logger.error("ROI functionality not available")
        return False
    
    logger.info("Testing ROI pooler...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        shape = (32, 32, 16)  # Smaller for testing
        
        # Create beta volume
        beta_path = temp_path / "beta_test.nii.gz"
        beta_img = create_mock_beta_volume(shape, beta_path)
        
        # Create ROI masks in expected directory structure
        roi_dir = temp_path / "nsddata" / "ppdata" / "subj01" / "anat"
        roi_dir.mkdir(parents=True)
        roi_files = create_mock_roi_masks(shape, roi_dir, n_rois=3)
        
        # Mock NSDLayout to return our test path
        class MockNSDLayout:
            def __init__(self, *args, **kwargs):
                pass  # Ignore any arguments
                
            def roi_masks_path(self, subject, full_url=True):
                pattern = str(roi_dir / "*roi*.nii.gz")
                if full_url:
                    return pattern
                return pattern.replace("s3://natural-scenes-dataset/", "")
        
        # Temporarily patch the import
        import fmri2img.data.roi as roi_module
        original_layout = getattr(roi_module, 'NSDLayout', None)
        roi_module.NSDLayout = MockNSDLayout
        
        try:
            # Test ROI pooler
            pooler = ROIPooler("subj01", min_voxels=10)
            
            # Fit on sample beta
            pooler.fit(str(beta_path))
            
            logger.info(f"Fitted {len(pooler.rois)} ROIs")
            logger.info(f"ROI names: {pooler.names()}")
            
            # Test pooling on a volume
            test_vol = np.random.randn(*shape).astype(np.float32)
            pooled = pooler.pool(test_vol)
            
            logger.info(f"Input shape: {test_vol.shape}")
            logger.info(f"Pooled shape: {pooled.shape}")
            logger.info(f"Pooled values: {pooled}")
            
            # Verify results
            assert pooled.shape == (len(pooler.rois),), f"Wrong pooled shape: {pooled.shape}"
            assert pooled.dtype == np.float32, f"Wrong dtype: {pooled.dtype}"
            assert not np.any(np.isnan(pooled)), "Found NaN values in pooled result"
            
            logger.info("✓ ROI pooler test passed!")
            return True
            
        finally:
            # Restore original
            if original_layout:
                roi_module.NSDLayout = original_layout


def test_roi_def():
    """Test ROIDef dataclass."""
    logger.info("Testing ROIDef...")
    
    indices = np.array([10, 20, 30, 40])
    roi = ROIDef(name="test_roi", mask_indices=indices)
    
    assert roi.name == "test_roi"
    assert np.array_equal(roi.mask_indices, indices)
    
    logger.info("✓ ROIDef test passed!")
    return True


def main():
    logger.info("Running ROI functionality tests...")
    
    success = True
    
    # Test ROIDef
    if not test_roi_def():
        success = False
    
    # Test ROI pooler
    if not test_roi_pooler():
        success = False
    
    if success:
        logger.info("🎉 All ROI tests passed!")
        return 0
    else:
        logger.error("❌ Some ROI tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())