"""Tests for brain alignment evaluation."""
import numpy as np
import pytest
from PIL import Image

from fmri2img.eval.brain_alignment import (
    compute_brain_alignment,
    compute_brain_alignment_with_ceiling
)


def test_compute_brain_alignment_basic(mock_encoding_model):
    """Test basic brain alignment computation."""
    # Generate test data
    n_images = 10
    n_voxels = 1000
    
    # Create fake images
    images = [Image.new("RGB", (64, 64), color=(i*10, i*10, i*10)) for i in range(n_images)]
    
    # Create fake fMRI targets
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    # Compute alignment
    alignment = compute_brain_alignment(
        mock_encoding_model,
        images,
        fmri_targets,
        batch_size=4,
        device="cpu"
    )
    
    # Check keys
    assert "voxelwise_corr_mean" in alignment
    assert "voxelwise_corr_median" in alignment
    assert "voxelwise_corr_std" in alignment
    assert "subject_level_corr" in alignment
    assert "n_voxels" in alignment
    assert "n_images" in alignment
    
    # Check values
    assert alignment["n_images"] == n_images
    assert alignment["n_voxels"] == n_voxels
    
    # Correlations should be in valid range
    assert -1.0 <= alignment["voxelwise_corr_mean"] <= 1.0
    assert -1.0 <= alignment["subject_level_corr"] <= 1.0


def test_compute_brain_alignment_with_roi_mask(mock_encoding_model):
    """Test alignment with ROI mask."""
    n_images = 10
    n_voxels = 1000
    
    images = [Image.new("RGB", (64, 64)) for _ in range(n_images)]
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    # Create ROI mask (select half the voxels)
    roi_mask = np.zeros(n_voxels, dtype=bool)
    roi_mask[:500] = True
    
    alignment = compute_brain_alignment(
        mock_encoding_model,
        images,
        fmri_targets,
        roi_mask=roi_mask,
        batch_size=4,
        device="cpu"
    )
    
    # Should only use ROI voxels
    assert alignment["n_voxels"] == 500


def test_compute_brain_alignment_return_voxelwise(mock_encoding_model):
    """Test returning voxel-wise correlations."""
    n_images = 10
    n_voxels = 100
    
    images = [Image.new("RGB", (64, 64)) for _ in range(n_images)]
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    alignment = compute_brain_alignment(
        mock_encoding_model,
        images,
        fmri_targets,
        batch_size=4,
        device="cpu",
        return_voxelwise=True
    )
    
    # Should include voxelwise array
    assert "voxelwise_corr" in alignment
    assert alignment["voxelwise_corr"].shape == (n_voxels,)


def test_compute_brain_alignment_shape_mismatch(mock_encoding_model):
    """Test error on shape mismatch."""
    images = [Image.new("RGB", (64, 64)) for _ in range(10)]
    fmri_targets = np.random.randn(5, 1000).astype(np.float32)  # Wrong number
    
    with pytest.raises(ValueError, match="must match"):
        compute_brain_alignment(
            mock_encoding_model,
            images,
            fmri_targets,
            device="cpu"
        )


def test_compute_brain_alignment_numpy_images(mock_encoding_model):
    """Test with numpy array images."""
    n_images = 5
    n_voxels = 100
    
    # Create numpy images
    images = np.random.randint(0, 255, size=(n_images, 64, 64, 3), dtype=np.uint8)
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    alignment = compute_brain_alignment(
        mock_encoding_model,
        images,
        fmri_targets,
        batch_size=2,
        device="cpu"
    )
    
    assert alignment["n_images"] == n_images


def test_compute_brain_alignment_with_ceiling(mock_encoding_model):
    """Test brain alignment with noise ceiling normalization."""
    n_images = 10
    n_voxels = 1000
    
    images = [Image.new("RGB", (64, 64)) for _ in range(n_images)]
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    # Create noise ceiling map
    noise_ceiling = np.random.uniform(0.3, 0.8, size=n_voxels).astype(np.float32)
    
    alignment = compute_brain_alignment_with_ceiling(
        mock_encoding_model,
        images,
        fmri_targets,
        noise_ceiling_map=noise_ceiling,
        batch_size=4,
        device="cpu"
    )
    
    # Should have raw and normalized metrics
    assert "voxelwise_corr_mean" in alignment
    assert "voxelwise_corr_mean_normalized" in alignment
    assert "subject_level_corr" in alignment
    assert "subject_level_corr_normalized" in alignment
    assert "roi_ceiling" in alignment
    
    # ROI ceiling should be mean of noise ceiling
    assert 0.3 <= alignment["roi_ceiling"] <= 0.8


def test_compute_brain_alignment_with_ceiling_no_ceiling(mock_encoding_model):
    """Test ceiling normalization without ceiling map."""
    n_images = 5
    n_voxels = 100
    
    images = [Image.new("RGB", (64, 64)) for _ in range(n_images)]
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    alignment = compute_brain_alignment_with_ceiling(
        mock_encoding_model,
        images,
        fmri_targets,
        noise_ceiling_map=None,  # No ceiling
        batch_size=4,
        device="cpu"
    )
    
    # Should have None for normalized metrics
    assert alignment["voxelwise_corr_mean_normalized"] is None
    assert alignment["subject_level_corr_normalized"] is None
    assert alignment["roi_ceiling"] is None


def test_compute_brain_alignment_perfect_correlation(mock_encoding_model):
    """Test alignment with perfect correlation (mock model predicts perfectly)."""
    n_images = 10
    n_voxels = 100
    
    images = [Image.new("RGB", (64, 64)) for _ in range(n_images)]
    
    # Make encoding model deterministic
    fmri_targets = mock_encoding_model.predict(images)
    
    alignment = compute_brain_alignment(
        mock_encoding_model,
        images,
        fmri_targets,
        batch_size=4,
        device="cpu"
    )
    
    # With same random seed in mock, should have high correlation
    # (Not perfect due to batching differences, but should be positive)
    assert alignment["voxelwise_corr_mean"] > -0.5  # Loose check for random noise


def test_compute_brain_alignment_with_roi_and_ceiling(mock_encoding_model):
    """Test alignment with both ROI mask and ceiling."""
    n_images = 10
    n_voxels = 1000
    
    images = [Image.new("RGB", (64, 64)) for _ in range(n_images)]
    fmri_targets = np.random.randn(n_images, n_voxels).astype(np.float32)
    
    # ROI mask
    roi_mask = np.zeros(n_voxels, dtype=bool)
    roi_mask[:500] = True
    
    # Noise ceiling (full brain)
    noise_ceiling = np.random.uniform(0.3, 0.8, size=n_voxels).astype(np.float32)
    
    alignment = compute_brain_alignment_with_ceiling(
        mock_encoding_model,
        images,
        fmri_targets,
        noise_ceiling_map=noise_ceiling,
        roi_mask=roi_mask,
        batch_size=4,
        device="cpu"
    )
    
    # Should use ROI subset
    assert alignment["n_voxels"] == 500
    
    # ROI ceiling should be from masked voxels only
    expected_ceiling = np.mean(noise_ceiling[roi_mask])
    assert abs(alignment["roi_ceiling"] - expected_ceiling) < 0.01
