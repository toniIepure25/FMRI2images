"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
import torch
from pathlib import Path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output = tmp_path / "outputs"
    output.mkdir()
    return output


@pytest.fixture
def sample_fmri():
    """Generate sample fMRI data for testing."""
    # 10 samples x 1000 voxels
    return np.random.randn(10, 1000).astype(np.float32)


@pytest.fixture
def sample_clip_embeddings():
    """Generate sample CLIP embeddings for testing."""
    # 10 samples x 512 dimensions
    embeddings = np.random.randn(10, 512).astype(np.float32)
    # L2 normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return embeddings


@pytest.fixture
def sample_nsd_ids():
    """Generate sample NSD IDs for testing."""
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture
def mock_encoding_model():
    """Mock encoding model for testing."""
    class MockEncodingModel:
        def __init__(self):
            self.n_voxels = 1000
            
        def predict(self, images):
            """Generate fake fMRI predictions."""
            batch_size = len(images) if isinstance(images, list) else images.shape[0]
            return np.random.randn(batch_size, self.n_voxels).astype(np.float32)
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
    
    return MockEncodingModel()
