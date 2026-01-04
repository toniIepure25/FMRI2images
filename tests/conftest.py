"""Pytest configuration and shared fixtures.

This repo uses a `src/` layout. In CI or fresh environments, tests may be run
without installing the package in editable mode.

To keep tests runnable in that situation, we add the repository `src/` folder
to `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import pytest
import numpy as np
import torch


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
        def __init__(self, n_voxels: int = 1000):
            self.n_voxels = int(n_voxels)
            
        def predict(self, images):
            """Generate fake fMRI predictions."""
            batch_size = len(images) if isinstance(images, list) else images.shape[0]
            return np.random.randn(batch_size, self.n_voxels).astype(np.float32)

        def with_n_voxels(self, n_voxels: int):
            """Return a new mock model with a different output dimensionality."""
            return MockEncodingModel(n_voxels=n_voxels)
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
    
    return MockEncodingModel()
