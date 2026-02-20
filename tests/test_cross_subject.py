"""Tests for cross-subject alignment modules."""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from fmri2img.models.cross_subject import (
    RidgeAligner,
    SubjectAdapter,
    procrustes_align,
)


class TestRidgeAligner:
    def test_output_shape(self):
        aligner = RidgeAligner(input_dim=500, output_dim=128)
        x = torch.randn(10, 500)
        out = aligner(x)
        assert out.shape == (10, 128)

    def test_fit_and_forward(self):
        aligner = RidgeAligner(input_dim=100, output_dim=32, alpha=1.0)
        X = torch.randn(50, 100)
        Y = torch.randn(50, 32)
        aligner.fit(X, Y)
        pred = aligner(X)
        assert pred.shape == (50, 32)
        # After fitting, reconstruction error should be moderate
        error = (pred - Y).norm() / Y.norm()
        assert error < 2.0

    def test_fit_updates_weights(self):
        aligner = RidgeAligner(input_dim=50, output_dim=16)
        w_before = aligner.projection.weight.clone()
        X = torch.randn(30, 50)
        Y = torch.randn(30, 16)
        aligner.fit(X, Y)
        w_after = aligner.projection.weight
        assert not torch.allclose(w_before, w_after)


class TestSubjectAdapter:
    def test_output_shape(self):
        adapter = SubjectAdapter(d_model=256)
        x = torch.randn(4, 256)
        out = adapter(x)
        assert out.shape == (4, 256)

    def test_near_identity_at_init(self):
        adapter = SubjectAdapter(d_model=128)
        x = torch.randn(3, 128)
        out = adapter(x)
        # Should be close to identity at initialization
        diff = (out - x).abs().mean()
        assert diff < 0.5

    def test_gradients_flow(self):
        adapter = SubjectAdapter(d_model=64)
        x = torch.randn(2, 64, requires_grad=True)
        out = adapter(x)
        out.sum().backward()
        assert x.grad is not None

    def test_custom_bottleneck(self):
        adapter = SubjectAdapter(d_model=256, bottleneck_dim=32)
        x = torch.randn(4, 256)
        out = adapter(x)
        assert out.shape == (4, 256)


class TestProcrustesAlign:
    def test_identity_rotation(self):
        N, D = 50, 32
        np.random.seed(42)
        source = np.random.randn(N, D)
        R, aligned = procrustes_align(source, source)
        assert aligned.shape == (N, D)
        np.testing.assert_allclose(R, np.eye(D), atol=0.1)

    def test_known_rotation(self):
        N, D = 100, 16
        np.random.seed(42)
        source = np.random.randn(N, D)
        # Create a known orthogonal rotation
        Q, _ = np.linalg.qr(np.random.randn(D, D))
        target = source @ Q
        R, aligned = procrustes_align(source, target)
        # Procrustes should recover the rotation well
        residual = np.linalg.norm(aligned - target) / np.linalg.norm(target)
        assert residual < 0.01

    def test_rotation_is_orthogonal(self):
        N, D = 50, 8
        np.random.seed(42)
        source = np.random.randn(N, D)
        target = np.random.randn(N, D)
        R, _ = procrustes_align(source, target)
        assert R.shape == (D, D)
        np.testing.assert_allclose(R @ R.T, np.eye(D), atol=1e-6)
