"""Tests for Frequency-ROI interaction module."""

import pytest
import torch
import torch.nn.functional as F

from fmri2img.models.freq_roi import (
    FrequencyBandFilter,
    FreqROIProjection,
    FreqROITransformerEncoder,
)


class TestFrequencyBandFilter:
    def test_output_shape(self):
        filt = FrequencyBandFilter(n_voxels=100, n_bands=4)
        x = torch.randn(2, 100)
        bands = filt(x)
        assert bands.shape == (2, 4, 100)

    def test_bands_reconstruct_input(self):
        filt = FrequencyBandFilter(n_voxels=64, n_bands=4)
        x = torch.randn(3, 64)
        bands = filt(x)
        reconstructed = bands.sum(dim=1)
        assert torch.allclose(x, reconstructed, atol=0.1)

    def test_different_n_bands(self):
        for n in [2, 4, 8]:
            filt = FrequencyBandFilter(n_voxels=32, n_bands=n)
            x = torch.randn(2, 32)
            bands = filt(x)
            assert bands.shape == (2, n, 32)


class TestFreqROIProjection:
    def test_output_shape(self):
        proj = FreqROIProjection(n_voxels=200, d_model=128, n_bands=4)
        x = torch.randn(4, 200)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_band_weights_sum_to_one(self):
        proj = FreqROIProjection(n_voxels=100, d_model=64, n_bands=4)
        weights = proj.band_weights
        assert weights.shape == (4,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_gradients_flow(self):
        proj = FreqROIProjection(n_voxels=50, d_model=32, n_bands=4)
        x = torch.randn(2, 50, requires_grad=True)
        out = proj(x)
        out.sum().backward()
        assert x.grad is not None


class TestFreqROITransformerEncoder:
    @pytest.fixture
    def encoder(self):
        roi_dims = {"V1": 100, "V2": 80, "FFA": 50, "PPA": 70}
        return FreqROITransformerEncoder(
            roi_dims=roi_dims,
            d_model=64,
            nhead=4,
            num_layers=2,
            n_bands=4,
        )

    def test_forward_shape(self, encoder):
        B = 3
        V = sum(encoder.roi_sizes)
        x = torch.randn(B, V)
        out = encoder(x)
        assert out.shape == (B, 64)

    def test_return_roi_tokens(self, encoder):
        B = 2
        V = sum(encoder.roi_sizes)
        x = torch.randn(B, V)
        out = encoder(x, return_roi_tokens=True)
        assert out.cls_out.shape == (B, 64)
        assert out.roi_tokens.shape == (B, 4, 64)
        assert out.cls_to_roi_alpha.shape == (B, 4)

    def test_frequency_importance(self, encoder):
        importance = encoder.get_frequency_importance()
        assert len(importance) == 4  # 4 ROIs
        for roi_name, weights in importance.items():
            assert len(weights) == 4  # 4 bands
            assert abs(sum(weights) - 1.0) < 1e-4

    def test_gradients_flow(self, encoder):
        B = 2
        V = sum(encoder.roi_sizes)
        x = torch.randn(B, V, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None
