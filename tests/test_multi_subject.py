"""Integration tests for multi-subject ROI Transformer and dataset."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from fmri2img.models.multi_subject_encoder import MultiSubjectROITransformer
from fmri2img.models.roi_transformer import ROITransformerOutput
from fmri2img.models.roi_dcf import ROIDCFDecoder
from fmri2img.models.vmf_decoder import VonMisesFisherDecoder


ROI_NAMES = ["V1v", "V1d", "V2v", "V3v", "FFA1", "PPA", "other"]


def _make_roi_dims(base_sizes):
    """Build subject_roi_dims with slight per-subject variation."""
    subjects = {}
    for i, subj in enumerate(["subj01", "subj02"]):
        subjects[subj] = {
            name: size + i * 10 for name, size in zip(ROI_NAMES, base_sizes)
        }
    return subjects


BASE_SIZES = [50, 40, 35, 30, 20, 25, 100]
SUBJECT_ROI_DIMS = _make_roi_dims(BASE_SIZES)


class TestMultiSubjectROITransformer:
    @pytest.fixture
    def encoder(self):
        return MultiSubjectROITransformer(
            subject_roi_dims=SUBJECT_ROI_DIMS,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.0,
        )

    def test_single_subject_forward(self, encoder):
        n_vox_s01 = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(4, n_vox_s01)
        out = encoder(x, subject_ids=0)
        assert out.shape == (4, 64)

    def test_single_subject_second(self, encoder):
        n_vox_s02 = sum(SUBJECT_ROI_DIMS["subj02"].values())
        x = torch.randn(4, n_vox_s02)
        out = encoder(x, subject_ids=1)
        assert out.shape == (4, 64)

    def test_return_roi_tokens(self, encoder):
        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(2, n_vox)
        out = encoder(x, subject_ids=0, return_roi_tokens=True)
        assert isinstance(out, ROITransformerOutput)
        assert out.cls_out.shape == (2, 64)
        assert out.roi_tokens.shape == (2, len(ROI_NAMES), 64)
        assert out.cls_to_roi_alpha.shape == (2, len(ROI_NAMES))
        assert torch.allclose(
            out.cls_to_roi_alpha.sum(dim=-1),
            torch.ones(2),
            atol=1e-5,
        )

    def test_gradient_flow_single_subject(self, encoder):
        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(4, n_vox, requires_grad=True)
        out = encoder(x, subject_ids=0)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_per_subject_gradient_isolation(self, encoder):
        """Only the active subject's projections receive gradients."""
        encoder.zero_grad()
        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(4, n_vox)
        out = encoder(x, subject_ids=0)
        out.sum().backward()

        s01_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.subject_projections["subj01"].parameters()
        )
        s02_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.subject_projections["subj02"].parameters()
        )
        assert s01_has_grad, "subj01 projections should have gradients"
        assert not s02_has_grad, "subj02 projections should NOT have gradients"

    def test_subject_embedding_shape(self, encoder):
        assert encoder.subject_embedding.weight.shape == (2, 64)


class TestMultiSubjectWithDecoder:
    def test_vmf_decoder_integration(self):
        encoder = MultiSubjectROITransformer(
            subject_roi_dims=SUBJECT_ROI_DIMS,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        decoder = VonMisesFisherDecoder(
            input_dim=64,
            output_dim=128,
            hidden_dims=[64],
            kappa_mode="softplus",
        )

        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(4, n_vox)
        h = encoder(x, subject_ids=0)
        mu, kappa = decoder(h)

        assert mu.shape == (4, 128)
        assert kappa.shape == (4, 1)
        assert (kappa > 0).all()
        norms = mu.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_dcf_decoder_integration(self):
        encoder = MultiSubjectROITransformer(
            subject_roi_dims=SUBJECT_ROI_DIMS,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        decoder = ROIDCFDecoder(
            d_model=64,
            output_dim=128,
            n_rois=len(ROI_NAMES),
            shared=True,
            kappa_mode="softplus",
        )

        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(4, n_vox)
        enc_out = encoder(x, subject_ids=0, return_roi_tokens=True)
        mu_fused, kappa_c, per_mus, per_kappas, delta = decoder(
            enc_out.roi_tokens, enc_out.cls_to_roi_alpha, return_per_roi=True,
        )

        assert mu_fused.shape == (4, 128)
        assert kappa_c.shape == (4, 1)
        assert per_mus.shape == (4, len(ROI_NAMES), 128)
        assert per_kappas.shape == (4, len(ROI_NAMES), 1)
        assert delta.shape == (4, 1)
        assert (per_kappas > 0).all()

    def test_end_to_end_gradient_flow(self):
        encoder = MultiSubjectROITransformer(
            subject_roi_dims=SUBJECT_ROI_DIMS,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        decoder = VonMisesFisherDecoder(
            input_dim=64,
            output_dim=128,
            hidden_dims=[64],
            kappa_mode="softplus",
        )

        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        x = torch.randn(4, n_vox, requires_grad=True)
        h = encoder(x, subject_ids=0)
        mu, kappa = decoder(h)

        target = F.normalize(torch.randn(4, 128), dim=-1)
        loss = 1 - (mu * target).sum(dim=-1).mean() + kappa.mean() * 0.01
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestKappaActivationModes:
    """Verify backward compat: both kappa modes work in multi-subject pipeline."""

    def test_bounded_sigmoid_mode(self):
        encoder = MultiSubjectROITransformer(
            subject_roi_dims=SUBJECT_ROI_DIMS,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        decoder = VonMisesFisherDecoder(
            input_dim=64, output_dim=128, hidden_dims=[64],
            kappa_mode="bounded_sigmoid", kappa_min=1.0, kappa_max=50.0,
        )
        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        h = encoder(torch.randn(2, n_vox), subject_ids=0)
        _, kappa = decoder(h)
        assert (kappa >= 1.0).all()
        assert (kappa <= 50.0).all()

    def test_softplus_mode(self):
        encoder = MultiSubjectROITransformer(
            subject_roi_dims=SUBJECT_ROI_DIMS,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        decoder = VonMisesFisherDecoder(
            input_dim=64, output_dim=128, hidden_dims=[64],
            kappa_mode="softplus",
        )
        n_vox = sum(SUBJECT_ROI_DIMS["subj01"].values())
        h = encoder(torch.randn(2, n_vox), subject_ids=0)
        _, kappa = decoder(h)
        assert (kappa >= 1.0).all()
