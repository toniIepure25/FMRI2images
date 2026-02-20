"""Tests for vMF mixture posterior sampling."""

import pytest
import torch
import torch.nn.functional as F

from fmri2img.inference.vmf_mixture import (
    _sample_vmf_wood,
    sample_vmf_mixture,
    compute_mixture_energy_score,
    select_best_from_mixture,
    VMFMixtureSamples,
)


@pytest.fixture
def roi_predictions():
    """Synthetic per-ROI vMF predictions."""
    B, R, D = 4, 5, 64
    torch.manual_seed(42)
    mus = F.normalize(torch.randn(B, R, D), p=2, dim=-1)
    kappas = torch.rand(B, R, 1) * 50 + 1  # [1, 51]
    alphas = F.softmax(torch.randn(B, R), dim=-1)
    return mus, kappas, alphas


class TestVMFSampling:
    def test_sample_shape(self):
        B, D = 3, 32
        mu = F.normalize(torch.randn(B, D), p=2, dim=-1)
        kappa = torch.ones(B) * 10.0
        samples = _sample_vmf_wood(mu, kappa, num_samples=5)
        assert samples.shape == (B, 5, D)

    def test_samples_are_unit_norm(self):
        B, D = 4, 64
        mu = F.normalize(torch.randn(B, D), p=2, dim=-1)
        kappa = torch.ones(B) * 50.0
        samples = _sample_vmf_wood(mu, kappa, num_samples=10)
        norms = samples.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_high_kappa_concentrates_near_mu(self):
        B, D = 2, 32
        mu = F.normalize(torch.randn(B, D), p=2, dim=-1)
        kappa = torch.ones(B) * 500.0
        samples = _sample_vmf_wood(mu, kappa, num_samples=50)
        cos_sim = (samples * mu.unsqueeze(1)).sum(dim=-1)
        assert cos_sim.mean() > 0.95

    def test_moderate_kappa_concentrates(self):
        """Moderate kappa (50) should concentrate well around mu in D=32."""
        B, D = 4, 32
        mu = F.normalize(torch.randn(B, D), p=2, dim=-1)
        kappa = torch.ones(B) * 50.0
        samples = _sample_vmf_wood(mu, kappa, num_samples=20)
        cos_sim = (samples * mu.unsqueeze(1)).sum(dim=-1)
        assert cos_sim.mean() > 0.8


class TestVMFMixture:
    def test_proportional_sampling(self, roi_predictions):
        mus, kappas, alphas = roi_predictions
        result = sample_vmf_mixture(
            mus, kappas, alphas, num_samples=8, strategy="proportional"
        )
        assert result.samples.shape == (4, 8, 64)
        assert result.component_indices.shape == (4, 8)

    def test_categorical_sampling(self, roi_predictions):
        mus, kappas, alphas = roi_predictions
        result = sample_vmf_mixture(
            mus, kappas, alphas, num_samples=6, strategy="categorical"
        )
        assert result.samples.shape == (4, 6, 64)

    def test_samples_are_unit_norm(self, roi_predictions):
        mus, kappas, alphas = roi_predictions
        result = sample_vmf_mixture(mus, kappas, alphas, num_samples=8)
        norms = result.samples.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_component_indices_valid(self, roi_predictions):
        mus, kappas, alphas = roi_predictions
        R = mus.shape[1]
        result = sample_vmf_mixture(mus, kappas, alphas, num_samples=8)
        assert (result.component_indices >= 0).all()
        assert (result.component_indices < R).all()


class TestMixtureEnergyScore:
    def test_perfect_prediction_low_score(self):
        B, K, D = 2, 10, 32
        target = F.normalize(torch.randn(B, D), p=2, dim=-1)
        # Samples very close to target
        noise = torch.randn(B, K, D) * 0.01
        samples = F.normalize(target.unsqueeze(1) + noise, p=2, dim=-1)
        mixture = VMFMixtureSamples(
            samples=samples,
            component_indices=torch.zeros(B, K, dtype=torch.long),
            mu_consensus=target,
            kappa_consensus=torch.ones(B, 1) * 100,
            delta=torch.zeros(B, 1),
        )
        es = compute_mixture_energy_score(mixture, target)
        assert es.shape == (B,)
        assert (es < 0.3).all()  # Low energy score for good predictions

    def test_random_prediction_high_score(self):
        B, K, D = 2, 10, 32
        target = F.normalize(torch.randn(B, D), p=2, dim=-1)
        samples = F.normalize(torch.randn(B, K, D), p=2, dim=-1)
        mixture = VMFMixtureSamples(
            samples=samples,
            component_indices=torch.zeros(B, K, dtype=torch.long),
            mu_consensus=F.normalize(torch.randn(B, D), p=2, dim=-1),
            kappa_consensus=torch.ones(B, 1),
            delta=torch.ones(B, 1),
        )
        es = compute_mixture_energy_score(mixture, target)
        # Energy score should be higher for random predictions
        assert (es > 0.1).all()


class TestSelectBest:
    def test_closest_to_consensus(self, roi_predictions):
        mus, kappas, alphas = roi_predictions
        result = sample_vmf_mixture(mus, kappas, alphas, num_samples=8)
        best, idx = select_best_from_mixture(result, selection="closest_to_consensus")
        assert best.shape == (4, 64)
        assert idx.shape == (4,)

    def test_highest_likelihood(self, roi_predictions):
        mus, kappas, alphas = roi_predictions
        result = sample_vmf_mixture(mus, kappas, alphas, num_samples=8)
        best, idx = select_best_from_mixture(result, selection="highest_likelihood")
        assert best.shape == (4, 64)
