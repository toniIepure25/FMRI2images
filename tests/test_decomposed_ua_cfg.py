"""Tests for Decomposed Uncertainty-Aware CFG."""

import pytest
import numpy as np

from fmri2img.inference.decomposed_ua_cfg import DecomposedUACFG, DUACFGConfig


@pytest.fixture
def cfg():
    return DecomposedUACFG(DUACFGConfig(
        kappa_q10=5.0,
        kappa_q50=50.0,
        kappa_q90=200.0,
    ))


class TestKappaNormalization:
    def test_min_kappa_maps_to_zero(self, cfg):
        assert cfg.normalize_kappa(5.0) == pytest.approx(0.0, abs=1e-6)

    def test_max_kappa_maps_to_one(self, cfg):
        assert cfg.normalize_kappa(200.0) == pytest.approx(1.0, abs=1e-6)

    def test_mid_kappa(self, cfg):
        kn = cfg.normalize_kappa(102.5)
        assert 0.0 < kn < 1.0

    def test_below_min_clips_to_zero(self, cfg):
        assert cfg.normalize_kappa(1.0) == 0.0

    def test_above_max_clips_to_one(self, cfg):
        assert cfg.normalize_kappa(500.0) == 1.0

    def test_batch_normalization(self, cfg):
        kappas = np.array([5.0, 50.0, 200.0, 1.0, 500.0])
        kn = cfg.normalize_kappa(kappas)
        assert kn[0] == pytest.approx(0.0, abs=1e-6)
        assert kn[2] == pytest.approx(1.0, abs=1e-6)
        assert kn[3] == 0.0
        assert kn[4] == 1.0


class TestGuidanceScale:
    def test_high_kappa_high_guidance(self, cfg):
        w = cfg.guidance_scale(200.0)
        assert w == pytest.approx(cfg.cfg.w_max, abs=0.5)

    def test_low_kappa_low_guidance(self, cfg):
        w = cfg.guidance_scale(5.0)
        assert w == pytest.approx(cfg.cfg.w_min, abs=0.1)

    def test_guidance_in_range(self, cfg):
        for k in [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
            w = cfg.guidance_scale(k)
            assert cfg.cfg.w_min <= w <= cfg.cfg.w_max + 0.01

    def test_delta_damps_guidance(self, cfg):
        w_no_delta = cfg.guidance_scale(100.0)
        w_high_delta = cfg.guidance_scale(100.0, delta=0.9)
        assert w_high_delta < w_no_delta

    def test_batch_guidance(self, cfg):
        kappas = np.array([5.0, 100.0, 200.0])
        deltas = np.array([0.1, 0.5, 0.9])
        w = cfg.guidance_scale(kappas, deltas)
        assert len(w) == 3
        assert w[0] < w[2]  # Low kappa < high kappa


class TestEnsembleSize:
    def test_low_delta_single_sample(self, cfg):
        k = cfg.ensemble_size(0.1)
        assert k == cfg.cfg.k_min

    def test_high_delta_many_samples(self, cfg):
        k = cfg.ensemble_size(0.8)
        assert k == cfg.cfg.k_max

    def test_ensemble_in_range(self, cfg):
        for d in np.linspace(0, 1, 20):
            k = cfg.ensemble_size(float(d))
            assert cfg.cfg.k_min <= k <= cfg.cfg.k_max

    def test_batch_ensemble(self, cfg):
        deltas = np.array([0.0, 0.3, 0.7, 1.0])
        ks = cfg.ensemble_size(deltas)
        assert len(ks) == 4
        assert ks[0] <= ks[-1]


class TestDiffusionSteps:
    def test_high_confidence_few_steps(self, cfg):
        steps = cfg.diffusion_steps(200.0, 0.0)
        assert steps <= cfg.cfg.steps_min + 10

    def test_low_confidence_many_steps(self, cfg):
        steps = cfg.diffusion_steps(5.0, 0.9)
        assert steps >= cfg.cfg.steps_max - 10

    def test_steps_in_range(self, cfg):
        for k in [5.0, 50.0, 200.0]:
            for d in [0.0, 0.5, 1.0]:
                steps = cfg.diffusion_steps(k, d)
                assert cfg.cfg.steps_min <= steps <= cfg.cfg.steps_max


class TestAbstention:
    def test_confident_does_not_abstain(self, cfg):
        assert not cfg.should_abstain(200.0, 0.1)

    def test_uncertain_and_disagreeing_abstains(self, cfg):
        assert cfg.should_abstain(2.0, 0.9)

    def test_uncertain_but_agreeing_does_not_abstain(self, cfg):
        assert not cfg.should_abstain(2.0, 0.1)


class TestGenerationParams:
    def test_returns_all_keys(self, cfg):
        params = cfg.get_generation_params(100.0, 0.3)
        expected_keys = {
            "guidance_scale", "num_steps", "ensemble_k", "abstain",
            "kappa_norm", "delta", "confidence_level",
        }
        assert set(params.keys()) == expected_keys

    def test_batch_params(self, cfg):
        kappas = np.array([5.0, 100.0, 200.0])
        deltas = np.array([0.1, 0.4, 0.8])
        result = cfg.batch_generation_params(kappas, deltas)
        assert "guidance_scales" in result
        assert "ensemble_ks" in result
        assert len(result["guidance_scales"]) == 3
