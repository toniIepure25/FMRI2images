"""S2.0 preprocessing tests: train-only fit, zero-variance, leakage-safety."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_preprocessing as pp


def test_zscore_train_only_stats():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(5.0, 3.0, size=(40, 6))
    s = pp.FittedScaler.fit(Xtr, pp.PREPROC_ZSCORE_TRAIN_ONLY)
    Z = s.transform(Xtr)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-9)
    assert s.with_scaling and s.n_zero_var == 0


def test_historical_d0_is_center_only():
    rng = np.random.default_rng(1)
    Xtr = rng.normal(2.0, 4.0, size=(30, 5))
    s = pp.FittedScaler.fit(Xtr, pp.PREPROC_HISTORICAL_D0)
    Z = s.transform(Xtr)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    # scale is all ones (no scaling) -> std preserved
    assert np.allclose(Z.std(axis=0), Xtr.std(axis=0), atol=1e-9)
    assert not s.with_scaling and np.allclose(s.scale_, 1.0)


def test_zero_variance_recorded_not_divided():
    Xtr = np.random.default_rng(2).normal(size=(20, 4))
    Xtr[:, 1] = 7.0  # constant column -> zero variance
    s = pp.FittedScaler.fit(Xtr, pp.PREPROC_ZSCORE_TRAIN_ONLY)
    assert s.n_zero_var == 1 and s.zero_var_idx.tolist() == [1]
    Z = s.transform(Xtr)
    assert np.isfinite(Z).all()          # never divided by zero
    assert np.allclose(Z[:, 1], 0.0)     # constant column centers to 0, scale=1


def test_val_test_use_frozen_train_stats_no_leakage():
    rng = np.random.default_rng(3)
    Xtr = rng.normal(0, 1, size=(40, 6))
    # val/test on a DIFFERENT distribution -> transformed with TRAIN stats only
    Xval = rng.normal(10, 5, size=(20, 6))
    s = pp.FittedScaler.fit(Xtr, pp.PREPROC_ZSCORE_TRAIN_ONLY)
    Zval = s.transform(Xval)
    # manual application of train stats must match exactly (proves frozen stats)
    manual = (Xval - Xtr.mean(axis=0)) / np.where(Xtr.std(axis=0) < 1e-12, 1.0, Xtr.std(axis=0))
    assert np.allclose(Zval, manual)
    # and val stats do NOT become ~0/1 (they would if it refit on val)
    assert not np.allclose(Zval.mean(axis=0), 0, atol=1e-6)


def test_transform_dim_mismatch_raises():
    s = pp.FittedScaler.fit(np.zeros((10, 5)) + np.arange(5), pp.PREPROC_ZSCORE_TRAIN_ONLY)
    with pytest.raises(ValueError, match="feature dim"):
        s.transform(np.zeros((3, 4)))


def test_unresolved_policy_raises():
    with pytest.raises(ValueError, match="unresolved preprocessing policy"):
        pp.FittedScaler.fit(np.zeros((5, 3)), "SOMETHING_ELSE")
