"""S2.6R geometry unit certifications (data-free): d99 rule + alignment ground truth + bases."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import roy_geometry as rg


# ------------------------------------------------------------------ dimensionality (Part 33)
def test_d99_known_first_crossing():
    # peak = 1.00, thr = 0.99; first rank reaching >=0.99 is rank 4
    d, why = rg.d99_from_curve([0.20, 0.50, 0.80, 1.00, 0.995])
    assert d == 4 and why == "OK"


def test_d99_plateau_picks_smallest():
    # peak 1.0 at rank 3; ranks 3,4,5 all >= 0.99 -> choose smallest (3)
    d, why = rg.d99_from_curve([0.5, 0.9, 1.0, 1.0, 1.0])
    assert d == 3 and why == "OK"


def test_d99_early_crossing_before_peak():
    # rank 2 already >= 0.99*peak(=0.99) -> first crossing is rank 2
    d, _ = rg.d99_from_curve([0.10, 0.995, 0.50, 1.00])
    assert d == 2


def test_d99_negative_peak_not_evaluable():
    d, why = rg.d99_from_curve([-0.3, -0.2, -0.5])
    assert d is None and why == rg.NONPOSITIVE_PEAK_TAG


def test_d99_zero_peak_not_evaluable():
    d, why = rg.d99_from_curve([0.0, 0.0, 0.0])
    assert d is None and why == rg.NONPOSITIVE_PEAK_TAG


def test_d99_nan_handling():
    d, why = rg.d99_from_curve([np.nan, np.nan])
    assert d is None and why == rg.NO_FINITE_TAG
    # NaN below peak is skipped; first finite crossing chosen
    d2, _ = rg.d99_from_curve([0.2, np.nan, 1.0])
    assert d2 == 3


def test_fold_average_dimension_not_rounded():
    assert float(np.mean([2, 3, 3, 4])) == 3.0
    assert float(np.mean([2, 3, 3, 3])) == pytest.approx(2.75)  # NOT rounded to integer


# --------------------------------------------------------------- alignment ground truth (Part 34)
def _ortho(n, k, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, k)))
    return Q[:, :k]


def test_alignment_identical_subspaces_is_one():
    V = _ortho(20, 5, 0)
    X = np.random.default_rng(1).standard_normal((40, 20))
    out = rg.alignment_ratio(X, V, V.copy(), d=5)
    assert out["a_g"] == pytest.approx(1.0, abs=1e-10)


def test_alignment_orthogonal_subspaces_is_zero():
    # visual activity that lives IN the visual subspace -> orthogonal imagery subspace
    # captures ~no variance (the real-pipeline sanity: X_vis concentrated in span(Vvis)).
    Q = _ortho(20, 10, 2)
    Vvis, Vimg = Q[:, :5], Q[:, 5:10]          # mutually orthogonal
    Z = np.random.default_rng(3).standard_normal((40, 5))
    X = Z @ Vvis.T                             # X lies entirely within span(Vvis)
    out = rg.alignment_ratio(X, Vvis, Vimg, d=5)
    assert out["a_g"] == pytest.approx(0.0, abs=1e-9)


def test_alignment_sign_flip_invariant():
    V = _ortho(16, 4, 4)
    Vimg = V.copy(); Vimg[:, 2] *= -1.0
    X = np.random.default_rng(5).standard_normal((30, 16))
    a = rg.alignment_ratio(X, V, V, d=4)["a_g"]
    b = rg.alignment_ratio(X, V, Vimg, d=4)["a_g"]
    assert a == pytest.approx(b, abs=1e-12)


def test_alignment_within_subspace_rotation_invariant():
    V = _ortho(16, 4, 6)
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((4, 4)))
    Vrot = V @ Q                                # same span, rotated basis
    X = np.random.default_rng(8).standard_normal((30, 16))
    a = rg.alignment_ratio(X, V, V, d=4)["a_g"]
    b = rg.alignment_ratio(X, V, Vrot, d=4)["a_g"]
    assert a == pytest.approx(b, abs=1e-10)


def test_alignment_intermediate_rotation_known_value():
    # d=1: Vvis = e0, Vimg = cos t e0 + sin t e1. TV_img/TV_vis has a known closed form.
    rng = np.random.default_rng(9)
    n, p = 200, 4
    X = rng.standard_normal((n, p))
    t = 0.6
    Vvis = np.zeros((p, 1)); Vvis[0, 0] = 1.0
    Vimg = np.zeros((p, 1)); Vimg[0, 0] = np.cos(t); Vimg[1, 0] = np.sin(t)
    out = rg.alignment_ratio(X, Vvis, Vimg, d=1)
    # closed form with sample covariance C: (u^T C u)/(e0^T C e0)
    C = np.cov(X, rowvar=False, ddof=1)
    u = Vimg[:, 0]
    expected = (u @ C @ u) / (Vvis[:, 0] @ C @ Vvis[:, 0])
    assert out["a_g"] == pytest.approx(expected, rel=1e-9)


def test_alignment_no_clipping_above_one():
    # imagery dim aligned with a HIGHER-variance data direction than the visual dim -> a_g > 1
    rng = np.random.default_rng(10)
    base = rng.standard_normal((300, 3))
    base[:, 1] *= 5.0                            # axis 1 much higher variance than axis 0
    Vvis = np.array([[1.0], [0.0], [0.0]])
    Vimg = np.array([[0.0], [1.0], [0.0]])
    out = rg.alignment_ratio(base, Vvis, Vimg, d=1)
    assert out["a_g"] > 1.0                      # honestly reported, not clipped


def test_total_variance_projector_equivalence():
    # sum var(X @ V) == sum var(X @ V @ V.T) for orthonormal V (Part 18)
    V = _ortho(18, 6, 11)
    X = np.random.default_rng(12).standard_normal((50, 18))
    lhs = float(np.var(X @ V, axis=0, ddof=1).sum())
    rhs = rg.subspace_total_variance(X, V)
    assert lhs == pytest.approx(rhs, rel=1e-10)


# ------------------------------------------------------------------- basis extraction (Part 35)
def test_basis_orthonormal_and_matches_rrr():
    from fmri2img.mindcompiler.roy_method_reproduction import roy_public_rrr as rr
    rng = np.random.default_rng(13)
    Xtr = rng.standard_normal((24, 10))
    W = rng.standard_normal((10, 15))
    V = rg.extract_output_basis(Xtr, W)
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8)
    # W @ V_r V_r^T must equal the established reduced-rank construction
    r = 4
    W_rrr_ref = rr._rrr(W, Xtr, r)
    W_rrr_basis = W @ (V[:, :r] @ V[:, :r].T)
    assert np.allclose(W_rrr_ref, W_rrr_basis, atol=1e-9)


def test_projector_hash_sign_invariant():
    V = _ortho(12, 5, 14)
    Vflip = V.copy(); Vflip[:, 1] *= -1; Vflip[:, 3] *= -1
    assert rg.projector_hash(V[:, :3]) == rg.projector_hash(Vflip[:, :3])


def test_projector_hash_deterministic():
    V = _ortho(12, 5, 15)
    assert rg.projector_hash(V[:, :4]) == rg.projector_hash(V[:, :4])


def test_alignment_null_deterministic_and_bounded():
    V = _ortho(20, 12, 16)
    X = np.random.default_rng(17).standard_normal((40, 20))
    tv = rg.subspace_total_variance(X, V[:, :4])
    a = rg.alignment_null(X, V, tv, 4, "subj02", "V1", "fold_0", n=50)
    b = rg.alignment_null(X, V, tv, 4, "subj02", "V1", "fold_0", n=50)
    assert a == b and len(a) == 50 and all(x >= 0 for x in a)
