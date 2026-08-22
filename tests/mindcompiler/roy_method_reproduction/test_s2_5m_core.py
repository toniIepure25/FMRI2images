"""S2.5M core tests: per-target ridge (Lambda), reduced-rank, Roy pairings."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp
from fmri2img.mindcompiler.roy_method_reproduction import roy_public_rrr as rr


# --- per-target lambda -------------------------------------------------------

def test_per_target_lambda_differs_when_snr_differs():
    rng = np.random.default_rng(0)
    n_tr, n_va, p = 60, 40, 6
    Xtr = rng.standard_normal((n_tr, p)); Xva = rng.standard_normal((n_va, p))
    B = rng.standard_normal((p, 3))
    # target 0: clean (small noise -> small lambda); target 2: very noisy (large lambda)
    def mk(X, nrows):
        Y = X @ B
        Y[:, 0] += 0.05 * rng.standard_normal(nrows)
        Y[:, 2] += 8.0 * rng.standard_normal(nrows)
        return Y
    lam = rr.select_per_target_lambda(Xtr, mk(Xtr, n_tr), Xva, mk(Xva, n_va))
    assert lam.shape == (3,)
    # targets with different noise structure select DIFFERENT lambdas (per-target, not scalar)
    assert not np.all(lam == lam[0])


def test_lambda_all_in_public_grid():
    rng = np.random.default_rng(1)
    Xtr = rng.standard_normal((40, 5)); Ytr = rng.standard_normal((40, 4))
    Xva = rng.standard_normal((20, 5)); Yva = rng.standard_normal((20, 4))
    lam = rr.select_per_target_lambda(Xtr, Ytr, Xva, Yva)
    grid = rr.ridge_grid()
    assert grid.size == 100 and abs(grid[0] - 1e-3) < 1e-9 and abs(grid[-1] - 1e5) < 1e-3
    assert all(v in set(grid) for v in lam)


def test_no_scalar_lambda_fallback_regression():
    # If the code collapsed to one lambda, this SNR-varied case would return identical lambdas.
    rng = np.random.default_rng(2)
    Xtr = rng.standard_normal((80, 4)); Xva = rng.standard_normal((50, 4))
    B = rng.standard_normal((4, 5))
    noise = np.array([0.02, 0.5, 3.0, 10.0, 0.1])
    Ytr = Xtr @ B + noise * rng.standard_normal((80, 5))
    Yva = Xva @ B + noise * rng.standard_normal((50, 5))
    lam = rr.select_per_target_lambda(Xtr, Ytr, Xva, Yva)
    assert len(np.unique(lam)) >= 3            # genuinely per-target


def test_fit_W_lambda_uses_each_targets_lambda():
    rng = np.random.default_rng(3)
    Xtr = rng.standard_normal((30, 4)); Ytr = rng.standard_normal((30, 3))
    lambdas = np.array([1e-3, 1e2, 1e5])
    W = rr.fit_W_lambda(Xtr, Ytr, lambdas)
    # column i must equal the direct ridge solve at lambda_i
    for i, lam in enumerate(lambdas):
        Wl = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(4), Xtr.T @ Ytr)
        assert np.allclose(W[:, i], Wl[:, i])


# --- reduced rank ------------------------------------------------------------

def test_reduced_rank_matches_direct_construction():
    rng = np.random.default_rng(4)
    Xtr = rng.standard_normal((40, 6)); Ytr = rng.standard_normal((40, 6))
    W_lambda = rr.fit_W_lambda(Xtr, Ytr, np.full(6, 10.0))
    for rank in (1, 3, 6):
        W_rrr = rr._rrr(W_lambda, Xtr, rank)
        _, _, Vt = np.linalg.svd(Xtr @ W_lambda, full_matrices=False)
        P = Vt[:rank].T @ Vt[:rank]
        assert np.allclose(W_rrr, W_lambda @ P)


def test_operational_rank_within_cap_and_validation_selected():
    rng = np.random.default_rng(5)
    Xtr = rng.standard_normal((40, 6)); Ytr = rng.standard_normal((40, 6))
    Xva = rng.standard_normal((20, 6)); Yva = rng.standard_normal((20, 6))
    fit = rr.fit_select(Xtr, Ytr, Xva, Yva, rank_max=4)
    assert 1 <= fit.r_model <= 4
    assert fit.lambdas.shape == (6,)


# --- Roy pairings ------------------------------------------------------------

def test_v2v_derangement_within_identity_and_deterministic():
    seed = rp.child_seed(rp.root_digest(0), "subj02", "fold_0", "train", "A:L", "vis2vis")
    s, t = rp.vis2vis_pairs([10, 11, 12, 13], seed)
    assert s == [10, 11, 12, 13] and sorted(t) == [10, 11, 12, 13]
    assert all(a != b for a, b in zip(s, t))                 # no fixed point
    s2, t2 = rp.vis2vis_pairs([10, 11, 12, 13], seed)
    assert t == t2                                            # deterministic replay


def test_v2v_two_repeat_split_is_swap():
    seed = rp.child_seed(rp.root_digest(0), "subj02", "fold_0", "val", "A:L", "vis2vis")
    s, t = rp.vis2vis_pairs([4, 5], seed)
    assert (s, t) == ([4, 5], [5, 4])


def test_v2i_random_within_identity_bijective_deterministic():
    seed = rp.child_seed(rp.root_digest(0), "subj02", "fold_0", "train", "A:L", "vis2img")
    v, im = rp.vis2img_pairs([0, 1, 2, 3], [100, 101, 102, 103], seed)
    assert v == [0, 1, 2, 3] and sorted(im) == [100, 101, 102, 103]  # bijective
    v2, im2 = rp.vis2img_pairs([0, 1, 2, 3], [100, 101, 102, 103], seed)
    assert im == im2


def test_seeds_are_roi_and_beta_independent():
    # child_seed signature has no ROI/beta -> identical pairings across ROIs/betas
    a = rp.child_seed(rp.root_digest(0), "subj02", "fold_0", "train", "A:L", "vis2img")
    b = rp.child_seed(rp.root_digest(0), "subj02", "fold_0", "train", "A:L", "vis2img")
    assert a == b
    # different realization -> different seed
    assert rp.root_digest(0) != rp.root_digest(1)


def test_v2v_requires_two_rows():
    with pytest.raises(ValueError, match=">= 2 rows"):
        rp.vis2vis_pairs([7], 123)
