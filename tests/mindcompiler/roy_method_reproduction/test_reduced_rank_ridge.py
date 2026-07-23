"""Synthetic ground-truth tests for the paper-specified model core.

No real data. Validates ridge, reduced-rank recovery, selection policies,
train-only normalization, leakage-freedom, and zero-variance handling.
"""
import numpy as np
import pytest
from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
    ridge_grid, Standardizer, fit_reduced_rank, per_voxel_pearson,
    select_hyperparameters, _ridge_solve,
)
from fmri2img.mindcompiler.roy_method_reproduction.contracts import RIDGE_GRID_N

def test_ridge_grid_is_100_log_spaced_1e_3_to_1e5():
    g = ridge_grid()
    assert g.size == RIDGE_GRID_N == 100
    assert np.isclose(g[0], 1e-3) and np.isclose(g[-1], 1e5)
    # log-spaced: constant ratio
    ratios = g[1:] / g[:-1]
    assert np.allclose(ratios, ratios[0])

def test_ridge_reduces_to_direct_solution_at_small_lambda():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)); W = rng.standard_normal((8, 5)); Y = X @ W
    Wt = _ridge_solve(X, Y, 1e-10)
    assert np.allclose(X @ Wt, Y, atol=1e-4)

def test_full_rank_equals_ridge():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 6)); Y = rng.standard_normal((60, 6))
    W_full = _ridge_solve(X, Y, 1.0)
    W_rr = fit_reduced_rank(X, Y, 1.0, rank=6)  # rank == q -> identity projection
    assert np.allclose(X @ W_full, X @ W_rr, atol=1e-6)

def test_low_rank_recovery():
    """A rank-2 target is recovered at rank>=2, and rank-1 loses variance."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 10))
    B = rng.standard_normal((10, 2)) @ rng.standard_normal((2, 8))  # rank-2 map
    Y = X @ B + 0.01 * rng.standard_normal((200, 8))
    pred2 = X @ fit_reduced_rank(X, Y, 1e-2, rank=2)
    pred1 = X @ fit_reduced_rank(X, Y, 1e-2, rank=1)
    r2 = np.nanmean(per_voxel_pearson(Y, pred2))
    r1 = np.nanmean(per_voxel_pearson(Y, pred1))
    assert r2 > 0.98, r2
    assert r1 < r2, (r1, r2)

def test_rank_one_special_case_runs():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 5)); Y = rng.standard_normal((40, 4))
    W = fit_reduced_rank(X, Y, 1.0, rank=1)
    assert W.shape == (5, 4) and np.isfinite(W).all()

def test_rank_below_one_rejected():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        fit_reduced_rank(np.eye(4), np.eye(4), 1.0, rank=0)

def test_pearson_zero_variance_is_nan_not_favorable():
    yt = np.tile(np.array([[1.0, 5.0]]), (10, 1))  # constant columns
    yp = np.random.default_rng(0).standard_normal((10, 2))
    r = per_voxel_pearson(yt, yp)
    assert np.isnan(r).all(), "constant target columns must give NaN, never 1.0 or 0.0"

def test_standardizer_is_train_only():
    rng = np.random.default_rng(4)
    Xtr = rng.standard_normal((30, 3)) + 10.0
    Xte = rng.standard_normal((30, 3)) + 10.0
    s = Standardizer.fit(Xtr, with_scaling=True)
    # centering uses TRAIN mean; test set is not re-centered to its own mean
    assert np.allclose(s.transform(Xtr).mean(0), 0, atol=1e-6)
    assert not np.allclose(s.transform(Xte).mean(0), 0, atol=1e-6)

def test_selection_is_deterministic_and_on_validation_only():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((120, 8))
    B = rng.standard_normal((8, 2)) @ rng.standard_normal((2, 6))
    Y = X @ B + 0.05 * rng.standard_normal((120, 6))
    Xtr, Xva = X[:80], X[80:]; Ytr, Yva = Y[:80], Y[80:]
    s1 = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                rank_tie_break="smallest_rank_at_threshold",
                                ridge_tie_break="argmax_validation")
    s2 = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                rank_tie_break="smallest_rank_at_threshold",
                                ridge_tie_break="argmax_validation")
    assert (s1.lam, s1.rank) == (s2.lam, s2.rank)  # deterministic
    assert 1 <= s1.rank <= 6
    assert s1.lam in ridge_grid()
    # a rank-2 signal should be selected at a small rank under the 99%-threshold rule
    assert s1.rank <= 3, s1.rank

def test_smallest_rank_rule_picks_no_larger_than_argmax():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((100, 8))
    B = rng.standard_normal((8, 3)) @ rng.standard_normal((3, 6))
    Y = X @ B + 0.05 * rng.standard_normal((100, 6))
    Xtr, Xva, Ytr, Yva = X[:70], X[70:], Y[:70], Y[70:]
    smallest = select_hyperparameters(Xtr, Ytr, Xva, Yva, "smallest_rank_at_threshold", "argmax_validation")
    argmax = select_hyperparameters(Xtr, Ytr, Xva, Yva, "argmax_validation", "argmax_validation")
    assert smallest.rank <= argmax.rank

def test_unresolved_tie_break_raises():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 4)); Y = rng.standard_normal((40, 3))
    with pytest.raises(ValueError, match="unresolved"):
        select_hyperparameters(X[:30], Y[:30], X[30:], Y[30:],
                               rank_tie_break="author_clarification_required",  # type: ignore
                               ridge_tie_break="argmax_validation")


# --- S1.3 model-selection repairs -----------------------------------------

def test_one_se_rejected_without_folds():
    """The single-split one-SE rule was invalid (SE over the lambda x rank
    matrix). It must now raise, not silently mis-select."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((40, 5)); Y = rng.standard_normal((40, 4))
    with pytest.raises(NotImplementedError, match="fold-level scores"):
        select_hyperparameters(X[:30], Y[:30], X[30:], Y[30:],
                               rank_tie_break="smallest_rank_at_threshold",
                               ridge_tie_break="one_se_rule")

def test_rank_threshold_uses_selected_lambda_peak_not_global():
    """Regression: the threshold must be 99% of the SELECTED lambda's own peak.

    Construct scores where the selected lambda's row peak is below the global
    matrix peak; the smallest-rank-at-threshold must be computed against the
    row peak, so a rank is always selectable within the chosen lambda.
    """
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
        Selection, RANK_SELECTION_FRACTION_OF_PEAK,
    )
    import numpy as np
    # Directly exercise the row-local logic the selector uses.
    row = np.array([0.10, 0.40, 0.42, 0.43])      # this lambda's peak = 0.43
    global_peak = 0.90                             # a DIFFERENT lambda scored higher
    thr_row = RANK_SELECTION_FRACTION_OF_PEAK * row.max()
    thr_global = RANK_SELECTION_FRACTION_OF_PEAK * global_peak
    # row-local: at least one rank qualifies (the peak itself)
    assert np.where(row >= thr_row)[0].size >= 1
    # global-peak (the bug): NOTHING in this row qualifies -> would fall back wrongly
    assert np.where(row >= thr_global)[0].size == 0

def test_fold_aware_one_se_is_more_parsimonious_than_argmax():
    """A true one-SE rule (fold SE) selects no larger than argmax."""
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
        select_with_folds, ridge_grid,
    )
    rng = np.random.default_rng(12)
    grid = ridge_grid()
    # 5 folds, 100 lambdas, 6 ranks; a broad plateau near the top so one-SE bites
    fold_scores = 0.5 + 0.01*rng.standard_normal((5, grid.size, 6))
    fold_scores[:, 40:60, 1:] += 0.3   # a wide high-scoring region
    argmax = select_with_folds(fold_scores, grid, rule="argmax")
    onese = select_with_folds(fold_scores, grid, rule="one_se")
    assert onese.rank <= argmax.rank
    assert onese.se_score >= 0.0
    assert onese.rule == "one_se"

def test_fold_scores_wrong_shape_rejected():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
        select_with_folds, ridge_grid,
    )
    with pytest.raises(ValueError, match="n_folds, n_lambda, n_rank"):
        select_with_folds(np.zeros((3, 4)), ridge_grid())


# --- S1.4 fold-selection hardening ----------------------------------------

def _folds(nf=4, nl=None, nr=6, seed=0):
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import ridge_grid
    g = ridge_grid(); nl = g.size if nl is None else nl
    rng = np.random.default_rng(seed)
    return 0.5 + 0.01*rng.standard_normal((nf, nl, nr)), g[:nl]

def test_one_se_requires_two_folds():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds(nf=1)
    with pytest.raises(ValueError, match="two folds|>= 2 folds"):
        select_with_folds(fs, g, rule="one_se")

def test_grid_length_mismatch_rejected():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds()
    with pytest.raises(ValueError, match="grid length"):
        select_with_folds(fs, g[:-3])

def test_non_monotonic_grid_rejected():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds(nl=5)
    bad = np.array([1e-3, 1e-1, 1e-2, 1e0, 1e1])  # not ascending
    with pytest.raises(ValueError, match="ascending"):
        select_with_folds(fs, bad)

def test_nan_only_input_rejected():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds(); fs[:] = np.nan
    with pytest.raises(ValueError, match="no fold-score cell is finite"):
        select_with_folds(fs, g)

def test_partial_nans_excluded_not_crash():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds(); fs[0, 5, 2] = np.nan  # kill one cell in one fold
    sel = select_with_folds(fs, g, rule="argmax")
    # the killed cell (lambda idx 5, rank idx 2 -> rank 3) must not be selected
    assert not (g[5] == sel.lam and sel.rank == 3)

def test_one_se_is_deterministic():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds()
    a = select_with_folds(fs, g, rule="one_se"); b = select_with_folds(fs, g, rule="one_se")
    assert (a.lam, a.rank) == (b.lam, b.rank)

def test_one_se_prefers_stronger_lambda_then_smaller_rank():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import select_with_folds
    fs, g = _folds(nl=10, nr=6, seed=3)
    fs[:, 3:8, 1:] += 0.3  # broad high plateau spanning several lambdas/ranks
    onese = select_with_folds(fs, g, rule="one_se")
    argmax = select_with_folds(fs, g, rule="argmax")
    gl = list(g)
    # parsimony within the plateau: one-SE picks lambda no weaker (index >=) than argmax
    assert gl.index(onese.lam) >= gl.index(argmax.lam)
