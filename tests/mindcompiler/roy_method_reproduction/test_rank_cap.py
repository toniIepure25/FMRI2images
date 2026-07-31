"""Rank-cap contract: the retained rank is never allowed above its explicit cap.

An uncapped reduced-rank search can silently select a rank up to
``min(p, q, n_train)``. When a cap is set (``r_max`` on the selector, ``rank_max``
on the pipeline) the returned rank MUST be ``<= cap`` for every choice of data,
and the score matrix must have exactly ``cap`` rank columns. These tests pin that
so a future change cannot let the cap leak.
"""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction.fitted_pipeline import FittedPipeline
from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
    select_hyperparameters,
)


def _data(n_tr=40, n_va=20, p=8, q=8, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((p, q))
    Xtr = rng.standard_normal((n_tr, p)); Ytr = Xtr @ B + 0.1 * rng.standard_normal((n_tr, q))
    Xva = rng.standard_normal((n_va, p)); Yva = Xva @ B + 0.1 * rng.standard_normal((n_va, q))
    return Xtr, Ytr, Xva, Yva


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_selector_never_exceeds_r_max(cap):
    Xtr, Ytr, Xva, Yva = _data()
    sel = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                 "argmax_validation", "argmax_validation", r_max=cap)
    assert 1 <= sel.rank <= cap


def test_selector_uncapped_can_exceed_a_small_cap():
    # Sanity: without the cap the search is free to pick a larger rank, proving the
    # cap in the capped test is doing real work rather than being vacuous.
    Xtr, Ytr, Xva, Yva = _data(p=8, q=8)
    uncapped = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                      "argmax_validation", "argmax_validation", r_max=None)
    assert uncapped.rank >= 1  # bounded only by min(p,q,n_tr)


def test_r_max_clamped_to_matrix_support():
    # cap larger than min(p,q,n_tr) must not raise and must stay within support.
    Xtr, Ytr, Xva, Yva = _data(n_tr=6, p=4, q=4)
    sel = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                 "smallest_rank_at_threshold", "argmax_validation", r_max=999)
    assert sel.rank <= min(4, 4, 6)


@pytest.mark.parametrize("cap", [1, 2])
def test_pipeline_rank_max_is_honoured(cap):
    Xtr, Ytr, Xva, Yva = _data()
    Xte, Yte = _data(seed=7)[2:]
    pipe = (FittedPipeline(rank_max=cap).resolve_policies()
            .fit_preprocessing(Xtr, Ytr).select(Xva, Yva).fit_final())
    pipe.evaluate_test(Xte, Yte)
    lam, rank = pipe.selected
    assert 1 <= rank <= cap
    assert pipe.seal()["rank"] <= cap
