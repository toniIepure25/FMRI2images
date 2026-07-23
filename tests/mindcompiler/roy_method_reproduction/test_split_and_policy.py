"""Tests for the paper-specified split protocol and the no-silent-ambiguity contract."""
import numpy as np
import pytest
from fmri2img.mindcompiler.roy_method_reproduction.contracts import (
    ReproductionPolicy, DENOISING_VARIANTS, RIDGE_GRID_N, VOXEL_SNR_PERCENTILE,
)
from fmri2img.mindcompiler.roy_method_reproduction.split_protocol import (
    build_repeat_level_split, split_is_valid,
)

def _identity(n_id=12, reps=8):
    return np.repeat(np.arange(n_id), reps)

def test_split_is_4_2_2_and_all_identities_present():
    ident = _identity()
    s = build_repeat_level_split(ident, seed=0)
    v = split_is_valid(s, ident)
    assert all(v.values()), v
    # 4/2/2 of 8 per identity
    assert s.train.size == 12*4 and s.val.size == 12*2 and s.test.size == 12*2

def test_split_is_deterministic_under_seed():
    ident = _identity()
    a = build_repeat_level_split(ident, seed=7)
    b = build_repeat_level_split(ident, seed=7)
    assert np.array_equal(a.train, b.train) and np.array_equal(a.test, b.test)

def test_split_rejects_wrong_repeat_count():
    ident = np.repeat(np.arange(5), 7)  # 7 repeats, not 8
    with pytest.raises(ValueError, match="expected 8"):
        build_repeat_level_split(ident, seed=0)

def test_policy_defaults_refuse_to_guess():
    p = ReproductionPolicy()
    assert "denoising" in p.unresolved()
    with pytest.raises(ValueError, match="unresolved ambiguities"):
        p.require_resolved()

def test_policy_requires_every_ambiguity_set():
    # Resolve all but one -> still refuses.
    p = ReproductionPolicy(denoising="strict_out_of_fold", pairing="single_seed",
                           rank_tie_break="smallest_rank_at_threshold",
                           ridge_tie_break="argmax_validation",
                           n_pairing_realizations=0)  # still unset
    assert "n_pairing_realizations" in p.unresolved()
    with pytest.raises(ValueError):
        p.require_resolved()

def test_fully_resolved_policy_passes():
    p = ReproductionPolicy(denoising="strict_out_of_fold", pairing="average_over_realizations",
                           rank_tie_break="smallest_rank_at_threshold",
                           ridge_tie_break="one_se_rule", n_pairing_realizations=50)
    p.require_resolved()  # must not raise
    assert p.center_on_train_only is True  # leakage-safe default

def test_specified_constants_match_paper():
    assert VOXEL_SNR_PERCENTILE == 98.0
    assert RIDGE_GRID_N == 100
    assert set(DENOISING_VARIANTS) == {"no_denoising","strict_out_of_fold","fold_fit_all_outputs"}
