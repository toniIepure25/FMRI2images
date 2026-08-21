"""S2.4H data-free tests: immutability, no-rescue, descriptive utilities."""
from __future__ import annotations

import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_4h_audit as au


def test_frozen_primary_is_the_committed_value():
    assert au.S2_3_PRIMARY_IMMUTABLE["n_permutations"] == 5040
    assert au.S2_3_PRIMARY_IMMUTABLE["n_null_ge_observed"] == 276
    assert abs(au.S2_3_PRIMARY_IMMUTABLE["p_exact_one_sided"] - 276 / 5040) < 1e-15
    assert au.S2_3_PRIMARY_IMMUTABLE["classification"] == "H_A_CROSS_PARTICIPANT_PARTIAL"


def test_immutability_guard_accepts_frozen_and_rejects_altered():
    good = {"n_permutations": 5040, "n_null_ge_observed": 276, "p_exact_one_sided": 276 / 5040}
    au.assert_primary_immutable(good)  # no raise
    for bad in ({"n_permutations": 5040, "n_null_ge_observed": 275, "p_exact_one_sided": 275 / 5040},
                {"n_permutations": 5040, "n_null_ge_observed": 276, "p_exact_one_sided": 0.049}):
        with pytest.raises(au.PrimaryRescueError):
            au.assert_primary_immutable(bad)


def test_no_recompute_primary_api_exists():
    # the module must NOT expose any function that recomputes a primary p-value
    forbidden = [n for n in dir(au) if "permut" in n.lower() or "pvalue" in n.lower() or "recompute" in n.lower()]
    assert forbidden == []


def test_sign_counts_positive_is_b1_worse():
    c = au.sign_counts([0.1, -0.2, 0.3, 0.0], [0.1, 0.2, -0.3, 0.4])
    assert c["n_rel_positive"] == 2 and c["n_perf_positive"] == 3
    assert c["n_concordant_positive"] == 1 and c["n_roi"] == 4


def test_mad_and_median():
    assert au.median([1, 2, 3, 4]) == 2.5
    assert au.mad([1, 1, 1, 1]) == 0.0
    assert au.mad([0, 2, 4]) == 2.0   # median 2; abs devs 2,0,2 -> median 2


def test_rank_influence_shapes():
    S_rel = {"a": 0.1, "b": 0.3, "c": 0.2}
    S_perf = {"a": 0.3, "b": 0.1, "c": 0.2}
    ri = au.rank_influence(S_rel, S_perf)
    assert ri["a"]["rank_S_rel"] == 1 and ri["a"]["rank_S_perf"] == 3
    assert ri["a"]["abs_rank_diff"] == 2


def test_subj07_cannot_be_excluded_no_filter_helper():
    # there must be no exclusion/outlier-drop helper in the module
    assert not any("exclud" in n.lower() or "outlier" in n.lower() or "drop" in n.lower() for n in dir(au))
