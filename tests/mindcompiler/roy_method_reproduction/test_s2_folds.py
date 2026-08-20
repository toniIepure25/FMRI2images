"""S2.0 four-fold protocol tests (data-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf


def _tt(n_id=12, n_rep=8):
    rows = []
    bi = 0
    for state in ("vision", "imagery"):
        for i in range(n_id):
            for rep in range(n_rep):
                rows.append(dict(identity=f"ID{i:02d}", state=state, repeat=rep, beta_index0=bi))
                bi += 1
    return pd.DataFrame(rows)


def test_four_folds_have_4_2_2_and_all_identities():
    tt = _tt()
    folds = sf.build_four_folds(tt, fold_root_seed=1234)
    assert set(folds) == {"fold_0", "fold_1", "fold_2", "fold_3"}
    ok = sf.validate_folds(folds, tt)
    assert all(ok.values()), ok
    for per in folds.values():
        assert len(per) == 24  # 12 identities x 2 states
        for d in per.values():
            assert len(d["train"]) == 4 and len(d["val"]) == 2 and len(d["test"]) == 2


def test_no_repeat_overlap_within_fold():
    tt = _tt()
    folds = sf.build_four_folds(tt, fold_root_seed=7)
    for per in folds.values():
        for d in per.values():
            allrows = list(d["train"]) + list(d["val"]) + list(d["test"])
            assert len(allrows) == len(set(allrows)) == 8


def test_test_coverage_is_balanced_each_repeat_once():
    tt = _tt()
    folds = sf.build_four_folds(tt, fold_root_seed=1234)
    cov = sf.fold_coverage(folds, tt)
    # controlled design: every repeat appears in test EXACTLY once across 4 folds
    assert (cov["test"] == 1).all(), cov[cov["test"] != 1]
    # and the row budget adds up: train+val+test counts == n_folds
    assert (cov["train"] + cov["val"] + cov["test"] == sf.N_FOLDS).all()


def test_determinism_same_seed_identical():
    tt = _tt()
    a = sf.build_four_folds(tt, 1234)
    b = sf.build_four_folds(tt, 1234)
    for fk in a:
        for key in a[fk]:
            for s in ("train", "val", "test"):
                assert np.array_equal(a[fk][key][s], b[fk][key][s])


def test_different_seed_changes_folds():
    tt = _tt()
    a = sf.build_four_folds(tt, 1234)
    b = sf.build_four_folds(tt, 9999)
    diff = any(not np.array_equal(a[fk][key][s], b[fk][key][s])
               for fk in a for key in a[fk] for s in ("train", "val", "test"))
    assert diff


def test_folds_invariant_to_row_order():
    # Shuffling the trial-table row order must NOT change the fold assignment
    # (folds key off (identity,state,repeat), via a SHA child seed).
    tt = _tt()
    shuffled = tt.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = sf.build_four_folds(tt, 1234)
    b = sf.build_four_folds(shuffled, 1234)
    # compare by (identity,state,repeat) membership rather than raw row indices
    def by_key(folds, table):
        table = table.reset_index(drop=True)
        out = {}
        for fk, per in folds.items():
            for (ident, state), d in per.items():
                for s in ("train", "val", "test"):
                    reps = tuple(sorted(int(table.loc[r, "repeat"]) for r in d[s]))
                    out[(fk, ident, state, s)] = reps
        return out
    assert by_key(a, tt) == by_key(b, shuffled)


def test_rejects_wrong_repeat_count():
    tt = _tt(n_rep=7)  # 7 repeats
    with pytest.raises(ValueError, match="expected 8"):
        sf.build_four_folds(tt, 1234)


def test_spec_requires_partitionable_test_coverage():
    with pytest.raises(ValueError, match="n_folds\\*n_test"):
        sf.FoldSpec(n_folds=3).validate()


def test_manifest_csv_roundtrip_and_hash(tmp_path):
    tt = _tt()
    folds = sf.build_four_folds(tt, 1234)
    p = tmp_path / "fold_manifest.csv"
    h1 = sf.write_fold_manifest_csv(folds, tt, 1234, p)
    h2 = sf.write_fold_manifest_csv(folds, tt, 1234, p)
    assert h1 == h2 and len(h1) == 64
    df = pd.read_csv(p)
    assert len(df) == 4 * 24 * 8  # folds x (id,state) x repeats
    assert set(df["fold"]) == {"fold_0", "fold_1", "fold_2", "fold_3"}
    assert set(df["split"]) == {"train", "val", "test"}
