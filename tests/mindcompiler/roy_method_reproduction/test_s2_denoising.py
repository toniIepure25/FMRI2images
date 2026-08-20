"""S2.0 PART E -- D1 cross-fit denoising dependency-graph and leakage tests (data-free)."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_denoising as dn
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY


def _setup(n_id=3, n_vox=10, seed=0):
    """Synthetic vision fold: per identity 4 train / 2 val / 2 test rows."""
    rng = np.random.default_rng(seed)
    train_by, val_by, test_by = {}, {}, {}
    row_identity, row_beta = {}, {}
    r = 0
    for i in range(n_id):
        ident = f"ID{i}"
        rows = list(range(r, r + 8)); r += 8
        for x in rows:
            row_identity[x] = ident; row_beta[x] = 1000 + x
        train_by[ident] = rows[:4]; val_by[ident] = rows[4:6]; test_by[ident] = rows[6:8]
    n_trials = r
    M = rng.standard_normal((n_trials, n_vox))
    return M, train_by, val_by, test_by, row_identity, row_beta


def _run(policy="all_ordered_distinct", seed=None):
    M, tr, va, te, rid, rb = _setup()
    den, recs = dn.d1_crossfit_denoise(
        M, "fold_0", tr, va, te, rid, rb, lam=10.0, rank=3,
        pairing_policy=policy, pairing_seed=seed, preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY)
    fold_train = {int(x) for rows in tr.values() for x in rows}
    return M, tr, va, te, rid, den, recs, fold_train


def test_denoises_all_rows_and_passes_leakage_contract():
    M, tr, va, te, rid, den, recs, fold_train = _run()
    n_expected = sum(len(v) for v in tr.values()) + sum(len(v) for v in va.values()) + sum(len(v) for v in te.values())
    assert len(den) == n_expected
    for vec in den.values():
        assert vec.shape == (M.shape[1],) and np.isfinite(vec).all()
    v = dn.validate_denoising_leakage(recs, fold_train, rid)
    assert v == {"self_target": 0, "self_source": 0, "val_test_in_training": 0,
                 "cross_identity_target": 0, "training_row_not_train_split": 0}


def test_train_rows_are_leave_one_out():
    _, tr, _, _, _, _, recs, _ = _run()
    train_recs = [r for r in recs if r.split == "train"]
    for r in train_recs:
        # the denoised trial is NEVER in its model's training set (source or target)
        assert r.denoised_row not in r.train_target_rows
        assert r.denoised_row not in r.train_source_rows
        assert r.model_id == f"fold_0:LOO_excl_{r.denoised_row}"


def test_val_test_use_full_train_model_within_train_rows():
    _, _, va, te, _, _, recs, fold_train = _run()
    for r in [x for x in recs if x.split in ("val", "test")]:
        assert r.model_id == "fold_0:FULLTRAIN"
        used = set(r.train_source_rows) | set(r.train_target_rows)
        assert used <= fold_train                 # no val/test rows leak into training
        assert r.denoised_row not in fold_train    # the evaluated row was never trained on


def test_validator_detects_self_target():
    _, _, _, _, rid, _, recs, fold_train = _run()
    bad = recs[0]
    tampered = dn.DenoiseRecord(
        denoised_row=bad.denoised_row, identity=bad.identity, beta_index=bad.beta_index,
        fold=bad.fold, split=bad.split, model_id=bad.model_id,
        train_source_rows=bad.train_source_rows,
        train_target_rows=bad.train_target_rows + (bad.denoised_row,))  # inject self-target
    with pytest.raises(dn.DenoisingLeakageError, match="self_target"):
        dn.validate_denoising_leakage([tampered], fold_train, rid)


def test_validator_detects_val_row_in_training():
    _, _, va, _, rid, _, recs, fold_train = _run()
    val_row = next(iter(va.values()))[0]
    # a record whose training set illegally includes a validation row
    bad = dn.DenoiseRecord(
        denoised_row=999, identity="IDx", beta_index=0, fold="fold_0", split="test",
        model_id="m", train_source_rows=(val_row,), train_target_rows=(val_row,))
    with pytest.raises(dn.DenoisingLeakageError, match="training_row_not_train_split"):
        dn.validate_denoising_leakage([bad], fold_train, rid)


def test_derangement_requires_seed_and_is_deterministic():
    with pytest.raises(ValueError, match="derangement pairing requires a child seed"):
        dn.vis2vis_pairs_for_rows([0, 1, 2, 3], "single_deterministic_derangement", None)
    a = _run(policy="single_deterministic_derangement", seed=1234)[6]
    b = _run(policy="single_deterministic_derangement", seed=1234)[6]
    aid = [(r.split, r.denoised_row, r.train_target_rows) for r in a]
    bid = [(r.split, r.denoised_row, r.train_target_rows) for r in b]
    assert aid == bid


def test_pooled_pairs_are_within_identity():
    M, tr, va, te, rid, rb = _setup()
    S, T = dn.pooled_pairs(tr, "all_ordered_distinct", None)
    for s, t in zip(S, T):
        assert rid[int(s)] == rid[int(t)]      # never cross-identity
    # all_ordered_distinct: 4x3 per identity x 3 identities = 36 pooled train pairs
    assert len(S) == 3 * 4 * 3
