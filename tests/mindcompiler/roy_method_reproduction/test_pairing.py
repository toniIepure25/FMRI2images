"""Pairing-manifest contract: seed independence, policy correctness, P0/I0 parity.

P0/I0 are the historical smoke path; the manifest module MUST reproduce the
``smoke_pipeline`` pairs bit-for-bit (regression guard), so switching the runner
to manifests can never move the 13cc3f0 numbers. P1/I1 are sensitivity variants
whose only stochastic input is a stable, independent child seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import pairing as pr
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp


def _toy_splits(n_id=3, reps=8, seed=1234):
    """Small synthetic trial table + splits mirroring make_splits' structure."""
    rows = []
    for s, state in enumerate(("vision", "imagery")):
        for i in range(n_id):
            for rep in range(reps):
                bi = s * 100 + i * reps + rep
                rows.append(dict(beta_index0=bi, identity=f"ID{i}", state=state, repeat=rep))
    tt = pd.DataFrame(rows)
    splits = sp.make_splits(tt, seed)
    return tt, splits


# --- child-seed derivation ---------------------------------------------------

def test_child_seed_is_deterministic_and_hashseed_independent():
    a = pr.derive_child_seed(1234, "vis2vis", "P1", "train", "ID0")
    b = pr.derive_child_seed(1234, "vis2vis", "P1", "train", "ID0")
    assert a == b and 0 <= a < 2 ** 64
    # value is a pure function of the labels (SHA-256), not builtin hash():
    import hashlib
    key = "|".join(["1234", "vis2vis", "P1", "train", "ID0"])
    assert a == int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def test_child_seeds_are_independent_across_labels():
    seeds = {
        pr.derive_child_seed(1234, "vis2vis", "P1", "train", "ID0"),
        pr.derive_child_seed(1234, "vis2vis", "P1", "train", "ID1"),
        pr.derive_child_seed(1234, "vis2img", "I1", "train", "ID0"),
        pr.derive_child_seed(1234, "vis2vis", "P1", "test", "ID0"),
        pr.derive_child_seed(9999, "vis2vis", "P1", "train", "ID0"),
    }
    assert len(seeds) == 5  # all distinct


# --- P0 / I0 parity with the historical smoke path ---------------------------

@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_P0_matches_smoke_pipeline_all_ordered_distinct(part):
    tt, splits = _toy_splits()
    rows = pr.vis2vis_manifest(splits["vision"], splits["_tt"], part,
                               "all_ordered_distinct", root_seed=1234)
    s, t = pr.rows_to_arrays(rows)
    xs, ys = sp.vis2vis_pairs(splits["vision"], part, "all-ordered-distinct", 1234)
    assert np.array_equal(s, xs) and np.array_equal(t, ys)
    assert all(r.child_seed == -1 for r in rows)  # P0 consumes no seed


@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_I0_matches_smoke_pipeline_index_aligned(part):
    tt, splits = _toy_splits()
    rows = pr.vis2img_manifest(splits["vision"], splits["imagery"], splits["_tt"],
                               part, "historical_index_aligned", root_seed=1234)
    s, t = pr.rows_to_arrays(rows)
    xs, ys = sp.vis2img_pairs(splits["vision"], splits["imagery"], part)
    assert np.array_equal(s, xs) and np.array_equal(t, ys)
    assert all(r.child_seed == -1 for r in rows)


# --- P1 / I1 correctness -----------------------------------------------------

def test_P1_is_a_derangement_and_reproducible():
    tt, splits = _toy_splits()
    r1 = pr.vis2vis_manifest(splits["vision"], splits["_tt"], "train",
                             "deterministic_derangement", root_seed=1234)
    r2 = pr.vis2vis_manifest(splits["vision"], splits["_tt"], "train",
                             "deterministic_derangement", root_seed=1234)
    # reproducible under the same root
    assert [pr_.source_row for pr_ in r1] == [pr_.source_row for pr_ in r2]
    assert [pr_.target_row for pr_ in r1] == [pr_.target_row for pr_ in r2]
    # no fixed points within any identity
    for row in r1:
        assert row.source_row != row.target_row
    # different root -> different child seeds recorded
    r3 = pr.vis2vis_manifest(splits["vision"], splits["_tt"], "train",
                             "deterministic_derangement", root_seed=9999)
    assert {x.child_seed for x in r1} != {x.child_seed for x in r3}


def test_I1_is_within_identity_permutation_and_differs_from_I0():
    tt, splits = _toy_splits()
    i1 = pr.vis2img_manifest(splits["vision"], splits["imagery"], splits["_tt"],
                             "train", "independent_within_identity_permutation", 1234)
    i0 = pr.vis2img_manifest(splits["vision"], splits["imagery"], splits["_tt"],
                             "train", "historical_index_aligned", 1234)
    # I1 targets are a permutation of the SAME imagery rows I0 uses per identity
    by_id_i0, by_id_i1 = {}, {}
    for r in i0:
        by_id_i0.setdefault(r.identity, []).append(r.target_row)
    for r in i1:
        by_id_i1.setdefault(r.identity, []).append(r.target_row)
    for ident in by_id_i0:
        assert sorted(by_id_i0[ident]) == sorted(by_id_i1[ident])
    # and it is genuinely different from index-alignment for at least one identity
    assert any(by_id_i0[k] != by_id_i1[k] for k in by_id_i0)


def test_manifest_csv_roundtrip_has_full_provenance(tmp_path):
    tt, splits = _toy_splits()
    rows = pr.vis2vis_manifest(splits["vision"], splits["_tt"], "train",
                               "deterministic_derangement", root_seed=1234)
    path = tmp_path / "m.csv"
    pr.write_manifest_csv(rows, path)
    back = pd.read_csv(path)
    assert list(back.columns) == list(pr.PairingRow.__dataclass_fields__.keys())
    assert len(back) == len(rows)
    # beta indices align with the trial table
    for _, br in back.iterrows():
        assert br["source_beta_index0"] == tt.iloc[int(br["source_row"])]["beta_index0"]
