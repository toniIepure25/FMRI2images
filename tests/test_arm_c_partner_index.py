"""Invariant tests for the ARM-C partner index.

ARM-C is only a control if the pairing is right. Each test here corresponds to a
way the control silently stops being a control -- the failure mode is never a
crash, it is a number that looks fine and means something else.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmri2img.data.arm_c_partner import (
    build_arm_c_partner_index,
    partner_index_hash,
    partner_index_manifest,
    verify_partner_index,
)
from fmri2img.models.auxiliary_objectives import stable_seed


def _trials(n_images: int, reps: int, subjects: int = 1):
    """Synthetic split: `n_images` images x `reps` repetitions x `subjects`."""
    ids, subj = [], []
    for s in range(subjects):
        for img in range(n_images):
            for _ in range(reps):
                ids.append(img + 1000 * s)  # distinct images per subject
                subj.append(s)
    return np.array(ids), np.array(subj)


def test_derangement_no_image_maps_to_itself() -> None:
    """A self-mapped image IS ARM-B for those trials."""
    ids, _ = _trials(40, 3)
    p = build_arm_c_partner_index(ids, seed=stable_seed("split", 42))
    assert not np.any(ids[p] == ids), "self-mapping detected: ARM-C is compromised"


def test_repetition_consistency_all_trials_share_one_partner_image() -> None:
    """Per-trial permutation would leak the true target through repetitions."""
    ids, _ = _trials(30, 4)
    p = build_arm_c_partner_index(ids, seed=7)
    for img in np.unique(ids):
        rows = np.nonzero(ids == img)[0]
        assert np.unique(ids[p[rows]]).size == 1, (
            f"image {img} maps to multiple partner images"
        )


def test_partner_trial_choice_spreads_across_repetitions() -> None:
    """Source repetitions should not all collapse onto one partner trial.

    Collapsing would make the target distribution artificially low-variance and
    would not match ARM-B's exposure to real trial-to-trial variability.
    """
    ids, _ = _trials(20, 4)
    p = build_arm_c_partner_index(ids, seed=3)
    img = ids[0]
    rows = np.nonzero(ids == img)[0]
    assert np.unique(p[rows]).size > 1, (
        "all repetitions of one image mapped to the same partner trial"
    )


def test_subject_containment() -> None:
    """A partner from another subject would mix anatomies."""
    ids, subj = _trials(25, 3, subjects=4)
    p = build_arm_c_partner_index(ids, seed=11, subject_ids=subj)
    assert np.all(subj[p] == subj), "partner crosses subject boundary"


def test_split_containment_is_structural() -> None:
    """Only training rows are ever passed in, so validation cannot be selected.

    Containment is guaranteed by construction rather than filtered after the fact.
    """
    ids, _ = _trials(30, 2)
    train_mask = np.arange(ids.size) < 40
    train_ids = ids[train_mask]

    p = build_arm_c_partner_index(train_ids, seed=5)
    assert p.max() < train_ids.size, "partner index escaped the training rows"

    held_out = set(int(i) for i in np.unique(ids[~train_mask])) - set(
        int(i) for i in np.unique(train_ids)
    )
    verify_partner_index(p, train_ids, forbidden_nsd_ids=list(held_out))


def test_restart_and_process_stability() -> None:
    """Same inputs must reproduce the mapping bit for bit.

    Anything keyed on Python's salted hash(), wall-clock, or worker id would
    silently differ across dataloader workers and across a checkpoint resume.
    """
    ids, subj = _trials(35, 3, subjects=2)
    seed = stable_seed("split_hash_abc", 42)

    a = build_arm_c_partner_index(ids, seed=seed, subject_ids=subj)
    b = build_arm_c_partner_index(ids, seed=seed, subject_ids=subj)
    assert np.array_equal(a, b)
    assert partner_index_hash(a) == partner_index_hash(b)


def test_seed_sensitivity() -> None:
    ids, _ = _trials(40, 2)
    a = build_arm_c_partner_index(ids, seed=1)
    b = build_arm_c_partner_index(ids, seed=2)
    assert not np.array_equal(a, b), "different seeds produced the same mapping"


def test_stable_seed_is_process_independent() -> None:
    """The seed itself must not depend on a salted hash."""
    assert stable_seed("split_hash_abc", 42) == stable_seed("split_hash_abc", 42)
    assert stable_seed("split_hash_abc", 42) != stable_seed("split_hash_xyz", 42)


def test_manifest_records_content_not_just_seed() -> None:
    """Seed alone is insufficient provenance: the mapping also depends on the split.

    Two runs with the same seed but different splits would otherwise record
    identical provenance for different experiments.
    """
    ids, subj = _trials(20, 2, subjects=2)
    p = build_arm_c_partner_index(ids, seed=9, subject_ids=subj)
    m = partner_index_manifest(p, ids, seed=9, subject_ids=subj)

    assert m["kind"] == "arm_c_partner_index"
    assert m["is_derangement"] is True
    assert m["within_subject"] is True
    assert m["n_subject_groups"] == 2
    assert len(m["partner_index_sha256"]) == 64

    ids2 = ids.copy()
    ids2[-1] = 999  # a different split
    p2 = build_arm_c_partner_index(ids2, seed=9, subject_ids=subj)
    assert m["partner_index_sha256"] != partner_index_hash(p2), (
        "the same seed on a different split produced the same hash"
    )


def test_verify_catches_self_mapping() -> None:
    """The verifier must actually fire; a silent pass would be worthless."""
    ids, _ = _trials(10, 2)
    bad = np.arange(ids.size)  # every row maps to itself
    with pytest.raises(AssertionError, match="self-mapping"):
        verify_partner_index(bad, ids)


def test_verify_catches_broken_repetition_consistency() -> None:
    ids, _ = _trials(10, 3)
    p = build_arm_c_partner_index(ids, seed=1)
    rows = np.nonzero(ids == ids[0])[0]
    # Point one repetition at a different partner image.
    other = np.nonzero(ids[p] != ids[p[rows[0]]])[0][0]
    p[rows[0]] = p[other]
    with pytest.raises(AssertionError, match="partner images"):
        verify_partner_index(p, ids)


def test_verify_catches_subject_crossing() -> None:
    """Redirect a whole image to another subject, so ONLY containment breaks.

    Corrupting a single row would trip the repetition-consistency assert first
    and this test would pass for the wrong reason.
    """
    ids, subj = _trials(10, 2, subjects=2)
    p = build_arm_c_partner_index(ids, seed=1, subject_ids=subj)

    src_rows = np.nonzero(ids == ids[0])[0]
    other_subject_rows = np.nonzero(subj != subj[0])[0]
    foreign_img = ids[other_subject_rows[0]]
    foreign_rows = np.nonzero(ids == foreign_img)[0]
    for k, row in enumerate(src_rows):
        p[row] = foreign_rows[k % foreign_rows.size]

    # Repetition consistency still holds; only subject containment is violated.
    assert np.unique(ids[p[src_rows]]).size == 1
    with pytest.raises(AssertionError, match="crosses subjects"):
        verify_partner_index(p, ids, subject_ids=subj)


def test_verify_catches_forbidden_images_in_pool() -> None:
    ids, _ = _trials(10, 2)
    p = build_arm_c_partner_index(ids, seed=1)
    with pytest.raises(AssertionError, match="forbidden image"):
        verify_partner_index(p, ids, forbidden_nsd_ids=[int(ids[0])])


def test_single_image_subject_rejected() -> None:
    """A derangement is impossible with one image; fail loudly, not silently."""
    ids = np.array([5, 5, 5])
    with pytest.raises(ValueError, match="at least 2"):
        build_arm_c_partner_index(ids, seed=1)


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="no rows"):
        build_arm_c_partner_index(np.array([], dtype=np.int64), seed=1)


def test_arm_c_differs_from_arm_b_in_target_identity_only() -> None:
    """The partner supplies a different image's activity at the same row shape.

    ARM-B's target is features[i]; ARM-C's is features[partner[i]]. Same subject,
    same split, same array, same shape -- only the stimulus identity differs.
    """
    ids, subj = _trials(30, 3, subjects=2)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((ids.size, 64)).astype(np.float32)

    p = build_arm_c_partner_index(ids, seed=stable_seed("s", 42), subject_ids=subj)
    verify_partner_index(p, ids, subject_ids=subj)

    arm_b_target = features
    arm_c_target = features[p]

    assert arm_c_target.shape == arm_b_target.shape, "targets must be shape-matched"
    assert not np.allclose(arm_c_target, arm_b_target), "ARM-C target equals ARM-B's"
    assert np.all(ids[p] != ids), "ARM-C target carries a different image"
    assert np.all(subj[p] == subj), "ARM-C target stays within subject"


def test_scales_to_realistic_split() -> None:
    """~9k images x 3 reps for one subject, the real NSD single-subject scale."""
    ids, _ = _trials(9000, 3)
    p = build_arm_c_partner_index(ids, seed=stable_seed("real", 42))
    verify_partner_index(p, ids)
    assert p.size == 27000
