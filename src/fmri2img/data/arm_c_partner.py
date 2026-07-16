"""Partner-trial index construction for ARM-C (stimulus-shuffled neural targets).

ARM-C keeps the real ROI-activity target distribution but destroys its
stimulus-specific content, by pairing each trial with the activity recorded for a
**different image**. If ARM-B does not beat ARM-C, the neural-prediction objective
is exploiting marginal statistics or inter-ROI connectivity rather than
stimulus-specific neural content (reviewer objection O-3).

The control is only as good as the pairing. Four ways to get it silently wrong,
each of which turns ARM-C back into something it is not:

1. **Self-mapping.** An image paired with itself *is* ARM-B for those trials.
   Prevented by constructing a derangement.
2. **Per-trial permutation.** If repeated presentations of one image map to
   *different* partners, the true target leaks across repetitions. Prevented by
   permuting **image identities**, then resolving trials.
3. **Cross-split contamination.** A partner drawn from validation or SHARED1000
   would pull sealed data into training. Prevented by restricting the pool to the
   training split.
4. **Worker/restart drift.** A mapping that depends on ``hash()``, wall-clock, or
   dataloader worker id would differ across processes and resumes, making the run
   unreproducible. Prevented by deriving everything from
   :func:`~fmri2img.models.auxiliary_objectives.stable_seed` and materialising the
   full index up front.

The index is built **once**, eagerly, and is a plain array. That is the cheapest
way to guarantee properties 2 and 4 simultaneously.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from fmri2img.models.auxiliary_objectives import DeterministicImagePermutation

logger = logging.getLogger(__name__)


def build_arm_c_partner_index(
    nsd_ids: Sequence[int],
    seed: int,
    subject_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Map every training row to a partner row carrying a different image's activity.

    All rows supplied must already be **training-split rows for the arm's subject
    set**; the function never sees validation or SHARED1000 data, which is how
    split containment is guaranteed rather than merely checked.

    Partner-trial selection among a partner image's repetitions is deterministic:
    the source trial's rank ``k`` within its own image selects the partner's
    ``k mod n_partner_trials``-th trial (trials ordered by row index). This spreads
    source repetitions across partner repetitions instead of collapsing them onto
    one trial, and depends on nothing but the data order.

    Args:
        nsd_ids: (N,) image identity per training row.
        seed: Derived via ``stable_seed(split_hash, experiment_seed)``.
        subject_ids: (N,) subject per row. When given, partners are drawn **within
            subject**, so a partner never carries another subject's anatomy.

    Returns:
        (N,) int64 array; ``partner[i]`` indexes into the same row space as the
        input. ``nsd_ids[partner[i]] != nsd_ids[i]`` for every ``i``.

    Raises:
        ValueError: If any subject group has fewer than 2 unique images, which
            makes a derangement impossible.
    """
    ids = np.asarray(nsd_ids, dtype=np.int64)
    n = ids.size
    if n == 0:
        raise ValueError("no rows supplied")

    if subject_ids is None:
        groups = {0: np.arange(n)}
    else:
        subj = np.asarray(subject_ids)
        groups = {int(s): np.nonzero(subj == s)[0] for s in np.unique(subj)}

    partner = np.full(n, -1, dtype=np.int64)

    for subject, rows in groups.items():
        group_ids = ids[rows]
        uniq = np.unique(group_ids)
        if uniq.size < 2:
            raise ValueError(
                f"subject {subject} has {uniq.size} unique image(s); a derangement "
                "needs at least 2. ARM-C cannot be constructed for this split."
            )

        # Permute IMAGE identities (never trials) -- guarantees repetition
        # consistency and rules out self-mapping in one step.
        perm = DeterministicImagePermutation(uniq.tolist(), seed=seed + int(subject))

        # Trials of each image, ordered by row index: stable and data-determined.
        trials_of: Dict[int, List[int]] = {}
        for local, row in enumerate(rows):
            trials_of.setdefault(int(group_ids[local]), []).append(int(row))

        for img, src_rows in trials_of.items():
            partner_rows = trials_of[perm(img)]
            for k, src in enumerate(src_rows):
                partner[src] = partner_rows[k % len(partner_rows)]

    if np.any(partner < 0):
        raise RuntimeError("internal error: unassigned partner rows")
    if np.any(ids[partner] == ids):
        raise RuntimeError("internal error: an image was paired with itself")

    logger.info(
        "build_arm_c_partner_index: %d rows, %d subject group(s), seed=%d",
        n, len(groups), seed,
    )
    return partner


def partner_index_hash(partner: np.ndarray) -> str:
    """Content hash of the full mapping, for the run manifest.

    Recording only the seed is not enough: the mapping also depends on the split
    and on row order, so two runs with the same seed but a different split would
    record identical provenance for different experiments.
    """
    return hashlib.sha256(np.ascontiguousarray(partner, dtype=np.int64).tobytes()).hexdigest()


def partner_index_manifest(
    partner: np.ndarray,
    nsd_ids: Sequence[int],
    seed: int,
    subject_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Provenance record for ARM-C, for `05_EXPERIMENT_REGISTRY.csv`."""
    ids = np.asarray(nsd_ids, dtype=np.int64)
    return {
        "kind": "arm_c_partner_index",
        "seed": int(seed),
        "n_rows": int(partner.size),
        "n_unique_images": int(np.unique(ids).size),
        "n_subject_groups": 1 if subject_ids is None else int(np.unique(subject_ids).size),
        "partner_index_sha256": partner_index_hash(partner),
        "is_derangement": bool(not np.any(ids[partner] == ids)),
        "within_subject": subject_ids is not None,
    }


def verify_partner_index(
    partner: np.ndarray,
    nsd_ids: Sequence[int],
    subject_ids: Optional[Sequence[int]] = None,
    forbidden_nsd_ids: Optional[Sequence[int]] = None,
) -> None:
    """Assert every ARM-C invariant. Cheap; run it at construction, every run.

    Args:
        partner: Output of :func:`build_arm_c_partner_index`.
        nsd_ids: Image identity per row.
        subject_ids: Subject per row, if partners must stay within subject.
        forbidden_nsd_ids: Images that must never appear in the target pool --
            validation images and SHARED1000.

    Raises:
        AssertionError: On any violation.
    """
    ids = np.asarray(nsd_ids, dtype=np.int64)

    assert partner.shape == ids.shape, "partner index has the wrong shape"
    assert partner.min() >= 0 and partner.max() < ids.size, "partner row out of range"
    assert not np.any(ids[partner] == ids), "self-mapping: ARM-C degenerates into ARM-B"

    # Repetition consistency: all trials of one image share one partner image.
    for img in np.unique(ids):
        rows = np.nonzero(ids == img)[0]
        partner_imgs = np.unique(ids[partner[rows]])
        assert partner_imgs.size == 1, (
            f"image {img} maps to {partner_imgs.size} partner images; trials of one "
            "image must share a partner or the true target leaks via repetitions"
        )

    if subject_ids is not None:
        subj = np.asarray(subject_ids)
        assert np.all(subj[partner] == subj), (
            "a partner crosses subjects; ARM-C would mix anatomies"
        )

    if forbidden_nsd_ids is not None:
        forbidden = set(int(i) for i in forbidden_nsd_ids)
        present = forbidden.intersection(int(i) for i in np.unique(ids))
        assert not present, (
            f"{len(present)} forbidden image(s) are in the ARM-C pool "
            "(validation or SHARED1000 leaked into the training rows)"
        )
