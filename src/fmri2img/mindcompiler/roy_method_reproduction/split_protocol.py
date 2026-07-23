"""Paper-specified repeat-level split for Roy et al. reproduction.

Explicit in the paper (no ambiguity in structure): for every stimulus identity, of
its 8 repeats, 4 -> train, 2 -> validation, 2 -> test, and **all identities appear in
every split**. This is the repeat-level protocol whose identifiability limits the
synthetic work already characterised; S1 reproduces it faithfully before any
content-held-out modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction.contracts import (
    SPLIT_TEST_REPEATS,
    SPLIT_TRAIN_REPEATS,
    SPLIT_VAL_REPEATS,
)


@dataclass(frozen=True)
class RepeatSplit:
    """Row indices for one repeat-level split. All identities present in each."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def build_repeat_level_split(
    identity: np.ndarray,
    seed: int,
    n_repeats_expected: int = 8,
) -> RepeatSplit:
    """Assign repeats of each identity to train/val/test (4/2/2).

    Args:
        identity: (T,) stimulus identity per trial row.
        seed: RNG seed for the per-identity repeat shuffle (recorded in the manifest).
        n_repeats_expected: expected repeats per identity (8 in NSD-Imagery vision).

    Returns:
        A :class:`RepeatSplit` of row indices. Every identity contributes to all
        three partitions.

    Raises:
        ValueError: if an identity does not have exactly ``n_repeats_expected`` rows,
            which would make the 4/2/2 allocation ill-defined.
    """
    rng = np.random.default_rng(seed)
    tr: List[int] = []
    va: List[int] = []
    te: List[int] = []
    for k in np.unique(identity):
        rows = np.where(identity == k)[0]
        if rows.size != n_repeats_expected:
            raise ValueError(
                f"identity {k} has {rows.size} repeats, expected {n_repeats_expected}; "
                "the 4/2/2 allocation requires the documented repeat count"
            )
        perm = rng.permutation(rows)
        tr.extend(perm[:SPLIT_TRAIN_REPEATS].tolist())
        va.extend(perm[SPLIT_TRAIN_REPEATS:SPLIT_TRAIN_REPEATS + SPLIT_VAL_REPEATS].tolist())
        te.extend(perm[SPLIT_TRAIN_REPEATS + SPLIT_VAL_REPEATS:].tolist())
    return RepeatSplit(np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te)))


def split_is_valid(split: RepeatSplit, identity: np.ndarray) -> Dict[str, bool]:
    """Check the paper's invariants: disjoint partitions, all identities everywhere."""
    tr, va, te = set(split.train.tolist()), set(split.val.tolist()), set(split.test.tolist())
    all_ids = set(np.unique(identity).tolist())
    return {
        "disjoint": not (tr & va) and not (tr & te) and not (va & te),
        "covers_all_rows": len(tr | va | te) == identity.size,
        "all_identities_in_train": set(identity[list(tr)].tolist()) == all_ids,
        "all_identities_in_val": set(identity[list(va)].tolist()) == all_ids,
        "all_identities_in_test": set(identity[list(te)].tolist()) == all_ids,
    }
