"""O1 identity-held-out fold construction (grouping unit = stimulus identity, never trial).

12 identities = 6 simple (stim_set A) + 6 naturalistic (stim_set B). Outer: 6 folds, each
holding out ONE simple + ONE naturalistic identity (every identity is a test identity exactly
once). Inner (within each outer's 10 training identities): 5 folds, each holding out one
remaining simple + one remaining naturalistic identity. Deterministic (sorted) pairing -- no
randomness needed; leakage-free by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Fold:
    test: Tuple[str, ...]
    train: Tuple[str, ...]


def _split_families(identities: Sequence[str], family: Dict[str, str]) -> Tuple[List[str], List[str]]:
    simple = sorted(i for i in identities if family[i] == "simple")
    nat = sorted(i for i in identities if family[i] == "naturalistic")
    if len(simple) != len(nat):
        raise ValueError(f"unequal families: {len(simple)} simple vs {len(nat)} naturalistic")
    return simple, nat


def outer_folds(identities: Sequence[str], family: Dict[str, str]) -> List[Fold]:
    """6 outer folds; fold k tests (simple[k], naturalistic[k]); every identity tested once."""
    simple, nat = _split_families(identities, family)
    allids = tuple(sorted(identities))
    folds = []
    for k in range(len(simple)):
        test = (simple[k], nat[k])
        train = tuple(i for i in allids if i not in test)
        folds.append(Fold(test=test, train=train))
    return folds


def inner_folds(train_identities: Sequence[str], family: Dict[str, str]) -> List[Fold]:
    """5 inner folds over the 10 outer-train identities; each holds out 1 simple + 1 naturalistic."""
    simple, nat = _split_families(train_identities, family)
    train_set = tuple(sorted(train_identities))
    folds = []
    for j in range(len(simple)):
        test = (simple[j], nat[j])
        train = tuple(i for i in train_set if i not in test)
        folds.append(Fold(test=test, train=train))
    return folds


def certify_no_leakage(outer: List[Fold], inner_by_outer: List[List[Fold]]) -> dict:
    """Assert every identity is tested exactly once (outer) and inner tests never touch outer test."""
    tested = [i for f in outer for i in f.test]
    once = len(tested) == len(set(tested))
    each_fold_mixed = all(len(f.test) == 2 for f in outer)
    inner_ok = True
    for of, inners in zip(outer, inner_by_outer):
        for inf in inners:
            if set(inf.test) & set(of.test):
                inner_ok = False
            if not set(inf.train + inf.test) <= set(of.train):
                inner_ok = False           # inner never sees the outer test identities
    inner_cover = all(sorted(i for f in inners for i in f.test) == sorted(of.train)
                      for of, inners in zip(outer, inner_by_outer))
    return {"every_identity_tested_once": once, "each_outer_fold_1simple_1nat": each_fold_mixed,
            "inner_disjoint_from_outer_test": inner_ok, "inner_covers_outer_train_once": inner_cover,
            "n_outer": len(outer), "n_inner_per_outer": [len(x) for x in inner_by_outer]}
