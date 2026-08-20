"""S2.0 four-fold 4/2/2 protocol with controlled, deterministic coverage.

The S1 smoke used ONE 4/2/2 realization. S2.0 requires FOUR folds. For every
(identity, state) the 8 analyzed repeats are partitioned per fold into 4 train /
2 validation / 2 test.

**Controlled coverage.** Because ``n_folds * n_test == n_repeats`` (4 * 2 == 8),
the four test-pairs exactly PARTITION the 8 repeats: **each repeat appears in the
test split exactly once across the four folds** (balanced test coverage, verified
by :func:`fold_coverage`). Train/validation coverage is deterministic and reported
as-is (not claimed balanced unless it is).

Determinism is by stable child seed (SHA-256 over root + identity + state +
version), so folds are identical across processes and unaffected by
``PYTHONHASHSEED`` -- never Python's builtin ``hash()``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fmri2img.mindcompiler.roy_method_reproduction.pairing import derive_child_seed

FOLD_DERIVATION_VERSION = "s2_folds_v1"
N_FOLDS = 4
N_REPEATS = 8
N_TRAIN, N_VAL, N_TEST = 4, 2, 2


@dataclass(frozen=True)
class FoldSpec:
    n_folds: int = N_FOLDS
    n_repeats: int = N_REPEATS
    n_train: int = N_TRAIN
    n_val: int = N_VAL
    n_test: int = N_TEST

    def validate(self) -> None:
        if self.n_train + self.n_val + self.n_test != self.n_repeats:
            raise ValueError("train+val+test must equal n_repeats")
        if self.n_folds * self.n_test != self.n_repeats:
            raise ValueError(
                "controlled test coverage requires n_folds*n_test == n_repeats "
                f"(got {self.n_folds}*{self.n_test} != {self.n_repeats})")


def build_four_folds(tt: pd.DataFrame, fold_root_seed: int,
                     spec: FoldSpec = FoldSpec()) -> Dict[str, Dict[Tuple[str, str], Dict[str, np.ndarray]]]:
    """Build ``n_folds`` deterministic 4/2/2 splits per (identity, state).

    Args:
        tt: trial table with columns ``identity``, ``state``, ``repeat``,
            ``beta_index0`` (row index = trial row).
        fold_root_seed: run-level seed; per-(identity,state) child seeds derive
            from it deterministically.
        spec: fold geometry (validated).

    Returns:
        ``{"fold_k": {(identity, state): {"train"/"val"/"test": row-index array}}}``.
    """
    spec.validate()
    tt = tt.reset_index(drop=True)
    folds: Dict[str, Dict[Tuple[str, str], Dict[str, np.ndarray]]] = {
        f"fold_{k}": {} for k in range(spec.n_folds)}
    for (ident, state), g in tt.groupby(["identity", "state"], sort=True):
        rows = g.sort_values("repeat")
        if len(rows) != spec.n_repeats:
            raise ValueError(
                f"({ident},{state}) has {len(rows)} repeats, expected {spec.n_repeats}")
        row_of_repeat = rows["_index"].values if "_index" in rows else rows.index.values
        seed = derive_child_seed(fold_root_seed, "fold", state, ident, FOLD_DERIVATION_VERSION)
        perm = np.random.default_rng(seed).permutation(spec.n_repeats)
        ordered_rows = row_of_repeat[perm]  # repeats in the permuted order
        for k in range(spec.n_folds):
            test_pos = list(range(k * spec.n_test, (k + 1) * spec.n_test))
            test = ordered_rows[test_pos]
            rest = np.array([ordered_rows[i] for i in range(spec.n_repeats)
                             if i not in test_pos])
            train = rest[:spec.n_train]
            val = rest[spec.n_train:spec.n_train + spec.n_val]
            folds[f"fold_{k}"][(ident, state)] = {
                "train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    return folds


def validate_folds(folds: Dict, tt: pd.DataFrame, spec: FoldSpec = FoldSpec()) -> Dict[str, bool]:
    """Check per-fold invariants: 4/2/2 counts, all identities, no repeat overlap."""
    idents = set(tt["identity"].unique())
    ok: Dict[str, bool] = {}
    for fk, per in folds.items():
        idents_present = {ident for (ident, _st) in per}
        counts_ok = all(
            len(d["train"]) == spec.n_train and len(d["val"]) == spec.n_val
            and len(d["test"]) == spec.n_test for d in per.values())
        no_overlap = all(
            len(set(d["train"]) | set(d["val"]) | set(d["test"]))
            == spec.n_repeats for d in per.values())
        ok[fk] = bool(counts_ok and no_overlap and idents_present == idents)
    return ok


def fold_coverage(folds: Dict, tt: pd.DataFrame) -> pd.DataFrame:
    """Per-(identity,state,repeat) count of appearances in train/val/test across folds."""
    tt = tt.reset_index(drop=True)
    rep_of_row = {i: (tt.loc[i, "identity"], tt.loc[i, "state"], int(tt.loc[i, "repeat"]))
                  for i in tt.index}
    counts: Dict[Tuple, Dict[str, int]] = {}
    for per in folds.values():
        for d in per.values():
            for split in ("train", "val", "test"):
                for r in d[split]:
                    key = rep_of_row[int(r)]
                    counts.setdefault(key, {"train": 0, "val": 0, "test": 0})[split] += 1
    rows = [dict(identity=k[0], state=k[1], repeat=k[2], **v) for k, v in sorted(counts.items())]
    return pd.DataFrame(rows)


def write_fold_manifest_csv(folds: Dict, tt: pd.DataFrame, fold_root_seed: int, path) -> str:
    """Write the row-level fold manifest and return its sha256."""
    tt = tt.reset_index(drop=True)
    recs: List[dict] = []
    for fk, per in folds.items():
        for (ident, state), d in per.items():
            for split in ("train", "val", "test"):
                for r in d[split]:
                    r = int(r)
                    recs.append(dict(
                        fold=fk, identity=ident, state=state, split=split,
                        trial_row=r, beta_index0=int(tt.loc[r, "beta_index0"]),
                        repeat=int(tt.loc[r, "repeat"]),
                        fold_root_seed=fold_root_seed,
                        derivation_version=FOLD_DERIVATION_VERSION))
    df = pd.DataFrame(recs).sort_values(
        ["fold", "state", "identity", "split", "repeat"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
