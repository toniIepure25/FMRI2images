"""Row-level pairing manifests with stable, independent child-seed derivation.

Four pairing policies produce (source_trial, target_trial) lists per partition:

* ``P0`` vis2vis ``all_ordered_distinct`` -- every ordered distinct pair per
  identity (the historical 13cc3f0 smoke path); deterministic from the split
  alone (no pairing seed consumed).
* ``P1`` vis2vis ``deterministic_derangement`` -- one no-fixed-point permutation
  per identity, seeded by a stable child seed. Sensitivity variant.
* ``I0`` vis2img ``historical_index_aligned`` -- within-identity index alignment
  of vision->imagery repeats (the historical smoke path); deterministic from the
  split alone.
* ``I1`` vis2img ``independent_within_identity_permutation`` -- an independent
  permutation of the identity's imagery repeats, seeded by a stable child seed.
  Sensitivity variant.

**Seed independence.** Randomized policies (P1, I1) never consume the root seed
directly. Each per-identity RNG is seeded by ``derive_child_seed(root, policy,
partition, identity)`` -- a SHA-256 of the canonicalised labels, so it is stable
across processes (independent of ``PYTHONHASHSEED``) and statistically
independent across (policy, partition, identity). Two policies sharing one root
seed therefore do not share randomness.

Every policy emits a flat list of :class:`PairingRow` carrying full provenance
(row indices, beta indices, the child seed actually used), which serialises to a
row-level CSV manifest for audit.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal

import numpy as np

Vis2VisPolicy = Literal["all_ordered_distinct", "deterministic_derangement"]
Vis2ImgPolicy = Literal["historical_index_aligned", "independent_within_identity_permutation"]

#: Policies that consume no pairing seed (fully determined by the split).
SPLIT_DETERMINISTIC = {"all_ordered_distinct", "historical_index_aligned"}


def derive_child_seed(root_seed: int, *labels) -> int:
    """Stable 64-bit child seed from a root seed and string labels.

    Uses SHA-256 of the ``|``-joined canonical labels rather than Python's builtin
    ``hash`` so the value is identical across interpreters and unaffected by
    ``PYTHONHASHSEED``. Distinct label tuples give independent seeds.

    Args:
        root_seed: the run-level pairing seed.
        *labels: canonical descriptors (policy, partition, identity, ...).

    Returns:
        A 64-bit non-negative integer seed suitable for ``np.random.default_rng``.
    """
    key = "|".join([str(int(root_seed))] + [str(x) for x in labels])
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class PairingRow:
    """One (source -> target) pairing with full provenance."""

    policy: str
    partition: str
    identity: str
    pair_index: int
    source_row: int
    target_row: int
    source_beta_index0: int
    target_beta_index0: int
    child_seed: int  # -1 when the policy consumes no seed


def _beta0(tt, row: int) -> int:
    return int(tt.iloc[int(row)]["beta_index0"])


def _idents(split: Dict) -> List[str]:
    return [k for k in split if k != "_tt"]


def vis2vis_manifest(vsplit: Dict, tt, part: str, policy: Vis2VisPolicy,
                     root_seed: int) -> List[PairingRow]:
    """Row-level vis2vis pairing manifest for one partition."""
    rows: List[PairingRow] = []
    for ident in _idents(vsplit):
        r = np.asarray(vsplit[ident][part])
        if policy == "all_ordered_distinct":
            pairs = [(int(r[a]), int(r[c])) for a in range(len(r))
                     for c in range(len(r)) if a != c]
            seed_used = -1
        elif policy == "deterministic_derangement":
            seed_used = derive_child_seed(root_seed, "vis2vis", "P1", part, ident)
            rng = np.random.default_rng(seed_used)
            perm = rng.permutation(len(r))
            while np.any(perm == np.arange(len(r))):  # enforce no fixed point
                perm = rng.permutation(len(r))
            pairs = [(int(r[a]), int(r[perm[a]])) for a in range(len(r))]
        else:
            raise ValueError(f"unknown vis2vis policy {policy!r}")
        for k, (s, t) in enumerate(pairs):
            rows.append(PairingRow(policy, part, ident, k, s, t,
                                   _beta0(tt, s), _beta0(tt, t), seed_used))
    return rows


def vis2img_manifest(vsplit: Dict, isplit: Dict, tt, part: str,
                     policy: Vis2ImgPolicy, root_seed: int) -> List[PairingRow]:
    """Row-level vis2img pairing manifest for one partition."""
    rows: List[PairingRow] = []
    for ident in _idents(vsplit):
        vr = np.asarray(vsplit[ident][part])
        ir = np.asarray(isplit[ident][part])
        n = min(len(vr), len(ir))
        if policy == "historical_index_aligned":
            tgt = list(range(n))
            seed_used = -1
        elif policy == "independent_within_identity_permutation":
            seed_used = derive_child_seed(root_seed, "vis2img", "I1", part, ident)
            rng = np.random.default_rng(seed_used)
            tgt = rng.permutation(n).tolist()
        else:
            raise ValueError(f"unknown vis2img policy {policy!r}")
        for k in range(n):
            s, t = int(vr[k]), int(ir[tgt[k]])
            rows.append(PairingRow(policy, part, ident, k, s, t,
                                   _beta0(tt, s), _beta0(tt, t), seed_used))
    return rows


def rows_to_arrays(rows: List[PairingRow]):
    """Collapse manifest rows to ``(source_rows, target_rows)`` int arrays."""
    return (np.array([r.source_row for r in rows]),
            np.array([r.target_row for r in rows]))


def write_manifest_csv(rows: List[PairingRow], path) -> None:
    """Write a row-level pairing manifest CSV (one line per pair)."""
    fields = list(PairingRow.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
