"""S2.5M public-method-aligned pairing (vis2vis within-split derangement; vis2img random).

Corrects the S2.5R material gaps. Author seeds are unknown; we derive deterministic
independent-reproduction seeds cryptographically from a PUBLIC immutable literal (the
DOI). Seeds intentionally EXCLUDE ROI and beta version, so the SAME pairings are used
across all 7 ROIs and both beta branches within a participant/fold -> controlled
comparisons. AUTHOR_RANDOM_SEED_NOT_PUBLICLY_IDENTIFIABLE.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np

DOI = "10.1101/2025.09.02.672180"
POLICY_VERSION = "PAIRING_V1"
V2V_POLICY = "V2V_ROY_WITHIN_SPLIT_DERANGEMENT_V1"
V2I_POLICY = "V2I_ROY_RANDOM_WITHIN_IDENTITY_V1"


def root_digest(realization: int = 0) -> str:
    return hashlib.sha256(f"{DOI}|S2.5M|{POLICY_VERSION}|realization={realization}".encode()).hexdigest()


def child_seed(root_hex: str, participant: str, fold: str, split: str, identity: str,
               model: str) -> int:
    """64-bit child seed. Excludes ROI and beta by contract (shared pairings)."""
    key = "|".join([root_hex, participant, fold, split, identity, model, POLICY_VERSION])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    return perm


def vis2vis_pairs(rows: Sequence[int], seed: int) -> Tuple[List[int], List[int]]:
    """Within-identity, within-split one-to-one derangement (source != target)."""
    rows = list(map(int, rows))
    if len(rows) < 2:
        raise ValueError("derangement needs >= 2 rows")
    perm = _derangement(len(rows), seed)
    return rows, [rows[int(p)] for p in perm]


def vis2img_pairs(vision_rows: Sequence[int], imagery_rows: Sequence[int], seed: int
                  ) -> Tuple[List[int], List[int]]:
    """Random one-to-one pairing of vision->imagery within identity/split (no derangement).

    Vision and imagery are separate trials; equal ordinal repeat index has no self-trial
    meaning, so NO fixed-point constraint is imposed. Random permutation without replacement.
    """
    v = list(map(int, vision_rows)); im = list(map(int, imagery_rows))
    n = min(len(v), len(im))
    perm = np.random.default_rng(seed).permutation(len(im))[:n]
    return v[:n], [im[int(p)] for p in perm]


def pooled_vis2vis(train_by_ident, root_hex, participant, fold, split):
    """Pooled within-identity vis2vis derangement across identities for one split."""
    S: List[int] = []; T: List[int] = []
    for ident in sorted(train_by_ident):
        cs = child_seed(root_hex, participant, fold, split, ident, "vis2vis")
        s, t = vis2vis_pairs(train_by_ident[ident], cs)
        S.extend(s); T.extend(t)
    return np.array(S, dtype=int), np.array(T, dtype=int)
