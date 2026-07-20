"""Can a vis2img-style model distinguish a neural transformation from identity lookup?

Gate M1 asks whether Roy et al.'s vision-to-imagery result identifies a
generalizable neural transformation, or whether it is explained by
repeated-condition identity.

The structural problem: with 12 stimulus identities and 8-16 repeats each, any
split over *repeats* necessarily places every identity in train, validation and
test. A model can then achieve high prediction by learning, in effect, "this
perception pattern belongs to stimulus k, so emit stimulus k's average imagery
pattern" -- **a lookup table, containing no transformation at all**.

This module simulates two generative worlds with known ground truth and asks
whether the estimator can tell them apart:

* ``identity_template`` -- imagery depends ONLY on which stimulus it was. There
  is no mapping from the perception *trial* to the imagery *trial*. Any apparent
  "transformation" is spurious.
* ``true_transform`` -- imagery is a genuine linear function of the single-trial
  perception pattern, over and above stimulus identity.

Under a repeat-level split both worlds can look identical. Under an
identity-held-out split they must not. That contrast is the diagnostic, and this
module exists to check that our analysis has it before any real data is touched.

No GPU, no real data, pure NumPy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)

World = Literal["identity_template", "true_transform", "mixture"]


@dataclass
class SimConfig:
    """Generative settings for one synthetic dataset.

    Attributes:
        n_identities: Number of distinct stimuli (NSD-Imagery has 12).
        n_repeats: Repeats per identity per state.
        n_voxels: Voxel count.
        latent_dim: Rank of the true transformation, when one exists.
        noise: Trial noise standard deviation.
        transform_weight: Mixture weight on the genuine transformation.
    """

    n_identities: int = 12
    n_repeats: int = 8
    n_voxels: int = 200
    latent_dim: int = 5
    noise: float = 1.0
    transform_weight: float = 0.5


def simulate(cfg: SimConfig, world: World, seed: int = 0):
    """Generate paired perception/imagery trials under a known world.

    Returns:
        ``(perception, imagery, identity)`` with shapes ``(T, V)``, ``(T, V)``
        and ``(T,)`` where ``T = n_identities * n_repeats``.
    """
    rng = np.random.default_rng(seed)
    n_id, n_rep, V = cfg.n_identities, cfg.n_repeats, cfg.n_voxels

    # Each stimulus has a mean perception pattern and a mean imagery pattern.
    perc_mean = rng.standard_normal((n_id, V))
    img_template = rng.standard_normal((n_id, V))

    # A genuine low-rank transformation from single-trial perception -> imagery.
    W = (rng.standard_normal((V, cfg.latent_dim)) @ rng.standard_normal((cfg.latent_dim, V))) / np.sqrt(V)

    identity = np.repeat(np.arange(n_id), n_rep)
    perception = perc_mean[identity] + cfg.noise * rng.standard_normal((n_id * n_rep, V))

    if world == "identity_template":
        # Imagery depends ONLY on identity. The perception TRIAL is irrelevant.
        imagery = img_template[identity] + cfg.noise * rng.standard_normal((n_id * n_rep, V))
    elif world == "true_transform":
        # Imagery is a function of the single perception trial.
        imagery = perception @ W + cfg.noise * rng.standard_normal((n_id * n_rep, V))
    elif world == "mixture":
        w = cfg.transform_weight
        imagery = (
            w * (perception @ W)
            + (1 - w) * img_template[identity]
            + cfg.noise * rng.standard_normal((n_id * n_rep, V))
        )
    else:
        raise ValueError(f"unknown world: {world!r}")

    return perception, imagery, identity


def _reduced_rank_fit(X: np.ndarray, Y: np.ndarray, rank: int, ridge: float = 1.0) -> np.ndarray:
    """Ridge-regularised reduced-rank regression, as in the vis2img model."""
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    B = np.linalg.solve(XtX, X.T @ Y)
    U, S, Vt = np.linalg.svd(X @ B, full_matrices=False)
    r = min(rank, S.size)
    P = Vt[:r].T @ Vt[:r]
    return B @ P


def _mean_voxel_r(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Mean across voxels of the Pearson r between predicted and measured."""
    yt = Y_true - Y_true.mean(0)
    yp = Y_pred - Y_pred.mean(0)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt**2).sum(0) * (yp**2).sum(0)) + 1e-12
    return float(np.mean(num / den))


def evaluate(
    cfg: SimConfig,
    world: World,
    split: Literal["repeat", "identity_heldout"],
    rank: int = 5,
    seed: int = 0,
) -> Tuple[float, float]:
    """Fit vis2img and an identity-template baseline under one split protocol.

    ``split="repeat"`` reproduces Roy et al.'s protocol: every stimulus identity
    appears in both train and test, only the *repeats* differ.
    ``split="identity_heldout"`` holds out whole identities, so no test stimulus
    was ever seen during fitting.

    Returns:
        ``(r_vis2img, r_identity_baseline)``. The baseline is NaN under the
        identity-held-out split, where it is undefined by construction -- a held
        out identity has no training repeats to average.
    """
    P, I, ident = simulate(cfg, world, seed=seed)

    if split == "repeat":
        # Every identity in both halves; split on repeat index.
        rep_idx = np.tile(np.arange(cfg.n_repeats), cfg.n_identities)
        train = rep_idx < cfg.n_repeats // 2
    else:
        # Whole identities held out.
        n_train_id = max(1, cfg.n_identities // 2)
        train = ident < n_train_id

    test = ~train
    W_hat = _reduced_rank_fit(P[train], I[train], rank=rank)
    r_model = _mean_voxel_r(I[test], P[test] @ W_hat)

    if split == "repeat":
        # B-ID1: per-identity mean of TRAINING imagery repeats.
        templates = np.zeros((cfg.n_identities, cfg.n_voxels))
        for k in range(cfg.n_identities):
            m = train & (ident == k)
            templates[k] = I[m].mean(0) if m.any() else 0.0
        r_baseline = _mean_voxel_r(I[test], templates[ident[test]])
    else:
        r_baseline = float("nan")  # undefined: no training repeats of a held-out identity

    return r_model, r_baseline
