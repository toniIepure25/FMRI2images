"""Matched auxiliary-control objectives for the NCD arm family.

Implements the control protocol in
``docs/research/pcd_program/25_MATCHED_AUXILIARY_CONTROL_PROTOCOL.md``, which exists
to answer one question: does the **neural-prediction target** matter, or does **any**
matched auxiliary objective regularize the decoder?

A masked-ROI objective is an auxiliary task, and auxiliary tasks regularize. Without
these controls a positive ARM-B result is uninterpretable, and the paper does not
survive review (objection O-1).

Arms
----
======  ==========================  ====================================================
Arm     ``AuxObjective``            Isolates
======  ==========================  ====================================================
A       ``NONE``                    the floor (retrieval only)
B       ``MASKED_NEURAL``           the hypothesis
C       ``SHUFFLED_NEURAL``         stimulus-specific content vs marginal/connectivity
D       ``RANDOM_TARGET``           any target vs a neural target
E       ``SELF_RECONSTRUCTION``     anatomical structure vs generic reconstruction
F       ``NONE`` + tuned reg        generic regularization (the decisive comparison)
G       ``MASKED_NEURAL`` + pseudo  anatomical grouping vs merely having groups
H       ``ROI_AUTOENCODE``          ROI organisation vs cross-ROI predictive dependency
======  ==========================  ====================================================

Capacity parity is a hard requirement: every arm allocates the same auxiliary heads
even when its objective is disabled, so parameter counts match across arms. The
objective selects the **target**, never the head.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


class AuxObjective(str, Enum):
    """Auxiliary objective selecting what the ROI heads are trained to predict."""

    NONE = "none"
    MASKED_NEURAL = "masked_neural"
    SHUFFLED_NEURAL = "shuffled_neural"
    RANDOM_TARGET = "random_target"
    SELF_RECONSTRUCTION = "self_reconstruction"
    ROI_AUTOENCODE = "roi_autoencode"

    @property
    def uses_masking(self) -> bool:
        """Whether the objective needs masked ROI context.

        ``ROI_AUTOENCODE`` (arm H) deliberately does not: it predicts each ROI from
        its own token, so no ROI is ever predicted from another. Masks are still
        *drawn* for every arm to hold masking frequency constant (protocol §1).
        """
        return self in {
            AuxObjective.MASKED_NEURAL,
            AuxObjective.SHUFFLED_NEURAL,
            AuxObjective.RANDOM_TARGET,
            AuxObjective.SELF_RECONSTRUCTION,
        }

    @property
    def is_neural_target(self) -> bool:
        """Whether the target is real recorded activity for the predicted ROI."""
        return self in {AuxObjective.MASKED_NEURAL, AuxObjective.ROI_AUTOENCODE}


def stable_seed(*parts: object) -> int:
    """Derive a deterministic 32-bit seed from arbitrary run identifiers.

    ``hash()`` is salted per process in Python, so it cannot be used for anything
    that must reproduce across runs or appear in a manifest.

    Args:
        *parts: Values whose ``str`` forms identify the run (split hash, seed, ...).

    Returns:
        A seed in ``[0, 2**32)``.
    """
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class DeterministicImagePermutation:
    """Fixed permutation over unique image IDs, for ARM-C (shuffled neural targets).

    ARM-C keeps the real ROI-activity target distribution but destroys its
    stimulus-specific content, by pairing each image with a *different* image's
    activity. If ARM-B does not beat ARM-C, the objective is exploiting marginal
    statistics or inter-ROI connectivity rather than stimulus-specific neural
    content (reviewer objection O-3).

    Two properties are essential and easy to get wrong:

    * **All trials of one image map to the same permuted image.** Permuting per
      *trial* would let repeated presentations leak the true target.
    * **The permutation is derangement-like**: no image maps to itself (where the
      set allows), or the arm silently becomes ARM-B for those images.

    Args:
        image_ids: Unique image IDs in the training split.
        seed: Derived via :func:`stable_seed` from split hash and run seed.
    """

    def __init__(self, image_ids: Sequence[int], seed: int) -> None:
        uniq = sorted(set(int(i) for i in image_ids))
        if len(uniq) < 2:
            raise ValueError("need >= 2 unique images to permute")

        g = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(uniq), generator=g).tolist()

        # Cyclic shift of a random order: a derangement by construction, so no
        # image can be paired with itself.
        shifted = order[1:] + order[:1]
        self._map: Dict[int, int] = {uniq[a]: uniq[b] for a, b in zip(order, shifted)}
        self.seed = int(seed)
        self.n_images = len(uniq)

        assert all(k != v for k, v in self._map.items()), "permutation must be a derangement"
        logger.info(
            "DeterministicImagePermutation: %d images, seed=%d (derangement)",
            self.n_images,
            self.seed,
        )

    def __call__(self, image_id: int) -> int:
        """Map an image ID to its permuted partner."""
        return self._map[int(image_id)]

    def map_batch(self, image_ids: torch.Tensor) -> torch.Tensor:
        """Map a batch of image IDs.

        Args:
            image_ids: (B,) integer image IDs.

        Returns:
            (B,) permuted image IDs on the same device.
        """
        return torch.tensor(
            [self._map[int(i)] for i in image_ids.tolist()],
            dtype=image_ids.dtype,
            device=image_ids.device,
        )

    def manifest(self) -> Dict[str, object]:
        """Provenance record for the run manifest."""
        return {
            "kind": "deterministic_image_permutation",
            "seed": self.seed,
            "n_images": self.n_images,
            "is_derangement": True,
        }


def make_pseudo_roi_groups(
    roi_sizes: "OrderedDict[str, int]",
    total_voxels: int,
    seed: int,
) -> "OrderedDict[str, torch.Tensor]":
    """Random voxel groups matched to real ROIs, for ARM-G.

    ARM-G asks whether *anatomical* grouping matters or whether merely *having*
    groups of the right shape is sufficient. The pseudo-ROIs therefore match the
    real ROIs in group count and in each group's dimensionality, which holds the
    parameter budget and masking probability constant.

    Multiple seeds are required in use: a single fixed shuffle is an anecdote, not
    a null distribution. This repeats the ``random.Random(42)`` defect found in PCD
    (``03_HYPOTHESES.md`` H2) if called with one seed.

    Args:
        roi_sizes: Real ROI name -> voxel count, in canonical order.
        total_voxels: Total voxels available to partition.
        seed: Seed for the partition.

    Returns:
        Pseudo-ROI name -> voxel index tensor, matching ``roi_sizes`` shapes.

    Raises:
        ValueError: If the ROI sizes do not fit in ``total_voxels``.
    """
    needed = sum(roi_sizes.values())
    if needed > total_voxels:
        raise ValueError(
            f"pseudo-ROI sizes need {needed} voxels but only {total_voxels} available"
        )

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total_voxels, generator=g)

    groups: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    cursor = 0
    for name, size in roi_sizes.items():
        groups[f"pseudo_{name}"] = perm[cursor : cursor + size].clone().sort().values
        cursor += size

    logger.info(
        "make_pseudo_roi_groups: %d groups matched to real ROI sizes, seed=%d",
        len(groups),
        seed,
    )
    return groups


class RandomTargetBank(torch.nn.Module):
    """Fixed random prediction targets for ARM-D.

    ARM-D asks whether *any* prediction target regularizes, independent of neural
    content. Targets are fixed (not resampled per step) so the task is learnable,
    and are variance-matched to the real activity they replace so the loss scale is
    comparable (protocol §1).

    Stored as buffers so they are checkpointed and reproduce exactly on reload.

    Args:
        roi_sizes: ROI name -> voxel count (the dimensionality to match).
        seed: Seed for the fixed targets.
        target_std: Standard deviation to match, from the real activity.
    """

    def __init__(
        self,
        roi_sizes: "OrderedDict[str, int]",
        seed: int = 0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.roi_names: List[str] = list(roi_sizes)
        self.target_std = float(target_std)
        g = torch.Generator().manual_seed(int(seed))
        for i, (name, size) in enumerate(roi_sizes.items()):
            t = torch.randn(size, generator=g) * target_std
            self.register_buffer(f"_target_{i}", t)

    def forward(self, roi_name: str, batch_size: int) -> torch.Tensor:
        """
        Args:
            roi_name: ROI whose fixed target is requested.
            batch_size: Batch dimension to expand to.

        Returns:
            (batch_size, n_voxels) fixed random target.
        """
        i = self.roi_names.index(roi_name)
        return getattr(self, f"_target_{i}").unsqueeze(0).expand(batch_size, -1)


def assert_param_parity(
    counts: Dict[str, int],
    tolerance: float = 0.02,
) -> None:
    """Assert arm parameter counts fall within the matching tolerance.

    The control protocol permits a documented spread of at most 2%. A larger spread
    means an arm has a capacity advantage and the comparison is confounded.

    Args:
        counts: Arm name -> trainable parameter count.
        tolerance: Maximum permitted relative spread.

    Raises:
        AssertionError: If the spread exceeds ``tolerance``.
    """
    if not counts:
        raise ValueError("no arms supplied")
    lo, hi = min(counts.values()), max(counts.values())
    spread = (hi - lo) / lo if lo else float("inf")
    if spread > tolerance:
        detail = ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))
        raise AssertionError(
            f"arm parameter spread {spread:.2%} exceeds tolerance {tolerance:.2%} "
            f"({detail}). Arms are not capacity-matched; any comparison between "
            "them is confounded (protocol 25 section 1)."
        )
    logger.info("assert_param_parity: %d arms within %.2f%% spread", len(counts), spread * 100)
