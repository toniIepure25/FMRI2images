"""Contracts for an independent method reproduction of Roy et al. (2025).

This package implements **only components explicitly specified in the paper**. Every
unresolved implementation choice (see
``docs/research/mindcompiler/30_ROY_IMPLEMENTATION_AMBIGUITY_REGISTRY.csv``) is
represented as an explicit policy whose default demands clarification, so no silent
default can ever resolve an ambiguity.

Scope, stated honestly (see `27_ROY_S1_DATA_AND_CODE_INVENTORY.md`):

* ``ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE`` -- no public author implementation located.
* ``INDEPENDENT_METHOD_REPRODUCTION_FEASIBLE_AFTER_DATA_ACCESS`` -- this package.

This is not a claim that reproduction is "impossible in principle" (that earlier
statement was withdrawn); it is a method reproduction with preregistered sensitivity
analyses over the ambiguous components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- Explicitly specified constants (paper Methods) -------------------------

#: Voxel inclusion: > this percentile of NSD-core voxelwise SNR, within each ROI.
VOXEL_SNR_PERCENTILE: float = 98.0

#: ROIs analysed.
ROIS: tuple[str, ...] = ("V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal")

#: Ridge grid: 100 log-spaced values from 1e-3 to 1e5, selected on validation.
RIDGE_GRID_N: int = 100
RIDGE_GRID_LO: float = 1e-3
RIDGE_GRID_HI: float = 1e5

#: Repeat allocation per stimulus identity (of 8): train / validation / test.
SPLIT_TRAIN_REPEATS: int = 4
SPLIT_VAL_REPEATS: int = 2
SPLIT_TEST_REPEATS: int = 2

#: Rank selection targets performance within this fraction of peak validation r.
RANK_SELECTION_FRACTION_OF_PEAK: float = 0.99


# --- Ambiguity policies: default forces explicit clarification --------------

DenoisingPolicy = Literal[
    "no_denoising",              # D0
    "strict_out_of_fold",        # D1
    "fold_fit_all_outputs",      # D2
    "author_clarification_required",  # default / D3-pending
]

PairingPolicy = Literal[
    "single_seed",
    "average_over_realizations",
    "author_clarification_required",
]

RankTieBreak = Literal[
    "smallest_rank_at_threshold",
    "argmax_validation",
    "author_clarification_required",
]

RidgeTieBreak = Literal["argmax_validation", "one_se_rule", "author_clarification_required"]


@dataclass(frozen=True)
class ReproductionPolicy:
    """Every ambiguous choice is explicit; defaults refuse to guess.

    A run that leaves any field at ``author_clarification_required`` must fail loudly
    at construction of the analysis, never proceed on a silent default.
    """

    denoising: DenoisingPolicy = "author_clarification_required"
    pairing: PairingPolicy = "author_clarification_required"
    rank_tie_break: RankTieBreak = "author_clarification_required"
    ridge_tie_break: RidgeTieBreak = "author_clarification_required"
    n_pairing_realizations: int = 0            # 0 = unset; must be chosen explicitly
    n_null_permutations: int = 1000            # our floor, not the paper's (unspecified)
    center_on_train_only: bool = True          # leakage-safe default (registry: partial)
    pooled_out_of_fold: bool = True            # registry default

    def unresolved(self) -> list[str]:
        """List policy fields still demanding clarification."""
        bad = []
        for name in ("denoising", "pairing", "rank_tie_break", "ridge_tie_break"):
            if getattr(self, name) == "author_clarification_required":
                bad.append(name)
        if self.n_pairing_realizations <= 0:
            bad.append("n_pairing_realizations")
        return bad

    def require_resolved(self) -> None:
        """Raise unless every ambiguity has been given an explicit value.

        This is the guard that makes silent ambiguity resolution impossible.
        """
        missing = self.unresolved()
        if missing:
            raise ValueError(
                "ReproductionPolicy has unresolved ambiguities that must be set "
                f"explicitly (no silent defaults): {missing}. See "
                "30_ROY_IMPLEMENTATION_AMBIGUITY_REGISTRY.csv."
            )


#: The preregistered denoising sensitivity variants (compare all; never cherry-pick).
DENOISING_VARIANTS: tuple[DenoisingPolicy, ...] = (
    "no_denoising",
    "strict_out_of_fold",
    "fold_fit_all_outputs",
)
