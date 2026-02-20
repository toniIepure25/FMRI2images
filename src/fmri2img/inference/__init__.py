"""Inference utilities for probabilistic decoding (allocation + selection)."""

from .vmf_mixture import (
    sample_vmf_mixture,
    compute_mixture_energy_score,
    select_best_from_mixture,
    VMFMixtureSamples,
)
from .decomposed_ua_cfg import DecomposedUACFG, DUACFGConfig

__all__ = [
    "sample_vmf_mixture",
    "compute_mixture_energy_score",
    "select_best_from_mixture",
    "VMFMixtureSamples",
    "DecomposedUACFG",
    "DUACFGConfig",
]
