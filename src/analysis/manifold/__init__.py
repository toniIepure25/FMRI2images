"""Manifold analysis modules for fMRI-to-CLIP neural decoding evaluation.

Provides modular, publication-quality analysis of decoded brain embeddings:
retrieval, hubness, RSA, neighborhood preservation, repeat stability,
uncertainty calibration, semantic probing, and aggregate scoring (NMAS).
"""

from analysis.manifold.config import ManifoldAnalysisConfig, load_config
from analysis.manifold.data_loading import AnalysisData, load_analysis_data

__all__ = [
    "ManifoldAnalysisConfig",
    "load_config",
    "AnalysisData",
    "load_analysis_data",
]
