"""
I/O utilities for NSD data access.

Provides filesystem abstraction (local / S3), NIfTI loading,
NSD layout resolution, and robust image loading with fallbacks.
"""

from .s3 import (
    get_s3_filesystem,
    NIfTILoader,
    HDF5Loader,
    CSVLoader,
)
from .nsd_layout import NSDLayout, get_nsd_layout
from .image_loader import RobustImageLoader

__all__ = [
    "get_s3_filesystem",
    "NIfTILoader",
    "HDF5Loader",
    "CSVLoader",
    "NSDLayout",
    "get_nsd_layout",
    "RobustImageLoader",
]
