"""
Data Package
============

Data loading, preprocessing, and indexing for NSD dataset.
"""

# Preprocessing
from .preprocess import NSDPreprocessor

# CLIP cache
from .clip_cache import CLIPCache

# Data loaders
from .loaders import (
    DataLoaderFactory,
    FMRIDataset,
    train_val_test_split,
    extract_features_and_targets
)

# Index utilities
from .nsd_index_reader import read_subject_index

__all__ = [
    # Preprocessing
    "NSDPreprocessor",
    
    # CLIP
    "CLIPCache",
    
    # Data loaders
    "DataLoaderFactory",
    "FMRIDataset",
    "train_val_test_split",
    "extract_features_and_targets",
    
    # Index
    "read_subject_index",
]
