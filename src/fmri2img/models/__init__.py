"""
fMRI-to-Image Models
===================

Neural and linear models for mapping fMRI activity to image representations.
"""

from .ridge import RidgeEncoder
from .clip_adapter import CLIPAdapter, save_adapter, load_adapter

__all__ = ["RidgeEncoder", "CLIPAdapter", "save_adapter", "load_adapter"]
