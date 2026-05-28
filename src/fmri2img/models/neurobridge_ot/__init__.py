"""
NeuroBridge-OT: Subject-Adaptive Optimal-Transport ROI Token Decoder
====================================================================

A novel architecture for fMRI-to-CLIP visual decoding that bridges
high-performance subject-specific retrieval and practical cross-subject
transfer via learned ROI tokens, optimal-transport alignment, and
few-shot hyper-adapters.
"""

from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel

__all__ = ["NeuroBridgeOTModel"]
