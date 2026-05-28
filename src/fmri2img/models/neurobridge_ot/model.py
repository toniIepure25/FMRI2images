"""
NeuroBridge-OT Model — Top-Level Architecture
================================================

Integrates all modules (A-H) into a single nn.Module:
    A. ROI Tokenizer -> B. Subject Fingerprint -> C. OT Alignment ->
    D. Semantic Transformer -> E. Decoding Heads

With optional:
    F. Teacher distillation (external, applied in training loop).
    G. Hyper-adapter modulation.
    H. Composite loss.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fmri2img.models.neurobridge_ot.decoding_heads import DecodingHeads
from fmri2img.models.neurobridge_ot.hyper_adapter import HyperNetwork
from fmri2img.models.neurobridge_ot.optimal_transport import OptimalTransportAlignment
from fmri2img.models.neurobridge_ot.roi_tokenizer import create_roi_tokenizer
from fmri2img.models.neurobridge_ot.semantic_transformer import SemanticTransformer
from fmri2img.models.neurobridge_ot.subject_fingerprint import SubjectFingerprint

logger = logging.getLogger(__name__)


class NeuroBridgeOTModel(nn.Module):
    """NeuroBridge-OT: Subject-Adaptive Optimal-Transport ROI Token Decoder.

    End-to-end architecture for fMRI-to-CLIP visual decoding with:
    - Learned ROI tokenization (variable subject anatomy).
    - Optimal-transport alignment to canonical cortical space.
    - Shared semantic transformer backbone.
    - Multi-target decoding heads with vMF uncertainty.
    - Optional hyper-adapter for few-shot subject adaptation.

    Args:
        config: Full model configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Core dimensions
        self.d_model = config.get("d_model", 768)
        self.output_dim = config.get("output_dim", 768)
        self.n_rois = config.get("n_rois", 17)
        n_canonical = config.get("n_canonical_tokens", self.n_rois)

        # A. ROI Tokenizer
        tokenizer_config = config.get("tokenizer", {})
        tokenizer_config.setdefault("n_rois", self.n_rois)
        tokenizer_config.setdefault("d_model", self.d_model)
        self.tokenizer = create_roi_tokenizer(tokenizer_config)

        # B. Subject Fingerprint (optional, needed for hyper-adapter)
        self.use_fingerprint = config.get("use_fingerprint", False)
        if self.use_fingerprint:
            fp_config = config.get("fingerprint", {})
            self.fingerprint = SubjectFingerprint(
                n_rois=self.n_rois,
                fingerprint_dim=fp_config.get("fingerprint_dim", 128),
                n_subjects=fp_config.get("n_subjects", 8),
                use_subject_embedding=fp_config.get("use_subject_embedding", True),
                roi_stat_dim=fp_config.get("roi_stat_dim", 5),
            )

        # C. Optimal Transport Alignment
        ot_config = config.get("optimal_transport", {})
        self.ot_alignment = OptimalTransportAlignment(
            n_canonical_tokens=n_canonical,
            d_model=self.d_model,
            n_source_tokens=self.n_rois,
            epsilon=ot_config.get("epsilon", 0.1),
            n_sinkhorn_iters=ot_config.get("n_sinkhorn_iters", 50),
            cost_type=ot_config.get("cost_type", "cosine"),
            use_anatomical_prior=ot_config.get("use_anatomical_prior", False),
            alignment_mode=ot_config.get("alignment_mode", "ot"),
        )

        # D. Semantic Transformer
        trans_config = config.get("transformer", {})
        self.transformer = SemanticTransformer(
            d_model=self.d_model,
            n_heads=trans_config.get("n_heads", 12),
            n_layers=trans_config.get("n_layers", 6),
            dim_feedforward=trans_config.get("dim_feedforward"),
            dropout=trans_config.get("dropout", 0.1),
            drop_path_rate=trans_config.get("drop_path_rate", 0.15),
            n_canonical_tokens=n_canonical,
            n_subjects=trans_config.get("n_subjects", 8),
            use_subject_embedding=trans_config.get("use_subject_embedding", False),
            activation=trans_config.get("activation", "gelu"),
            gradient_checkpointing=trans_config.get("gradient_checkpointing", False),
        )

        # E. Decoding Heads
        heads_config = config.get("heads", {})
        heads_config.setdefault("d_model", self.d_model)
        heads_config.setdefault("output_dim", self.output_dim)
        heads_config.setdefault("n_canonical_tokens", n_canonical)
        self.heads = DecodingHeads(heads_config)

        # G. Hyper-Adapter (optional)
        self.use_hyper_adapter = config.get("use_hyper_adapter", False)
        if self.use_hyper_adapter:
            ha_config = config.get("hyper_adapter", {})
            fp_dim = config.get("fingerprint", {}).get("fingerprint_dim", 128)
            self.hyper_network = HyperNetwork(
                fingerprint_dim=fp_dim,
                d_model=self.d_model,
                n_target_layers=trans_config.get("n_layers", 6),
                adapter_type=ha_config.get("adapter_type", "film"),
                lora_rank=ha_config.get("lora_rank", 8),
                hidden_dim=ha_config.get("hidden_dim", 512),
            )

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "NeuroBridgeOTModel: total_params=%d, trainable=%d", n_params, n_trainable
        )

    def forward(
        self,
        fmri: Tensor,
        roi_indices: Dict[str, Tensor],
        subject_id: Optional[Tensor] = None,
        roi_stats: Optional[Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Tensor]:
        """Full forward pass through NeuroBridge-OT.

        Args:
            fmri: Raw fMRI voxel activations, shape (B, V_subj).
            roi_indices: Dict mapping ROI names to voxel index tensors.
            subject_id: Subject identifiers, shape (B,).
            roi_stats: Precomputed ROI statistics for fingerprint, shape (B, n_rois, stat_dim).
            return_intermediates: Whether to include intermediate tensors in output.

        Returns:
            Dict with model outputs:
                'clip_embedding': (B, output_dim) normalized CLIP prediction.
                'vmf_mu': (B, output_dim) vMF mean direction (if enabled).
                'vmf_kappa': (B,) vMF concentration (if enabled).
                'ot_cost': Scalar OT transport cost.
                'transport_matrix': (B, M, K) transport plan.
                'cls_embedding': (B, d_model) raw CLS embedding.
                Plus optional token predictions, confidence, intermediates.
        """
        # B. Subject fingerprint (optional)
        adapters = None
        fingerprint = None
        if self.use_fingerprint and roi_stats is not None:
            fingerprint = self.fingerprint(roi_stats, subject_id)
            if self.use_hyper_adapter:
                adapters = self.hyper_network(fingerprint)

        # A. ROI Tokenization
        roi_tokens, roi_mask = self.tokenizer(
            fmri, roi_indices, subject_id=subject_id, adapters=adapters
        )  # (B, n_rois, d_model), (B, n_rois)

        # C. Optimal Transport Alignment
        ot_output = self.ot_alignment(roi_tokens, roi_mask)
        aligned_tokens = ot_output["aligned_tokens"]  # (B, K, d_model)
        ot_cost = ot_output["ot_cost"]
        transport_matrix = ot_output["transport_matrix"]

        # D. Semantic Transformer
        transformer_output = self.transformer(
            aligned_tokens,
            subject_id=subject_id,
            token_mask=None,  # All aligned tokens are valid after OT
            return_all_layers=return_intermediates,
        )
        cls_embedding = transformer_output["cls_embedding"]  # (B, d_model)
        ctx_roi_tokens = transformer_output["roi_tokens"]  # (B, K, d_model)

        # E. Decoding Heads
        head_outputs = self.heads(cls_embedding, ctx_roi_tokens)

        # Assemble final outputs
        outputs: Dict[str, Tensor] = {
            "clip_embedding": head_outputs["clip_embedding"],
            "cls_embedding": cls_embedding,
            "ot_cost": ot_cost,
            "transport_matrix": transport_matrix,
        }

        if "vmf_mu" in head_outputs:
            outputs["vmf_mu"] = head_outputs["vmf_mu"]
            outputs["vmf_kappa"] = head_outputs["vmf_kappa"]

        if "token_predictions" in head_outputs:
            outputs["token_predictions"] = head_outputs["token_predictions"]

        if "confidence" in head_outputs:
            outputs["confidence"] = head_outputs["confidence"]

        if return_intermediates:
            outputs["roi_tokens_raw"] = roi_tokens
            outputs["roi_mask"] = roi_mask
            outputs["aligned_tokens"] = aligned_tokens
            outputs["ctx_roi_tokens"] = ctx_roi_tokens
            outputs["canonical_bank"] = ot_output["canonical_bank"]
            if fingerprint is not None:
                outputs["fingerprint"] = fingerprint
            if "all_layers" in transformer_output:
                outputs["all_layers"] = transformer_output["all_layers"]

        return outputs
