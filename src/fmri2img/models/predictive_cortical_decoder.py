"""
Predictive Cortical Decoder (PCD)
==================================

A novel architecture that models the visual processing hierarchy as a
predictive coding system (Rao & Ballard, 1999).  Instead of treating all
brain regions equally, PCD groups ROIs into four levels mirroring the
known ventral visual stream hierarchy and propagates only **prediction
errors** from lower to higher levels.

Architecture:
    fMRI → ROI tokenisation → Level-wise encoding with top-down prediction
    → Hierarchical error-driven aggregation → vMF decoder (mu, kappa)

Hierarchical levels (from the 17 NSD ROIs):
    Level 1 (Early Visual)      : V1v, V1d, V2v, V2d
    Level 2 (Mid Visual)        : V3v, V3d, V3A, V3B, V4
    Level 3 (Category-Selective): FFA1, FFA2, PPA, EBA, OFA, OPA, RSC
    Level 4 (Residual/Global)   : nsdgeneral_other

Neuroscience hypothesis tested: hierarchical prediction errors along the
ventral stream carry more information than raw activations for visual
decoding.  If true, this validates predictive coding theory in the
decoding setting.

References:
    - Rao, R.P. & Ballard, D.H. (1999) Predictive coding in visual cortex
    - Friston, K. (2005) A theory of cortical responses
    - Clark, A. (2013) Whatever next? Predictive brains, situated agents
    - Allen et al. (2022) NSD dataset
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmri2img.models.vmf_decoder import VonMisesFisherDecoder, kappa_activation

logger = logging.getLogger(__name__)

# Canonical hierarchy mapping: ROI name → level index (0-based)
HIERARCHY_LEVELS: Dict[str, int] = {
    "V1v": 0, "V1d": 0, "V2v": 0, "V2d": 0,
    "V3v": 1, "V3d": 1, "V3A": 1, "V3B": 1, "V4": 1,
    "FFA1": 2, "FFA2": 2, "PPA": 2, "EBA": 2, "OFA": 2, "OPA": 2, "RSC": 2,
    "nsdgeneral_other": 3,
}

LEVEL_NAMES: List[str] = [
    "early_visual",    # V1/V2
    "mid_visual",      # V3/V4
    "category_select", # FFA/PPA/EBA/OFA/OPA/RSC
    "residual",        # nsdgeneral_other
]

N_LEVELS = 4


@dataclass
class PCDOutput:
    """Container for PCD forward pass outputs.

    Attributes:
        mu:              (B, D) fused mean direction on the hypersphere.
        kappa:           (B, 1) fused concentration parameter.
        level_outputs:   List of (B, d_model) per-level hidden states.
        prediction_errors: List of (B, n_tokens_l, d_model) raw prediction
                           error tensors for levels 1-2 (3 entries, None for L0).
        level_kappas:    (B, N_LEVELS) per-level concentration parameters.
        level_weights:   (B, N_LEVELS) learned aggregation weights (softmax).
    """
    mu: torch.Tensor
    kappa: torch.Tensor
    level_outputs: List[torch.Tensor] = field(default_factory=list)
    prediction_errors: List[Optional[torch.Tensor]] = field(default_factory=list)
    level_kappas: Optional[torch.Tensor] = None
    level_weights: Optional[torch.Tensor] = None


class ROIProjection(nn.Module):
    """Project variable-size ROI voxels to a fixed-dim token."""

    def __init__(self, n_voxels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_voxels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LevelTransformerEncoder(nn.Module):
    """Small Transformer encoder for a single hierarchical level.

    Processes ``n_tokens`` ROI tokens (from a single level) through a
    lightweight Transformer.  A learnable [CLS] token aggregates
    information, producing a single (B, d_model) level representation.

    Args:
        d_model:        Hidden / token dimension.
        nhead:          Number of attention heads.
        num_layers:     Number of Transformer encoder layers.
        dropout:        Dropout probability.
        max_tokens:     Maximum number of input tokens (for pos embed sizing).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_tokens: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_tokens + 1, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, n_tokens, d_model)

        Returns:
            cls_out: (B, d_model) — [CLS] representation for this level.
        """
        B, T, _ = tokens.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)
        x = x + self.pos_embed[:, :T + 1, :]
        x = self.transformer(x)
        return self.norm(x[:, 0])  # [CLS]


class PredictionHead(nn.Module):
    """Cross-level prediction head: predict next level's token representations.

    Given the current level's [CLS] representation, predicts the expected
    token embeddings for the next level.  The prediction error (actual
    minus predicted) is what the next level's encoder processes.

    Args:
        d_model:     Hidden dimension.
        n_target_tokens: Number of tokens in the target (next) level.
        dropout:     Dropout probability.
    """

    def __init__(self, d_model: int, n_target_tokens: int, dropout: float = 0.1):
        super().__init__()
        self.n_target = n_target_tokens
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_target_tokens * d_model),
        )
        self.d_model = d_model

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_model) — current level's [CLS] representation.

        Returns:
            predicted: (B, n_target_tokens, d_model) — predicted token
                       embeddings for the next level.
        """
        out = self.predictor(h)
        return out.view(-1, self.n_target, self.d_model)


class HierarchicalAggregator(nn.Module):
    """Attention-weighted aggregation of per-level representations.

    Learns data-dependent weights over the 4 level representations via a
    single-head attention mechanism (query is a learnable vector).

    Args:
        d_model:   Hidden dimension.
        n_levels:  Number of hierarchy levels (default 4).
        dropout:   Dropout probability.
    """

    def __init__(self, d_model: int, n_levels: int = N_LEVELS, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(
        self, level_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            level_outputs: List of (B, d_model) tensors, one per level.

        Returns:
            aggregated: (B, d_model) — weighted combination.
            weights:    (B, n_levels) — level attention weights.
        """
        B = level_outputs[0].shape[0]
        stacked = torch.stack(level_outputs, dim=1)  # (B, L, d_model)

        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        k = self.key_proj(stacked)         # (B, L, d_model)
        v = self.value_proj(stacked)       # (B, L, d_model)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, 1, L)
        weights = F.softmax(attn, dim=-1)                      # (B, 1, L)
        out = torch.bmm(weights, v).squeeze(1)                 # (B, d_model)

        return self.norm(out), weights.squeeze(1)


class PerLevelKappaHeads(nn.Module):
    """Produces a scalar kappa for each hierarchical level.

    Used for per-level uncertainty decomposition: measuring which levels
    of the visual hierarchy are most confident for a given stimulus.

    Args:
        d_model:    Hidden dimension.
        n_levels:   Number of levels.
        kappa_min:  Lower bound for kappa.
        kappa_max:  Upper bound for kappa.
        kappa_mode: Kappa activation mode.
    """

    def __init__(
        self,
        d_model: int,
        n_levels: int = N_LEVELS,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        kappa_mode: str = "softplus",
    ):
        super().__init__()
        self.n_levels = n_levels
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_mode = kappa_mode
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_levels)
        ])

    def forward(self, level_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            level_outputs: List of (B, d_model) tensors.

        Returns:
            level_kappas: (B, n_levels) — per-level concentrations.
        """
        kappas = []
        for i, (head, h) in enumerate(zip(self.heads, level_outputs)):
            raw = head(h)
            k = kappa_activation(
                raw, mode=self.kappa_mode,
                kappa_min=self.kappa_min, kappa_max=self.kappa_max,
            )
            kappas.append(k.squeeze(-1))
        return torch.stack(kappas, dim=1)  # (B, n_levels)


class PredictiveCorticalDecoder(nn.Module):
    """Predictive Cortical Decoder — full architecture.

    Implements the predictive coding processing pipeline:

        1. ROI tokenisation: flat fMRI → per-ROI tokens
        2. Group tokens by hierarchical level
        3. Level 1: encode early visual tokens → h1
        4. Predict level 2 tokens from h1; compute error = actual - predicted
        5. Level 2: encode prediction errors → h2
        6. Predict level 3 from h2; compute error
        7. Level 3: encode prediction errors → h3
        8. Level 4: encode residual tokens directly → h4
        9. Aggregate h1..h4 via learned attention → h_fused
       10. vMF decoder → (mu, kappa)

    Supports multi-subject training via per-subject ROI projections with
    a shared predictive backbone.

    Args:
        roi_indices:     OrderedDict mapping ROI name → index array.
        d_model:         Hidden / token dimension (default 768).
        nhead:           Number of attention heads per level encoder.
        layers_per_level: Number of Transformer layers per level encoder.
        dropout:         Dropout probability.
        output_dim:      CLIP embedding dimension (default 768).
        kappa_min:       Lower bound for kappa.
        kappa_max:       Upper bound for kappa.
        kappa_mode:      Kappa activation mode.
        enable_per_level_kappa: Whether to produce per-level kappas.
        ablation_mode:   One of "full", "no_errors", "reversed", "random".
                         Controls the predictive coding component for ablations.
        subject_roi_indices: Dict[subject_id, OrderedDict] for multi-subject.
    """

    def __init__(
        self,
        roi_indices: Optional[OrderedDict] = None,
        d_model: int = 768,
        nhead: int = 12,
        layers_per_level: int = 2,
        dropout: float = 0.1,
        output_dim: int = 768,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        kappa_mode: str = "softplus",
        enable_per_level_kappa: bool = True,
        ablation_mode: str = "full",
        subject_roi_indices: Optional[Dict[str, OrderedDict]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.ablation_mode = ablation_mode
        self.enable_per_level_kappa = enable_per_level_kappa
        self._multi_subject = subject_roi_indices is not None

        if self._multi_subject:
            self._init_multi_subject(subject_roi_indices, d_model, dropout)
            sample_subj = next(iter(subject_roi_indices))
            sample_indices = subject_roi_indices[sample_subj]
        else:
            assert roi_indices is not None, "roi_indices required for single-subject"
            self._init_single_subject(roi_indices, d_model, dropout)
            sample_indices = roi_indices

        self.roi_names = list(sample_indices.keys())
        self.n_rois = len(self.roi_names)

        # Build hierarchy grouping from ROI names
        self._level_roi_indices = self._build_level_groups(self.roi_names)
        self._n_tokens_per_level = [len(rois) for rois in self._level_roi_indices]

        # Handle ablation_mode == "reversed": swap the hierarchy order
        if ablation_mode == "reversed":
            self._level_roi_indices = list(reversed(self._level_roi_indices))
            self._n_tokens_per_level = list(reversed(self._n_tokens_per_level))
        elif ablation_mode == "random":
            import random
            all_roi_idxs = list(range(self.n_rois))
            random.Random(42).shuffle(all_roi_idxs)
            cumulative = 0
            self._level_roi_indices = []
            for nt in self._n_tokens_per_level:
                self._level_roi_indices.append(all_roi_idxs[cumulative:cumulative + nt])
                cumulative += nt

        # Register level-token indices as persistent buffers (avoids
        # creating new tensors on every forward call)
        for lev_idx, indices in enumerate(self._level_roi_indices):
            self.register_buffer(
                f"_level_idx_{lev_idx}",
                torch.tensor(indices, dtype=torch.long),
            )

        # Per-level Transformer encoders
        self.level_encoders = nn.ModuleList([
            LevelTransformerEncoder(
                d_model=d_model, nhead=nhead,
                num_layers=layers_per_level, dropout=dropout,
                max_tokens=max(self._n_tokens_per_level[i], 1),
            )
            for i in range(N_LEVELS)
        ])

        # Cross-level prediction heads (level 0→1, 1→2)
        # Level 3 (residual) doesn't get predictions from level 2
        self.prediction_heads = nn.ModuleList([
            PredictionHead(d_model, self._n_tokens_per_level[i + 1], dropout)
            for i in range(N_LEVELS - 2)  # 0→1, 1→2
        ])

        # Error normalization layers
        self.error_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(N_LEVELS - 2)
        ])

        # Hierarchical aggregator
        self.aggregator = HierarchicalAggregator(d_model, N_LEVELS, dropout)

        # vMF decoder head
        self.vmf_decoder = VonMisesFisherDecoder(
            input_dim=d_model,
            output_dim=output_dim,
            hidden_dims=[d_model],
            activation="gelu",
            dropout=dropout,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_mode=kappa_mode,
        )

        # Per-level kappa heads (for neuroscience analysis)
        if enable_per_level_kappa:
            self.level_kappa_heads = PerLevelKappaHeads(
                d_model, N_LEVELS, kappa_min, kappa_max, kappa_mode,
            )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "PredictiveCorticalDecoder: %d ROIs → %d levels, "
            "d_model=%d, heads=%d, layers/level=%d, "
            "output_dim=%d, ablation=%s, multi_subject=%s, "
            "params=%s",
            self.n_rois, N_LEVELS, d_model, nhead, layers_per_level,
            output_dim, ablation_mode, self._multi_subject,
            f"{n_params:,}",
        )

    @property
    def input_dim(self) -> int:
        """For compatibility with UnifiedModel that reads encoder.output_dim."""
        return self.d_model

    @property
    def output_dim_prop(self) -> int:
        return self.output_dim

    def _init_single_subject(
        self, roi_indices: OrderedDict, d_model: int, dropout: float
    ) -> None:
        """Build per-ROI projections for a single subject."""
        self.roi_projections = nn.ModuleList([
            ROIProjection(len(idx), d_model, dropout)
            for idx in roi_indices.values()
        ])
        # Register index buffers
        for i, (name, idx) in enumerate(roi_indices.items()):
            buf = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, dtype=torch.long)
            self.register_buffer(f"_roi_idx_{i}", buf)

    def _init_multi_subject(
        self,
        subject_roi_indices: Dict[str, OrderedDict],
        d_model: int,
        dropout: float,
    ) -> None:
        """Build per-subject ROI projections with shared backbone."""
        self.subject_projections = nn.ModuleDict()
        self._subject_list = sorted(subject_roi_indices.keys())
        self._subject_to_int = {s: i for i, s in enumerate(self._subject_list)}

        for subj in self._subject_list:
            indices = subject_roi_indices[subj]
            projs = nn.ModuleList([
                ROIProjection(len(idx), d_model, dropout)
                for idx in indices.values()
            ])
            self.subject_projections[subj] = projs
            for i, (name, idx) in enumerate(indices.items()):
                buf = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, dtype=torch.long)
                self.register_buffer(f"_roi_idx_{subj}_{i}", buf)

        # Subject embeddings for conditioning
        n_subjects = len(self._subject_list)
        self.subject_embed = nn.Embedding(n_subjects, d_model)
        logger.info(
            "PCD multi-subject: %d subjects, %d ROIs each",
            n_subjects, len(next(iter(subject_roi_indices.values()))),
        )

    def _build_level_groups(self, roi_names: List[str]) -> List[List[int]]:
        """Map ROI token indices to hierarchical levels.

        Returns list of 4 lists, each containing token position indices
        within the full ROI token sequence.
        """
        levels: List[List[int]] = [[] for _ in range(N_LEVELS)]
        for i, name in enumerate(roi_names):
            # Handle subdivided ROI names (e.g., "nsdgeneral_other_0")
            base_name = name
            for suffix_len in range(len(name)):
                candidate = name[:len(name) - suffix_len]
                if candidate in HIERARCHY_LEVELS:
                    base_name = candidate
                    break
                if candidate.endswith("_") and candidate[:-1] in HIERARCHY_LEVELS:
                    base_name = candidate[:-1]
                    break

            level = HIERARCHY_LEVELS.get(base_name, 3)  # default to residual
            levels[level].append(i)

        for lev_idx, indices in enumerate(levels):
            level_rois = [roi_names[j] for j in indices]
            logger.info(
                "  PCD Level %d (%s): %d tokens %s",
                lev_idx, LEVEL_NAMES[lev_idx], len(indices), level_rois,
            )

        return levels

    def _tokenize_single(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize for single-subject mode.

        Args:
            x: (B, V) flat fMRI vector.

        Returns:
            (B, n_rois, d_model) projected ROI tokens.
        """
        tokens = []
        for i in range(self.n_rois):
            idx = getattr(self, f"_roi_idx_{i}")
            roi_voxels = x[:, idx]
            tokens.append(self.roi_projections[i](roi_voxels))
        return torch.stack(tokens, dim=1)

    def _tokenize_multi(
        self, x: torch.Tensor, subject_ids: torch.Tensor
    ) -> torch.Tensor:
        """Tokenize for multi-subject mode.

        Each sample uses its own subject's ROI projection.

        Args:
            x:           (B, V_max) padded fMRI vector.
            subject_ids: (B,) integer subject IDs.

        Returns:
            (B, n_rois, d_model) projected ROI tokens.
        """
        B = x.shape[0]
        tokens = torch.zeros(
            B, self.n_rois, self.d_model,
            device=x.device, dtype=x.dtype,
        )

        for subj_int, subj_id in enumerate(self._subject_list):
            mask_indices = (subject_ids == subj_int).nonzero(as_tuple=True)[0]
            if mask_indices.numel() == 0:
                continue
            subj_x = x[mask_indices]
            projs = self.subject_projections[subj_id]
            subj_tokens = []
            for i in range(self.n_rois):
                idx = getattr(self, f"_roi_idx_{subj_id}_{i}")
                roi_voxels = subj_x[:, idx]
                subj_tokens.append(projs[i](roi_voxels))
            tokens[mask_indices] = torch.stack(subj_tokens, dim=1)

        return tokens

    def _gather_level_tokens(
        self, all_tokens: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Extract tokens belonging to a specific hierarchy level.

        Args:
            all_tokens: (B, n_rois, d_model)
            level:      0-3

        Returns:
            (B, n_tokens_level, d_model)
        """
        idx_tensor: torch.Tensor = getattr(self, f"_level_idx_{level}")
        if idx_tensor.numel() == 0:
            return all_tokens[:, :0, :]
        return all_tokens[:, idx_tensor, :]

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> PCDOutput:
        """
        Full predictive cortical processing pipeline.

        Args:
            x:              (B, V) flat fMRI input.
            subject_ids:    (B,) integer subject IDs (multi-subject mode).
            return_details: If True, populates all fields of PCDOutput.

        Returns:
            PCDOutput with mu, kappa, and optionally per-level details.
        """
        # Step 1: ROI tokenisation
        if self._multi_subject and subject_ids is not None:
            all_tokens = self._tokenize_multi(x, subject_ids)
            # Add subject embeddings
            subj_emb = self.subject_embed(subject_ids)  # (B, d_model)
            all_tokens = all_tokens + subj_emb.unsqueeze(1)
        else:
            all_tokens = self._tokenize_single(x)

        # Step 2: Hierarchical predictive coding
        level_outputs: List[torch.Tensor] = []
        prediction_errors: List[Optional[torch.Tensor]] = [None]  # no error for L0

        # Level 0: encode early visual tokens directly
        l0_tokens = self._gather_level_tokens(all_tokens, 0)
        h0 = self.level_encoders[0](l0_tokens)
        level_outputs.append(h0)

        if self.ablation_mode == "no_errors":
            # Ablation: skip prediction errors, encode raw tokens at each level
            for lev in range(1, N_LEVELS):
                lev_tokens = self._gather_level_tokens(all_tokens, lev)
                h = self.level_encoders[lev](lev_tokens)
                level_outputs.append(h)
                prediction_errors.append(None)
        else:
            # Levels 1-2: prediction error driven
            prev_h = h0
            for lev in range(1, N_LEVELS - 1):  # levels 1, 2
                lev_tokens = self._gather_level_tokens(all_tokens, lev)
                predicted = self.prediction_heads[lev - 1](prev_h)
                error = lev_tokens - predicted
                error = self.error_norms[lev - 1](error)
                prediction_errors.append(error)
                h = self.level_encoders[lev](error)
                level_outputs.append(h)
                prev_h = h

            # Level 3 (residual): encode directly (no prediction from above)
            l3_tokens = self._gather_level_tokens(all_tokens, N_LEVELS - 1)
            h3 = self.level_encoders[N_LEVELS - 1](l3_tokens)
            level_outputs.append(h3)
            prediction_errors.append(None)

        # Step 3: hierarchical aggregation
        aggregated, weights = self.aggregator(level_outputs)

        # Step 4: vMF decoder
        dec_out = self.vmf_decoder(aggregated)
        mu, kappa = dec_out[0], dec_out[1]

        # Step 5: per-level kappas
        level_kappas = None
        if self.enable_per_level_kappa:
            level_kappas = self.level_kappa_heads(level_outputs)

        return PCDOutput(
            mu=mu,
            kappa=kappa,
            level_outputs=level_outputs if return_details else [],
            prediction_errors=prediction_errors if return_details else [],
            level_kappas=level_kappas,
            level_weights=weights,
        )

    def get_prediction_error_magnitudes(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract prediction error magnitudes per level.

        Utility for neuroscience analysis: returns the L2 norm of
        prediction errors at each level, enabling analysis of how much
        information each level adds beyond lower levels.

        Returns:
            Dict with keys like "L1_to_L2_error", values (B,).
        """
        out = self.forward(x, subject_ids=subject_ids, return_details=True)
        result = {}
        for i, err in enumerate(out.prediction_errors):
            if err is not None:
                # Mean L2 norm across tokens
                err_mag = err.norm(dim=-1).mean(dim=-1)  # (B,)
                result[f"L{i-1}_to_L{i}_error"] = err_mag
        if out.level_kappas is not None:
            result["level_kappas"] = out.level_kappas
        if out.level_weights is not None:
            result["level_weights"] = out.level_weights
        return result
