"""Neural-Constrained Decoder (NCD).

An fMRI-to-CLIP retrieval decoder whose intermediate representations are
constrained by a **masked-ROI neural-prediction objective**: a random subset of
ROI nodes is masked before encoding, and the model must predict the true
activity of those ROIs from the remaining cortical context.

The scientific question (``docs/research/pcd_program/17_PRIMARY_THESIS_SELECTION.md``)
is whether that constraint improves perception-to-imagery transfer at matched
perception performance. ``lambda_neural`` is the single manipulated variable;
everything else is held fixed across arms.

Terminology
-----------
This module claims a **neural-prediction constraint** and nothing stronger. It
does *not* claim identifiability in the statistical sense (uniqueness up to a
known transformation), and it makes **no claim about cortical implementation**.
See decision records D-002, D-006, D-007.

Design constraints inherited from the Gate 0 audit
--------------------------------------------------
* **Every interpreted quantity has an identifying objective.** ``PCD``'s
  per-level kappa heads were interpreted but never received a gradient
  (findings T8/T13), yielding stable, plausible, meaningless figures. Here the
  only interpreted quantity is ``y_hat`` (masked-ROI predictions), supervised by
  ``L_neural``. Enforced by ``tests/test_ncd_gradient_flow.py``.
* **No unrestricted ``nsdgeneral_other`` bypass.** In PCD that ROI held ~64% of
  voxels and fed the aggregator directly (finding T7), so the hierarchy could be
  decorative while the model still scored. Here it is a parameter-budgeted
  context node behind a frozen random projection.
* **Per-ROI nodes, never pooled.** Category-selective ROIs (FFA1, FFA2, PPA,
  EBA, OFA, OPA, RSC) stay separate so per-ROI contrasts are possible at all.
* **Low-rank subject adaptation.** PCD's unrestricted per-subject projections
  were ~57% of its parameters and it overfit by 82pp across four rounds of
  regularization (F-001).

References:
    - Kneeland et al. (2025) NSD-Imagery (CVPR) — the transfer gap this targets
    - Allen et al. (2022) NSD
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmri2img.models.auxiliary_objectives import AuxObjective, RandomTargetBank

logger = logging.getLogger(__name__)

#: ROI treated as a capacity-constrained context node rather than a normal node.
CONTEXT_ROI = "nsdgeneral_other"


@dataclass
class NCDOutput:
    """Container for NCD forward outputs.

    Attributes:
        mu: (B, output_dim) L2-normalised CLIP direction for retrieval.
        neural_pred: ROI name -> (B, n_voxels) predicted activity, populated
            only for masked ROIs when a mask is supplied. This is the sole
            interpreted quantity and it is supervised by ``L_neural``.
        masked_rois: Names of the ROIs masked on this forward pass.
        roi_states: (B, R, d) post-encoder ROI node states, for analysis only.
    """

    mu: torch.Tensor
    neural_pred: Dict[str, torch.Tensor] = field(default_factory=dict)
    masked_rois: List[str] = field(default_factory=list)
    roi_states: Optional[torch.Tensor] = None


class LowRankROIProjection(nn.Module):
    """Project one ROI's voxels to a node embedding with low-rank subject adaptation.

    Implements ``h_r = LN((W_r + U_r^(s) V_r^(s)^T) y_r + b_r)`` from the formal
    specification. ``W_r`` is shared across subjects; only the rank-``k``
    factors are subject-specific, which is what keeps per-subject capacity small.

    Args:
        n_voxels: Voxel count for this ROI (may differ per subject).
        d_model: Node embedding dimension.
        subjects: Subject IDs to allocate adapters for. Empty for single-subject.
        rank: Adapter rank ``k``. Must be much smaller than ``d_model``.
        dropout: Dropout applied after the node non-linearity.
    """

    def __init__(
        self,
        n_voxels: int,
        d_model: int,
        subjects: Optional[List[str]] = None,
        rank: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_voxels = n_voxels
        self.d_model = d_model
        self.rank = rank

        self.shared = nn.Linear(n_voxels, d_model)

        self.subject_u = nn.ParameterDict()
        self.subject_v = nn.ParameterDict()
        for subj in subjects or []:
            # U starts at zero so the adapter is an exact no-op at init: the
            # model begins as the shared projection and earns any deviation.
            self.subject_u[subj] = nn.Parameter(torch.zeros(d_model, rank))
            self.subject_v[subj] = nn.Parameter(
                torch.randn(n_voxels, rank) * (1.0 / math.sqrt(n_voxels))
            )

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, subject: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            y: (B, n_voxels) voxel activity for this ROI.
            subject: Subject ID selecting the adapter, or None for shared-only.

        Returns:
            (B, d_model) ROI node embedding.
        """
        out = self.shared(y)
        if subject is not None and subject in self.subject_u:
            # (B, V) @ (V, k) -> (B, k); then (B, k) @ (k, d) -> (B, d)
            out = out + (y @ self.subject_v[subject]) @ self.subject_u[subject].t()
        return self.drop(self.act(self.norm(out)))


class ConstrainedContextNode(nn.Module):
    """Capacity-budgeted encoder for the large residual ROI (``nsdgeneral_other``).

    In PCD this ROI held ~64% of voxels and was projected with a full learned
    ``Linear(n_voxels, d_model)`` that fed the aggregator directly (finding T7),
    so it could dominate the representation regardless of the hierarchy. Here a
    **frozen random projection** compresses the voxels and only a small ``d x d``
    map is learned, costing ``d^2`` parameters instead of ``d * n_voxels``.

    The random projection is registered as a buffer so it is saved with the
    checkpoint and reproduced exactly on reload.

    Args:
        n_voxels: Voxel count of the residual ROI.
        d_model: Node embedding dimension.
        dropout: Dropout applied after the non-linearity.
        seed: Seed for the frozen projection, for reproducibility.
    """

    def __init__(
        self, n_voxels: int, d_model: int, dropout: float = 0.1, seed: int = 0
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        proj = torch.randn(n_voxels, d_model, generator=gen) / math.sqrt(n_voxels)
        self.register_buffer("frozen_proj", proj)

        self.learned = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, n_voxels) residual-ROI voxel activity.

        Returns:
            (B, d_model) context node embedding.
        """
        return self.drop(self.act(self.norm(self.learned(y @ self.frozen_proj))))


class AttentionPool(nn.Module):
    """Pool ROI node states into a single cortical state via learned-query attention.

    Args:
        d_model: Node embedding dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, R, d_model) ROI node states.

        Returns:
            (B, d_model) pooled cortical state.
        """
        b = h.shape[0]
        q = self.query.expand(b, -1, -1)
        attn = torch.bmm(q, self.key(h).transpose(1, 2)) / self.scale
        w = F.softmax(attn, dim=-1)
        return self.norm(torch.bmm(w, self.value(h)).squeeze(1))


class NeuralConstrainedDecoder(nn.Module):
    """fMRI-to-CLIP decoder with a masked-ROI neural-prediction constraint.

    Args:
        roi_indices: ROI name -> voxel index tensor. Single-subject form, or the
            template used for shapes in multi-subject mode.
        subject_roi_indices: subject -> (ROI name -> voxel index tensor) for
            multi-subject training. Takes precedence over ``roi_indices``.
        d_model: Node embedding dimension.
        nhead: Attention heads in the lateral encoder.
        num_layers: Lateral encoder depth.
        dropout: Dropout probability.
        output_dim: CLIP target dimension. Held fixed across all arms — target
            dimensionality is a confound (the competing hypothesis of
            Kneeland et al. 2025), not a free parameter.
        adapter_rank: Rank ``k`` of the subject adapters.
        mask_ratio: Fraction of ROI nodes masked per forward pass.
        context_roi: ROI routed through the constrained context node.
        seed: Seed for the frozen context projection.
    """

    def __init__(
        self,
        roi_indices: Optional[OrderedDict] = None,
        subject_roi_indices: Optional[Dict[str, OrderedDict]] = None,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 768,
        adapter_rank: int = 8,
        mask_ratio: float = 0.3,
        context_roi: str = CONTEXT_ROI,
        seed: int = 0,
        aux_objective: AuxObjective = AuxObjective.MASKED_NEURAL,
    ) -> None:
        super().__init__()
        if subject_roi_indices is None and roi_indices is None:
            raise ValueError("one of roi_indices or subject_roi_indices is required")

        self.d_model = d_model
        self.output_dim = output_dim
        self.mask_ratio = mask_ratio
        self.context_roi = context_roi
        self.aux_objective = AuxObjective(aux_objective)
        self._multi_subject = subject_roi_indices is not None

        if self._multi_subject:
            self.subjects = sorted(subject_roi_indices)
            template = subject_roi_indices[self.subjects[0]]
        else:
            self.subjects = []
            template = roi_indices

        self.roi_names: List[str] = list(template.keys())
        self.n_rois = len(self.roi_names)

        # Voxel indices per subject (or a single pseudo-subject) as buffers.
        index_source = (
            subject_roi_indices if self._multi_subject else {"_single": roi_indices}
        )
        for subj, indices in index_source.items():
            for i, idx in enumerate(indices.values()):
                buf = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx)
                self.register_buffer(f"_idx_{subj}_{i}", buf.long())

        # Per-ROI encoders. The context ROI gets a capacity-budgeted path.
        self.roi_encoders = nn.ModuleDict()
        for name in self.roi_names:
            n_vox = len(template[name])
            if name == context_roi:
                self.roi_encoders[name] = ConstrainedContextNode(
                    n_vox, d_model, dropout, seed
                )
            else:
                self.roi_encoders[name] = LowRankROIProjection(
                    n_vox, d_model, self.subjects, adapter_rank, dropout
                )

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_rois, d_model) * 0.02)
        if self._multi_subject:
            self.subject_embed = nn.Embedding(len(self.subjects), d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model)
        self.retrieval_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        # Neural-prediction heads: the ONLY interpreted quantity, and the loss
        # reaches them (test_ncd_gradient_flow.py). Skipped for the context ROI,
        # whose frozen projection makes its voxels unreconstructable by design.
        self.neural_heads = nn.ModuleDict(
            {
                name: nn.Linear(d_model, len(template[name]))
                for name in self.roi_names
                if name != context_roi
            }
        )

        self.predictable_rois = [n for n in self.roi_names if n != context_roi]

        # --- Matched-control machinery (protocol 25) -------------------------
        # Every arm allocates the SAME heads above. What follows only supplies
        # alternative *targets*, so parameter counts stay matched across arms.
        pred_sizes = OrderedDict(
            (name, len(template[name])) for name in self.predictable_rois
        )

        # ARM-D: fixed random targets, variance-matched, dimensionality-matched.
        self.random_targets = RandomTargetBank(pred_sizes, seed=seed, target_std=1.0)

        # ARM-E: generic (non-anatomical) reconstruction targets. Same voxel
        # count as each ROI but drawn from an arbitrary index set, so the target
        # carries comparable information with no anatomical identity.
        first_key = next(iter(index_source))
        total_vox = 1 + max(
            int(getattr(self, f"_idx_{first_key}_{i}").max()) for i in range(self.n_rois)
        )
        g = torch.Generator().manual_seed(seed + 1)
        for i, name in enumerate(self.predictable_rois):
            pool = torch.randperm(total_vox, generator=g)[: pred_sizes[name]]
            self.register_buffer(f"_generic_idx_{i}", pool.sort().values)

        logger.info(
            "NCD: %d ROI nodes (%d predictable, context=%s), d_model=%d, "
            "rank=%d, mask_ratio=%.2f, aux=%s, multi_subject=%s, params=%s",
            self.n_rois,
            len(self.predictable_rois),
            context_roi if context_roi in self.roi_names else "none",
            d_model,
            adapter_rank,
            mask_ratio,
            self.aux_objective.value,
            self._multi_subject,
            f"{sum(p.numel() for p in self.parameters()):,}",
        )

    def _voxels(self, x: torch.Tensor, subj_key: str, i: int) -> torch.Tensor:
        return x[:, getattr(self, f"_idx_{subj_key}_{i}")]

    def sample_mask(self, generator: Optional[torch.Generator] = None) -> List[str]:
        """Sample the ROI subset to mask for one forward pass.

        Returns:
            Names of ROIs to mask. Never includes the context ROI.
        """
        n_mask = max(1, int(round(self.mask_ratio * len(self.predictable_rois))))
        perm = torch.randperm(len(self.predictable_rois), generator=generator)
        return [self.predictable_rois[i] for i in perm[:n_mask].tolist()]

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        masked_rois: Optional[List[str]] = None,
        return_states: bool = False,
    ) -> NCDOutput:
        """Encode fMRI, optionally under an ROI mask, and decode to CLIP.

        Masked ROIs are replaced by ``mask_token`` **before** the encoder, so the
        voxels of a masked ROI never enter the computation that predicts them.
        Leakage-safety is therefore structural and per-batch, not a property of
        the train/val split.

        Args:
            x: (B, V) flat fMRI activity.
            subject_ids: (B,) integer subject indices; required in multi-subject
                mode. All samples in a batch must share a subject.
            masked_rois: ROIs to mask. None disables the constraint path.
            return_states: Also return post-encoder ROI states for analysis.

        Returns:
            NCDOutput with ``mu`` always populated and ``neural_pred`` populated
            for the masked ROIs.
        """
        subj_key = "_single"
        subject: Optional[str] = None
        if self._multi_subject:
            if subject_ids is None:
                raise ValueError("subject_ids required in multi-subject mode")
            uniq = torch.unique(subject_ids)
            if uniq.numel() != 1:
                raise ValueError(
                    "NCD requires subject-homogeneous batches (got "
                    f"{uniq.numel()} subjects); group the sampler by subject."
                )
            subject = self.subjects[int(uniq.item())]
            subj_key = subject

        masked = set(masked_rois or [])

        nodes = []
        for i, name in enumerate(self.roi_names):
            y = self._voxels(x, subj_key, i)
            enc = self.roi_encoders[name]
            h = enc(y) if name == self.context_roi else enc(y, subject)
            nodes.append(h)
        h = torch.stack(nodes, dim=1)  # (B, R, d)

        if masked:
            keep = torch.tensor(
                [name not in masked for name in self.roi_names],
                device=h.device,
            ).view(1, -1, 1)
            h = torch.where(keep, h, self.mask_token.to(h.dtype))

        pre_encoder = h  # (B, R, d) — needed by ARM-H, which must not see context
        h = h + self.pos_embed
        if self._multi_subject:
            h = h + self.subject_embed(subject_ids).unsqueeze(1)

        states = self.encoder(h)  # (B, R, d) — lateral messages only
        mu = F.normalize(self.retrieval_head(self.pool(states)), dim=-1)

        neural_pred: Dict[str, torch.Tensor] = {}
        if self.aux_objective is AuxObjective.ROI_AUTOENCODE:
            # ARM-H: predict each ROI from its OWN pre-encoder token. No ROI is
            # ever predicted from another, which is precisely what separates ROI
            # organisation from cross-ROI predictive dependency. Masking is
            # irrelevant here but is still drawn upstream to hold masking
            # frequency constant across arms (protocol 25 section 1).
            for name in self.predictable_rois:
                idx = self.roi_names.index(name)
                neural_pred[name] = self.neural_heads[name](pre_encoder[:, idx])
        else:
            for name in masked:
                idx = self.roi_names.index(name)
                neural_pred[name] = self.neural_heads[name](states[:, idx])

        return NCDOutput(
            mu=mu,
            neural_pred=neural_pred,
            masked_rois=sorted(masked),
            roi_states=states if return_states else None,
        )

    def _aux_target(
        self,
        name: str,
        x: torch.Tensor,
        subj_key: str,
        shuffled_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build the auxiliary target for one ROI under the active arm.

        The arm selects the *target*; the head is identical across arms so that
        capacity stays matched (protocol 25 section 1).
        """
        obj = self.aux_objective
        roi_pos = self.roi_names.index(name)
        pred_pos = self.predictable_rois.index(name)

        if obj in (AuxObjective.MASKED_NEURAL, AuxObjective.ROI_AUTOENCODE):
            return self._voxels(x, subj_key, roi_pos)

        if obj is AuxObjective.SHUFFLED_NEURAL:
            if shuffled_x is None:
                raise ValueError(
                    "SHUFFLED_NEURAL (ARM-C) requires shuffled_x: the fMRI of the "
                    "permuted partner image. Pass it from the dataloader using "
                    "DeterministicImagePermutation, which maps whole images (never "
                    "trials) so repetitions cannot leak the true target."
                )
            return self._voxels(shuffled_x, subj_key, roi_pos)

        if obj is AuxObjective.RANDOM_TARGET:
            return self.random_targets(name, x.shape[0])

        if obj is AuxObjective.SELF_RECONSTRUCTION:
            return x[:, getattr(self, f"_generic_idx_{pred_pos}")]

        raise ValueError(f"no target defined for objective {obj}")

    def auxiliary_loss(
        self,
        x: torch.Tensor,
        out: NCDOutput,
        subject_ids: Optional[torch.Tensor] = None,
        shuffled_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Auxiliary objective loss for the active arm, normalised per voxel.

        This is the identifying objective for ``out.neural_pred``. Without it the
        heads would be an interpreted quantity with no gradient — the exact
        failure documented as F-002.

        Args:
            x: (B, V) flat fMRI activity (supplies targets for most arms).
            out: Output of :meth:`forward` carrying ``neural_pred``.
            subject_ids: (B,) subject indices; required in multi-subject mode.
            shuffled_x: (B, V) fMRI of the permuted partner image. Required by
                ARM-C (``SHUFFLED_NEURAL``) and ignored otherwise.

        Returns:
            Scalar loss. Zero (and gradient-free) under ARM-A/F
            (``AuxObjective.NONE``) or when nothing was predicted.
        """
        if self.aux_objective is AuxObjective.NONE or not out.neural_pred:
            return x.new_zeros(())

        subj_key = "_single"
        if self._multi_subject:
            if subject_ids is None:
                raise ValueError("subject_ids required in multi-subject mode")
            subj_key = self.subjects[int(torch.unique(subject_ids).item())]

        total = x.new_zeros(())
        for name, pred in out.neural_pred.items():
            target = self._aux_target(name, x, subj_key, shuffled_x)
            # float32 loss math even under autocast, per repo convention.
            total = total + F.mse_loss(pred.float(), target.float())
        return total / len(out.neural_pred)

    def neural_prediction_loss(
        self,
        x: torch.Tensor,
        out: NCDOutput,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backwards-compatible alias for :meth:`auxiliary_loss`.

        Retained so existing NCD tests and configs keep working after the arm
        family landed. New code should call :meth:`auxiliary_loss`.
        """
        return self.auxiliary_loss(x, out, subject_ids=subject_ids)
