"""
BrainBits Information Bottleneck for Neural Decoding
=====================================================

Implements a **learned linear bottleneck** that compresses fMRI representations
to a tunable dimensionality, enabling measurement of how much neural signal
the decoder actually extracts vs how much the generative prior fills in.

Inspired by Mayo et al. (2024) "BrainBits: How Much of the Brain are
Generative Reconstruction Methods Using?" (NeurIPS 2024), which showed that
~30-50 dimensions suffice for most existing methods.

Our contributions beyond BrainBits:
    1. **Bottleneck Regularizer**: An auxiliary loss that PENALIZES the model
       when a smaller bottleneck retains too much performance — forcing the
       encoder to actually use more brain dimensions.
    2. **Differentiable rank control**: Soft rank via singular value decay,
       enabling gradient-based optimization of effective dimensionality.
    3. **Neural Information Ratio (NIR)**: Normalized metric for fair comparison.

Architecture:
    fMRI (V voxels) -> Encoder -> z (D dims) -> Bottleneck(rank r) -> z_r (r dims)
    
    During training:
        - Main loss: L_main on full z (no bottleneck)
        - Bottleneck reg: L_bn = -|L_main(z) - L_main(z_r)| * lambda_bn
          (maximizes performance DROP under compression)

    During evaluation:
        - Sweep bottleneck rank r from 1 to D
        - Report performance curve and NIR at each rank

References:
    - Mayo et al. (2024) BrainBits, NeurIPS 2024
    - Tishby et al. (2000) Information Bottleneck Method
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LearnedLinearBottleneck(nn.Module):
    """
    Learned linear bottleneck with soft rank control.

    Compresses D-dimensional embeddings through a rank-r linear projection.
    The effective rank is controlled by a learnable singular value mask.

    Parameters
    ----------
    input_dim : int
        Embedding dimension (e.g., 768 for CLIP).
    max_rank : int
        Maximum bottleneck rank (defaults to input_dim).
    initial_rank : int
        Starting effective rank (soft initialization).
    temperature : float
        Temperature for soft rank mask (lower = sharper cutoff).
    """

    def __init__(
        self,
        input_dim: int,
        max_rank: Optional[int] = None,
        initial_rank: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_rank = max_rank or input_dim
        self.temperature = temperature

        # Learned projection matrices (low-rank factorization)
        self.U = nn.Linear(input_dim, self.max_rank, bias=False)
        self.V = nn.Linear(self.max_rank, input_dim, bias=False)

        # Learnable log-singular-values for soft rank control
        if initial_rank is not None:
            init_logits = torch.zeros(self.max_rank)
            init_logits[:initial_rank] = 3.0   # high -> keep
            init_logits[initial_rank:] = -3.0  # low -> suppress
        else:
            init_logits = torch.zeros(self.max_rank)

        self.sv_logits = nn.Parameter(init_logits)

        logger.info(
            "LearnedLinearBottleneck: dim=%d, max_rank=%d, initial_rank=%s, temp=%.2f",
            input_dim, self.max_rank,
            initial_rank if initial_rank else "full",
            temperature,
        )

    def _compute_mask(self, hard_rank: Optional[int] = None) -> torch.Tensor:
        """Compute soft or hard rank mask over dimensions."""
        if hard_rank is not None:
            mask = torch.zeros(self.max_rank, device=self.sv_logits.device)
            mask[:hard_rank] = 1.0
            return mask
        return torch.sigmoid(self.sv_logits / self.temperature)

    def forward(
        self,
        z: torch.Tensor,
        hard_rank: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply bottleneck compression.

        Parameters
        ----------
        z : (B, D) input embeddings
        hard_rank : if set, use hard cutoff at this rank (for eval sweeps)

        Returns
        -------
        z_compressed : (B, D) compressed-then-reconstructed embeddings
        """
        mask = self._compute_mask(hard_rank)  # (max_rank,)

        # Project down, mask, project back
        h = self.U(z)             # (B, max_rank)
        h = h * mask.unsqueeze(0)  # apply dimension mask
        z_r = self.V(h)           # (B, D)

        return z_r

    def effective_rank(self) -> float:
        """Compute effective rank (sum of sigmoid activations)."""
        with torch.no_grad():
            mask = torch.sigmoid(self.sv_logits / self.temperature)
            return float(mask.sum().item())

    def get_dimension_importances(self) -> torch.Tensor:
        """Return sorted dimension importances (sigmoid of logits)."""
        with torch.no_grad():
            return torch.sigmoid(self.sv_logits / self.temperature).sort(
                descending=True
            ).values


class BottleneckRegularizer(nn.Module):
    """
    Auxiliary loss that encourages the model to USE more brain dimensions.

    Core idea: If compressing to rank r barely hurts performance, the model
    is not using those extra dimensions. We penalize this by maximizing the
    performance drop under compression.

    L_bn = -lambda_bn * max(0, L_full - L_compressed)

    Where:
        L_full = loss on full-rank embeddings
        L_compressed = loss on bottleneck-compressed embeddings

    This INCREASES when the model relies more on higher dimensions,
    incentivizing richer neural signal extraction.

    Parameters
    ----------
    bottleneck : LearnedLinearBottleneck module
    lambda_bn : weight for the regularizer
    compression_rank : rank to use for the compressed version
    warmup_epochs : number of epochs before activating (let main loss stabilize)
    """

    def __init__(
        self,
        bottleneck: LearnedLinearBottleneck,
        lambda_bn: float = 0.1,
        compression_rank: int = 32,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.bottleneck = bottleneck
        self.lambda_bn = lambda_bn
        self.compression_rank = compression_rank
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def forward(
        self,
        z_full: torch.Tensor,
        targets: torch.Tensor,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute bottleneck regularization loss.

        Parameters
        ----------
        z_full : (B, D) full embeddings from encoder
        targets : (B, D) CLIP ground truth
        loss_fn : callable(predictions, targets) -> scalar loss

        Returns
        -------
        reg_loss : scalar regularization loss (to be ADDED to main loss)
        metrics : dict with diagnostic values
        """
        if self._current_epoch < self.warmup_epochs:
            return torch.tensor(0.0, device=z_full.device), {
                "bn_reg_loss": 0.0,
                "bn_effective_rank": self.bottleneck.effective_rank(),
                "bn_warmup": True,
            }

        # Compress through bottleneck at fixed rank
        z_compressed = self.bottleneck(z_full, hard_rank=self.compression_rank)

        # Normalize both for fair comparison (vMF operates on unit sphere)
        z_compressed_norm = F.normalize(z_compressed, p=2, dim=-1)

        # Compute loss on compressed version
        loss_compressed = loss_fn(z_compressed_norm, targets)

        # Compute loss on full version (should already be computed, but
        # recompute for gradient flow through bottleneck)
        z_full_norm = F.normalize(z_full, p=2, dim=-1)
        loss_full = loss_fn(z_full_norm, targets)

        # Regularizer: MAXIMIZE the drop (negative sign because we minimize)
        performance_drop = (loss_compressed - loss_full).clamp(min=0)
        reg_loss = -self.lambda_bn * performance_drop

        metrics = {
            "bn_reg_loss": float(reg_loss.item()),
            "bn_loss_full": float(loss_full.item()),
            "bn_loss_compressed": float(loss_compressed.item()),
            "bn_performance_drop": float(performance_drop.item()),
            "bn_effective_rank": self.bottleneck.effective_rank(),
        }

        return reg_loss, metrics


class NeuralInformationRatio:
    """
    Neural Information Ratio (NIR) computation.

    NIR(r) = (perf(r) - perf_random) / (perf_ceiling - perf_random)

    Measures what fraction of achievable performance comes from the first
    r brain dimensions, normalized between random baseline and ceiling.

    Parameters
    ----------
    random_baseline : performance with random/null brain signal
    ceiling : maximum achievable performance (e.g., noise ceiling)
    """

    def __init__(self, random_baseline: float, ceiling: float):
        self.random_baseline = random_baseline
        self.ceiling = ceiling
        self._range = ceiling - random_baseline

    def compute(self, performance: float) -> float:
        """Compute NIR for a single performance value."""
        if self._range <= 0:
            return 0.0
        return (performance - self.random_baseline) / self._range

    def compute_curve(
        self,
        performances: Dict[int, float],
    ) -> Dict[int, float]:
        """
        Compute NIR curve from {rank: performance} dictionary.

        Parameters
        ----------
        performances : {bottleneck_rank: metric_value} mapping

        Returns
        -------
        {bottleneck_rank: NIR_value} mapping
        """
        return {rank: self.compute(perf) for rank, perf in performances.items()}

    def area_under_nir_curve(self, nir_curve: Dict[int, float]) -> float:
        """Compute area under the NIR curve (higher = uses more dimensions)."""
        if not nir_curve:
            return 0.0
        sorted_items = sorted(nir_curve.items())
        ranks = [r for r, _ in sorted_items]
        nirs = [n for _, n in sorted_items]

        area = 0.0
        for i in range(len(ranks) - 1):
            width = ranks[i + 1] - ranks[i]
            height = (nirs[i] + nirs[i + 1]) / 2
            area += width * height

        max_rank = ranks[-1]
        if max_rank > 0:
            area /= max_rank

        return area


@torch.no_grad()
def compute_bottleneck_curve(
    model: nn.Module,
    bottleneck: LearnedLinearBottleneck,
    dataloader: torch.utils.data.DataLoader,
    metric_fn,
    ranks: Optional[list] = None,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[int, float]:
    """
    Sweep bottleneck rank and compute performance at each level.

    Parameters
    ----------
    model : encoder model producing embeddings
    bottleneck : trained bottleneck layer
    dataloader : evaluation data
    metric_fn : callable(all_preds, all_gts) -> float metric
    ranks : list of ranks to evaluate (default: logarithmic sweep)
    device : compute device
    max_batches : limit batches for speed

    Returns
    -------
    {rank: metric_value} dictionary for plotting BrainBits curves
    """
    model.eval()
    bottleneck.eval()

    if ranks is None:
        max_r = bottleneck.max_rank
        ranks = sorted(set(
            [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, max_r]
        ))
        ranks = [r for r in ranks if r <= max_r]

    results = {}

    for rank in ranks:
        all_preds = []
        all_gts = []

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                fmri, targets = batch[0], batch[1]
            else:
                continue

            fmri = fmri.to(device).float()
            targets = targets.to(device).float()

            # Get full embeddings from model
            output = model(fmri)
            if isinstance(output, tuple):
                mu = output[0]
            else:
                mu = output

            # Apply bottleneck at fixed rank
            mu_compressed = bottleneck(mu, hard_rank=rank)
            mu_compressed = F.normalize(mu_compressed, p=2, dim=-1)

            all_preds.append(mu_compressed.cpu().numpy())
            all_gts.append(targets.cpu().numpy())

        import numpy as np
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_gts_np = np.concatenate(all_gts, axis=0)

        metric_value = metric_fn(all_preds_np, all_gts_np)
        results[rank] = metric_value

        logger.info("Bottleneck rank=%d: metric=%.4f", rank, metric_value)

    return results
