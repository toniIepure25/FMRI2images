"""
Von Mises-Fisher losses for hyperspherical probabilistic embeddings.

The vMF distribution is the natural probability distribution for data
living on the unit hypersphere S^{d-1}:

    p(z | mu, kappa) = C_d(kappa) * exp(kappa * mu^T z)

where C_d(kappa) is the normalising constant involving modified Bessel
functions.

Three components are provided:

1.  VonMisesFisherNLLLoss  --  proper negative log-likelihood under vMF.
    (requires the Bessel-based normaliser; kept unchanged)
2.  VonMisesFisherNCELoss  --  contrastive (InfoNCE-style) objective.
    Key insight: the log-normaliser log C_d(kappa_q) is constant across
    all keys for a fixed query q and therefore cancels inside the
    softmax.  The logits simplify to  kappa_q * cos(mu_q, z_k) / tau
    with NO Bessel functions needed.
3.  kappa_regularizer      --  lightweight penalty on mean kappa to
    prevent unbounded growth (simpler alternative to full vMF-NLL).

References:
    - Banerjee et al. (2005)  Clustering on the Unit Hypersphere
    - Davidson et al. (2018)  Hyperspherical Variational Auto-Encoders
    - Hasnat et al. (2017)    vMF Mixture Model for Deep Clustering
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bessel-function utilities  (used by NLL only)
# ---------------------------------------------------------------------------

def _log_iv(v: float, z: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log I_v(z).

    Uses the relation  I_v(z) = exp(z) * I_v^e(z)  where I_v^e is the
    exponentially-scaled Bessel function, so log I_v(z) = z + log I_v^e(z).

    For v = 0 and v = 1 we have torch.special helpers; for other half-integer
    orders we fall back to a saddle-point approximation.
    """
    if abs(v) < 1e-6:
        return z + torch.log(torch.special.i0e(z) + 1e-30)
    elif abs(v - 1.0) < 1e-6:
        return z + torch.log(torch.special.i1e(z) + 1e-30)
    else:
        return z - 0.5 * torch.log(2.0 * math.pi * z.clamp(min=1e-6)) + \
               (v * v - 0.25) * 0.5 / z.clamp(min=1e-6)


def _log_vmf_normaliser(dim: int, kappa: torch.Tensor) -> torch.Tensor:
    """
    log C_d(kappa) for the vMF normalising constant.

    C_d(kappa) = kappa^{d/2-1} / ( (2*pi)^{d/2} * I_{d/2-1}(kappa) )

    Returns tensor of same shape as kappa.
    """
    half_d = dim / 2.0
    v = half_d - 1.0
    log_num = v * torch.log(kappa.clamp(min=1e-6))
    log_den = half_d * math.log(2 * math.pi) + _log_iv(v, kappa)
    return log_num - log_den


# ---------------------------------------------------------------------------
# vMF NLL loss  (unchanged — correctly uses the normaliser)
# ---------------------------------------------------------------------------

class VonMisesFisherNLLLoss(nn.Module):
    """
    Negative log-likelihood under vMF(mu, kappa) for unit-norm targets.

    NLL = -log C_d(kappa) - kappa * mu^T z

    Accepts **either** log_kappa or direct kappa via ``kappa_is_log``.

    Args:
        dim: Embedding dimensionality (e.g. 768 for ViT-L/14 CLIP)
        reduction: 'mean', 'sum', or 'none'
        kappa_is_log: If True, the second input is log(kappa) and will be
                      exponentiated.  If False, it is kappa directly.
    """

    def __init__(self, dim: int = 768, reduction: str = "mean",
                 kappa_is_log: bool = True):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.kappa_is_log = kappa_is_log
        logger.info(f"VonMisesFisherNLLLoss: dim={dim}, kappa_is_log={kappa_is_log}")

    def forward(
        self,
        mu: torch.Tensor,
        kappa_or_log_kappa: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu:                  (B, D) unit-norm mean directions
            kappa_or_log_kappa:  (B, 1) concentration or log-concentration
            target:              (B, D) unit-norm ground-truth embeddings
        """
        if self.kappa_is_log:
            kappa = kappa_or_log_kappa.exp().squeeze(-1)
        else:
            kappa = kappa_or_log_kappa.squeeze(-1)

        cos_sim = (mu * target).sum(dim=-1)
        log_c = _log_vmf_normaliser(self.dim, kappa)

        nll = -log_c - kappa * cos_sim

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll


# ---------------------------------------------------------------------------
# vMF-NCE contrastive loss  (FIXED — no Bessel in logits)
# ---------------------------------------------------------------------------

class VonMisesFisherNCELoss(nn.Module):
    """
    vMF-NCE: contrastive loss with kappa-scaled cosine logits.

    For query vMF(mu_q, kappa_q) and deterministic key z_k the logit is:

        logit(q, k) = kappa_q * cos(mu_q, z_k) / tau

    The log-normaliser log C_d(kappa_q) cancels in the cross-entropy
    softmax (it is constant across all keys for a fixed query), so we
    do NOT compute Bessel functions here.

    kappa encodes confidence: low kappa -> flat density -> all keys
    score similarly -> the model abstains.  High kappa -> peaked
    density -> only the correct key scores high.

    Args:
        tau:       Temperature (scales logits; default 0.07)
        use_queue: Whether to use memory-queue negatives
        kappa_is_log: If True, the input is log(kappa) and will be
                      exponentiated.  If False, it is kappa directly.
    """

    def __init__(self, tau: float = 0.07, use_queue: bool = True,
                 kappa_is_log: bool = False,
                 learnable_temperature: bool = False,
                 # Legacy kwargs accepted but ignored
                 dim: int = 768):
        super().__init__()
        self.use_queue = use_queue
        self.kappa_is_log = kappa_is_log
        self.learnable_temperature = learnable_temperature

        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / tau))
            )
            self.tau = None
            logger.info(
                "VonMisesFisherNCELoss: learnable_temperature=True (init tau=%.4f), "
                "use_queue=%s, kappa_is_log=%s",
                tau, use_queue, kappa_is_log,
            )
        else:
            self.tau = tau
            logger.info(
                "VonMisesFisherNCELoss: tau=%s, use_queue=%s, kappa_is_log=%s",
                tau, use_queue, kappa_is_log,
            )

    @property
    def effective_tau(self) -> torch.Tensor:
        """Current temperature (scalar), clamped for stability."""
        if self.learnable_temperature:
            return torch.exp(-self.logit_scale.clamp(max=4.6052))
        return self.tau

    def _score(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute kappa-scaled cosine logits (no Bessel functions).

        Args:
            mu:    (B, D) query mean directions (unit norm)
            kappa: (B,)   query concentrations (positive)
            keys:  (M, D) key embeddings (unit norm)

        Returns:
            (B, M) logits
        """
        cos_sim = mu @ keys.T                          # (B, M)
        tau = self.effective_tau
        logits = kappa.unsqueeze(1) * cos_sim / tau
        return logits.clamp(-80, 80)                   # (B, M)

    def forward(
        self,
        mu_query: torch.Tensor,
        kappa_or_log_kappa_query: torch.Tensor,
        key_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            mu_query:                    (B, D) unit-norm query means
            kappa_or_log_kappa_query:    (B, 1) concentration or log-concentration
            key_embeddings:              (B, D) unit-norm GT keys (positives on diagonal)
            queue:                       optional MemoryQueue for extra negatives
        """
        if self.kappa_is_log:
            kappa = kappa_or_log_kappa_query.exp().squeeze(-1)
        else:
            kappa = kappa_or_log_kappa_query.squeeze(-1)

        B = mu_query.size(0)

        logits = self._score(mu_query, kappa, key_embeddings)   # (B, B)

        if self.use_queue and queue is not None and queue.is_ready():
            queue_embs = queue.get_queue()
            logits_q = self._score(mu_query, kappa, queue_embs)  # (B, Q)
            logits = torch.cat([logits, logits_q], dim=1)        # (B, B+Q)

        labels = torch.arange(B, device=mu_query.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Kappa regularizer  (lightweight alternative to full vMF-NLL)
# ---------------------------------------------------------------------------

def kappa_regularizer(kappa: torch.Tensor, lambda_kappa: float = 0.01) -> torch.Tensor:
    """
    Penalise large kappa to prevent unbounded concentration growth.

    Args:
        kappa: (B, 1) or (B,) positive concentration values
        lambda_kappa: Regularisation strength

    Returns:
        Scalar penalty: lambda_kappa * mean(kappa)
    """
    return lambda_kappa * kappa.mean()


# ---------------------------------------------------------------------------
# kappa-SPCL: Concentration-Aware Self-Paced Contrastive Learning
# ---------------------------------------------------------------------------

class KappaSPCLVMFNCELoss(nn.Module):
    """
    Concentration-Aware Self-Paced Contrastive Learning (kappa-SPCL).

    Modifies vMF-NCE with per-sample importance weights derived from
    kappa, creating an automatic curriculum: samples the model is
    confident about (high kappa) contribute more to the gradient early in
    training.  As the curriculum temperature anneals, the model gradually
    attends to harder / noisier samples.

    Weighting scheme:

        w_i = softmax(kappa_i / T_curriculum)
        L   = - sum_i w_i * log(exp(s_{ii}) / sum_j exp(s_{ij}))

    where s_{ij} = kappa_i * cos(mu_i, z_j) / tau.

    T_curriculum is controlled externally via ``set_curriculum_temperature``
    and should be annealed from a high value (uniform weights) to a low
    value (concentrate on high-kappa samples) and optionally back up.

    Args:
        tau:                   Contrastive temperature.
        use_queue:             Whether to use memory-queue negatives.
        kappa_is_log:          Whether kappa inputs are in log space.
        initial_curriculum_t:  Starting curriculum temperature.
    """

    def __init__(
        self,
        tau: float = 0.07,
        use_queue: bool = True,
        kappa_is_log: bool = False,
        initial_curriculum_t: float = 50.0,
    ):
        super().__init__()
        self.tau = tau
        self.use_queue = use_queue
        self.kappa_is_log = kappa_is_log
        self.curriculum_t = initial_curriculum_t
        logger.info(
            "KappaSPCLVMFNCELoss: tau=%s, use_queue=%s, curriculum_t=%s",
            tau, use_queue, initial_curriculum_t,
        )

    def set_curriculum_temperature(self, t: float) -> None:
        """Update the curriculum temperature (call once per epoch)."""
        self.curriculum_t = max(t, 1e-6)

    def forward(
        self,
        mu_query: torch.Tensor,
        kappa_or_log_kappa_query: torch.Tensor,
        key_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            mu_query:                 (B, D) unit-norm query means.
            kappa_or_log_kappa_query: (B, 1) concentration or log-concentration.
            key_embeddings:           (B, D) unit-norm GT keys.
            queue:                    optional MemoryQueue for extra negatives.

        Returns:
            Scalar weighted contrastive loss.
        """
        if self.kappa_is_log:
            kappa = kappa_or_log_kappa_query.exp().squeeze(-1)
        else:
            kappa = kappa_or_log_kappa_query.squeeze(-1)

        B = mu_query.size(0)

        # Importance weights from kappa
        weights = F.softmax(kappa / self.curriculum_t, dim=0)  # (B,)

        cos_sim = mu_query @ key_embeddings.T  # (B, B)
        logits = (kappa.unsqueeze(1) * cos_sim / self.tau).clamp(-80, 80)

        if self.use_queue and queue is not None and queue.is_ready():
            queue_embs = queue.get_queue()
            cos_q = mu_query @ queue_embs.T  # (B, Q)
            logits_q = (kappa.unsqueeze(1) * cos_q / self.tau).clamp(-80, 80)
            logits = torch.cat([logits, logits_q], dim=1)  # (B, B+Q)

        labels = torch.arange(B, device=mu_query.device)
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")  # (B,)
        return (weights * per_sample_loss).sum()


# ---------------------------------------------------------------------------
# Multi-task vMF-NCE for ROI-DCF
# ---------------------------------------------------------------------------

class MultiTaskVMFNCELoss(nn.Module):
    """
    Multi-task vMF-NCE combining fused and per-ROI contrastive losses.

    Total loss:

        L = L_vmf_nce(mu_fused, kappa_consensus, z_GT)
            + lambda_aux * (1/R) * sum_r L_vmf_nce(mu_r, kappa_r, z_GT)

    The fused loss trains the attention-based weighting, while the
    auxiliary per-ROI losses ensure each brain region learns meaningful
    CLIP directions independently.

    Args:
        tau:          Temperature for all vMF-NCE sub-losses.
        use_queue:    Whether to use memory-queue negatives.
        lambda_aux:   Weight for the per-ROI auxiliary loss term.
        kappa_is_log: Whether kappa inputs are in log space.
    """

    def __init__(
        self,
        tau: float = 0.07,
        use_queue: bool = True,
        lambda_aux: float = 0.5,
        kappa_is_log: bool = False,
    ):
        super().__init__()
        self.tau = tau
        self.use_queue = use_queue
        self.lambda_aux = lambda_aux

        self.fused_loss = VonMisesFisherNCELoss(
            tau=tau, use_queue=use_queue, kappa_is_log=kappa_is_log,
        )
        self.roi_loss = VonMisesFisherNCELoss(
            tau=tau, use_queue=use_queue, kappa_is_log=kappa_is_log,
        )
        logger.info(
            "MultiTaskVMFNCELoss: tau=%s, use_queue=%s, lambda_aux=%s",
            tau, use_queue, lambda_aux,
        )

    def forward(
        self,
        mu_fused: torch.Tensor,
        kappa_consensus: torch.Tensor,
        key_embeddings: torch.Tensor,
        per_roi_mus: Optional[torch.Tensor] = None,
        per_roi_kappas: Optional[torch.Tensor] = None,
        queue: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_fused:        (B, D) fused mean direction.
            kappa_consensus: (B, 1) consensus concentration.
            key_embeddings:  (B, D) unit-norm GT keys (positives on diagonal).
            per_roi_mus:     (B, R, D) per-ROI mean directions (optional).
            per_roi_kappas:  (B, R, 1) per-ROI concentrations (optional).
            queue:           optional MemoryQueue for extra negatives.

        Returns:
            total_loss:  Scalar combined loss.
            fused_loss:  Scalar fused-only loss (for logging).
            aux_loss:    Scalar auxiliary per-ROI loss (for logging).
        """
        loss_fused = self.fused_loss(
            mu_fused, kappa_consensus, key_embeddings, queue=queue,
        )

        if per_roi_mus is None or per_roi_kappas is None or self.lambda_aux == 0.0:
            return loss_fused, loss_fused, torch.tensor(0.0, device=loss_fused.device)

        B, R, D = per_roi_mus.shape
        roi_losses = []
        for r in range(R):
            roi_loss_r = self.roi_loss(
                per_roi_mus[:, r],            # (B, D)
                per_roi_kappas[:, r],         # (B, 1)
                key_embeddings,
                queue=queue,
            )
            roi_losses.append(roi_loss_r)

        loss_aux = torch.stack(roi_losses).mean()
        total = loss_fused + self.lambda_aux * loss_aux
        return total, loss_fused, loss_aux
