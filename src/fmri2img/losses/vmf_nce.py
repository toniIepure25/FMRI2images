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

        logit(q, k) = kappa_q * sim(mu_q, z_k) / tau

    where sim is either cos (default) or arctanh(cos) for improved
    gradient dynamics near the poles.

    The log-normaliser log C_d(kappa_q) cancels in the cross-entropy
    softmax (it is constant across all keys for a fixed query), so we
    do NOT compute Bessel functions here.

    kappa encodes confidence: low kappa -> flat density -> all keys
    score similarly -> the model abstains.  High kappa -> peaked
    density -> only the correct key scores high.

    V11 additions:
        - CSLS-corrected logits (``use_csls_training``): applies
          differentiable Cross-domain Similarity Local Scaling inside
          the loss so gradients teach the encoder to avoid hub embeddings.
        - Inverted softmax component (``isf_weight``): column-normalised
          cross-entropy that directly penalises hub targets.

    Args:
        tau:       Temperature (scales logits; default 0.07)
        use_queue: Whether to use memory-queue negatives
        kappa_is_log: If True, the input is log(kappa) and will be
                      exponentiated.  If False, it is kappa directly.
        use_arctanh:  If True, apply arctanh to cosine similarities
                      before kappa scaling.
        margin_base:  Base additive margin for positive pairs (0 = disabled).
        margin_kappa_ref: Reference kappa for margin scaling.
                          margin_i = margin_base * kappa_i / (kappa_i + ref).
        use_csls_training: If True, apply differentiable CSLS correction
                           to the logit matrix before cross-entropy.
        csls_k: Number of nearest neighbours for CSLS hub estimation.
        isf_weight: Weight for the inverted-softmax loss component
                    (0 = disabled).  The final loss is
                    (1 - isf_weight) * standard_CE + isf_weight * ISF_CE.
    """

    def __init__(self, tau: float = 0.07, use_queue: bool = True,
                 kappa_is_log: bool = False,
                 learnable_temperature: bool = False,
                 use_arctanh: bool = False,
                 margin_base: float = 0.0,
                 margin_kappa_ref: float = 50.0,
                 label_smoothing: float = 0.0,
                 hard_negative_weight: float = 0.0,
                 hard_neg_k: int = 16,
                 use_csls_training: bool = False,
                 csls_k: int = 10,
                 isf_weight: float = 0.0,
                 # Legacy kwargs accepted but ignored
                 dim: int = 768):
        super().__init__()
        self.use_queue = use_queue
        self.kappa_is_log = kappa_is_log
        self.learnable_temperature = learnable_temperature
        self.use_arctanh = use_arctanh
        self.margin_base = margin_base
        self.margin_kappa_ref = margin_kappa_ref
        self.label_smoothing = label_smoothing
        self.hard_negative_weight = hard_negative_weight
        self.hard_neg_k = hard_neg_k
        self.use_csls_training = use_csls_training
        self.csls_k = csls_k
        self.isf_weight = isf_weight

        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / tau))
            )
            self.tau = None
        else:
            self.tau = tau

        logger.info(
            "VonMisesFisherNCELoss: tau=%s, use_queue=%s, kappa_is_log=%s, "
            "arctanh=%s, margin=%.2f, label_smoothing=%.2f, "
            "hard_neg=(%.2f, k=%d), csls_train=%s(k=%d), isf=%.2f",
            tau, use_queue, kappa_is_log, use_arctanh, margin_base,
            label_smoothing, hard_negative_weight, hard_neg_k,
            use_csls_training, csls_k, isf_weight,
        )

    @property
    def effective_tau(self) -> torch.Tensor:
        """Current temperature (scalar), clamped for stability."""
        if self.learnable_temperature:
            return torch.exp(-self.logit_scale.clamp(max=4.6052))
        return self.tau

    def _csls_correct(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable CSLS correction for contrastive logits.

        CSLS(x, y) = 2*s(x,y) - r_X(x) - r_Y(y)
        where r_X(x) = mean similarity of x to its k-NN in Y.

        All operations (topk, mean, subtract) are differentiable so
        gradients flow through the correction and teach the encoder to
        avoid producing hub embeddings.
        """
        k = min(self.csls_k, logits.shape[1] - 1, logits.shape[0] - 1)
        if k < 1:
            return logits
        r_x = logits.topk(k, dim=1).values.mean(dim=1)   # (B,)
        r_y = logits.topk(k, dim=0).values.mean(dim=0)    # (M,)
        return 2.0 * logits - r_x.unsqueeze(1) - r_y.unsqueeze(0)

    def _score(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        keys: torch.Tensor,
        positive_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kappa-scaled similarity logits with optional hard negative boosting.

        Args:
            mu:           (B, D) query mean directions (unit norm)
            kappa:        (B,)   query concentrations (positive)
            keys:         (M, D) key embeddings (unit norm)
            positive_idx: (B,)   column indices of positive keys (for masking).
                          If None, hard negative mining is skipped.

        Returns:
            (B, M) logits
        """
        cos_sim = mu @ keys.T                          # (B, M)
        if self.use_arctanh:
            cos_sim = torch.atanh(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        tau = self.effective_tau
        logits = kappa.unsqueeze(1) * cos_sim / tau

        if self.hard_negative_weight > 0 and positive_idx is not None:
            B, M = logits.shape
            neg_mask = torch.ones(B, M, dtype=torch.bool, device=logits.device)
            neg_mask[torch.arange(B, device=logits.device), positive_idx] = False
            top_k = min(self.hard_neg_k, int(neg_mask.sum(1).min().item()))
            if top_k > 0:
                masked_logits = logits.masked_fill(~neg_mask, -1e9)
                _, hard_idx = masked_logits.topk(top_k, dim=1)
                boost = torch.zeros_like(logits)
                boost.scatter_(1, hard_idx, self.hard_negative_weight)
                logits = logits + boost

        if self.use_csls_training:
            logits = self._csls_correct(logits)

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
        labels = torch.arange(B, device=mu_query.device)

        if self.use_queue and queue is not None and queue.is_ready():
            queue_embs = queue.get_queue()
            all_keys = torch.cat([key_embeddings, queue_embs], dim=0)  # (B+Q, D)
        else:
            all_keys = key_embeddings  # (B, D)

        logits = self._score(mu_query, kappa, all_keys, positive_idx=labels)

        # Kappa-adaptive additive margin on positive pairs (CosFace-style)
        if self.margin_base > 0:
            adaptive_margin = self.margin_base * kappa / (
                kappa + self.margin_kappa_ref
            )  # (B,)
            logits[torch.arange(B, device=logits.device), labels] -= adaptive_margin

        std_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        if self.isf_weight > 0:
            log_prob_isf = F.log_softmax(logits, dim=0)  # column-normalise
            isf_loss = -log_prob_isf[labels, labels].mean()
            return (1.0 - self.isf_weight) * std_loss + self.isf_weight * isf_loss

        return std_loss


class MixtureVonMisesFisherNCELoss(VonMisesFisherNCELoss):
    """InfoNCE-style loss for a multi-hypothesis vMF query distribution.

    The query is parameterised by M unit directions and concentrations, with an
    optional learned mixture weight per component. Candidate logits are

        logit(q, k) = logsumexp_j(log w_j + kappa_j * cos(mu_j, z_k) / tau)

    which defines a multi-modal hyperspherical retrieval expert while keeping
    the same contrastive training protocol as the single-vMF head.
    """

    def _score_mixture(
        self,
        component_mu: torch.Tensor,
        component_kappa: torch.Tensor,
        keys: torch.Tensor,
        component_logits: Optional[torch.Tensor] = None,
        positive_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if component_kappa.ndim == 3 and component_kappa.shape[-1] == 1:
            component_kappa = component_kappa[..., 0]
        cos_sim = torch.einsum("bmd,kd->bmk", component_mu, keys)
        if self.use_arctanh:
            cos_sim = torch.atanh(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        tau = self.effective_tau
        component_scores = component_kappa.unsqueeze(-1) * cos_sim / tau

        if component_logits is not None:
            log_w = F.log_softmax(component_logits, dim=1).unsqueeze(-1)
        else:
            m = component_mu.shape[1]
            log_w = component_scores.new_full((component_mu.shape[0], m, 1), -math.log(float(m)))

        logits = torch.logsumexp(component_scores + log_w, dim=1)

        if self.hard_negative_weight > 0 and positive_idx is not None:
            B, M = logits.shape
            neg_mask = torch.ones(B, M, dtype=torch.bool, device=logits.device)
            neg_mask[torch.arange(B, device=logits.device), positive_idx] = False
            top_k = min(self.hard_neg_k, int(neg_mask.sum(1).min().item()))
            if top_k > 0:
                masked_logits = logits.masked_fill(~neg_mask, -1e9)
                _, hard_idx = masked_logits.topk(top_k, dim=1)
                boost = torch.zeros_like(logits)
                boost.scatter_(1, hard_idx, self.hard_negative_weight)
                logits = logits + boost

        if self.use_csls_training:
            logits = self._csls_correct(logits)

        return logits.clamp(-80, 80)

    def forward(
        self,
        component_mu_query: torch.Tensor,
        component_kappa_or_log_query: torch.Tensor,
        key_embeddings: torch.Tensor,
        component_logits: Optional[torch.Tensor] = None,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if self.kappa_is_log:
            component_kappa = component_kappa_or_log_query.exp()
        else:
            component_kappa = component_kappa_or_log_query
        if component_kappa.ndim == 2:
            pass
        elif component_kappa.ndim == 3 and component_kappa.shape[-1] == 1:
            component_kappa = component_kappa[..., 0]
        else:
            raise ValueError(
                f"MixtureVonMisesFisherNCELoss expects component kappa with shape (B, M) or (B, M, 1); got {tuple(component_kappa.shape)}"
            )

        B = component_mu_query.size(0)
        labels = torch.arange(B, device=component_mu_query.device)
        if self.use_queue and queue is not None and queue.is_ready():
            queue_embs = queue.get_queue()
            all_keys = torch.cat([key_embeddings, queue_embs], dim=0)
        else:
            all_keys = key_embeddings

        logits = self._score_mixture(
            component_mu_query,
            component_kappa,
            all_keys,
            component_logits=component_logits,
            positive_idx=labels,
        )
        std_loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        if self.isf_weight > 0:
            log_prob_isf = F.log_softmax(logits, dim=0)
            isf_loss = -log_prob_isf[labels, labels].mean()
            return (1.0 - self.isf_weight) * std_loss + self.isf_weight * isf_loss
        return std_loss


# ---------------------------------------------------------------------------
# R-Drop regularization for vMF outputs  (Liang et al., 2021)
# ---------------------------------------------------------------------------

def vmf_rdrop_loss(
    mu1: torch.Tensor,
    kappa1: torch.Tensor,
    mu2: torch.Tensor,
    kappa2: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL divergence between two vMF distributions.

    Uses the closed-form approximation:
        KL(vMF_1 || vMF_2) ≈ kappa_1 * (1 - mu_1^T mu_2) + (kappa_1 - kappa_2) * A_d(kappa_1)
    where A_d is the ratio I_{d/2} / I_{d/2-1}.  For simplicity we use the
    dominant cosine term and kappa difference:

        D_sym ≈ (kappa_1 + kappa_2) * (1 - mu_1^T mu_2) + |kappa_1 - kappa_2|

    Args:
        mu1, mu2: (B, D) unit-norm mean directions from two forward passes.
        kappa1, kappa2: (B,) or (B, 1) concentrations from two forward passes.

    Returns:
        Scalar mean symmetric divergence.
    """
    k1 = kappa1.squeeze(-1)
    k2 = kappa2.squeeze(-1)
    cos_sim = (mu1 * mu2).sum(dim=-1)  # (B,)
    direction_div = (k1 + k2) * (1.0 - cos_sim)
    kappa_div = (k1 - k2).abs()
    return (direction_div + kappa_div).mean()


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
        hard_negative_weight: float = 0.0,
        hard_neg_k: int = 16,
        use_csls_training: bool = False,
        csls_k: int = 10,
        isf_weight: float = 0.0,
    ):
        super().__init__()
        self.tau = tau
        self.use_queue = use_queue
        self.kappa_is_log = kappa_is_log
        self.curriculum_t = initial_curriculum_t
        self.hard_negative_weight = hard_negative_weight
        self.hard_neg_k = hard_neg_k
        self.use_csls_training = use_csls_training
        self.csls_k = csls_k
        self.isf_weight = isf_weight
        logger.info(
            "KappaSPCLVMFNCELoss: tau=%s, use_queue=%s, curriculum_t=%s, "
            "hard_neg=(%.2f, k=%d), csls_train=%s, isf=%.2f",
            tau, use_queue, initial_curriculum_t,
            hard_negative_weight, hard_neg_k,
            use_csls_training, isf_weight,
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
        labels = torch.arange(B, device=mu_query.device)

        # Importance weights from kappa
        weights = F.softmax(kappa / self.curriculum_t, dim=0)  # (B,)

        if self.use_queue and queue is not None and queue.is_ready():
            all_keys = torch.cat([key_embeddings, queue.get_queue()], dim=0)
        else:
            all_keys = key_embeddings

        cos_sim = mu_query @ all_keys.T
        logits = (kappa.unsqueeze(1) * cos_sim / self.tau).clamp(-80, 80)

        if self.hard_negative_weight > 0:
            M = logits.size(1)
            neg_mask = torch.ones(B, M, dtype=torch.bool, device=logits.device)
            neg_mask[torch.arange(B, device=logits.device), labels] = False
            top_k = min(self.hard_neg_k, int(neg_mask.sum(1).min().item()))
            if top_k > 0:
                masked_logits = logits.masked_fill(~neg_mask, -1e9)
                _, hard_idx = masked_logits.topk(top_k, dim=1)
                boost = torch.zeros_like(logits)
                boost.scatter_(1, hard_idx, self.hard_negative_weight)
                logits = logits + boost

        if self.use_csls_training:
            k = min(self.csls_k, logits.shape[1] - 1, logits.shape[0] - 1)
            if k >= 1:
                r_x = logits.topk(k, dim=1).values.mean(dim=1)
                r_y = logits.topk(k, dim=0).values.mean(dim=0)
                logits = 2.0 * logits - r_x.unsqueeze(1) - r_y.unsqueeze(0)

        logits = logits.clamp(-80, 80)

        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")  # (B,)
        std_loss = (weights * per_sample_loss).sum()

        if self.isf_weight > 0:
            log_prob_isf = F.log_softmax(logits, dim=0)
            isf_loss = -log_prob_isf[labels, labels].mean()
            return (1.0 - self.isf_weight) * std_loss + self.isf_weight * isf_loss

        return std_loss


# ---------------------------------------------------------------------------
# Delta-SPCL: Disagreement-Aware Self-Paced Contrastive Learning
# ---------------------------------------------------------------------------

class DeltaSPCLVMFNCELoss(nn.Module):
    """
    Disagreement-Aware Self-Paced Contrastive Learning (Delta-SPCL).

    Extends kappa-SPCL by incorporating the directional disagreement
    score (delta) from SphericalConsensusFusion.  Samples are weighted
    by high confidence (kappa) AND low inter-ROI disagreement (delta):

        scoring_i = kappa_i - delta_weight * delta_i
        w_i       = softmax(scoring_i / T_curriculum)
        L         = sum_i w_i * CE_i

    When ROIs disagree (high delta), the fMRI signal is likely corrupted
    by noise, inattention, or mind-wandering.  The curriculum begins with
    high-agreement, high-confidence samples and gradually admits noisier
    trials as T_curriculum decreases.

    When delta is not provided (non-DCF models), falls back to pure
    kappa-SPCL weighting.

    Args:
        tau:                   Contrastive temperature.
        use_queue:             Whether to use memory-queue negatives.
        kappa_is_log:          Whether kappa inputs are in log space.
        initial_curriculum_t:  Starting curriculum temperature.
        delta_weight:          Scaling factor for delta penalty in the
                               scoring metric (higher = stronger penalty
                               for inter-ROI disagreement).
    """

    def __init__(
        self,
        tau: float = 1.0,
        use_queue: bool = True,
        kappa_is_log: bool = False,
        initial_curriculum_t: float = 50.0,
        delta_weight: float = 10.0,
    ):
        super().__init__()
        self.tau = tau
        self.use_queue = use_queue
        self.kappa_is_log = kappa_is_log
        self.curriculum_t = initial_curriculum_t
        self.delta_weight = delta_weight
        logger.info(
            "DeltaSPCLVMFNCELoss: tau=%s, use_queue=%s, curriculum_t=%s, "
            "delta_weight=%s",
            tau, use_queue, initial_curriculum_t, delta_weight,
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
        delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mu_query:                 (B, D) unit-norm query means.
            kappa_or_log_kappa_query: (B, 1) concentration or log-concentration.
            key_embeddings:           (B, D) unit-norm GT keys.
            queue:                    optional MemoryQueue for extra negatives.
            delta:                    (B, 1) directional disagreement from DCF.
                                      If None, falls back to kappa-only weighting.

        Returns:
            Scalar weighted contrastive loss.
        """
        if self.kappa_is_log:
            kappa = kappa_or_log_kappa_query.exp().squeeze(-1)
        else:
            kappa = kappa_or_log_kappa_query.squeeze(-1)

        B = mu_query.size(0)

        scoring = kappa
        if delta is not None:
            delta_val = delta.squeeze(-1)
            scoring = kappa - self.delta_weight * delta_val

        weights = F.softmax(scoring / self.curriculum_t, dim=0)  # (B,)

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
