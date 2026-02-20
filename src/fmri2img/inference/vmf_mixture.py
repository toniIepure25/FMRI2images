"""
vMF Mixture Posterior Sampling for ROI-DCF Generation
=====================================================

Instead of collapsing per-ROI vMF predictions to a single consensus
direction, sample from the full mixture posterior:

    p(z | f) = sum_r  alpha_r * vMF(z; mu_r, kappa_r)

This gives natural diversity in generated CLIP embeddings that reflects
which brain regions dominate — enabling uncertainty-aware ensemble
generation where high inter-ROI disagreement (delta) produces more
diverse samples.

Sampling from a vMF mixture:
    1. Choose component r ~ Categorical(alpha)
    2. Sample z ~ vMF(mu_r, kappa_r)

The vMF sampler uses the Wood (1994) rejection algorithm for the
concentration parameter, adapted for high-dimensional spheres.

References:
    - Wood, A.T.A. (1994) Simulation of the von Mises Fisher distribution
    - Mardia, K.V. & Jupp, P.E. (2000) Directional Statistics
    - Ulrich (1984) Computer generation of distributions on the m-sphere
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def _sample_vmf_wood(
    mu: torch.Tensor,
    kappa: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Sample from vMF(mu, kappa) using the Wood (1994) rejection method.

    Args:
        mu: (B, D) unit-norm mean directions.
        kappa: (B,) or (B, 1) concentration parameters.

    Returns:
        samples: (B, num_samples, D) unit-norm samples on S^{d-1}.
    """
    kappa = kappa.squeeze(-1) if kappa.dim() > 1 else kappa
    B, D = mu.shape
    device = mu.device
    dtype = mu.dtype

    samples_list = []
    for _ in range(num_samples):
        w = _rejection_sample_w(kappa, D, device, dtype)  # (B,)
        v = _sample_uniform_tangent(mu, device, dtype)     # (B, D)
        z = w.unsqueeze(-1) * mu + torch.sqrt(
            (1.0 - w ** 2).clamp(min=1e-10)
        ).unsqueeze(-1) * v
        z = F.normalize(z, p=2, dim=-1)
        samples_list.append(z)

    return torch.stack(samples_list, dim=1)  # (B, num_samples, D)


def _rejection_sample_w(
    kappa: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Rejection sampling for the w component (Wood 1994, Ulrich 1984).

    For high kappa and high dim, the acceptance rate is high so this
    converges quickly.
    """
    B = kappa.shape[0]
    m = (dim - 1) / 2.0
    b = (-2.0 * kappa + torch.sqrt(4.0 * kappa ** 2 + (dim - 1) ** 2)) / (dim - 1)
    a = (m * torch.log(m * (1.0 - b ** 2)) + m * math.log(m)
         - 0.5 * (dim - 1) * torch.log(1.0 - b ** 2))
    # Simplified: use the approximate beta distribution trick
    d_param = m

    w = torch.zeros(B, device=device, dtype=dtype)
    done = torch.zeros(B, device=device, dtype=torch.bool)

    for _ in range(max_iter):
        if done.all():
            break
        remaining = ~done
        n_remaining = remaining.sum().item()
        if n_remaining == 0:
            break

        eps = torch.distributions.Beta(d_param, d_param).sample(
            (n_remaining,)
        ).to(device=device, dtype=dtype)
        w_proposal = (1.0 - (1.0 + b[remaining]) * eps) / (
            1.0 - (1.0 - b[remaining]) * eps
        )

        t = 2.0 * kappa[remaining] * w_proposal + (dim - 1) * torch.log(
            (1.0 - w_proposal * b[remaining]).clamp(min=1e-30)
        ) - a[remaining]

        accept = torch.log(
            torch.rand(n_remaining, device=device, dtype=dtype).clamp(min=1e-30)
        ) < t

        idx = torch.where(remaining)[0]
        w[idx[accept]] = w_proposal[accept]
        done[idx[accept]] = True

    # Fill any remaining with mean direction (kappa -> inf => w -> 1)
    if not done.all():
        w[~done] = 1.0 - 1e-6
        logger.debug(
            "vMF rejection sampling: %d/%d samples hit max_iter",
            (~done).sum().item(), B,
        )

    return w


def _sample_uniform_tangent(
    mu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Sample a unit vector uniformly in the tangent plane of mu.

    Uses Householder reflection to map a random unit vector
    orthogonal to e_1 into one orthogonal to mu.
    """
    B, D = mu.shape
    v = torch.randn(B, D, device=device, dtype=dtype)
    # Project out the mu component
    proj = (v * mu).sum(dim=-1, keepdim=True) * mu
    v = v - proj
    v = F.normalize(v, p=2, dim=-1)
    return v


@dataclass
class VMFMixtureSamples:
    """Container for vMF mixture posterior samples."""
    samples: torch.Tensor           # (B, K, D) sampled embeddings
    component_indices: torch.Tensor  # (B, K) which ROI generated each sample
    mu_consensus: torch.Tensor      # (B, D) consensus direction
    kappa_consensus: torch.Tensor   # (B, 1) consensus concentration
    delta: torch.Tensor             # (B, 1) ROI disagreement


def sample_vmf_mixture(
    per_roi_mus: torch.Tensor,
    per_roi_kappas: torch.Tensor,
    alphas: torch.Tensor,
    num_samples: int = 8,
    consensus_mu: Optional[torch.Tensor] = None,
    consensus_kappa: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    strategy: str = "proportional",
) -> VMFMixtureSamples:
    """
    Sample from the vMF mixture posterior defined by per-ROI predictions.

    Args:
        per_roi_mus: (B, R, D) per-ROI unit-norm mean directions.
        per_roi_kappas: (B, R, 1) per-ROI concentrations.
        alphas: (B, R) attention-derived mixing weights (sum to 1).
        num_samples: Total number of samples to draw per batch element.
        consensus_mu: (B, D) precomputed consensus direction (optional).
        consensus_kappa: (B, 1) precomputed consensus concentration (optional).
        delta: (B, 1) precomputed disagreement score (optional).
        strategy: Sampling strategy:
            - "proportional": Draw from each component proportional to alpha
            - "categorical": Draw component index from Categorical(alpha) per sample
            - "top_k": Draw from top-k components by alpha weight

    Returns:
        VMFMixtureSamples with K samples per batch element.
    """
    B, R, D = per_roi_mus.shape
    device = per_roi_mus.device
    kappas_flat = per_roi_kappas.squeeze(-1)  # (B, R)

    all_samples = []
    all_components = []

    if strategy == "proportional":
        # Allocate samples proportional to mixing weights
        counts = (alphas * num_samples).round().long()  # (B, R)
        # Adjust to ensure exactly num_samples per batch element
        diff = num_samples - counts.sum(dim=-1)  # (B,)
        for b_idx in range(B):
            if diff[b_idx] != 0:
                best_roi = alphas[b_idx].argmax()
                counts[b_idx, best_roi] += diff[b_idx]

        for b_idx in range(B):
            b_samples = []
            b_components = []
            for r in range(R):
                n_r = int(counts[b_idx, r].item())
                if n_r <= 0:
                    continue
                s = _sample_vmf_wood(
                    per_roi_mus[b_idx:b_idx+1].expand(n_r, -1, -1)[:, r, :],
                    kappas_flat[b_idx, r].expand(n_r),
                    num_samples=1,
                ).squeeze(1)  # (n_r, D)
                b_samples.append(s)
                b_components.append(torch.full((n_r,), r, device=device))
            all_samples.append(torch.cat(b_samples, dim=0)[:num_samples])
            all_components.append(torch.cat(b_components, dim=0)[:num_samples])

    elif strategy == "categorical":
        component_dist = torch.distributions.Categorical(probs=alphas)
        component_idx = component_dist.sample((num_samples,)).T  # (B, num_samples)

        for b_idx in range(B):
            b_samples = []
            for k in range(num_samples):
                r = int(component_idx[b_idx, k].item())
                s = _sample_vmf_wood(
                    per_roi_mus[b_idx:b_idx+1, r],
                    kappas_flat[b_idx:b_idx+1, r],
                    num_samples=1,
                ).squeeze(1)  # (1, D)
                b_samples.append(s)
            all_samples.append(torch.cat(b_samples, dim=0))
            all_components.append(component_idx[b_idx])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    samples = torch.stack(all_samples, dim=0)       # (B, K, D)
    components = torch.stack(all_components, dim=0)  # (B, K)

    return VMFMixtureSamples(
        samples=samples,
        component_indices=components,
        mu_consensus=consensus_mu if consensus_mu is not None else per_roi_mus.mean(dim=1),
        kappa_consensus=consensus_kappa if consensus_kappa is not None else kappas_flat.mean(dim=-1, keepdim=True),
        delta=delta if delta is not None else torch.zeros(B, 1, device=device),
    )


def compute_mixture_energy_score(
    mixture_samples: VMFMixtureSamples,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Energy Score for the vMF mixture posterior (proper scoring rule).

    ES = E[||z - z_gt||] - 0.5 * E[||z - z'||]

    where z, z' are independent samples from the mixture, z_gt is ground truth.
    Evaluated on the unit sphere, so ||.|| uses Euclidean (chord) distance.
    The geodesic alternative is arccos(z^T z'), but chord distance is standard.

    Args:
        mixture_samples: VMFMixtureSamples with samples (B, K, D).
        target: (B, D) ground truth unit-norm embeddings.

    Returns:
        (B,) Energy Scores per sample (lower is better).
    """
    samples = mixture_samples.samples  # (B, K, D)
    B, K, D = samples.shape

    target_exp = target.unsqueeze(1)  # (B, 1, D)
    dist_to_target = (samples - target_exp).norm(dim=-1)  # (B, K)
    term1 = dist_to_target.mean(dim=1)  # (B,)

    # Pairwise distances between samples
    # Use efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a^T b
    # For unit vectors: = 2 - 2 a^T b
    gram = torch.bmm(samples, samples.transpose(1, 2))  # (B, K, K)
    pairwise_sq = (2.0 - 2.0 * gram).clamp(min=0.0)
    pairwise_dist = pairwise_sq.sqrt()

    # Exclude diagonal
    mask = 1.0 - torch.eye(K, device=samples.device).unsqueeze(0)
    n_pairs = K * (K - 1)
    term2 = 0.5 * (pairwise_dist * mask).sum(dim=(1, 2)) / max(n_pairs, 1)

    return term1 - term2


def select_best_from_mixture(
    mixture_samples: VMFMixtureSamples,
    clip_reranker: Optional[torch.Tensor] = None,
    selection: str = "closest_to_consensus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select the best sample from the mixture for single-image generation.

    Args:
        mixture_samples: VMFMixtureSamples with samples (B, K, D).
        clip_reranker: (B, D) optional CLIP embedding for re-ranking
                       (e.g., from a generated image back through CLIP).
        selection: Strategy:
            - "closest_to_consensus": Pick sample closest to mu_consensus
            - "highest_likelihood": Pick sample with highest mixture log-density
            - "clip_rerank": Pick sample closest to clip_reranker

    Returns:
        best_sample: (B, D) selected embedding.
        best_idx: (B,) index of selected sample.
    """
    samples = mixture_samples.samples  # (B, K, D)
    B, K, D = samples.shape

    if selection == "closest_to_consensus":
        mu = mixture_samples.mu_consensus.unsqueeze(1)  # (B, 1, D)
        cos = (samples * mu).sum(dim=-1)  # (B, K)
        best_idx = cos.argmax(dim=1)
    elif selection == "clip_rerank" and clip_reranker is not None:
        ref = clip_reranker.unsqueeze(1)  # (B, 1, D)
        cos = (samples * ref).sum(dim=-1)
        best_idx = cos.argmax(dim=1)
    elif selection == "highest_likelihood":
        mu = mixture_samples.mu_consensus.unsqueeze(1)
        kappa = mixture_samples.kappa_consensus  # (B, 1)
        log_lik = kappa * (samples * mu).sum(dim=-1)  # (B, K)
        best_idx = log_lik.argmax(dim=1)
    else:
        best_idx = torch.zeros(B, dtype=torch.long, device=samples.device)

    best_sample = samples[torch.arange(B, device=samples.device), best_idx]
    return best_sample, best_idx
