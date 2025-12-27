from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def cosine_select(samples: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Select sample with highest cosine similarity to target."""
    if samples.dim() == 2:
        samples_flat = samples
    else:
        samples_flat = samples.view(samples.shape[0], -1)
    target_flat = target.view(1, -1)
    sims = F.cosine_similarity(samples_flat, target_flat.expand_as(samples_flat), dim=-1)
    best_idx = int(torch.argmax(sims).item())
    return samples[best_idx], best_idx, sims


def likelihood_select(samples: torch.Tensor, log_likelihoods: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
    if log_likelihoods.shape[0] != samples.shape[0]:
        raise ValueError("log_likelihoods length must match samples")
    best_idx = int(torch.argmax(log_likelihoods).item())
    return samples[best_idx], best_idx, log_likelihoods


def oracle_select(samples: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
    # Oracle uses ground-truth target → same as cosine selection upper bound
    return cosine_select(samples, target)


def select_sample(
    samples: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    rule: str = "cosine",
    log_likelihoods: Optional[torch.Tensor] = None,
    allow_oracle: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Select the best sample given a rule.

    Args:
        samples: Tensor of shape (num_samples, dim)
        target: Ground truth embedding (required for cosine/oracle)
        rule: Selection rule ('cosine', 'likelihood', 'oracle')
        log_likelihoods: Optional log-likelihoods for 'likelihood' rule

    Returns:
        selected_sample: Tensor
        index: int index of selected sample
        scores: Tensor of scores used for selection
    """
    rule = rule.lower()
    if rule in {"cosine", "oracle"} and target is None:
        raise ValueError("target is required for cosine/oracle selection")
    if rule == "cosine":
        return cosine_select(samples, target)  # type: ignore[arg-type]
    if rule == "likelihood":
        if log_likelihoods is None:
            raise ValueError("log_likelihoods required for likelihood selection")
        return likelihood_select(samples, log_likelihoods)
    if rule == "oracle":
        if not allow_oracle:
            raise PermissionError("oracle selection is disabled; set allow_oracle=True to enable")
        return oracle_select(samples, target)  # type: ignore[arg-type]
    raise ValueError(f"Unknown selection rule: {rule}")
