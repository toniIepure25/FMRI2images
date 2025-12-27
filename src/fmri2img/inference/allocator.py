from typing import Sequence, Tuple

import torch


def _next_lower(value: int, allowed: Sequence[int]) -> int:
    sorted_vals = sorted(set(allowed))
    for v in reversed(sorted_vals):
        if v < value:
            return v
    return value


def _next_higher(value: int, allowed: Sequence[int]) -> int:
    sorted_vals = sorted(set(allowed))
    for v in sorted_vals:
        if v > value:
            return v
    return value


def _enforce_budget(k_values: torch.Tensor, allowed: Sequence[int], target_total: int) -> torch.Tensor:
    """Deterministically adjust K allocations to hit a target budget."""
    allowed_set = sorted(set(allowed))
    ks = k_values.clone()
    total = int(ks.sum().item())
    n = ks.numel()

    # Early exit
    if total == target_total:
        return ks

    # Reduce budget by stepping down from largest allocations
    if total > target_total:
        # iterate in descending order of K
        while total > target_total:
            # pick index with largest K (stable by lower index first)
            max_k = ks.max().item()
            max_indices = (ks == max_k).nonzero(as_tuple=True)[0]
            if max_indices.numel() == 0:
                break
            idx = int(max_indices[0].item())
            lower = _next_lower(int(max_k), allowed_set)
            if lower == max_k:  # cannot reduce further
                break
            ks[idx] = lower
            total = int(ks.sum().item())
    else:
        # Increase budget by stepping up from smallest allocations
        while total < target_total:
            min_k = ks.min().item()
            min_indices = (ks == min_k).nonzero(as_tuple=True)[0]
            if min_indices.numel() == 0:
                break
            idx = int(min_indices[0].item())
            higher = _next_higher(int(min_k), allowed_set)
            if higher == min_k:  # cannot increase further
                break
            ks[idx] = higher
            total = int(ks.sum().item())
    # Final clamp to allowed set
    for i in range(n):
        if ks[i].item() not in allowed_set:
            # snap to nearest allowed
            diffs = [abs(int(ks[i].item()) - v) for v in allowed_set]
            ks[i] = allowed_set[int(torch.tensor(diffs).argmin().item())]
    return ks


def allocate_k_fixed(num_items: int, k_base: int) -> torch.Tensor:
    return torch.full((num_items,), k_base, dtype=torch.int64)


def allocate_k_adaptive_quantile(
    uncertainties: torch.Tensor,
    quantiles: Sequence[float],
    k_bins: Sequence[int],
    enforce_budget: bool,
    k_base: int,
    k_set: Sequence[int],
    budget_correction: str = "deterministic",
) -> torch.Tensor:
    if len(quantiles) < 2 or len(k_bins) != len(quantiles) - 1:
        raise ValueError("quantiles must be len>=2 and k_bins length must be len(quantiles)-1")
    if uncertainties.numel() == 0:
        return torch.tensor([], dtype=torch.int64)

    n = uncertainties.numel()
    # Percentile ranks on descending uncertainty (0 = most uncertain)
    sorted_idx = torch.argsort(uncertainties, descending=True)
    ranks = torch.empty_like(sorted_idx, dtype=torch.float32)
    ranks[sorted_idx] = torch.arange(n, device=uncertainties.device, dtype=torch.float32) / float(n)

    ks = torch.empty(n, dtype=torch.int64)
    for i in range(len(k_bins)):
        lower_q, upper_q = quantiles[i], quantiles[i + 1]
        mask = (ranks >= lower_q) & (ranks < upper_q if i < len(k_bins) - 1 else ranks <= upper_q)
        ks[mask] = k_bins[i]

    if enforce_budget and budget_correction == "deterministic":
        target_total = n * k_base
        ks = _enforce_budget(ks, k_set, target_total)
    return ks


def allocate_k_adaptive_mapping(
    uncertainties: torch.Tensor,
    thresholds: Sequence[float],
    k_values: Sequence[int],
    enforce_budget: bool,
    k_base: int,
    k_set: Sequence[int],
) -> torch.Tensor:
    if len(k_values) != len(thresholds) + 1:
        raise ValueError("k_values length must be thresholds length + 1")
    if uncertainties.numel() == 0:
        return torch.tensor([], dtype=torch.int64)

    ks = torch.empty_like(uncertainties, dtype=torch.int64)
    for idx, u in enumerate(uncertainties):
        bucket = sum(float(u.item()) >= t for t in thresholds)
        ks[idx] = k_values[bucket]

    if enforce_budget:
        ks = _enforce_budget(ks, k_set, uncertainties.numel() * k_base)
    return ks


def allocate_k(
    uncertainties: torch.Tensor,
    policy: str = "fixed_k",
    k_base: int = 1,
    k_set: Sequence[int] = (1,),
    enforce_budget: bool = True,
    adaptive_quantile: Tuple[Sequence[float], Sequence[int], str] = ((), (), "deterministic"),
    adaptive_mapping: Tuple[Sequence[float], Sequence[int]] = ((), ()),
) -> torch.Tensor:
    """Allocate sample counts K per item based on uncertainty-driven policy."""
    policy = policy.lower()
    if policy == "fixed_k":
        ks = allocate_k_fixed(uncertainties.numel(), k_base)
    elif policy == "adaptive_quantile":
        quantiles, k_bins, budget_correction = adaptive_quantile
        ks = allocate_k_adaptive_quantile(
            uncertainties,
            quantiles=quantiles,
            k_bins=k_bins,
            enforce_budget=enforce_budget,
            k_base=k_base,
            k_set=k_set,
            budget_correction=budget_correction,
        )
    elif policy == "adaptive_mapping":
        thresholds, k_values = adaptive_mapping
        ks = allocate_k_adaptive_mapping(
            uncertainties,
            thresholds=thresholds,
            k_values=k_values,
            enforce_budget=enforce_budget,
            k_base=k_base,
            k_set=k_set,
        )
    else:
        raise ValueError(f"Unknown sampling policy: {policy}")

    # Ensure Ks are from allowed set
    allowed = set(k_set)
    ks = ks.clone()
    for i in range(ks.numel()):
        if int(ks[i].item()) not in allowed:
            ks[i] = min(k_set, key=lambda v: abs(v - int(ks[i].item())))
    return ks
