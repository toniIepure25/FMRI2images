"""Probabilistic inference utilities (sampling + selection + logging)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Optional, Any, List

import torch

from fmri2img.inference.allocator import allocate_k
from fmri2img.inference.selector import select_sample
from fmri2img.training.losses import gaussian_nll


@dataclass
class SamplingConfig:
    sampling_policy: str = "fixed_k"
    k_base: int = 1
    k_set: Sequence[int] = (1,)
    adaptive_quantile: Tuple[Sequence[float], Sequence[int], str] = ((), (), "deterministic")
    adaptive_mapping: Tuple[Sequence[float], Sequence[int]] = ((), ())
    logvar_min: float = -8.0
    logvar_max: float = 2.0
    variance_floor: float = 1e-6
    clip_space: str = "normalized"
    enforce_budget: bool = True


@dataclass
class TrialResult:
    trial_id: int
    stimulus_id: int
    split: str
    subject: str
    uncertainty: float
    embedding_error: Optional[float]
    nll: Optional[float]
    mu_cosine: Optional[float]
    mu_mse: Optional[float]
    K_assigned: int
    K_base: int
    K_set: Sequence[int]
    sampling_policy: str
    budget_target: int
    budget_actual: int
    selection_rule: str
    allow_oracle: bool
    chosen_k: int
    best_score: float
    score_min: float
    score_mean: float
    score_max: float
    diffusion_calls: int
    diffusion_steps: int
    guidance_scale: float
    seed_base: int
    wall_time_ms: float
    is_oracle_run: bool


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, logvar_min: float, logvar_max: float, variance_floor: float) -> torch.Tensor:
    logvar_clamped = logvar.clamp(logvar_min, logvar_max)
    var = torch.exp(logvar_clamped).clamp_min(variance_floor)
    std = torch.sqrt(var)
    eps = torch.randn_like(std)
    return mu + eps * std


def run_probabilistic_trial(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    K_i: int,
    selection_rule: str,
    allow_oracle: bool,
    sampling_cfg: SamplingConfig,
    generate_fn: Callable[[torch.Tensor, int], Any],
    embed_image_fn: Callable[[Any], torch.Tensor],
    seed_base: int,
    trial_id: int,
    guidance_scale: float,
    diffusion_steps: int,
) -> Dict[str, Any]:
    """
    Run sampling + selection for a single trial.
    generate_fn returns an image-like object given (z, seed).
    embed_image_fn maps that object to a CLIP embedding tensor.
    """
    scores: List[torch.Tensor] = []
    images: List[Any] = []
    zs: List[torch.Tensor] = []
    log_liks: List[torch.Tensor] = []

    t0 = time.time()

    for k in range(K_i):
        z = reparameterize(mu, logvar, sampling_cfg.logvar_min, sampling_cfg.logvar_max, sampling_cfg.variance_floor)
        if sampling_cfg.clip_space == "normalized":
            z = torch.nn.functional.normalize(z, dim=-1)
        zs.append(z)
        img = generate_fn(z, seed_base + trial_id * 1000 + k)
        images.append(img)
        phi = embed_image_fn(img)
        if sampling_cfg.clip_space == "normalized":
            phi = torch.nn.functional.normalize(phi, dim=-1)
        if selection_rule == "likelihood":
            ll = -gaussian_nll(
                mu.unsqueeze(0),
                logvar.unsqueeze(0),
                phi.unsqueeze(0),
                clamp_min=sampling_cfg.logvar_min,
                clamp_max=sampling_cfg.logvar_max,
                variance_floor=sampling_cfg.variance_floor,
                reduction="mean",
                space=sampling_cfg.clip_space,
            )
            log_liks.append(ll.squeeze())
            scores.append(ll.squeeze())
        else:
            cos = torch.nn.functional.cosine_similarity(phi.unsqueeze(0), target.unsqueeze(0), dim=-1).squeeze()
            scores.append(cos)
            log_liks.append(cos)

    score_tensor = torch.stack(scores)
    loglik_tensor = torch.stack(log_liks)
    zs_tensor = torch.stack(zs)

    selected, idx, score_used = select_sample(
        samples=zs_tensor,
        target=target,
        rule=selection_rule,
        log_likelihoods=loglik_tensor,
        allow_oracle=allow_oracle,
    )
    chosen_image = images[int(idx)]

    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "chosen_z": selected,
        "chosen_image": chosen_image,
        "scores": score_tensor,
        "chosen_score": score_used[idx],
        "chosen_k": int(idx),
        "log_likelihoods": loglik_tensor,
        "elapsed_ms": elapsed_ms,
        "diffusion_calls": K_i,
        "guidance_scale": guidance_scale,
        "diffusion_steps": diffusion_steps,
    }


def run_probabilistic_trials(
    mus: torch.Tensor,
    logvars: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
    sampling_cfg: SamplingConfig,
    selection_rule: str,
    allow_oracle: bool,
    generate_fn: Callable[[torch.Tensor, int], Any],
    embed_image_fn: Callable[[Any], torch.Tensor],
    seed_base: int,
    guidance_scale: float,
    diffusion_steps: int,
    stimulus_ids: Sequence[int],
    subject: str,
    split: str,
) -> Tuple[List[TrialResult], List[Any], Dict[str, float]]:
    Ks = allocate_k(
        uncertainties,
        policy=sampling_cfg.sampling_policy,
        k_base=sampling_cfg.k_base,
        k_set=sampling_cfg.k_set,
        enforce_budget=sampling_cfg.enforce_budget,
        adaptive_quantile=sampling_cfg.adaptive_quantile,
        adaptive_mapping=sampling_cfg.adaptive_mapping,
    )
    budget_target = int(len(uncertainties) * sampling_cfg.k_base)
    budget_actual = int(Ks.sum().item())

    trial_results: List[TrialResult] = []
    chosen_images: List[Any] = []

    for i in range(len(uncertainties)):
        K_i = int(Ks[i].item())
        out = run_probabilistic_trial(
            mu=mus[i],
            logvar=logvars[i],
            target=targets[i],
            K_i=K_i,
            selection_rule=selection_rule,
            allow_oracle=allow_oracle,
            sampling_cfg=sampling_cfg,
            generate_fn=generate_fn,
            embed_image_fn=embed_image_fn,
            seed_base=seed_base,
            trial_id=i,
            guidance_scale=guidance_scale,
            diffusion_steps=diffusion_steps,
        )

        scores = out["scores"]
        trial_results.append(
            TrialResult(
                trial_id=i,
                stimulus_id=int(stimulus_ids[i]),
                split=split,
                subject=subject,
                uncertainty=float(uncertainties[i].item()),
                embedding_error=None,
                nll=float(out["log_likelihoods"].mean().item()) if selection_rule == "likelihood" else None,
                mu_cosine=None,
                mu_mse=None,
                K_assigned=K_i,
                K_base=sampling_cfg.k_base,
                K_set=sampling_cfg.k_set,
                sampling_policy=sampling_cfg.sampling_policy,
                budget_target=budget_target,
                budget_actual=budget_actual,
                selection_rule=selection_rule,
                allow_oracle=allow_oracle,
                chosen_k=int(out["chosen_k"]),
                best_score=float(out["chosen_score"].item()),
                score_min=float(scores.min().item()),
                score_mean=float(scores.mean().item()),
                score_max=float(scores.max().item()),
                diffusion_calls=out["diffusion_calls"],
                diffusion_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                seed_base=seed_base,
                wall_time_ms=out["elapsed_ms"],
                is_oracle_run=selection_rule == "oracle",
            )
        )
        chosen_images.append(out["chosen_image"])

    budget_stats = {
        "budget_target": float(budget_target),
        "budget_actual": float(budget_actual),
        "mean_K": float(Ks.float().mean().item()),
        "max_K": int(Ks.max().item()) if len(Ks) > 0 else 0,
        "min_K": int(Ks.min().item()) if len(Ks) > 0 else 0,
    }

    return trial_results, chosen_images, budget_stats
