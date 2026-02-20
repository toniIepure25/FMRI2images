"""
Decomposed Uncertainty-Aware Classifier-Free Guidance (DUA-CFG)
===============================================================

Principled mapping from dual uncertainty signals (kappa, delta) to
diffusion generation parameters.  Unlike the heuristic UA-CFG that
collapses kappa and delta into a single confidence scalar, DUA-CFG
treats them as distinct uncertainty types:

    kappa (concentration)  ->  aleatoric / within-region noise
        Controls: CFG guidance scale w
        Rationale: Low kappa means noisy data -> moderate CFG to
                   avoid hallucinating details; high kappa -> strong CFG.

    delta (ROI disagreement) ->  epistemic-like / between-region conflict
        Controls: Number of ensemble samples K and diffusion steps
        Rationale: High delta means ROIs disagree about content ->
                   need multiple diverse samples to cover possibilities.

Theoretical connection:
    The decomposition is a special case of the stochastic optimal control
    framework for adaptive guidance (Karras et al., 2025) where the
    "classifier confidence" comes from a vMF mixture posterior.

Usage:
    cfg = DecomposedUACFG(config)
    for kappa, delta in predictions:
        w = cfg.guidance_scale(kappa, delta)
        K = cfg.ensemble_size(delta)
        steps = cfg.diffusion_steps(kappa, delta)

References:
    - Ho & Salimans (2022) Classifier-Free Diffusion Guidance
    - Karras et al. (2025) Adaptive Diffusion Guidance via SOC
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DUACFGConfig:
    """Configuration for Decomposed UA-CFG."""
    # Guidance scale mapping (kappa -> w)
    w_min: float = 1.5
    w_max: float = 12.0
    gamma_w: float = 1.0         # Power curve exponent for kappa -> w

    # Ensemble size mapping (delta -> K)
    k_min: int = 1
    k_max: int = 8
    delta_threshold_low: float = 0.2   # Below this: K = k_min (confident)
    delta_threshold_high: float = 0.6  # Above this: K = k_max (conflicting)

    # Diffusion steps mapping (combined kappa + delta -> steps)
    steps_min: int = 10
    steps_max: int = 50
    gamma_steps: float = 0.5    # Power curve exponent

    # Abstention threshold
    abstain_kappa_quantile: float = 0.05  # Abstain if kappa below this quantile
    abstain_delta_threshold: float = 0.85  # Abstain if delta above this

    # Calibration
    kappa_q10: float = 1.0
    kappa_q50: float = 50.0
    kappa_q90: float = 200.0


class DecomposedUACFG:
    """
    Decomposed Uncertainty-Aware CFG for diffusion generation.

    Maps dual uncertainty signals to independent generation parameters,
    treating kappa (aleatoric) and delta (epistemic-like) as distinct
    information sources.
    """

    def __init__(self, config: Union[DUACFGConfig, Dict]):
        if isinstance(config, dict):
            config = DUACFGConfig(**{
                k: v for k, v in config.items()
                if k in DUACFGConfig.__dataclass_fields__
            })
        self.cfg = config
        logger.info(
            "DecomposedUACFG: w=[%.1f, %.1f], K=[%d, %d], "
            "steps=[%d, %d], gamma_w=%.2f",
            config.w_min, config.w_max, config.k_min, config.k_max,
            config.steps_min, config.steps_max, config.gamma_w,
        )

    def load_calibration(self, path: Path) -> None:
        """Load kappa calibration quantiles from a JSON file."""
        with open(path) as f:
            cal = json.load(f)
        self.cfg.kappa_q10 = cal.get("q10", self.cfg.kappa_q10)
        self.cfg.kappa_q50 = cal.get("q50", self.cfg.kappa_q50)
        self.cfg.kappa_q90 = cal.get("q90", self.cfg.kappa_q90)
        logger.info(
            "Loaded calibration: q10=%.2f, q50=%.2f, q90=%.2f",
            self.cfg.kappa_q10, self.cfg.kappa_q50, self.cfg.kappa_q90,
        )

    def normalize_kappa(self, kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map raw kappa to [0, 1] using calibration quantiles."""
        normed = (kappa - self.cfg.kappa_q10) / (
            self.cfg.kappa_q90 - self.cfg.kappa_q10 + 1e-8
        )
        return np.clip(normed, 0.0, 1.0) if isinstance(kappa, np.ndarray) else max(0.0, min(1.0, normed))

    def guidance_scale(
        self,
        kappa: Union[float, np.ndarray],
        delta: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Compute per-sample CFG guidance scale from kappa (aleatoric).

        w = w_min + kappa_norm^gamma * (w_max - w_min)

        Delta modulates weakly: high disagreement slightly reduces guidance
        to avoid amplifying conflicting signals.

        Args:
            kappa: Raw concentration value(s).
            delta: ROI disagreement score(s) in [0, 1], optional.

        Returns:
            Guidance scale(s).
        """
        kn = self.normalize_kappa(kappa)
        w_range = self.cfg.w_max - self.cfg.w_min

        if isinstance(kn, np.ndarray):
            w = self.cfg.w_min + np.power(kn, self.cfg.gamma_w) * w_range
            if delta is not None:
                damping = 1.0 - 0.15 * np.clip(delta, 0.0, 1.0)
                w = self.cfg.w_min + (w - self.cfg.w_min) * damping
        else:
            w = self.cfg.w_min + (kn ** self.cfg.gamma_w) * w_range
            if delta is not None:
                damping = 1.0 - 0.15 * max(0.0, min(1.0, delta))
                w = self.cfg.w_min + (w - self.cfg.w_min) * damping

        return w

    def ensemble_size(
        self,
        delta: Union[float, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """
        Compute number of ensemble samples from delta (epistemic-like).

        Low delta (ROIs agree) -> K_min (single sample sufficient).
        High delta (ROIs disagree) -> K_max (need diversity).

        Args:
            delta: ROI disagreement score(s) in [0, 1].

        Returns:
            Number of ensemble samples K.
        """
        lo = self.cfg.delta_threshold_low
        hi = self.cfg.delta_threshold_high
        k_range = self.cfg.k_max - self.cfg.k_min

        if isinstance(delta, np.ndarray):
            t = np.clip((delta - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            k = self.cfg.k_min + np.round(t * k_range).astype(int)
            return k
        else:
            t = max(0.0, min(1.0, (delta - lo) / (hi - lo + 1e-8)))
            return self.cfg.k_min + round(t * k_range)

    def diffusion_steps(
        self,
        kappa: Union[float, np.ndarray],
        delta: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[int, np.ndarray]:
        """
        Compute diffusion steps from combined uncertainty.

        High confidence (high kappa, low delta) -> fewer steps.
        Low confidence -> more steps for careful denoising.

        Args:
            kappa: Raw concentration value(s).
            delta: ROI disagreement score(s) in [0, 1], optional.

        Returns:
            Number of diffusion steps.
        """
        kn = self.normalize_kappa(kappa)
        step_range = self.cfg.steps_max - self.cfg.steps_min

        # Invert: more steps for LOW confidence
        if isinstance(kn, np.ndarray):
            confidence = np.power(kn, self.cfg.gamma_steps)
            if delta is not None:
                confidence = confidence * (1.0 - 0.5 * np.clip(delta, 0.0, 1.0))
            steps = self.cfg.steps_max - np.round(confidence * step_range).astype(int)
            return np.clip(steps, self.cfg.steps_min, self.cfg.steps_max)
        else:
            confidence = kn ** self.cfg.gamma_steps
            if delta is not None:
                confidence = confidence * (1.0 - 0.5 * max(0.0, min(1.0, delta)))
            steps = self.cfg.steps_max - round(confidence * step_range)
            return max(self.cfg.steps_min, min(self.cfg.steps_max, steps))

    def should_abstain(
        self,
        kappa: float,
        delta: float,
    ) -> bool:
        """
        Determine whether to abstain from generation.

        Abstain when both uncertainty sources indicate the prediction
        is unreliable.
        """
        kn = self.normalize_kappa(kappa)
        return (kn < self.cfg.abstain_kappa_quantile and
                delta > self.cfg.abstain_delta_threshold)

    def get_generation_params(
        self,
        kappa: float,
        delta: float,
    ) -> Dict:
        """
        Get all generation parameters for a single sample.

        Returns:
            Dict with keys: guidance_scale, num_steps, ensemble_k, abstain,
                           kappa_norm, confidence_level
        """
        kn = self.normalize_kappa(kappa)
        abstain = self.should_abstain(kappa, delta)

        if abstain:
            confidence_level = "abstain"
        elif kn > 0.7 and delta < 0.2:
            confidence_level = "high"
        elif kn < 0.3 or delta > 0.5:
            confidence_level = "low"
        else:
            confidence_level = "medium"

        return {
            "guidance_scale": self.guidance_scale(kappa, delta),
            "num_steps": self.diffusion_steps(kappa, delta),
            "ensemble_k": self.ensemble_size(delta),
            "abstain": abstain,
            "kappa_norm": kn,
            "delta": delta,
            "confidence_level": confidence_level,
        }

    def batch_generation_params(
        self,
        kappas: np.ndarray,
        deltas: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute generation parameters for a batch of samples.

        Args:
            kappas: (N,) raw concentration values.
            deltas: (N,) ROI disagreement scores.

        Returns:
            Dict of arrays: guidance_scales, num_steps, ensemble_ks,
                           kappa_norms, abstain_mask.
        """
        return {
            "guidance_scales": self.guidance_scale(kappas, deltas),
            "num_steps": self.diffusion_steps(kappas, deltas),
            "ensemble_ks": self.ensemble_size(deltas),
            "kappa_norms": self.normalize_kappa(kappas),
            "abstain_mask": np.array([
                self.should_abstain(float(k), float(d))
                for k, d in zip(kappas, deltas)
            ]),
        }
