"""
Kappa calibration for uncertainty-aware diffusion inference.

Collects kappa values from the validation set, computes quantiles,
and persists the result as a JSON artifact.  The calibration is then
loaded at inference time to normalise per-sample kappa into [0, 1]
for dynamic guidance-scale / step-count mapping.

Usage (training):
    cal = compute_kappa_calibration(model, val_loader, device)
    save_calibration(cal, checkpoint_dir / "kappa_calibration.json")

Usage (inference):
    cal = load_calibration(checkpoint_dir / "kappa_calibration.json")
    kappa_norm = normalise_kappa(kappa, cal)
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_kappa_calibration(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    kappa_is_log: bool = False,
) -> Dict[str, float]:
    """
    Forward-pass the validation set and collect kappa statistics.

    Args:
        model:        Trained UnifiedModel in eval mode
        val_loader:   Validation DataLoader yielding (fmri, embedding) batches
        device:       "cpu" or "cuda:N"
        kappa_is_log: If True, decoder outputs log(kappa) (legacy)

    Returns:
        Dictionary with keys: q10, q50, q90, mean, std, min, max, n_samples
    """
    model.eval()
    all_kappa: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            fmri = batch[0].to(device)
            output = model(fmri)
            if isinstance(output, tuple):
                _, aux = output
            else:
                continue

            if aux is None:
                continue

            kappa = aux.squeeze(-1)
            if kappa_is_log:
                kappa = kappa.exp()

            all_kappa.extend(kappa.cpu().tolist())

    if len(all_kappa) == 0:
        logger.warning("No kappa values collected — is the model vMF?")
        return {"q10": 0.0, "q50": 0.0, "q90": 0.0,
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "n_samples": 0}

    arr = np.array(all_kappa)
    q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9]).tolist()

    calibration = {
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_samples": len(all_kappa),
    }

    logger.info(
        f"Kappa calibration (n={calibration['n_samples']}): "
        f"q10={q10:.2f}  q50={q50:.2f}  q90={q90:.2f}  "
        f"mean={calibration['mean']:.2f}  std={calibration['std']:.2f}"
    )

    return calibration


def save_calibration(calibration: Dict[str, float], path: Path) -> None:
    """Persist calibration artifact to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Saved kappa calibration to {path}")


def load_calibration(path: Path) -> Dict[str, float]:
    """Load calibration artifact from JSON."""
    path = Path(path)
    with open(path) as f:
        cal = json.load(f)
    logger.info(
        f"Loaded kappa calibration from {path} "
        f"(q10={cal['q10']:.2f}, q90={cal['q90']:.2f}, n={cal.get('n_samples', '?')})"
    )
    return cal


def normalise_kappa(
    kappa: float,
    calibration: Dict[str, float],
) -> float:
    """
    Map a raw kappa value to [0, 1] using calibration quantiles.

    kappa_norm = clamp((kappa - q10) / (q90 - q10), 0, 1)
    """
    q10 = calibration["q10"]
    q90 = calibration["q90"]
    denom = q90 - q10
    if denom < 1e-8:
        return 0.5
    return max(0.0, min(1.0, (kappa - q10) / denom))
