import math

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from fmri2img.models.encoders import (  # noqa: E402
    ProbabilisticMultiLayerTwoStageEncoder,
    TwoStageEncoder,
)
from fmri2img.training.train_two_stage import (  # noqa: E402
    train_epoch_multilayer,
    evaluate_epoch_multilayer,
    validate_probabilistic_setup,
)
from fmri2img.training.losses import ProbabilisticMultiLayerLoss  # noqa: E402
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


class _TinyProbDataset(Dataset):
    def __init__(self, n: int, input_dim: int):
        self.X = torch.randn(n, input_dim)
        # Use small dims for speed but match encoder defaults (final=512)
        self.targets = {
            'final': torch.randn(n, 512)
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}


def test_probabilistic_metrics_populated_cpu():
    input_dim = 8
    dataset = _TinyProbDataset(n=4, input_dim=input_dim)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ProbabilisticMultiLayerTwoStageEncoder(
        input_dim=input_dim,
        latent_dim=8,
        n_blocks=1,
        head_hidden_dim=16,
        enabled_layers=['final'],
        uncertainty='diag',
        clip_space='normalized',
    )

    criterion = ProbabilisticMultiLayerLoss(
        layer_weights={'final': 1.0},
        use_mse=False,
        mse_weight=0.0,
        kl_weight_max=1e-4,
        kl_anneal_epochs=1,
        text_clip_weight=0.0,
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)

    prob_cfg = {
        'nll_weight': 1.0,
        'cosine_aux_weight': 0.0,
        'variance_penalty_weight': 0.0,
        'nll_reduction': 'mean',
        'logvar_min': -8.0,
        'logvar_max': 2.0,
        'variance_floor': 1e-6,
        'clip_space': 'normalized',
    }

    # One tiny train/eval pass
    train_epoch_multilayer(
        model, loader, optimizer, criterion, device='cpu', epoch=1,
        probabilistic=True, prob_cfg=prob_cfg
    )
    metrics = evaluate_epoch_multilayer(
        model, loader, device='cpu', probabilistic=True, prob_cfg=prob_cfg
    )

    for key in ['nll', 'logvar_mean', 'uncertainty_mean', 'embedding_error_mean']:
        assert key in metrics
        assert metrics[key] is not None
        assert not math.isnan(metrics[key])


def test_probabilistic_guard_raises_for_deterministic_model():
    model = TwoStageEncoder(input_dim=8, latent_dim=8, n_blocks=1, dropout=0.1, head_type='linear', head_hidden_dim=8)
    prob_cfg = {'nll_weight': 1.0, 'predict_logvar': True}
    with pytest.raises(ValueError):
        validate_probabilistic_setup(model, prob_cfg, probabilistic_enabled=True)


def test_probabilistic_metrics_single_val_has_no_nan():
    # Single-sample loader to exercise small-split robustness
    input_dim = 8
    dataset = _TinyProbDataset(n=1, input_dim=input_dim)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ProbabilisticMultiLayerTwoStageEncoder(
        input_dim=input_dim,
        latent_dim=8,
        n_blocks=1,
        head_hidden_dim=16,
        enabled_layers=['final'],
        uncertainty='diag',
        clip_space='normalized',
    )

    prob_cfg = {
        'nll_weight': 1.0,
        'cosine_aux_weight': 0.0,
        'variance_penalty_weight': 0.0,
        'nll_reduction': 'mean',
        'logvar_min': -8.0,
        'logvar_max': 2.0,
        'variance_floor': 1e-6,
        'clip_space': 'normalized',
    }

    metrics = evaluate_epoch_multilayer(
        model, loader, device='cpu', probabilistic=True, prob_cfg=prob_cfg
    )

    for key in ['uncertainty_std', 'logvar_std', 'spearman_u_e']:
        assert key in metrics
        assert metrics[key] == metrics[key]  # not NaN
        assert metrics[key] >= 0 or key == 'spearman_u_e'
