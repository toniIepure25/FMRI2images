import torch
from fmri2img.training.losses import gaussian_nll, variance_penalty


def test_nll_decreases_as_mu_matches_target():
    torch.manual_seed(0)
    target = torch.randn(4, 8)
    logvar = torch.zeros_like(target)
    mu_far = target + 0.5
    mu_close = target + 0.05

    nll_far = gaussian_nll(mu_far, logvar, target, reduction="mean", space="normalized")
    nll_close = gaussian_nll(mu_close, logvar, target, reduction="mean", space="normalized")

    assert nll_close.item() < nll_far.item()


def test_scalar_vs_diag_logvar_matches_when_constant():
    torch.manual_seed(0)
    target = torch.randn(3, 6)
    mu = torch.randn(3, 6)
    logvar_scalar = torch.zeros(3, 1)
    logvar_diag = torch.zeros_like(target)

    nll_scalar = gaussian_nll(mu, logvar_scalar, target, reduction="mean", space="normalized")
    nll_diag = gaussian_nll(mu, logvar_diag, target, reduction="mean", space="normalized")

    assert torch.isclose(nll_scalar, nll_diag, atol=1e-6)


def test_logvar_clamp_and_variance_floor_applied():
    torch.manual_seed(0)
    target = torch.zeros(2, 4)
    mu = torch.zeros(2, 4)
    logvar = torch.full((2, 4), 5.0)  # above clamp_max

    nll = gaussian_nll(
        mu,
        logvar,
        target,
        reduction="mean",
        clamp_min=-2.0,
        clamp_max=1.0,
        variance_floor=1e-3,
        space="normalized",
    )

    # Expected variance uses clamp_max=1.0
    expected_var = torch.exp(torch.tensor(1.0)) + 1e-3
    expected = 0.5 * torch.log(expected_var)
    # mean over dims then batch
    expected_loss = expected.mean()

    assert torch.isclose(nll, expected_loss, atol=1e-4)


def test_variance_penalty_respects_clamp():
    logvar = torch.tensor([[-10.0, 0.0, 5.0]])
    pen = variance_penalty(logvar, clamp_min=-2.0, clamp_max=1.0)
    # Clamped values: [-2, 0, 1] -> exp -> [e^-2, 1, e^1]
    expected = torch.exp(torch.tensor([-2.0, 0.0, 1.0])).mean()
    assert torch.isclose(pen, expected)
