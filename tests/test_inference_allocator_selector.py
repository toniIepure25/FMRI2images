import torch

from fmri2img.inference.allocator import (
    allocate_k,
    allocate_k_adaptive_mapping,
    allocate_k_adaptive_quantile,
)
from fmri2img.inference.selector import select_sample


def test_fixed_k_allocation():
    uncertainties = torch.tensor([0.1, 0.5, 0.9])
    ks = allocate_k(uncertainties, policy="fixed_k", k_base=2, k_set=(1, 2, 4))
    assert ks.tolist() == [2, 2, 2]


def test_adaptive_quantile_allocation_with_budget():
    uncertainties = torch.tensor([0.9, 0.8, 0.5, 0.2, 0.1])
    quantiles = [0.0, 0.10, 0.30, 0.60, 0.85, 1.0]
    k_bins = [16, 8, 4, 2, 1]
    ks = allocate_k_adaptive_quantile(
        uncertainties,
        quantiles=quantiles,
        k_bins=k_bins,
        enforce_budget=True,
        k_base=1,
        k_set=[1, 2, 4, 8, 16],
        budget_correction="deterministic",
    )
    assert ks.sum().item() == uncertainties.numel() * 1
    assert all(int(k.item()) in {1, 2, 4, 8, 16} for k in ks)


def test_adaptive_mapping_allocation():
    uncertainties = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    thresholds = [0.2, 0.4, 0.6, 0.8]
    k_values = [1, 2, 4, 8, 16]
    ks = allocate_k_adaptive_mapping(
        uncertainties,
        thresholds=thresholds,
        k_values=k_values,
        enforce_budget=False,
        k_base=1,
        k_set=k_values,
    )
    assert ks.tolist() == [1, 2, 4, 8, 16]


def test_cosine_and_likelihood_selection():
    samples = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([1.0, 0.0])

    selected_cosine, idx_cos, scores_cos = select_sample(samples, target=target, rule="cosine")
    assert idx_cos == 0
    assert torch.allclose(selected_cosine, samples[0])
    assert scores_cos.shape[0] == 2

    log_likelihoods = torch.tensor([0.1, 0.3])
    selected_ll, idx_ll, scores_ll = select_sample(samples, rule="likelihood", log_likelihoods=log_likelihoods)
    assert idx_ll == 1
    assert torch.allclose(selected_ll, samples[1])
    assert torch.allclose(scores_ll, log_likelihoods)

    selected_oracle, idx_oracle, _ = select_sample(samples, target=target, rule="oracle", allow_oracle=True)
    assert idx_oracle == 0
    assert torch.allclose(selected_oracle, samples[0])


def test_oracle_gate_blocks_without_flag():
    samples = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([1.0, 0.0])
    try:
        select_sample(samples, target=target, rule="oracle", allow_oracle=False)
        assert False, "oracle selection should be blocked when allow_oracle is False"
    except PermissionError:
        assert True
