"""Gradient-flow and hierarchy-direction tests for the Predictive Cortical Decoder.

These tests encode two Gate 0 audit findings so they cannot silently regress:

1. ``PerLevelKappaHeads`` receives **no gradient** from any loss that depends
   only on ``(mu, kappa)`` -- which is exactly what ``PCDModel.forward``
   exposes to the training loop.  Any scientific claim about "per-level
   uncertainty" therefore describes randomly-initialised weights.
   ``test_per_level_kappa_heads_are_unsupervised`` documents the current
   (broken) contract; ``test_per_level_kappa_heads_trainable_when_supervised``
   shows the heads are reachable once a loss actually touches them.

2. Cross-level prediction flows **lower -> higher** (level k predicts the
   tokens of level k+1).  This is feed-forward residual extraction, not
   Rao & Ballard (1999) predictive coding, which requires predictions to
   descend from higher to lower levels.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from fmri2img.models.predictive_cortical_decoder import (
    HIERARCHY_LEVELS,
    LEVEL_NAMES,
    PredictiveCorticalDecoder,
)

N_VOXELS_PER_ROI = 12


def _build_model(**overrides) -> PredictiveCorticalDecoder:
    """Construct a small single-subject PCD over the 17 canonical ROIs."""
    roi_indices: OrderedDict = OrderedDict()
    cursor = 0
    for name in HIERARCHY_LEVELS:
        roi_indices[name] = torch.arange(
            cursor, cursor + N_VOXELS_PER_ROI, dtype=torch.long
        )
        cursor += N_VOXELS_PER_ROI

    kwargs = dict(
        roi_indices=roi_indices,
        d_model=32,
        nhead=4,
        layers_per_level=1,
        output_dim=16,
        enable_per_level_kappa=True,
    )
    kwargs.update(overrides)
    model = PredictiveCorticalDecoder(**kwargs)
    model._test_n_voxels = cursor  # type: ignore[attr-defined]
    return model


def _forward(model: PredictiveCorticalDecoder, batch: int = 4):
    x = torch.randn(batch, model._test_n_voxels)  # type: ignore[attr-defined]
    return model(x, return_details=True)


def test_per_level_kappa_heads_are_unsupervised() -> None:
    """A (mu, kappa)-only loss must not reach the per-level kappa heads.

    This is the exact signal ``PCDModel.forward`` hands to the training loop;
    ``level_kappas`` is stashed in ``_last_pcd_extras`` and read by no loss.
    Regression guard for the Gate 0 finding: these heads never train.
    """
    torch.manual_seed(0)
    model = _build_model()
    out = _forward(model)

    loss = out.mu.pow(2).sum() + out.kappa.sum()
    loss.backward()

    for i, head in enumerate(model.level_kappa_heads.heads):
        assert head.weight.grad is None, (
            f"level_kappa_heads[{i}] ({LEVEL_NAMES[i]}) unexpectedly received a "
            "gradient. If a per-level kappa objective was added, update this "
            "test and the claim registry -- per-level uncertainty may now be "
            "an identifiable quantity."
        )

    # Every other component must train, proving the loss itself is live.
    for name, module in [
        ("prediction_heads", model.prediction_heads),
        ("level_encoders", model.level_encoders),
        ("aggregator", model.aggregator),
        ("vmf_decoder", model.vmf_decoder),
        ("roi_projections", model.roi_projections),
    ]:
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad) > 0
            for p in module.parameters()
        ), f"{name} received no gradient; the probe loss is not live."


def test_level_kappas_are_in_graph_but_orphaned() -> None:
    """``level_kappas`` has a grad_fn yet is never consumed.

    This is why the bug is easy to miss on inspection: the tensor looks
    connected, but nothing backpropagates through it.
    """
    torch.manual_seed(0)
    model = _build_model()
    out = _forward(model)
    assert out.level_kappas is not None
    assert out.level_kappas.grad_fn is not None
    assert out.level_kappas.shape == (4, len(LEVEL_NAMES))


def test_per_level_kappa_heads_trainable_when_supervised() -> None:
    """The heads are reachable -- they simply lack an objective today."""
    torch.manual_seed(0)
    model = _build_model()
    out = _forward(model)

    out.level_kappas.sum().backward()

    for i, head in enumerate(model.level_kappa_heads.heads):
        assert head.weight.grad is not None, (
            f"level_kappa_heads[{i}] is unreachable even under direct "
            "supervision, which would indicate a wiring bug rather than a "
            "missing loss term."
        )


def test_prediction_flows_low_to_high_not_rao_ballard() -> None:
    """Prediction head k must map level k -> tokens of level k+1.

    Canonical predictive coding (Rao & Ballard, 1999) predicts *downward*.
    This asserts the implemented direction is the opposite, so the mismatch
    between code and the module's docstring cannot be forgotten.
    """
    torch.manual_seed(0)
    model = _build_model()

    n_tokens = model._n_tokens_per_level
    assert n_tokens == [4, 5, 7, 1], f"unexpected level sizes: {n_tokens}"

    assert len(model.prediction_heads) == 2
    for k, head in enumerate(model.prediction_heads):
        assert head.n_target == n_tokens[k + 1], (
            f"prediction_heads[{k}] targets {head.n_target} tokens but level "
            f"{k + 1} has {n_tokens[k + 1]}. The head predicts the HIGHER "
            "level from the LOWER level (feed-forward), which is the "
            "direction this test pins down."
        )


def test_residual_level_bypasses_predictive_chain() -> None:
    """``nsdgeneral_other`` (level 3) is encoded directly, not as a residual.

    Level 3 holds the majority of voxels yet receives no prediction and
    contributes an unmediated path into the aggregator.  Any claim that the
    decoder is 'driven by prediction errors' must account for this bypass.
    """
    torch.manual_seed(0)
    model = _build_model()
    out = _forward(model)

    # Errors exist only for levels 1 and 2; levels 0 and 3 are raw.
    assert out.prediction_errors[0] is None, "level 0 must be raw"
    assert out.prediction_errors[1] is not None, "level 1 must be an error"
    assert out.prediction_errors[2] is not None, "level 2 must be an error"
    assert out.prediction_errors[3] is None, "level 3 (residual) must be raw"


@pytest.mark.parametrize("mode", ["full", "no_errors", "reversed", "random"])
def test_ablation_modes_construct_and_run(mode: str) -> None:
    """All four ablation arms must build and produce finite outputs."""
    torch.manual_seed(0)
    model = _build_model(ablation_mode=mode)
    out = _forward(model)

    assert torch.isfinite(out.mu).all(), f"non-finite mu in ablation={mode}"
    assert torch.isfinite(out.kappa).all(), f"non-finite kappa in ablation={mode}"
    assert (out.kappa > 0).all(), f"kappa must stay positive in ablation={mode}"
