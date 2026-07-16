"""Contract tests for the Neural-Constrained Decoder (NCD).

These encode the design constraints the Gate 0 audit produced. They are not
incidental unit tests -- each one guards a specific documented failure:

* ``test_neural_heads_receive_gradient`` -- the T8/F-002 contract. PCD shipped an
  interpreted quantity (per-level kappa) that no loss reached, so it stayed at
  random init and produced stable, plausible, meaningless figures. NCD's only
  interpreted quantity is ``neural_pred``, and the loss must reach it.
* ``test_masked_roi_voxels_cannot_influence_own_prediction`` -- structural
  leakage safety of the masked-ROI objective.
* ``test_context_node_cannot_dominate_by_capacity`` -- finding T7, where
  ``nsdgeneral_other`` held ~64% of voxels and bypassed the hierarchy.
* ``test_subject_capacity_is_low_rank`` -- finding F-001, where unrestricted
  per-subject projections were ~57% of parameters and overfit by 82pp.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from fmri2img.models.neural_constrained_decoder import (
    CONTEXT_ROI,
    NeuralConstrainedDecoder,
)

ROI_VOXELS = OrderedDict(
    [
        ("V1v", 20), ("V1d", 20), ("V2v", 20), ("V2d", 20),
        ("V3v", 16), ("V3d", 16), ("V3A", 12), ("V3B", 12), ("V4", 20),
        ("FFA1", 10), ("FFA2", 10), ("PPA", 12), ("EBA", 12),
        ("OFA", 10), ("OPA", 10), ("RSC", 10),
        (CONTEXT_ROI, 400),
    ]
)
CATEGORY_SELECTIVE = ["FFA1", "FFA2", "PPA", "EBA", "OFA", "OPA", "RSC"]


def _indices(order=ROI_VOXELS):
    idx, cursor = OrderedDict(), 0
    for name, n in order.items():
        idx[name] = torch.arange(cursor, cursor + n, dtype=torch.long)
        cursor += n
    return idx, cursor


def _model(subjects=None, **kw):
    idx, total = _indices()
    kwargs = dict(d_model=32, nhead=4, num_layers=2, output_dim=16, adapter_rank=4)
    kwargs.update(kw)
    if subjects:
        model = NeuralConstrainedDecoder(
            subject_roi_indices={s: idx for s in subjects}, **kwargs
        )
    else:
        model = NeuralConstrainedDecoder(roi_indices=idx, **kwargs)
    return model, total


def test_neural_heads_receive_gradient() -> None:
    """The identifying objective must reach the interpreted quantity.

    This is the test PCD never had. If it fails, ``neural_pred`` is an
    uninterpretable random projection and no claim may rest on it.
    """
    torch.manual_seed(0)
    model, v = _model()
    x = torch.randn(4, v)

    masked = ["V1v", "FFA1"]
    out = model(x, masked_rois=masked)
    loss = model.neural_prediction_loss(x, out) + out.mu.pow(2).sum()
    loss.backward()

    for name in masked:
        w = model.neural_heads[name].weight
        assert w.grad is not None and torch.count_nonzero(w.grad) > 0, (
            f"neural_heads[{name}] received no gradient. The neural-prediction "
            "constraint is not identifying anything -- this is the F-002 failure."
        )


def test_retrieval_only_loss_leaves_neural_heads_unsupervised() -> None:
    """Documents the lambda_neural=0 arm: heads exist but are NOT trained.

    This is intentional -- it is the control arm. The point is that in that arm
    ``neural_pred`` must never be interpreted, exactly as PCD's kappa should not
    have been. The asymmetry with the test above is the whole contract.
    """
    torch.manual_seed(0)
    model, v = _model()
    x = torch.randn(4, v)

    out = model(x, masked_rois=["V1v"])
    out.mu.pow(2).sum().backward()  # retrieval only: lambda_neural = 0

    assert model.neural_heads["V1v"].weight.grad is None, (
        "With a retrieval-only loss the neural heads must be unreachable. If "
        "they now train, some other loss touches them and the lambda_neural=0 "
        "control arm is contaminated."
    )


def test_masked_roi_voxels_cannot_influence_own_prediction() -> None:
    """Structural leakage safety: masking happens before the encoder.

    Perturbing a masked ROI's voxels must not change its own prediction. If it
    does, the model is reading the answer and every neural-prediction number is
    void.
    """
    torch.manual_seed(0)
    model, v = _model()
    model.eval()

    x1 = torch.randn(2, v)
    x2 = x1.clone()
    masked = ["V2v"]
    roi_slice = model._idx__single_2  # V2v is index 2
    x2[:, roi_slice] = torch.randn_like(x2[:, roi_slice]) * 50.0  # violent perturbation

    with torch.no_grad():
        p1 = model(x1, masked_rois=masked).neural_pred["V2v"]
        p2 = model(x2, masked_rois=masked).neural_pred["V2v"]

    assert torch.allclose(p1, p2, atol=1e-6), (
        "A masked ROI's own voxels changed its prediction -- the mask is not "
        "applied before the encoder and the objective is leaking."
    )


def test_unmasked_roi_voxels_do_influence_prediction() -> None:
    """Sanity counterpart: context ROIs must actually matter.

    Without this, the previous test could pass trivially on a model that ignores
    its input entirely.
    """
    torch.manual_seed(0)
    model, v = _model()
    model.eval()

    x1 = torch.randn(2, v)
    x2 = x1.clone()
    x2[:, model._idx__single_0] = torch.randn_like(x2[:, model._idx__single_0]) * 50.0

    with torch.no_grad():
        p1 = model(x1, masked_rois=["V2v"]).neural_pred["V2v"]
        p2 = model(x2, masked_rois=["V2v"]).neural_pred["V2v"]

    assert not torch.allclose(p1, p2, atol=1e-4), (
        "Perturbing an unmasked ROI did not change the masked ROI's prediction; "
        "the model is not using cortical context at all."
    )


def test_context_node_cannot_dominate_by_capacity() -> None:
    """Finding T7: the residual ROI must not buy its way into the representation.

    Its learned cost is d^2, independent of its voxel count, versus a normal
    ROI's d * n_voxels.
    """
    torch.manual_seed(0)
    model, _ = _model()

    ctx = model.roi_encoders[CONTEXT_ROI]
    learned = sum(p.numel() for p in ctx.parameters() if p.requires_grad)
    d = model.d_model

    assert learned <= 2 * d * d + 4 * d, (
        f"context node has {learned} learned params; expected ~d^2={d * d}. "
        "The parameter budget that prevents the T7 bypass is gone."
    )
    assert "frozen_proj" in dict(ctx.named_buffers()), "projection must be a buffer"
    assert not ctx.frozen_proj.requires_grad, "context projection must stay frozen"

    # And it must be cheaper than an unrestricted projection would have been.
    unrestricted = d * ROI_VOXELS[CONTEXT_ROI]
    assert learned < unrestricted


def test_subject_capacity_is_low_rank() -> None:
    """Finding F-001: per-subject capacity must be a small fraction of the model."""
    torch.manual_seed(0)
    model, _ = _model(subjects=["subj01", "subj02", "subj05", "subj07"])

    subject_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "subject_u" in n or "subject_v" in n
    )
    total = sum(p.numel() for p in model.parameters())
    frac = subject_params / total

    assert frac < 0.25, (
        f"per-subject parameters are {frac:.1%} of the model. PCD's were ~57% "
        "and it overfit by 82pp; low-rank adaptation is the corrective."
    )


def test_subject_adapter_is_noop_at_init() -> None:
    """U initialised to zero: the model starts as the shared projection."""
    torch.manual_seed(0)
    model, v = _model(subjects=["subj01", "subj02"])
    model.eval()
    x = torch.randn(2, v)

    with torch.no_grad():
        a = model(x, subject_ids=torch.zeros(2, dtype=torch.long)).mu
        b = model(x, subject_ids=torch.ones(2, dtype=torch.long)).mu

    # Subject embeddings still differ, so mu differs; the ROI *projections* must not.
    proj = model.roi_encoders["V1v"]
    with torch.no_grad():
        y = x[:, model._idx_subj01_0]
        assert torch.allclose(proj(y, "subj01"), proj(y, "subj02"), atol=1e-6), (
            "subject adapters are not a no-op at init; U must start at zero so "
            "any subject-specific deviation is earned, not assumed."
        )
    assert a.shape == b.shape


def test_per_roi_nodes_are_never_pooled() -> None:
    """Category-selective ROIs stay separate, enabling per-ROI contrasts.

    PCD pooled FFA/PPA/EBA/OFA/OPA/RSC into one level, which made ROI-specific
    claims impossible in principle.
    """
    torch.manual_seed(0)
    model, _ = _model()

    for roi in CATEGORY_SELECTIVE:
        assert roi in model.roi_names, f"{roi} missing as a node"
        assert roi in model.neural_heads, f"{roi} has no neural-prediction head"
    assert model.n_rois == len(ROI_VOXELS)
    assert CONTEXT_ROI not in model.neural_heads, (
        "the context ROI must not have a prediction head: its frozen projection "
        "makes its voxels unreconstructable by design"
    )


def test_no_mask_means_no_neural_prediction() -> None:
    """Without a mask the constraint path is inert and the loss is a true zero."""
    torch.manual_seed(0)
    model, v = _model()
    x = torch.randn(3, v)

    out = model(x)
    assert out.neural_pred == {}
    assert out.masked_rois == []

    loss = model.neural_prediction_loss(x, out)
    assert loss.item() == 0.0
    assert loss.grad_fn is None, "a no-op loss must not carry a graph"


def test_sample_mask_excludes_context_roi() -> None:
    """The context ROI is never masked -- it has no prediction head."""
    torch.manual_seed(0)
    model, _ = _model(mask_ratio=0.9)

    for _ in range(20):
        assert CONTEXT_ROI not in model.sample_mask()


@pytest.mark.parametrize("ratio", [0.15, 0.3, 0.5])
def test_mask_ratio_dose(ratio: float) -> None:
    """Mask ratio controls the dose; at least one ROI is always masked."""
    torch.manual_seed(0)
    model, _ = _model(mask_ratio=ratio)
    n = len(model.sample_mask())
    assert n == max(1, round(ratio * len(model.predictable_rois)))


def test_mu_is_unit_norm() -> None:
    """Retrieval output lives on the hypersphere."""
    torch.manual_seed(0)
    model, v = _model()
    out = model(torch.randn(5, v))
    assert torch.allclose(out.mu.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_mixed_subject_batch_is_rejected() -> None:
    """Subject-homogeneous batches are required; fail loudly, never silently."""
    torch.manual_seed(0)
    model, v = _model(subjects=["subj01", "subj02"])
    with pytest.raises(ValueError, match="subject-homogeneous"):
        model(torch.randn(4, v), subject_ids=torch.tensor([0, 0, 1, 1]))
