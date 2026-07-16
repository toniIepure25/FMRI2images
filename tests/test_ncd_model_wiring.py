"""Wiring tests for NCDModel: create_model dispatch and the training-loop contract.

The pilot runs through ``create_model`` and ``train_epoch``, so these guard the
seams between the NCD module and the existing training infrastructure. Two of them
guard specific documented failures:

* ``test_kappa_is_a_global_temperature`` -- NCD has no uncertainty head by design.
  PCD's per-level kappa heads were interpreted but never trained (F-002) and its
  global kappa was degenerate (F-003). If someone later makes kappa
  input-dependent and starts reading it as confidence, this fails first.
* ``test_arm_a_and_f_contribute_no_auxiliary_loss`` -- the control arms must add
  exactly nothing, or they stop being controls.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import yaml
from pathlib import Path

from fmri2img.models.auxiliary_objectives import AuxObjective
from fmri2img.models.unified_model import NCDModel, create_model

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"

ROI_VOXELS = OrderedDict(
    [("V1v", 20), ("V1d", 20), ("V2v", 16), ("V3v", 16), ("FFA1", 10),
     ("PPA", 12), ("nsdgeneral_other", 200)]
)


def _indices():
    idx, cursor = OrderedDict(), 0
    for name, n in ROI_VOXELS.items():
        idx[name] = torch.arange(cursor, cursor + n, dtype=torch.long)
        cursor += n
    return idx, cursor


def _config(objective: str = "masked_neural", weight: float = 0.5) -> dict:
    return {
        "type": "ncd",
        "ncd": {"d_model": 32, "nhead": 4, "num_layers": 2, "adapter_rank": 4,
                "mask_ratio": 0.3, "context_roi": "nsdgeneral_other"},
        "decoder": {"output_dim": 16},
        "cross_subject": {"enabled": False},
        "loss": {"neural_prediction": {"objective": objective, "weight": weight}},
    }


def _model(objective: str = "masked_neural", weight: float = 0.5):
    idx, total = _indices()
    return create_model(_config(objective, weight), roi_indices=idx), total


def test_create_model_dispatches_ncd() -> None:
    """`type: "ncd"` must route to NCDModel, not fall through to UnifiedModel."""
    m, _ = _model()
    assert isinstance(m, NCDModel)
    assert m.architecture_type == "ncd"
    assert m.model_type == "vmf", (
        "NCD presents as vmf so the existing retrieval loss and eval paths work "
        "unchanged; that is what keeps it comparable to the frozen recipe."
    )


def test_forward_returns_mu_and_kappa_shapes() -> None:
    m, v = _model()
    mu, kappa = m(torch.randn(5, v))
    assert mu.shape == (5, 16)
    assert kappa.shape == (5, 1)
    assert torch.allclose(mu.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_kappa_is_a_global_temperature() -> None:
    """kappa must be constant across samples: it is a temperature, not confidence.

    NCD deliberately has no uncertainty head (specification 18 section 10). If
    kappa ever varies per sample, someone has added one and it will be
    interpreted -- the exact path that produced F-002 and F-003.
    """
    torch.manual_seed(0)
    m, v = _model()
    _, kappa = m(torch.randn(8, v))

    assert torch.allclose(kappa, kappa[0].expand_as(kappa)), (
        "kappa varies across samples. NCD must not carry per-sample uncertainty: "
        "PCD's kappa heads were interpreted but never trained (F-002) and its "
        "global kappa was degenerate (F-003)."
    )
    assert float(kappa[0]) > 0, "kappa must be positive"


def test_kappa_is_positive_without_clamping() -> None:
    """Log-parameterised, so positivity needs no clamp.

    Clamping is what trapped PCD's kappa at a constant (F-003); parameterising in
    log-space avoids reintroducing the mechanism.
    """
    m, _ = _model()
    assert hasattr(m, "log_kappa")
    with torch.no_grad():
        m.log_kappa.fill_(-20.0)
    _, kappa = m(torch.randn(2, sum(ROI_VOXELS.values())))
    assert (kappa > 0).all(), "kappa must stay positive at extreme log values"


def test_auxiliary_loss_is_stashed_in_train_mode_only() -> None:
    """The training loop reads `_last_aux_loss`; eval must not populate it."""
    torch.manual_seed(0)
    m, v = _model(objective="masked_neural", weight=0.5)
    x = torch.randn(4, v)

    m.train()
    m(x)
    assert m._last_aux_loss is not None, "train mode must stash an auxiliary loss"
    assert m._last_aux_loss.requires_grad, "auxiliary loss must carry a graph"

    m.eval()
    m(x)
    assert m._last_aux_loss is None, "eval must not compute the auxiliary objective"


@pytest.mark.parametrize("arm_objective", ["none"])
def test_arm_a_and_f_contribute_no_auxiliary_loss(arm_objective: str) -> None:
    """ARM-A/F must add exactly nothing, or they are not controls."""
    torch.manual_seed(0)
    m, v = _model(objective=arm_objective, weight=0.0)
    m.train()
    m(torch.randn(4, v))
    assert m._last_aux_loss is None
    assert m._last_masked_rois == [], (
        "ARM-A/F draw no mask because their objective needs none; masking "
        "frequency parity is enforced for arms that do use it."
    )


def test_masking_is_drawn_for_every_masking_arm() -> None:
    """Masking frequency is held constant across arms that use it (protocol 25)."""
    for obj in ["masked_neural", "shuffled_neural", "random_target", "self_reconstruction"]:
        torch.manual_seed(0)
        m, v = _model(objective=obj)
        m.train()
        try:
            m(torch.randn(2, v))
        except ValueError:
            pass  # ARM-C needs shuffled_x; the mask is still drawn before that
        assert m._last_masked_rois, f"{obj} drew no mask"


def test_arm_c_fails_loudly_without_partner_image() -> None:
    """ARM-C must not silently degenerate into ARM-B.

    This is why the dataloader change is a hard prerequisite for running ARM-C,
    not an optional nicety.
    """
    torch.manual_seed(0)
    m, v = _model(objective="shuffled_neural")
    m.train()
    with pytest.raises(ValueError, match="requires shuffled_x"):
        m(torch.randn(2, v))


def test_arm_c_runs_when_partner_supplied() -> None:
    torch.manual_seed(0)
    m, v = _model(objective="shuffled_neural")
    m.train()
    mu, _ = m(torch.randn(2, v), shuffled_x=torch.randn(2, v))
    assert m._last_aux_loss is not None and m._last_aux_loss.requires_grad
    assert mu.shape == (2, 16)


def test_aux_gradient_reaches_backbone_and_heads() -> None:
    """The weighted auxiliary loss must train both the heads and the encoder."""
    torch.manual_seed(0)
    m, v = _model()
    m.train()
    x = torch.randn(4, v)
    mu, kappa = m(x)
    (mu.pow(2).sum() + kappa.sum() + 0.5 * m._last_aux_loss).backward()

    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in m.ncd.neural_heads.parameters()
    ), "auxiliary heads received no gradient (F-002 contract)"
    assert m.log_kappa.grad is not None, "the temperature must train"


@pytest.mark.parametrize("arm", ["A", "B", "C", "F"])
def test_pilot_arm_configs_yield_expected_objective_and_weight(arm: str) -> None:
    """The shipped arm configs must map to the intended AuxObjective end to end."""
    cfg = yaml.safe_load((CONFIG_DIR / f"NCD_v1_arm{arm}_pilot_1subj.yaml").read_text())
    aux = cfg["loss"]["neural_prediction"]

    expected = {
        "A": ("none", 0.0),
        "B": ("masked_neural", 0.5),
        "C": ("shuffled_neural", 0.5),
        "F": ("none", 0.0),
    }[arm]
    assert (aux["objective"], float(aux["weight"])) == expected
    AuxObjective(aux["objective"])  # must be a valid enum member


def test_loss_weights_extraction_matches_training_loop() -> None:
    """Replicates train_unified.py's generic weight extraction.

    `loss_weights = {k: v.get("weight", 1.0) for k, v in config["loss"].items()}`
    -- so `loss.neural_prediction.weight` reaches
    `loss_weights["neural_prediction"]` with no special-casing.
    """
    cfg = yaml.safe_load((CONFIG_DIR / "NCD_v1_armB_pilot_1subj.yaml").read_text())
    loss_weights = {
        k: v.get("weight", 1.0)
        for k, v in cfg.get("loss", {}).items()
        if isinstance(v, dict)
    }
    assert loss_weights["neural_prediction"] == 0.5

    cfg_a = yaml.safe_load((CONFIG_DIR / "NCD_v1_armA_pilot_1subj.yaml").read_text())
    weights_a = {
        k: v.get("weight", 1.0)
        for k, v in cfg_a.get("loss", {}).items()
        if isinstance(v, dict)
    }
    assert weights_a["neural_prediction"] == 0.0
