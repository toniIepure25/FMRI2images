"""Config-diff validation for the NCD matched-arm family.

Protocol 25 section 1 binds every arm to identical training conditions; only the
auxiliary objective (and, for ARM-F alone, the regularization knobs it exists to
tune) may vary. A config drift is invisible in results and would silently confound
the comparison that the whole thesis rests on -- so it is asserted here rather than
trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"
ARMS = ["A", "B", "C", "F"]

#: Keys permitted to differ between arms. Anything else differing is a bug.
PERMITTED_DIFFS = {
    "experiment.name",
    "experiment.description",
    "experiment.tags",
    "experiment.arm",
    "loss.neural_prediction.weight",
    "loss.neural_prediction.objective",
    "paths.output_dir",
}

#: ARM-F is the tuned-regularization control; it alone may move these.
ARM_F_PERMITTED = {
    "model.ncd.dropout",
    "training.fmri_noise_std",
    "training.voxel_dropout",
    "training.stochastic_depth",
    "training.optimizer.weight_decay",
}


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def _load(arm: str) -> Dict[str, Any]:
    path = CONFIG_DIR / f"NCD_v1_arm{arm}_pilot_1subj.yaml"
    assert path.exists(), f"missing arm config: {path}"
    return _flatten(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("arm", ARMS)
def test_arm_config_parses_and_declares_identity(arm: str) -> None:
    """Every arm must declare its identity so the run manifest can record it."""
    cfg = _load(arm)
    assert cfg["experiment.arm"] == arm
    assert cfg["model.type"] == "ncd"
    assert "loss.neural_prediction.objective" in cfg
    assert "loss.neural_prediction.weight" in cfg


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "A"])
def test_arms_differ_from_baseline_only_in_permitted_keys(arm: str) -> None:
    """The core matching contract, enforced key by key."""
    base, other = _load("A"), _load(arm)
    assert set(base) == set(other), (
        f"ARM-{arm} has a different config SHAPE than ARM-A: "
        f"{set(base) ^ set(other)}"
    )

    allowed = PERMITTED_DIFFS | (ARM_F_PERMITTED if arm == "F" else set())
    offenders = {
        k: (base[k], other[k])
        for k in base
        if base[k] != other[k] and k not in allowed
    }
    assert not offenders, (
        f"ARM-{arm} differs from ARM-A in keys that must be held identical: "
        f"{offenders}. Arms are not matched and any comparison is confounded "
        "(protocol 25 section 1)."
    )


@pytest.mark.parametrize("arm", ARMS)
def test_all_arms_hold_the_confounds_fixed(arm: str) -> None:
    """Target dimensionality, optimizer, schedule, masking, selection rule.

    Target dimensionality is the competing hypothesis of Kneeland et al. 2025
    (22 section 2), so it is a confound to control, never a knob to tune.
    """
    base, other = _load("A"), _load(arm)
    for key in [
        "model.decoder.output_dim",
        "model.ncd.d_model",
        "model.ncd.num_layers",
        "model.ncd.nhead",
        "model.ncd.adapter_rank",
        "model.ncd.mask_ratio",
        "training.batch_size",
        "training.num_epochs",
        "training.optimizer.lr",
        "training.warmup_epochs",
        "training.gradient_accumulation_steps",
        "evaluation.checkpoint_selection_metric",
        "data.seed",
        "data.split_by_image",
        "data.exclude_shared1000",
        "data.embedding_column",
    ]:
        assert base[key] == other[key], (
            f"ARM-{arm} changed {key} ({base[key]} -> {other[key]}). This must be "
            "identical across arms."
        )


def test_only_arm_f_moves_regularization() -> None:
    """ARM-F is the tuned-regularization control; no other arm may tune reg.

    If ARM-B also moved these knobs, a positive ARM-B result could not be
    attributed to the neural target -- which is objection O-1 exactly.
    """
    a, f = _load("A"), _load("F")
    moved = {k for k in ARM_F_PERMITTED if k in a and a[k] != f[k]}
    assert moved, "ARM-F must actually differ in regularization, or it is not a control"

    for arm in ["B", "C"]:
        cfg = _load(arm)
        for k in ARM_F_PERMITTED:
            if k in a:
                assert cfg[k] == a[k], (
                    f"ARM-{arm} moved {k}, which only ARM-F may tune. A positive "
                    "ARM-B could then be explained by regularization (O-1)."
                )


def test_no_arm_touches_imagery_or_the_sealed_set() -> None:
    """Zero-shot invariant (24 section 2) and the sealed SHARED1000 rule.

    If any imagery data reached fitting or selection, our setting would collapse
    into Spera et al. (2026) and the contribution would be forfeit (23 section 5).
    """
    for arm in ARMS:
        cfg = _load(arm)
        assert cfg["data.exclude_shared1000"] is True, f"ARM-{arm} unsealed SHARED1000"
        assert cfg["evaluation.checkpoint_selection_metric"] == "val_r@1", (
            f"ARM-{arm} selects on something other than the declared metric"
        )
        blob = " ".join(str(v).lower() for v in cfg.values())
        assert "imagery" not in blob, (
            f"ARM-{arm} references imagery in its config. Imagery is a sealed "
            "zero-shot test: it may not touch training, selection, or lambda."
        )
