"""Contract tests for the NCD matched auxiliary-control family (ARM-A..H).

These guard the protocol in
``docs/research/pcd_program/25_MATCHED_AUXILIARY_CONTROL_PROTOCOL.md``. They exist
because a masked-ROI objective is an auxiliary task, and auxiliary tasks
regularize: without a capacity-matched control family, a positive ARM-B result is
uninterpretable (reviewer objection O-1).

The tests that matter most here are the ones that can fail *silently* in a way that
would fabricate a positive result:

* ``test_all_arms_are_parameter_matched`` -- an arm with a capacity advantage would
  win for the wrong reason.
* ``test_shuffled_permutation_is_a_derangement`` / ``test_..._maps_whole_images`` --
  a permutation that maps an image to itself, or that permutes per *trial*, turns
  ARM-C silently back into ARM-B and destroys the control.
* ``test_arm_h_never_predicts_one_roi_from_another`` -- if ARM-H can see other ROIs,
  it stops isolating predictive dependency.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from fmri2img.models.auxiliary_objectives import (
    AuxObjective,
    DeterministicImagePermutation,
    RandomTargetBank,
    assert_param_parity,
    make_pseudo_roi_groups,
    stable_seed,
)
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

ALL_ARMS = {
    "A": AuxObjective.NONE,
    "B": AuxObjective.MASKED_NEURAL,
    "C": AuxObjective.SHUFFLED_NEURAL,
    "D": AuxObjective.RANDOM_TARGET,
    "E": AuxObjective.SELF_RECONSTRUCTION,
    "F": AuxObjective.NONE,          # + tuned regularization (a training-time knob)
    "G": AuxObjective.MASKED_NEURAL,  # + pseudo-ROI grouping (an input-time knob)
    "H": AuxObjective.ROI_AUTOENCODE,
}


def _indices(order=ROI_VOXELS):
    idx, cursor = OrderedDict(), 0
    for name, n in order.items():
        idx[name] = torch.arange(cursor, cursor + n, dtype=torch.long)
        cursor += n
    return idx, cursor


def _model(aux=AuxObjective.MASKED_NEURAL, **kw):
    idx, total = _indices()
    kwargs = dict(
        d_model=32, nhead=4, num_layers=2, output_dim=16,
        adapter_rank=4, aux_objective=aux,
    )
    kwargs.update(kw)
    return NeuralConstrainedDecoder(roi_indices=idx, **kwargs), total


# --------------------------------------------------------------------------
# Capacity parity -- protocol section 1
# --------------------------------------------------------------------------

def test_all_arms_are_parameter_matched() -> None:
    """Every arm must allocate identical capacity.

    The objective selects the target, never the head. An arm with more parameters
    would win for a reason that has nothing to do with the hypothesis.
    """
    counts = {}
    for arm, obj in ALL_ARMS.items():
        torch.manual_seed(0)
        m, _ = _model(aux=obj)
        counts[arm] = sum(p.numel() for p in m.parameters() if p.requires_grad)

    assert len(set(counts.values())) == 1, (
        f"arms are not exactly parameter-matched: {counts}. Every arm must "
        "allocate the same heads even when its objective is disabled."
    )
    assert_param_parity(counts, tolerance=0.02)  # must not raise


def test_param_parity_rejects_a_capacity_advantage() -> None:
    """The parity assertion must actually fire; a silent pass would be worthless."""
    with pytest.raises(AssertionError, match="not capacity-matched"):
        assert_param_parity({"A": 1_000_000, "B": 1_100_000}, tolerance=0.02)


# --------------------------------------------------------------------------
# Gradient reachability, per arm -- the F-002 contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arm,obj",
    [(a, o) for a, o in ALL_ARMS.items() if o is not AuxObjective.NONE],
)
def test_enabled_arms_reach_their_aux_heads(arm: str, obj: AuxObjective) -> None:
    """Each enabled objective must actually train the heads it interprets."""
    torch.manual_seed(0)
    m, v = _model(aux=obj)
    x = torch.randn(4, v)
    shuffled = torch.randn(4, v) if obj is AuxObjective.SHUFFLED_NEURAL else None

    masked = ["V1v", "FFA1"]
    out = m(x, masked_rois=masked)
    loss = m.auxiliary_loss(x, out, shuffled_x=shuffled)
    assert loss.requires_grad, f"ARM-{arm}: auxiliary loss carries no graph"
    loss.backward()

    reached = [
        n for n, p in m.neural_heads.named_parameters()
        if p.grad is not None and torch.count_nonzero(p.grad) > 0
    ]
    assert reached, (
        f"ARM-{arm} ({obj.value}): no auxiliary head received gradient. The "
        "objective is not identifying anything -- this is the F-002 failure."
    )


@pytest.mark.parametrize("arm", ["A", "F"])
def test_disabled_arms_produce_a_true_zero_loss(arm: str) -> None:
    """ARM-A/F have no auxiliary objective: the loss must be a graph-free zero."""
    torch.manual_seed(0)
    m, v = _model(aux=AuxObjective.NONE)
    x = torch.randn(4, v)

    out = m(x, masked_rois=["V1v"])
    loss = m.auxiliary_loss(x, out)
    assert loss.item() == 0.0
    assert loss.grad_fn is None, (
        f"ARM-{arm}: a disabled objective must not carry a graph, or it silently "
        "contributes gradient and stops being a control."
    )


def test_arm_c_refuses_to_run_without_its_partner_image() -> None:
    """ARM-C must fail loudly rather than silently degenerate into ARM-B."""
    torch.manual_seed(0)
    m, v = _model(aux=AuxObjective.SHUFFLED_NEURAL)
    x = torch.randn(2, v)
    out = m(x, masked_rois=["V1v"])
    with pytest.raises(ValueError, match="requires shuffled_x"):
        m.auxiliary_loss(x, out)


# --------------------------------------------------------------------------
# Arm-specific semantics
# --------------------------------------------------------------------------

def test_arm_c_target_differs_from_arm_b_target() -> None:
    """The shuffled target must genuinely differ from the true one."""
    torch.manual_seed(0)
    mb, v = _model(aux=AuxObjective.MASKED_NEURAL)
    torch.manual_seed(0)
    mc, _ = _model(aux=AuxObjective.SHUFFLED_NEURAL)

    x = torch.randn(4, v)
    shuffled = torch.randn(4, v)
    tb = mb._aux_target("V1v", x, "_single", None)
    tc = mc._aux_target("V1v", x, "_single", shuffled)

    assert not torch.allclose(tb, tc), (
        "ARM-C target equals ARM-B target: the control is inert."
    )
    assert tb.shape == tc.shape, "targets must be dimensionality-matched"


def test_arm_d_targets_are_fixed_across_calls() -> None:
    """ARM-D targets must be fixed, not resampled, or the task is unlearnable."""
    torch.manual_seed(0)
    m, v = _model(aux=AuxObjective.RANDOM_TARGET)
    t1 = m._aux_target("V1v", torch.randn(3, v), "_single", None)
    t2 = m._aux_target("V1v", torch.randn(3, v), "_single", None)
    assert torch.equal(t1, t2), "random targets must be fixed across steps"
    assert t1.shape == (3, ROI_VOXELS["V1v"]), "must be dimensionality-matched"


def test_arm_e_target_is_non_anatomical_but_dimension_matched() -> None:
    """ARM-E reconstructs arbitrary voxels, not an anatomically-defined ROI."""
    torch.manual_seed(0)
    m, v = _model(aux=AuxObjective.SELF_RECONSTRUCTION)
    generic = m._generic_idx_0
    anatomical = m._idx__single_0  # V1v

    assert generic.numel() == anatomical.numel(), "must be dimensionality-matched"
    assert not torch.equal(generic, anatomical), (
        "ARM-E's target index set equals the real ROI's: it is not non-anatomical."
    )


def test_arm_h_never_predicts_one_roi_from_another() -> None:
    """ARM-H isolates ROI organisation from cross-ROI predictive dependency.

    Perturbing OTHER ROIs must not change a given ROI's prediction. If it does,
    ARM-H reads cortical context and no longer separates the two factors.
    """
    torch.manual_seed(0)
    m, v = _model(aux=AuxObjective.ROI_AUTOENCODE)
    m.eval()

    x1 = torch.randn(2, v)
    x2 = x1.clone()
    # Perturb every ROI except V1v (index 0).
    for i in range(1, m.n_rois):
        sl = getattr(m, f"_idx__single_{i}")
        x2[:, sl] = torch.randn_like(x2[:, sl]) * 50.0

    with torch.no_grad():
        p1 = m(x1).neural_pred["V1v"]
        p2 = m(x2).neural_pred["V1v"]

    assert torch.allclose(p1, p2, atol=1e-6), (
        "ARM-H's prediction for V1v changed when other ROIs changed. It is using "
        "cross-ROI context and therefore does not isolate predictive dependency."
    )


def test_arm_b_does_predict_one_roi_from_another() -> None:
    """Counterpart: ARM-B must use cortical context, or it is not the hypothesis."""
    torch.manual_seed(0)
    m, v = _model(aux=AuxObjective.MASKED_NEURAL)
    m.eval()

    x1 = torch.randn(2, v)
    x2 = x1.clone()
    x2[:, m._idx__single_1] = torch.randn_like(x2[:, m._idx__single_1]) * 50.0

    with torch.no_grad():
        p1 = m(x1, masked_rois=["V1v"]).neural_pred["V1v"]
        p2 = m(x2, masked_rois=["V1v"]).neural_pred["V1v"]

    assert not torch.allclose(p1, p2, atol=1e-4), (
        "ARM-B ignored a change in another ROI: it is not using cortical context."
    )


# --------------------------------------------------------------------------
# Deterministic shuffling -- ARM-C provenance
# --------------------------------------------------------------------------

def test_shuffled_permutation_is_a_derangement() -> None:
    """No image may map to itself, or ARM-C silently becomes ARM-B for it."""
    perm = DeterministicImagePermutation(list(range(200)), seed=stable_seed("split", 42))
    for i in range(200):
        assert perm(i) != i, f"image {i} maps to itself; ARM-C is compromised"


def test_shuffled_permutation_maps_whole_images_not_trials() -> None:
    """All trials of one image must receive the same permuted partner.

    Permuting per trial would let repeated presentations of an image leak its true
    target across the batch.
    """
    perm = DeterministicImagePermutation(list(range(50)), seed=7)
    trials = torch.tensor([3, 3, 3, 11, 11])
    mapped = perm.map_batch(trials)
    assert len(set(mapped[:3].tolist())) == 1, "trials of image 3 mapped inconsistently"
    assert len(set(mapped[3:].tolist())) == 1, "trials of image 11 mapped inconsistently"


def test_shuffled_permutation_is_reproducible_and_seed_sensitive() -> None:
    """Same seed reproduces; different seed differs. Required for the manifest."""
    a = DeterministicImagePermutation(list(range(100)), seed=1)
    b = DeterministicImagePermutation(list(range(100)), seed=1)
    c = DeterministicImagePermutation(list(range(100)), seed=2)

    assert [a(i) for i in range(100)] == [b(i) for i in range(100)]
    assert [a(i) for i in range(100)] != [c(i) for i in range(100)]
    assert a.manifest()["seed"] == 1
    assert a.manifest()["is_derangement"] is True


def test_stable_seed_is_deterministic_across_processes() -> None:
    """Python's hash() is salted per process; the manifest needs a stable seed."""
    assert stable_seed("split_abc", 42) == stable_seed("split_abc", 42)
    assert stable_seed("split_abc", 42) != stable_seed("split_abc", 43)
    assert 0 <= stable_seed("x") < 2**32


# --------------------------------------------------------------------------
# Pseudo-ROI grouping -- ARM-G
# --------------------------------------------------------------------------

def test_pseudo_roi_groups_match_real_roi_shapes() -> None:
    """ARM-G must match group count and per-group dimensionality exactly.

    That is what holds parameter budget and masking probability constant, so the
    only thing varying is whether the grouping is anatomical.
    """
    sizes = OrderedDict((k, v) for k, v in ROI_VOXELS.items())
    _, total = _indices()
    groups = make_pseudo_roi_groups(sizes, total, seed=0)

    assert len(groups) == len(sizes), "group count must match"
    for (real, n), (pseudo, idx) in zip(sizes.items(), groups.items()):
        assert idx.numel() == n, f"{pseudo} has {idx.numel()} voxels, expected {n}"
        assert pseudo == f"pseudo_{real}"


def test_pseudo_roi_groups_are_disjoint_and_in_range() -> None:
    """A voxel may not appear in two pseudo-ROIs."""
    sizes = OrderedDict((k, v) for k, v in ROI_VOXELS.items())
    _, total = _indices()
    groups = make_pseudo_roi_groups(sizes, total, seed=3)

    seen = torch.cat(list(groups.values()))
    assert seen.numel() == len(torch.unique(seen)), "pseudo-ROIs overlap"
    assert int(seen.min()) >= 0 and int(seen.max()) < total


def test_pseudo_roi_groups_are_seed_sensitive() -> None:
    """A single fixed shuffle is an anecdote; ARM-G needs a seed distribution.

    This is the ``random.Random(42)`` defect found in PCD (03_HYPOTHESES H2).
    """
    sizes = OrderedDict((k, v) for k, v in ROI_VOXELS.items())
    _, total = _indices()
    g0 = make_pseudo_roi_groups(sizes, total, seed=0)
    g1 = make_pseudo_roi_groups(sizes, total, seed=1)
    same = all(torch.equal(a, b) for a, b in zip(g0.values(), g1.values()))
    assert not same, "different seeds produced identical pseudo-ROIs"


def test_pseudo_roi_groups_reject_oversized_request() -> None:
    sizes = OrderedDict([("a", 100), ("b", 100)])
    with pytest.raises(ValueError, match="only 150 available"):
        make_pseudo_roi_groups(sizes, 150, seed=0)


def test_random_target_bank_is_variance_matched_and_checkpointed() -> None:
    """ARM-D targets must be variance-matched and reload identically."""
    sizes = OrderedDict([("V1v", 2000), ("FFA1", 2000)])
    bank = RandomTargetBank(sizes, seed=0, target_std=1.0)

    t = bank("V1v", 4)
    assert t.shape == (4, 2000)
    assert abs(float(t[0].std()) - 1.0) < 0.1, "targets are not variance-matched"
    assert "_target_0" in dict(bank.named_buffers()), "targets must be buffers"
