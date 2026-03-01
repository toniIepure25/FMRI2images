---
name: vMF Novel Losses
overview: Add three vMF-native loss/augmentation improvements (Delta-SPCL, vMF-SoftCLIP, Slerp MixCo) to the N-series experiments. All three fit within the existing architecture and require no new data extraction.
todos:
  - id: delta-spcl
    content: "Add DeltaSPCLVMFNCELoss to vmf_nce.py: extends KappaSPCL with optional delta-based sample weighting, configurable delta_weight"
    status: pending
  - id: vmf-softclip
    content: "Add VMFSoftCLIPLoss to softclip.py: uses kappa as dynamic student temperature, supports queue + symmetric mode, teacher_tau=0.05"
    status: pending
  - id: slerp-mixco
    content: "Add slerp() to mixco.py and use_slerp parameter to mixco_augment: spherical interpolation for CLIP targets with proper edge-case handling"
    status: pending
  - id: wire-training-loop
    content: "Wire new losses into train_unified.py: pass delta to SPCL, route kappa to VMFSoftCLIP, pass use_slerp to mixco_augment"
    status: pending
  - id: v6-configs
    content: Create N1v6-N4v6 configs based on v5 + new features. Add v6 entries to ablation ladder and aggregation scripts
    status: pending
  - id: tests
    content: Add tests for DeltaSPCLVMFNCELoss, VMFSoftCLIPLoss, and slerp()
    status: pending
isProject: false
---

# Implement vMF-Native Loss Improvements for N-Series

## Proposal Assessment Summary


| Proposal     | Verdict         | Applies to | Expected Impact   | Complexity |
| ------------ | --------------- | ---------- | ----------------- | ---------- |
| Delta-SPCL   | FITS            | N4v5 only  | Low (1-3% R@1)    | Medium     |
| vMF-SoftCLIP | FITS            | N1-N4 v5   | Medium (3-5% R@1) | Medium     |
| vMF-Slerp    | FITS (marginal) | All MixCo  | Low (<1% R@1)     | Low        |


All three are implementable within the current architecture, require no CLIP patch tokens, and can be independently toggled via config.

## Versioning Decision

These changes should go into **v6 configs** (N1v6-N4v6), NOT modify v5. Rationale:

- v5 isolates the critical **tau=1.0 kappa collapse fix** -- the single most impactful change
- v6 adds novel loss formulations on top of v5
- This allows clean A/B comparison: v5 results show the kappa fix alone; v6 results show the additional contribution of the new losses
- If v5 already reaches 60%+ R@1, the value of v6 changes is different than if v5 stays at 35%

The code changes (new loss classes, slerp utility) go into the library; v6 configs enable them.

---

## 1. Delta-SPCL (`DeltaSPCLVMFNCELoss`)

**File:** [src/fmri2img/losses/vmf_nce.py](src/fmri2img/losses/vmf_nce.py)

Add a new class `DeltaSPCLVMFNCELoss` after the existing `KappaSPCLVMFNCELoss` (line 358). Key differences from the proposal:

- Reuse `_score()` pattern from `VonMisesFisherNCELoss` instead of inline logit computation (avoid code duplication, respect tau/clamp settings)
- Make the delta weight configurable (`delta_weight` parameter, default 10.0) instead of hardcoded
- Accept `delta` as an optional parameter (graceful fallback when delta is not available)
- Keep the existing `set_curriculum_temperature()` API for training loop compatibility

**File:** [scripts/training/train_unified.py](scripts/training/train_unified.py)

At the `vmf_nce_spcl` loss call site (line ~803), pass `delta` from `model._last_dcf_extras` when available:

```python
if "vmf_nce_spcl" in losses and is_vmf:
    dcf_extras = getattr(model, "_last_dcf_extras", {})
    delta = dcf_extras.get("delta")
    l = losses["vmf_nce_spcl"](pred, aux, gt_embedding, queue=queue, delta=delta)
```

Same pattern for the validation loop (line ~991).

In `setup_losses()`, when `vmf_nce_spcl.use_delta: true` is set, instantiate `DeltaSPCLVMFNCELoss` instead of `KappaSPCLVMFNCELoss`.

**Config:** Only `N4v6_full_system.yaml` enables this (the only experiment with both SPCL and DCF):

```yaml
loss:
  vmf_nce_spcl:
    enabled: true
    use_delta: true        # NEW: use DeltaSPCLVMFNCELoss
    delta_weight: 10.0     # NEW: scaling factor for delta penalty
    tau: 1.0
    initial_curriculum_t: 50.0
```

---

## 2. vMF-SoftCLIP (`VMFSoftCLIPLoss`)

**File:** [src/fmri2img/losses/softclip.py](src/fmri2img/losses/softclip.py)

Add a new class `VMFSoftCLIPLoss` after `SoftCLIPLoss` (line 98). Key design:

- **Teacher distribution** (unchanged): `softmax(gt @ all_keys.T / teacher_tau)` -- CLIP-CLIP similarity with fixed temperature
- **Student distribution** (novel): `log_softmax(kappa.unsqueeze(1) * mu @ all_keys.T)` -- kappa acts as per-sample inverse temperature
- Must support queue negatives (current SoftCLIP does)
- Must support symmetric mode (current SoftCLIP does)
- `teacher_tau` should default to `0.05` (matching current SoftCLIP tau) -- NOT the proposal's `0.005` which would create an extreme scale mismatch with kappa~23

**Calibration note:** With `teacher_tau=0.05`, teacher logit scale is `1/0.05 = 20`. With kappa~23 (from v5 epoch 1), student logit scale is ~23. These are comparable, which means the KL divergence will properly encourage kappa to track actual prediction quality.

**File:** [scripts/training/train_unified.py](scripts/training/train_unified.py)

At the SoftCLIP call site, check if `vmf_softclip: true` is configured. When enabled for N-series, pass `(pred, aux)` instead of just `pred`:

```python
if _epoch_softclip and _softclip_loss_obj is not None:
    if vmf_softclip_enabled and is_vmf:
        sc_loss = _softclip_loss_obj(pred, aux, gt_embedding, queue=queue)
    else:
        sc_loss = _softclip_loss_obj(pred_for_softclip, gt_embedding, queue=queue)
```

In `setup_losses()`, when `softclip.vmf_mode: true` is set and model is vMF, instantiate `VMFSoftCLIPLoss` instead of `SoftCLIPLoss`.

**Config:** All N-series v6 configs get `vmf_mode: true` in softclip section:

```yaml
loss:
  softclip:
    enabled: true
    vmf_mode: true      # NEW: use VMFSoftCLIPLoss with kappa as student temperature
    teacher_tau: 0.05   # Teacher distribution temperature
    use_queue: true
    symmetric: true
```

---

## 3. Spherical MixCo (Slerp)

**File:** [src/fmri2img/losses/mixco.py](src/fmri2img/losses/mixco.py)

Add a `slerp()` function and modify `mixco_augment()` to accept a `use_slerp: bool = False` parameter. When enabled, replace line 50:

```python
# Current:
clip_mixed = lam_v * clip_emb + (1 - lam_v) * clip_emb[perm]
# With slerp:
clip_mixed = slerp(clip_emb, clip_emb[perm], 1.0 - lam)
```

The slerp implementation must fix the proposal's edge-case bug -- use `sin_theta_0.clamp(min=1e-6)` instead of the all-or-nothing check.

**Note:** fMRI features are NOT on the unit sphere, so `fmri_mixed` stays as linear interpolation. Only CLIP targets get slerp.

**File:** [scripts/training/train_unified.py](scripts/training/train_unified.py)

Pass `use_slerp=mixco_cfg.get("use_slerp", False)` to `mixco_augment()` at line ~859.

**Config:** All N-series v6 configs:

```yaml
mixco:
  enabled: true
  use_slerp: true    # NEW: spherical interpolation for CLIP targets
  alpha: 0.2
```

---

## 4. v6 Config Creation

Create 4 new config files based on v5, adding the new features:

- `configs/experiments/N1v6_vmf_nce.yaml` -- v5 + vMF-SoftCLIP + Slerp MixCo
- `configs/experiments/N2v6_roi_transformer.yaml` -- v5 + vMF-SoftCLIP + Slerp MixCo
- `configs/experiments/N3v6_roi_dcf.yaml` -- v5 + vMF-SoftCLIP + Slerp MixCo
- `configs/experiments/N4v6_full_system.yaml` -- v5 + vMF-SoftCLIP + Slerp MixCo + Delta-SPCL

## 5. Tests

Add test cases in `tests/`:

- `DeltaSPCLVMFNCELoss` forward pass with and without delta
- `VMFSoftCLIPLoss` forward pass, symmetric mode, queue support
- `slerp()` correctness (unit norm output, boundary cases, per-element edge case)

## 6. Wire into ablation scripts

Add v6 entries to [scripts/training/run_ablation_ladder.sh](scripts/training/run_ablation_ladder.sh) `CONFIGS` array and [scripts/evaluation/aggregate_ablation.py](scripts/evaluation/aggregate_ablation.py) `EXPERIMENTS` list. Do NOT change the default `EXPERIMENT_ORDER` -- v6 is run explicitly with `ONLY=N` and `--start N1v6` or by editing the order.