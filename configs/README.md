# Configuration System

Hierarchical YAML configuration for the fMRI-to-Image neural decoding pipeline.
All configs inherit from `base.yaml` and can be overridden at runtime.

## Directory Structure

```
configs/
├── base.yaml                        # Global defaults (Phase 2: ViT-L/14, 768-D)
│
├── experiments/                     # Ablation ladder (B0-B1-N1-N2-N3-N4)
│   ├── B0_deterministic.yaml        # Strong deterministic baseline
│   ├── B1_gaussian.yaml             # Probabilistic Gaussian baseline
│   ├── N1_vmf_nce.yaml              # Novel: vMF-NCE (MLP encoder)
│   ├── N2_roi_transformer.yaml      # Novel: ROI Transformer + vMF-NCE
│   ├── N3_roi_dcf.yaml              # Novel: ROI-DCF consensus fusion
│   └── N4_full_system.yaml          # Novel: Full system (flagship)
│
├── training/                        # Training recipes
│   ├── dev_fast.yaml                # Fast development loop
│   ├── ridge_baseline.yaml          # Linear baseline
│   └── adapter_vitl14.yaml          # CLIP adapter (512->768)
│
├── inference/                       # Diffusion generation
│   ├── production.yaml              # Balanced quality/speed
│   ├── fast_inference.yaml          # Speed-optimized (25 steps)
│   └── highres_quality.yaml         # Publication quality (200 steps)
│
└── system/                          # Infrastructure
    ├── clip.yaml                    # CLIP model defaults
    ├── data.yaml                    # NSD dataset paths
    └── logging.yaml                 # Logging settings
```

## Quick Start

```bash
# Run a Phase 2 experiment
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1_vmf_nce.yaml --gpu 0

# Run the full ablation ladder
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" --gpu 0

# Generate images from a trained model
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint experimental_results/N4_full_system/best_model.pt

# Quick dev iteration
python3 scripts/training/train_unified.py \
    --config configs/training/dev_fast.yaml --gpu 0
```

## Inheritance

All configs inherit from `base.yaml` via the `_base_` key:

```yaml
_base_: ../base.yaml

training:
  batch_size: 8        # Override specific fields
```

Runtime overrides:

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1_vmf_nce.yaml \
    --override "training.batch_size=8" \
    --override "training.num_epochs=50"
```

## Ablation Ladder

6 experiments: 2 strong baselines + 4 novel contributions.
Consecutive row differences isolate each innovation.

| ID | Config | What It Proves | Novel? |
|----|--------|---------------|--------|
| **B0** | `B0_deterministic` | Strong standard baseline (MSE + InfoNCE + queue + PCR) | No |
| **B1** | `B1_gaussian` | Probabilistic Gaussian baseline (Gaussian-NCE + KL) | No |
| **N1** | `N1_vmf_nce` | vMF > Gaussian on S^{d-1} (distribution matches geometry) | **Yes** |
| **N2** | `N2_roi_transformer` | Brain-topology inductive bias > flat MLP | **Yes** |
| **N3** | `N3_roi_dcf` | Per-ROI directional consensus > single-head | **Yes** |
| **N4** | `N4_full_system` | Full system best overall (DCF + Mixture + DUA-CFG + SPCL) | **Yes** |

## Phase History

- **Phase 1** (completed): ViT-B/32 (512-D), TwoStageEncoder, InfoNCE.
  Results in `experimental_results/exp001_baseline_ultimate/`.
- **Phase 2** (current): ViT-L/14 (768-D), ROI Transformer, vMF-NCE, ROI-DCF.
  Configs = `experiments/B0-N4`.
