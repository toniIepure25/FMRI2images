# Configuration System

Hierarchical YAML configuration for the fMRI-to-Image neural decoding pipeline.
All configs inherit from `base.yaml` and can be overridden at runtime.

## Directory Structure

```
configs/
├── base.yaml                        # Global defaults (Phase 2: ViT-L/14, 768-D)
│
├── experiments/                     # Ablation ladder (EXP0-EXP14)
│   ├── exp0_baseline.yaml           # Deterministic MLP baseline
│   ├── exp1_preproc.yaml            # + center_pcr preprocessing
│   ├── exp2_queue.yaml              # + memory queue
│   ├── exp3_gaussian_nll.yaml       # + Gaussian NLL
│   ├── exp4_gaussian_nce.yaml       # + Gaussian-NCE contrastive
│   ├── exp5_kl_anneal.yaml          # + KL annealing
│   ├── exp6_whiten.yaml             # Ablation: whitening vs PCR
│   ├── exp7_vmf_nce.yaml            # vMF-NCE (MLP encoder)
│   ├── exp8_roi_transformer.yaml    # ROI Transformer + vMF-NCE
│   ├── exp9_roi_dcf.yaml            # ROI-DCF consensus fusion
│   ├── exp10_vmf_mixture.yaml       # + vMF mixture sampling
│   ├── exp11_dual_ua_cfg.yaml       # + decomposed UA-CFG
│   ├── exp12_ceiling_temperature.yaml # + noise-ceiling temperature
│   ├── exp13_kappa_spcl.yaml        # + kappa-SPCL curriculum
│   └── exp14_full_system.yaml       # Full system (flagship)
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
    --config configs/experiments/exp7_vmf_nce.yaml --gpu 0

# Run the full ablation ladder
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" --gpu 0

# Generate images from a trained model
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint experimental_results/exp14_full_system/best_model.pt

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
    --config configs/experiments/exp7_vmf_nce.yaml \
    --override "training.batch_size=8" \
    --override "training.num_epochs=50"
```

## Ablation Ladder

Each experiment adds one component to isolate its effect:

| Exp | What Changes | Hypothesis |
|-----|-------------|-----------|
| EXP0 | Deterministic MLP baseline | Lower bound |
| EXP1 | + center_pcr preprocessing | Reduces hubness |
| EXP2 | + memory queue (Q=8192) | Harder negatives |
| EXP3 | + Gaussian NLL | Captures uncertainty |
| EXP4 | + Gaussian-NCE | Distribution-aware contrastive |
| EXP5 | + KL annealing + free-bits | Stabilizes training |
| EXP6 | Whitening ablation | PCR vs whitening |
| **EXP7** | **vMF-NCE** | **Matches S^{d-1} geometry** |
| **EXP8** | **ROI Transformer** | **Brain-topology inductive bias** |
| **EXP9** | **ROI-DCF consensus** | **Per-ROI directional experts** |
| EXP10 | + vMF mixture sampling | Diversity in generation |
| EXP11 | + decomposed UA-CFG | Principled uncertainty guidance |
| EXP12 | + noise-ceiling temperature | Measurement-aware calibration |
| EXP13 | + kappa-SPCL curriculum | Self-paced learning |
| **EXP14** | **Full system** | **All contributions combined** |

Bold = novel contributions (EXP7-EXP9 core, EXP14 flagship).

## Phase History

- **Phase 1** (completed): ViT-B/32 (512-D), TwoStageEncoder, InfoNCE.
  Results in `experimental_results/exp001_baseline_ultimate/`.
- **Phase 2** (current): ViT-L/14 (768-D), ROI Transformer, vMF-NCE, ROI-DCF.
  Configs = `experiments/exp0-exp14`.
