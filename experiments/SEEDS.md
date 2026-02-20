# Seed and Determinism Policy

This repository targets **reproducible, paper-grade experiments**. All seeds are set
via YAML configs (`training.seed`, `evaluation.seed`, `inference.seed`). The global
default is `42` (from `configs/base.yaml`).

## Required Seeds

| Component | How | Config Key |
|-----------|-----|------------|
| PyTorch | `torch.manual_seed(seed)` | `training.seed` |
| CUDA | `torch.cuda.manual_seed_all(seed)` | `training.seed` |
| NumPy | `np.random.seed(seed)` | `training.seed` |
| Python | `random.seed(seed)` | `training.seed` |
| DataLoader | `worker_init_fn` + seeded `torch.Generator` | `training.seed` |
| Diffusion | Per-trial: `base_seed + trial_id * 1000 + k` | `inference.seed` |
| vMF Sampling | Seeded `torch.Generator` for rejection sampling | `inference.seed` |
| Evaluation | Bootstrap CIs, 2AFC pair sampling | `evaluation.seed` |

## cuDNN Determinism

For exact reproducibility across runs:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:2
```

And in code:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

**Caveat**: Deterministic mode disables some fast GPU kernels. Use only for final
reported numbers, not during hyperparameter search.

## Multi-Seed Reporting

For statistical significance in the paper, run each experiment with 3 seeds
(`42`, `123`, `2024`) and report `mean +/- std` with paired t-tests across subjects.
