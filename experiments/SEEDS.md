# Seed & Determinism Policy

This repository targets **reproducible paper-grade runs**. Use these defaults unless a config overrides them.

- **Global seed**: `42` (matches `training.seed` in `configs/base.yaml`).
- **PyTorch**: set `torch.manual_seed`, `torch.cuda.manual_seed_all`, enable `torch.backends.cudnn.deterministic = True`, and set `torch.backends.cudnn.benchmark = False` for deterministic kernels when reproducibility matters.
- **NumPy**: `np.random.seed(training.seed)`.
- **Python**: `random.seed(training.seed)`.
- **DataLoader**: set `worker_init_fn` to seed workers, and `generator` to a seeded `torch.Generator` when shuffling.
- **Diffusion sampling**: pass an explicit seed to the sampler; for adaptive-K runs, keep a deterministic per-trial seed schedule (e.g., `base_seed + trial_id * 1000 + k`).
- **Logvar/uncertainty heads**: sampling uses reparameterization noise; use the seeded RNG from PyTorch for determinism in validation/inference when you need identical draws. For reporting, prefer deterministic mean or a fixed `eps` seed unless ablation requires stochasticity.
- **CUDA/cuDNN determinism caveat**: some ops have non-deterministic kernels on GPU. If exact determinism is required, set `CUBLAS_WORKSPACE_CONFIG=:16:8` or `:4096:2` (see PyTorch docs) before running.

When running large grids, record the seed in your run name (see `experiments/RUNS.md`).
