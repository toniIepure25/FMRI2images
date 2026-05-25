# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo at a glance

Python research codebase for fMRI-to-image neural decoding on the Natural Scenes Dataset (NSD), CLIP embeddings, and diffusion-based reconstruction. A separate Vite+React+TypeScript demo app (`demo_thesis/`) presents the work for thesis defense. The two are independent — `demo_thesis/` has its own `CLAUDE.md`, `package.json`, and toolchain.

**Best documented result:** 77.2% R@1 on SHARED1000 from a frozen compact+legacy fixed tri-fusion retrieval system (V35+N1v28a). After that endpoint the bottleneck is ranking inside the shortlist, not shortlist recall.

## Scope guard (read this first)

The active web/frontend app lives **only** inside `demo_thesis/`. For UI, frontend, layout, styling, React, Tailwind, Vite, or demo presentation tasks:

- Edit only files under `demo_thesis/`; treat it as a separate project root and run frontend commands from inside it.
- Do **not** modify `src/fmri2img/`, root `scripts/`, `configs/`, `tests/`, experiment outputs, reconstruction artifacts, data files, model files, or research pipeline files unless explicitly asked.
- Root files may be read for context only.

For Python/research/pipeline tasks, stay out of `demo_thesis/` unless asked.

## Architecture

```
src/fmri2img/     Library package (importable, NO CLI code — no argparse, no __main__)
scripts/          Thin CLI entry points organized by category (training/, evaluation/,
                  build/, reconstruction/, analysis/, orchestration/, utils/, diagnostics/,
                  preprocessing/)
tests/            pytest suite
configs/          YAML configs — base.yaml + experiments/ (B-series + N-series) + training/
                  + inference/ + system/
data/             NSD parquet indices (data/indices/nsd_index/subject=<id>/index.parquet)
cache/            Generated caches (preproc, preextracted fMRI, CLIP, HF) + .markers/
outputs/          Run outputs, reports, clip caches, SHARED1000 eval results
docs/             Guides, paper drafts, technical docs
demo_thesis/      Independent Vite+React app (see its own CLAUDE.md)
```

**Hard rule:** library code goes in `src/fmri2img/`, never in `scripts/`. Some older scripts violate this (1000+ lines) — refactor incrementally only when already touching them; do not preemptively split.

## First-time setup

```bash
cp .env.example .env             # then: set -a && source .env && set +a
pip install -e ".[train,diffusion]"
make preflight                   # python/cuda/disk/env checks
```

The Makefile auto-includes `.env`. When invoking Python scripts directly (not via `make`), source `.env` first or paths will be wrong on cluster pods.

## Data pipeline order (must be respected)

```
make index                  # build NSD parquet index per subject (needed before preprocess)
make preprocess             # fit scaler + reliability + PCA on the index
make clip-cache             # build the CLIP embedding cache for all images
# optional, used by specific experiment families:
make token-clip-cache       # 257×768 token-level CLIP cache (MindEye-style targets)
make bigg-token-cache       # 257×1280 ViT-bigG/14 token cache (V27+ experiments)
```

The Makefile uses idempotent marker files under `cache/.markers/` — re-running a target is safe and skips completed steps.

## Common Make targets

| Target | Notes |
|---|---|
| `make help` | Self-documented target list |
| `make preflight` / `make doctor` | Quick / thorough readiness checks |
| `make smoke` | Smoke test |
| `make test` | `python -m pytest tests/ -v` (full suite) |
| `make test-quick` | Fast subset: `-x --ignore=tests/unit -k "not slow"` |
| `make train CONFIG=configs/experiments/B0_deterministic.yaml` | Single experiment via `scripts/training/train_unified.py` |
| `make ablation` | Full B0-N4 ablation ladder. Filter with `ONLY=N1v28a`. Checkpoint policy via `SAVE_CKPT=all\|best\|no` |
| `make full-pipeline` | Train + reconstruct + evaluate + aggregate |
| `make ridge` | Ridge regression baseline |
| `make eval-shared1000 ENCODER_CKPT=...` | Paper-grade SHARED1000 benchmark (strategies: `single best_of_8 boi_lite`, seeds `0 1 2`) |
| `make summarize-shared1000` | Aggregate SHARED1000 across subjects |
| `make aggregate` | Build paper tables from `experimental_results/` |
| `make paper` | Build paper artifacts via `scripts/analysis/build_paper_artifacts.py` |

### Running a single test

```bash
python -m pytest tests/test_losses.py -v               # one file
python -m pytest tests/test_losses.py::test_vmf -v     # one test
python -m pytest tests/unit/ -v                        # only unit tests
```

### Important env vars (most targets honor these)

- `SUBJECT` (default `subj01`)
- `GPU` (default `0`), `DEVICE` (default `cuda`)
- `CONFIG` — required by `make train`
- `LIMIT` — cap sample count for quick experiments
- `ONLY` — filter ablation to a single experiment ID (e.g. `ONLY=N1v28a`)
- `SUBJECTS` — multi-subject targets (default `subj01 subj02 subj05 subj07`)
- `CACHE_ROOT`, `OUTPUT_ROOT`, `CHECKPOINT_ROOT`, `NSD_DATA_ROOT` from `.env`

## Experiment naming

Ablation ladder uses two families, all under `configs/experiments/`:
- **B-series** (`B0`…`B6`, `Bxv2`…): deterministic baselines, increasing features.
- **N-series** (`N1v10`…`N1v30d`): novel/advanced variants — vMF-NCE, token-targets, cross-subject, dual-head, bigG, rerank heads, etc.

For the frozen 77.2% retrieval recipe: compact score `csls`, legacy score `csls`, fusion `normalized_weighted`, normalization `zscore`, shortlist 150, weights α/β/γ = 0.3/0.0/0.7. See `docs/EXPERIMENT_CONTEXT.md` and the frozen export in `docs/thesis/results/final_outputs/best_system`.

## Code quality

- Formatter: `black` (line-length 100); imports: `isort` (profile black); linter: `ruff` (target py310). Config in `pyproject.toml`.
- Type hints required on function/method signatures; Google-style docstrings on public APIs.
- `pathlib.Path` for filesystem ops, never hardcoded absolute paths.
- Use `logging` (module logger), not `print`.
- Loss math runs in float32 even under AMP — keep `autocast` at the training-loop level, not inside loss functions.
- Use `torch.logsumexp` / clamped `log(x.clamp(min=1e-8))` / `F.normalize(..., dim=-1)`; device-agnostic (`next(model.parameters()).device`), never `.cuda()`.

## Research discipline (non-obvious, enforce in writeups/analysis)

- **Never fabricate numbers.** Every reported metric must trace to a JSON/CSV in `outputs/`, `experimental_results/`, or `RaportPaper3/`. If a result doesn't exist yet, say "pending experiment."
- Cite as `Author et al., Year`. Key refs: Allen et al. 2022 (NSD), Scotti et al. 2024 (MindEye2), Banerjee et al. 2005 (vMF), Davidson et al. 2018 (Hyperspherical VAE), Ozcelik & VanRullen 2023 (Brain Diffuser), Naselaris et al. 2011 (encoding/decoding review).
- All experiments must be fully specified by a YAML config — no magic numbers in code.
- Never break existing configs/APIs without an explicit migration flag.

## Related docs (read these when context is needed)

- `README.md` — full project overview, results table, qualitative reconstructions
- `ARCHITECTURE.md` — full module tree and per-directory READMEs
- `AGENTS.md` — short orientation doc (overlaps with this file)
- `setup.md` — step-by-step cluster setup
- `docs/` — guides (`SETUP.md`, `RUNNING_EXPERIMENTS.md`, `EVALUATION_SUITE_GUIDE.md`), paper drafts, `EXPERIMENT_CONTEXT.md`
- `demo_thesis/CLAUDE.md` — frontend-specific guidance (port 3000, provenance system, `npm` scripts)
