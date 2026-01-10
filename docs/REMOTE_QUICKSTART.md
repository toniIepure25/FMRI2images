# Remote GPU Pod Quickstart

This is the canonical **fresh machine** flow (0 → runnable).

## 0) Clone

Clone the repo and `cd` into it.

## 1) Configure once

```bash
cp .env.example .env
# edit NSD_DATA_ROOT, CACHE_ROOT, OUTPUT_ROOT, CHECKPOINT_ROOT (and HF_TOKEN if needed)
```

Tip for pods: set these to mounted volumes (e.g. `/mnt/data`, `/mnt/cache`, `/mnt/outputs`).

## 2) Install

```bash
make setup
```

## 3) Doctor (recommended)

Runs preflight + dataset verification + model cache check.

```bash
make doctor
```

## 4) Prepare (idempotent)

Stages prerequisites (dataset check, model snapshot cache, index, preprocessing, CLIP cache).

```bash
make prepare
```

## 5) Run an experiment

```bash
make exp EXP=experiments/novel_subj01.yaml
```

Override example:

```bash
make exp EXP=experiments/novel_subj01.yaml OVERRIDES="training.lr=1e-4 training.epochs=50"
```

## Outputs / caches

- Outputs: `$OUTPUT_ROOT` (defaults to `./outputs`)
- Caches: `$CACHE_ROOT` (defaults to `./cache`)
- Checkpoints: `$CHECKPOINT_ROOT` (defaults to `./checkpoints`)
