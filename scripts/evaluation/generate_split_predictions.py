#!/usr/bin/env python3
"""Generate compact (768-D) predictions for a data split from a saved checkpoint.

Lightweight inference-only script for V39 reranker cache building.
Loads fMRI features directly and applies z-scoring manually — does NOT
require token caches, rerank caches, or pretrained model inits.

Memory: ~4 GB (model ~2GB + features ~2GB). Safe on shared H100.

Saves: {prefix}_predictions_compact.npy, {prefix}_ground_truth_compact.npy,
       {prefix}_kappas.npy, {prefix}_nsd_ids.npy

Usage:
    python scripts/evaluation/generate_split_predictions.py \
        --checkpoint experimental_results/V35_legacy_teacher_distill/subj01/checkpoint_best.pt \
        --split train --subject subj01

    python scripts/evaluation/generate_split_predictions.py \
        --checkpoint /home/jovyan/local-data/experiment_archive/N1v28a_dual_head/subj01/checkpoint_best.pt \
        --output-dir experimental_results/N1v28a_dual_head/subj01 \
        --split train --subject subj01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def _infer_dim_from_state_dict(state_dict: dict, pattern: str) -> int | None:
    """Find output dim of a linear layer matching pattern."""
    for key, tensor in state_dict.items():
        if pattern in key and key.endswith(".weight"):
            return tensor.shape[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact predictions for a data split"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: parent of checkpoint)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Output dir: default to checkpoint's parent (experimental_results/EXP/SUBJ/)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. Load checkpoint ───────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt["model_state_dict"]
    logger.info("Epoch: %s", ckpt.get("epoch", "?"))

    # ── 2. Load fMRI features ────────────────────────────────────────────
    features_path = Path(f"cache/preextracted/subject={args.subject}/fmri_features.npy")
    if not features_path.exists():
        raise FileNotFoundError(f"fMRI features not found: {features_path}")
    all_features = np.load(features_path)  # (30000, ~15724)
    logger.info("fMRI features: %s (%.1f GB)", all_features.shape,
                all_features.nbytes / 1e9)

    # ── 3. Load index and split ──────────────────────────────────────────
    # Use the experiment's split.json to get the exact same train/val split
    split_path = output_dir / "split.json"
    if not split_path.exists():
        # Try checkpoint's parent
        split_path = ckpt_path.parent / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"split.json not found in {output_dir} or {ckpt_path.parent}. "
            "Cannot determine train/val split."
        )
    with open(split_path) as f:
        split_info = json.load(f)
    logger.info("Loaded split from %s", split_path)

    # Load the trial metadata to get per-trial nsdIds
    # Try multiple known locations
    index_candidates = [
        Path(f"cache/preextracted/subject={args.subject}/trial_meta.parquet"),
        Path(f"cache/preproc/subject={args.subject}/index.parquet"),
        Path(f"cache/preproc/subject={args.subject}/index.csv"),
    ]
    index_path = None
    for p in index_candidates:
        if p.exists():
            index_path = p
            break
    if index_path is None:
        raise FileNotFoundError(
            f"Trial metadata not found. Tried: {[str(p) for p in index_candidates]}"
        )
    if str(index_path).endswith(".parquet"):
        index_df = pd.read_parquet(index_path)
    else:
        index_df = pd.read_csv(index_path)
    # Column might be nsdId or nsd_id
    if "nsdId" in index_df.columns:
        nsd_ids_all = index_df["nsdId"].values
    elif "nsd_id" in index_df.columns:
        nsd_ids_all = index_df["nsd_id"].values
    else:
        raise KeyError(f"No nsdId column found. Columns: {index_df.columns.tolist()}")
    logger.info("Trial metadata: %s from %s", index_df.shape, index_path)

    # split.json may have trial-level indices OR image-level nsd_ids
    if "train_indices" in split_info:
        # Direct trial indices
        if args.split == "train":
            trial_indices = np.array(split_info["train_indices"])
        else:
            trial_indices = np.array(split_info["val_indices"])
    elif "train_nsd_ids" in split_info:
        # Image-level split: map nsd_ids back to trial indices
        if args.split == "train":
            split_nsd_set = set(split_info["train_nsd_ids"])
        else:
            split_nsd_set = set(split_info["val_nsd_ids"])
        trial_indices = np.array([i for i, nid in enumerate(nsd_ids_all)
                                  if int(nid) in split_nsd_set])
    else:
        raise KeyError(f"Unrecognized split.json format. Keys: {list(split_info.keys())}")

    logger.info("Split '%s': %d trials", args.split, len(trial_indices))

    nsd_ids = nsd_ids_all[trial_indices]
    features = all_features[trial_indices]
    del all_features  # Free memory

    # ── 4. Apply z-scoring ───────────────────────────────────────────────
    zscore_stats_dir = output_dir / "zscore_stats"
    if not zscore_stats_dir.exists():
        zscore_stats_dir = ckpt_path.parent / "zscore_stats"

    if zscore_stats_dir.exists():
        # Determine which session each trial belongs to
        if "session" in index_df.columns:
            sessions = index_df["session"].values[trial_indices]
        elif "sessionId" in index_df.columns:
            sessions = index_df["sessionId"].values[trial_indices]
        else:
            sessions = None

        if sessions is not None:
            unique_sessions = np.unique(sessions)
            applied = 0
            for sess in unique_sessions:
                mean_path = zscore_stats_dir / f"session_{sess}_mean.npy"
                std_path = zscore_stats_dir / f"session_{sess}_std.npy"
                if mean_path.exists() and std_path.exists():
                    mean = np.load(mean_path)
                    std = np.load(std_path)
                    std = np.where(std < 1e-6, 1.0, std)
                    mask = sessions == sess
                    features[mask] = (features[mask] - mean) / std
                    applied += 1
            logger.info("Applied z-scoring for %d/%d sessions", applied, len(unique_sessions))
        else:
            logger.warning("No session column found — skipping z-scoring")
    else:
        logger.warning("No zscore_stats directory found — skipping z-scoring")

    # ── 5. Load CLIP embeddings (768-D GT) ───────────────────────────────
    emb_col = config.get("data", {}).get("embedding_column", "embedding")
    clip_path = Path("outputs/clip_cache/clip_multilayer.parquet")
    if not clip_path.exists():
        clip_path = Path("outputs/clip_cache/clip.parquet")
    if not clip_path.exists():
        raise FileNotFoundError("CLIP cache not found")

    clip_df = pd.read_parquet(clip_path)
    logger.info("CLIP cache: %d rows, columns: %s", len(clip_df), list(clip_df.columns)[:5])

    # Build nsdId -> embedding lookup
    if emb_col not in clip_df.columns and "fused" in clip_df.columns:
        emb_col = "fused"
    elif emb_col not in clip_df.columns:
        emb_col = "embedding"
    clip_lookup = {row["nsdId"]: np.asarray(row[emb_col], dtype=np.float32)
                   for _, row in clip_df.iterrows()}
    del clip_df

    # ── 6. Build model ───────────────────────────────────────────────────
    from fmri2img.models.unified_model import UnifiedModel

    input_dim = features.shape[1]
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "vmf")

    # Inject input_dim into encoder config so UnifiedModel can build the encoder
    model_cfg.setdefault("encoder", {})["input_dim"] = input_dim

    # Infer head dimensions from state_dict for vmf_triple
    if model_type == "vmf_triple":
        decoder_cfg = model_cfg.get("decoder", {})
        rich_dim = _infer_dim_from_state_dict(state_dict, "regression_head")
        rerank_dim = _infer_dim_from_state_dict(state_dict, "rerank_head")
        if rich_dim:
            decoder_cfg["rich_target_dim"] = rich_dim
        if rerank_dim:
            decoder_cfg["rerank_dim"] = rerank_dim
        model_cfg["decoder"] = decoder_cfg
        logger.info("vmf_triple: rich_dim=%s, rerank_dim=%s", rich_dim, rerank_dim)

    config["model"] = model_cfg
    model = UnifiedModel(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s...", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s...", len(unexpected), unexpected[:3])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s, %dM params", model_type, n_params // 1_000_000)

    # ── 7. Run inference ─────────────────────────────────────────────────
    n_trials = len(features)
    bs = args.batch_size
    all_preds = []
    all_kappas = []

    logger.info("Running inference: %d trials, batch_size=%d", n_trials, bs)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device != "cpu"),
                                              dtype=torch.bfloat16):
        for start in range(0, n_trials, bs):
            end = min(start + bs, n_trials)
            fmri_batch = torch.from_numpy(features[start:end]).to(device, dtype=torch.float32)

            output = model(fmri_batch)

            if isinstance(output, tuple):
                pred = output[0]  # compact mu (768-D)
                aux = output[1] if len(output) > 1 else {}
                if isinstance(aux, dict):
                    kappa = aux.get("kappa", aux.get("concentration"))
                    if kappa is not None:
                        all_kappas.append(kappa.squeeze(-1).cpu().float().numpy())
                elif torch.is_tensor(aux):
                    all_kappas.append(aux.squeeze(-1).cpu().float().numpy())
            else:
                pred = output

            all_preds.append(pred.cpu().float().numpy())

            if (start // bs) % 100 == 0:
                logger.info("  %d/%d trials", end, n_trials)

    preds = np.concatenate(all_preds)
    kappas = np.concatenate(all_kappas) if all_kappas else None
    logger.info("Raw predictions: %s", preds.shape)

    # ── 8. Image-level averaging ─────────────────────────────────────────
    unique_nsd = np.unique(nsd_ids)
    n_images = len(unique_nsd)
    emb_dim = preds.shape[1]
    logger.info("Averaging: %d trials -> %d images", n_trials, n_images)

    preds_img = np.zeros((n_images, emb_dim), dtype=np.float32)
    gts_img = np.zeros((n_images, 768), dtype=np.float32)
    kappas_img = np.zeros(n_images, dtype=np.float32) if kappas is not None else None

    for i, nid in enumerate(unique_nsd):
        mask = nsd_ids == nid
        preds_img[i] = preds[mask].mean(axis=0)
        if nid in clip_lookup:
            gts_img[i] = clip_lookup[nid]
        if kappas is not None:
            kappas_img[i] = kappas[mask[:len(kappas)]].mean()

    # L2-normalize
    nrm = np.linalg.norm(preds_img, axis=-1, keepdims=True)
    preds_img = preds_img / np.maximum(nrm, 1e-8)

    # ── 9. Save ──────────────────────────────────────────────────────────
    prefix = args.split

    np.save(metrics_dir / f"{prefix}_predictions.npy", preds_img)
    np.save(metrics_dir / f"{prefix}_predictions_compact.npy", preds_img)
    np.save(metrics_dir / f"{prefix}_ground_truth.npy", gts_img)
    np.save(metrics_dir / f"{prefix}_ground_truth_compact.npy", gts_img)
    np.save(metrics_dir / f"{prefix}_nsd_ids.npy", unique_nsd)
    if kappas_img is not None:
        np.save(metrics_dir / f"{prefix}_kappas.npy", kappas_img)

    # Sanity check
    cos_sim = (preds_img * gts_img).sum(axis=-1).mean()
    logger.info("Saved %d images to %s", n_images, metrics_dir)
    logger.info("Sanity: mean_cos_sim=%.4f (expect ~0.4-0.5 for train)", cos_sim)
    logger.info("Done.")


if __name__ == "__main__":
    main()
