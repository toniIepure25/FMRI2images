#!/usr/bin/env python3
"""Train and evaluate V39 union-shortlist residual reranker.

This script:
1. Loads pre-built union shortlist caches (from build_union_shortlist_cache.py)
2. Trains a small per-candidate MLP reranker on the TRAIN split
3. Selects the best checkpoint on VAL
4. Freezes and evaluates once on SHARED1000
5. Reports comparison against all baselines

The reranker is a residual decision model: it takes expert-derived candidate
features and learns to resolve disagreements between experts on a per-query,
per-candidate basis. It is NOT another embedding model.

Usage:
    python scripts/training/train_union_shortlist_reranker.py \
        experimental_results/V35_legacy_teacher_distill/subj01 \
        experimental_results/N1v28a_dual_head/subj01 \
        --shortlist-k 100

Outputs:
    {tri_results_dir}/diagnostics/v39_reranker_summary.json
    {tri_results_dir}/metrics/val_v39_reranker_metrics.json
    {tri_results_dir}/metrics/shared1000_v39_reranker_metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from sweep_tri_fusion_retrieval import (  # noqa: E402
    _align_common_ids,
    _build_split_scores,
    _gt_rank_from_scores,
    _load_legacy_split,
    _load_tri_split,
    _metrics_from_gt_rank,
    _nearby_experiment_dirs,
    _save_json,
)

# Add src to path for model import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from fmri2img.models.union_shortlist_reranker import (  # noqa: E402
    CandidateReranker,
    shortlist_cross_entropy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Cache loading ──────────────────────────────────────────────────────


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """Load union shortlist cache .npz file."""
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    logger.info("Loaded cache %s: features=%s, labels=%s",
                path.name, data["features"].shape, data["labels"].shape)
    return data


def _load_meta(path: Path) -> dict[str, Any]:
    """Load cache metadata JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path) as f:
        return json.load(f)


# ── Dataset helpers ────────────────────────────────────────────────────


class UnionShortlistDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapping a union shortlist cache."""

    def __init__(self, cache: dict[str, np.ndarray]) -> None:
        self.features = torch.from_numpy(cache["features"].astype(np.float32))
        self.labels = torch.from_numpy(cache["labels"].astype(np.float32))
        self.shortlists = cache["shortlists"]
        self.sizes = cache["sizes"]
        self.mask = torch.from_numpy(cache["shortlists"] >= 0)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],  # (max_size, F)
            "labels": self.labels[idx],      # (max_size,)
            "mask": self.mask[idx],          # (max_size,)
        }


# ── Evaluation ─────────────────────────────────────────────────────────


def _evaluate_reranker(
    model: CandidateReranker,
    dataset: UnionShortlistDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate reranker on a full split."""
    model.eval()
    n = len(dataset)
    shortlists = dataset.shortlists  # (N, max_size)
    sizes = dataset.sizes            # (N,)

    all_logits = np.empty((n, shortlists.shape[1]), dtype=np.float32)
    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_features = dataset.features[start:end].to(device)
        batch_mask = dataset.mask[start:end].to(device)
        with torch.no_grad():
            logits = model(batch_features, batch_mask)
        all_logits[start:end] = logits.cpu().numpy()

    # Rerank: for each query, pick the candidate with the highest logit
    labels = dataset.labels.numpy()
    has_gt = labels.any(axis=1)  # (N,) which queries have GT in shortlist

    # Compute GT rank after reranking
    gt_ranks = np.full(n, n, dtype=np.int32)  # default: worst possible rank
    for i in range(n):
        valid = shortlists[i, :sizes[i]]
        valid_logits = all_logits[i, :sizes[i]]
        order = np.argsort(-valid_logits)
        reranked = valid[order]
        # GT for query i is gallery index i
        gt_pos = np.where(reranked == i)[0]
        if len(gt_pos) > 0:
            gt_ranks[i] = int(gt_pos[0]) + 1
        # If GT not in shortlist, rank stays at n

    metrics = _metrics_from_gt_rank(gt_ranks)

    # Failure analysis
    gt_absent = int(np.sum(~has_gt))
    gt_present_miss = int(np.sum(has_gt & (gt_ranks > 1)))
    gt_present_hit = int(np.sum(has_gt & (gt_ranks == 1)))

    metrics["gt_absent_count"] = gt_absent
    metrics["gt_absent_rate"] = float(gt_absent / n)
    metrics["gt_present_miss_count"] = gt_present_miss
    metrics["gt_present_hit_count"] = gt_present_hit
    metrics["oracle_recall"] = float(has_gt.mean())

    return metrics


def _evaluate_baselines_from_cache(
    cache: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute baseline metrics directly from cached features."""
    features = cache["features"]     # (N, max_size, F)
    labels = cache["labels"]         # (N, max_size)
    shortlists = cache["shortlists"] # (N, max_size)
    sizes = cache["sizes"]           # (N,)
    n = features.shape[0]

    baselines = {}
    # Feature indices (from FEATURE_NAMES in build_union_shortlist_cache.py)
    score_indices = {
        "compact_raw": 0,
        "compact_csls": 1,
        "rerank": 2,
        "legacy_raw": 3,
        "legacy_csls": 4,
    }

    for name, feat_idx in score_indices.items():
        local_scores = features[:, :, feat_idx]
        gt_ranks = np.full(n, n, dtype=np.int32)
        for i in range(n):
            valid = shortlists[i, :sizes[i]]
            valid_scores = local_scores[i, :sizes[i]]
            order = np.argsort(-valid_scores)
            reranked = valid[order]
            gt_pos = np.where(reranked == i)[0]
            if len(gt_pos) > 0:
                gt_ranks[i] = int(gt_pos[0]) + 1
        baselines[name] = _metrics_from_gt_rank(gt_ranks)

    return baselines


# ── Training ───────────────────────────────────────────────────────────


def _train_reranker(
    train_dataset: UnionShortlistDataset,
    val_dataset: UnionShortlistDataset,
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 30,
    batch_size: int = 256,
    eval_every: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train the reranker and return the best model + metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CandidateReranker(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Reranker: %d parameters, hidden=%d, layers=%d, dropout=%.2f",
                n_params, hidden_dim, num_layers, dropout)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)

    # Feature normalization (fit on train only)
    train_features = train_dataset.features.numpy()
    train_mask = train_dataset.mask.numpy()
    valid_features = train_features[train_mask]
    feat_mean = valid_features.mean(axis=0, keepdims=True).astype(np.float32)
    feat_std = valid_features.std(axis=0, keepdims=True).astype(np.float32)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    # Apply normalization
    feat_mean_t = torch.from_numpy(feat_mean).to(device)
    feat_std_t = torch.from_numpy(feat_std).to(device)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - feat_mean_t) / feat_std_t

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    best_payload: dict[str, Any] | None = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = normalize(batch["features"].to(device))
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            logits = model(features, mask)
            loss = shortlist_cross_entropy(logits, labels, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch == 1 or epoch % eval_every == 0 or epoch == max_epochs:
            # Normalize val features
            orig_features = val_dataset.features.clone()
            val_dataset.features = normalize(val_dataset.features.to(device)).cpu()
            val_metrics = _evaluate_reranker(model, val_dataset, device)
            val_dataset.features = orig_features

            row = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)

            val_r1 = val_metrics["R@1"]
            logger.info(
                "  Epoch %3d: loss=%.4f, val_R@1=%.1f%%, val_R@5=%.1f%%, val_MRR=%.3f",
                epoch, avg_loss, val_r1 * 100, val_metrics["R@5"] * 100, val_metrics["MRR"],
            )

            if best_payload is None or val_r1 > best_payload["val_R@1"]:
                best_payload = {
                    "epoch": epoch,
                    "val_R@1": val_r1,
                    "val_metrics": val_metrics,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "train_loss": avg_loss,
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

    assert best_payload is not None
    model.load_state_dict(best_payload["state_dict"])
    model.eval()

    return {
        "model": model,
        "best_epoch": best_payload["epoch"],
        "best_val_metrics": best_payload["val_metrics"],
        "best_train_loss": best_payload["train_loss"],
        "n_params": n_params,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "history": history,
    }


# ── Compare against tri-fusion baselines ───────────────────────────────


def _load_existing_metrics(metrics_dir: Path, prefix: str) -> dict[str, Any]:
    """Load existing baseline metrics from the metrics directory."""
    baselines = {}
    for suffix in [
        "fused_metrics",
        "tri_fused_metrics",
        "tri_gated_metrics",
    ]:
        path = metrics_dir / f"{prefix}_{suffix}.json"
        if path.exists():
            with open(path) as f:
                baselines[suffix] = json.load(f)
    return baselines


def _comparison_table(
    reranker_metrics: dict[str, float],
    cache_baselines: dict[str, dict[str, float]],
    existing_baselines: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    """Build comparison table."""
    table = {
        "split": split_name,
        "v39_reranker": {k: v for k, v in reranker_metrics.items()
                         if isinstance(v, (int, float))},
    }

    # Cache-derived baselines (within-shortlist)
    for name, metrics in cache_baselines.items():
        table[f"shortlist_{name}"] = metrics

    # Existing file-based baselines
    if "fused_metrics" in existing_baselines:
        fused = existing_baselines["fused_metrics"]
        table["two_expert_fused"] = {
            "R@1": fused.get("R@1", fused.get("fused_R@1", None)),
        }
    if "tri_fused_metrics" in existing_baselines:
        tri = existing_baselines["tri_fused_metrics"]
        best = tri.get("tri_fused_best", tri.get("tri_fused_frozen", tri))
        table["fixed_tri_fused"] = {
            "R@1": best.get("R@1", None),
        }

    # Compute gains
    reranker_r1 = reranker_metrics["R@1"]
    gains = {}
    for key in ["shortlist_compact_csls", "shortlist_legacy_csls", "two_expert_fused", "fixed_tri_fused"]:
        if key in table and table[key].get("R@1") is not None:
            baseline_r1 = table[key]["R@1"]
            gains[f"gain_over_{key}"] = round(reranker_r1 - baseline_r1, 4)
    table["gains_pp"] = gains

    return table


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate V39 union-shortlist residual reranker"
    )
    parser.add_argument(
        "tri_results_dir", type=str,
        help="Triple-head results dir, e.g. experimental_results/V35_legacy_teacher_distill/subj01",
    )
    parser.add_argument(
        "legacy_results_dir", type=str,
        help="Legacy N1v28a results dir, e.g. experimental_results/N1v28a_dual_head/subj01",
    )
    parser.add_argument("--shortlist-k", type=int, default=100, help="Shortlist K (default: 100)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dim (default: 64)")
    parser.add_argument("--num-layers", type=int, default=2, help="MLP layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (default: 0.1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max epochs (default: 200)")
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience (default: 30)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-on-val", action="store_true",
        help="Train on val split (if train cache unavailable). WARNING: this WILL overfit.",
    )
    parser.add_argument(
        "--skip-shared1000", action="store_true",
        help="Skip shared1000 evaluation",
    )
    args = parser.parse_args()

    tri_results_dir = Path(args.tri_results_dir)
    legacy_results_dir = Path(args.legacy_results_dir)

    if not tri_results_dir.exists():
        nearby = _nearby_experiment_dirs(tri_results_dir)
        raise FileNotFoundError(f"Not found: {tri_results_dir}. Nearby: {nearby}")

    cache_dir = tri_results_dir / "cache"
    metrics_dir = tri_results_dir / "metrics"
    diagnostics_dir = tri_results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    k = args.shortlist_k
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Step 1: Load caches ────────────────────────────────────────────

    # Try to load train cache first
    train_cache_path = cache_dir / f"union_shortlist_train_k{k}.npz"
    val_cache_path = cache_dir / f"union_shortlist_val_k{k}.npz"
    s1000_cache_path = cache_dir / f"union_shortlist_shared1000_k{k}.npz"

    have_train = train_cache_path.exists()
    have_val = val_cache_path.exists()

    if not have_val:
        raise FileNotFoundError(
            f"Val cache required but not found: {val_cache_path}\n"
            f"Run build_union_shortlist_cache.py first."
        )

    if have_train:
        logger.info("Loading TRAIN cache for training...")
        train_cache = _load_cache(train_cache_path)
        val_cache = _load_cache(val_cache_path)
        train_split_name = "train"
    elif args.train_on_val:
        logger.warning(
            "No train cache found. Training on VAL split as requested. "
            "This WILL overfit — results are for debugging only."
        )
        val_cache = _load_cache(val_cache_path)
        train_cache = val_cache
        train_split_name = "val (WARNING: same as eval)"
    else:
        # Split val cache into train/val subsets (80/20)
        logger.info("No train cache found. Splitting val cache 80/20 for train/val...")
        full_cache = _load_cache(val_cache_path)
        n_full = full_cache["features"].shape[0]
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_full)
        n_train = int(0.8 * n_full)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        train_cache = {key: v[train_idx] for key, v in full_cache.items()}
        val_cache = {key: v[val_idx] for key, v in full_cache.items()}
        train_split_name = f"val_80pct ({n_train} queries)"
        logger.info("  Train: %d queries, Val: %d queries", n_train, len(val_idx))

    train_dataset = UnionShortlistDataset(train_cache)
    val_dataset = UnionShortlistDataset(val_cache)
    input_dim = train_cache["features"].shape[-1]

    logger.info("Input dim: %d features per candidate", input_dim)
    logger.info("Train: %d queries, Val: %d queries", len(train_dataset), len(val_dataset))

    # ── Step 2: Train ──────────────────────────────────────────────────

    logger.info("Training reranker (train_split=%s)...", train_split_name)
    result = _train_reranker(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    model = result["model"]
    feat_mean_t = torch.from_numpy(result["feat_mean"]).to(device)
    feat_std_t = torch.from_numpy(result["feat_std"]).to(device)

    def normalize_dataset(ds: UnionShortlistDataset) -> UnionShortlistDataset:
        """Apply train-fitted normalization to a dataset (in-place features)."""
        orig = ds.features.clone()
        ds.features = ((ds.features.to(device) - feat_mean_t) / feat_std_t).cpu()
        return orig

    logger.info("Best epoch: %d, Val R@1: %.1f%%",
                result["best_epoch"], result["best_val_metrics"]["R@1"] * 100)

    # ── Step 3: Final evaluation on val ────────────────────────────────

    logger.info("Evaluating on VAL with best checkpoint...")
    orig_val_feats = normalize_dataset(val_dataset)
    val_metrics = _evaluate_reranker(model, val_dataset, device)
    val_dataset.features = orig_val_feats

    val_cache_baselines = _evaluate_baselines_from_cache(val_cache)
    val_existing = _load_existing_metrics(metrics_dir, "val") if metrics_dir.exists() else {}
    val_comparison = _comparison_table(val_metrics, val_cache_baselines, val_existing, "val")

    # ── Step 4: Evaluate on SHARED1000 ─────────────────────────────────

    s1000_comparison = None
    s1000_metrics = None
    if not args.skip_shared1000 and s1000_cache_path.exists():
        logger.info("Evaluating on SHARED1000 (frozen)...")
        s1000_cache = _load_cache(s1000_cache_path)
        s1000_dataset = UnionShortlistDataset(s1000_cache)
        orig_s1000_feats = normalize_dataset(s1000_dataset)
        s1000_metrics = _evaluate_reranker(model, s1000_dataset, device)
        s1000_dataset.features = orig_s1000_feats

        s1000_cache_baselines = _evaluate_baselines_from_cache(s1000_cache)
        s1000_existing = _load_existing_metrics(metrics_dir, "shared1000") if metrics_dir.exists() else {}
        s1000_comparison = _comparison_table(s1000_metrics, s1000_cache_baselines, s1000_existing, "shared1000")
    elif not args.skip_shared1000:
        logger.warning("Shared1000 cache not found: %s", s1000_cache_path)

    # ── Step 5: Print results ──────────────────────────────────────────

    print("\n" + "=" * 70)
    print("V39 UNION SHORTLIST RESIDUAL RERANKER — RESULTS")
    print("=" * 70)

    print(f"\nModel: {result['n_params']} params, hidden={result['hidden_dim']}, "
          f"layers={result['num_layers']}, dropout={result['dropout']}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Train split: {train_split_name}")

    for label, comp in [("VAL", val_comparison), ("SHARED1000", s1000_comparison)]:
        if comp is None:
            continue
        print(f"\n--- {label} ---")
        print(f"  V39 Reranker R@1:     {comp['v39_reranker']['R@1']:.1%}")
        print(f"  V39 Reranker R@5:     {comp['v39_reranker']['R@5']:.1%}")
        print(f"  V39 Reranker R@10:    {comp['v39_reranker']['R@10']:.1%}")
        print(f"  Oracle recall:        {comp['v39_reranker']['oracle_recall']:.1%}")
        print(f"  GT absent:            {comp['v39_reranker']['gt_absent_count']}")
        print(f"  GT present, missed:   {comp['v39_reranker']['gt_present_miss_count']}")
        print()
        for key, baseline in comp.items():
            if key.startswith("shortlist_") and isinstance(baseline, dict) and "R@1" in baseline:
                name = key.replace("shortlist_", "")
                print(f"  {name:30s} R@1: {baseline['R@1']:.1%}")
        if "two_expert_fused" in comp and comp["two_expert_fused"].get("R@1") is not None:
            print(f"  {'2-expert fused':30s} R@1: {comp['two_expert_fused']['R@1']:.1%}")
        if "fixed_tri_fused" in comp and comp["fixed_tri_fused"].get("R@1") is not None:
            print(f"  {'fixed tri-fusion':30s} R@1: {comp['fixed_tri_fused']['R@1']:.1%}")
        if "gains_pp" in comp:
            print()
            for gain_name, gain_val in comp["gains_pp"].items():
                sign = "+" if gain_val >= 0 else ""
                print(f"  {gain_name}: {sign}{gain_val*100:.1f} pp")

    # ── Step 6: Save ───────────────────────────────────────────────────

    summary = {
        "experiment": "V39_union_shortlist_residual_reranker",
        "hyperparameters": {
            "shortlist_k": k,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "training": {
            "train_split": train_split_name,
            "best_epoch": result["best_epoch"],
            "best_train_loss": result["best_train_loss"],
            "n_params": result["n_params"],
        },
        "val": val_comparison,
    }
    if s1000_comparison is not None:
        summary["shared1000"] = s1000_comparison

    # Save summary
    summary_path = diagnostics_dir / "v39_reranker_summary.json"
    _save_json(summary_path, summary)
    logger.info("Saved summary to %s", summary_path)

    # Save per-split metrics
    if metrics_dir.exists():
        val_metrics_path = metrics_dir / "val_v39_reranker_metrics.json"
        _save_json(val_metrics_path, {
            "v39_reranker": val_metrics,
            "comparison": val_comparison,
        })
        logger.info("Saved val metrics to %s", val_metrics_path)

        if s1000_metrics is not None:
            s1000_metrics_path = metrics_dir / "shared1000_v39_reranker_metrics.json"
            _save_json(s1000_metrics_path, {
                "v39_reranker": s1000_metrics,
                "comparison": s1000_comparison,
            })
            logger.info("Saved shared1000 metrics to %s", s1000_metrics_path)

    # Save model checkpoint
    ckpt_path = cache_dir / f"v39_reranker_k{k}_best.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "feat_mean": result["feat_mean"],
        "feat_std": result["feat_std"],
        "best_epoch": result["best_epoch"],
        "shortlist_k": k,
    }, ckpt_path)
    logger.info("Saved model checkpoint to %s", ckpt_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
