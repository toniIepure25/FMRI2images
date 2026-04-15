#!/usr/bin/env python3
"""Feature ablation for the union-shortlist reranker.

Trains the reranker with subsets of features (leave-one-group-out and
single-group-only) to identify which feature groups contribute most.

Usage:
    python scripts/evaluation/ablate_reranker_features.py \
        experimental_results/V35_legacy_teacher_distill/subj01 \
        experimental_results/N1v28a_dual_head/subj01 \
        --shortlist-k 100 \
        --train-cache-split train_oof \
        --run-tag v40_ablation
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from fmri2img.models.union_shortlist_reranker import (
    CandidateReranker,
    shortlist_cross_entropy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_GROUPS: dict[str, list[int]] = {
    "raw_scores": list(range(0, 5)),
    "normalized_ranks": list(range(5, 8)),
    "reciprocal_ranks": list(range(8, 11)),
    "zscore_scores": list(range(11, 14)),
    "margin_to_top1": list(range(14, 17)),
    "score_diffs": list(range(17, 20)),
    "rank_gaps": list(range(20, 22)),
    "source_flags": list(range(22, 27)),
    "agreement_structure": list(range(27, 33)),
    "topk_indicators": list(range(33, 39)),
    "kappa_query": [39],
    "vmf_component": list(range(40, 46)),
    "vmf_auxiliary": list(range(46, 53)),
    "calibration": list(range(53, 57)),
}


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    return data


def _mask_features(
    features: np.ndarray,
    keep_indices: list[int],
) -> np.ndarray:
    """Zero out all feature columns except those in keep_indices."""
    masked = np.zeros_like(features)
    for idx in keep_indices:
        if idx < features.shape[-1]:
            masked[..., idx] = features[..., idx]
    return masked


def _train_and_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_shortlists: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_shortlists: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Train a reranker and return val metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CandidateReranker(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_mask = train_shortlists >= 0
    val_mask = val_shortlists >= 0

    t_feat = torch.from_numpy(train_features).float()
    t_lab = torch.from_numpy(train_labels).float()
    t_mask = torch.from_numpy(train_mask)
    v_feat = torch.from_numpy(val_features).float()
    v_lab = torch.from_numpy(val_labels).float()
    v_mask = torch.from_numpy(val_mask)

    valid_train = t_feat[t_mask]
    feat_mean = valid_train.mean(dim=0)
    feat_std = valid_train.std(dim=0).clamp(min=1e-8)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - feat_mean.to(x.device)) / feat_std.to(x.device)

    best_val_r1 = -1.0
    patience_counter = 0
    batch_size = 256
    n_train = t_feat.shape[0]

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            bf = normalize(t_feat[idx].to(device))
            bl = t_lab[idx].to(device)
            bm = t_mask[idx].to(device)
            logits = model(bf, bm)
            loss = shortlist_cross_entropy(logits, bl, bm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if epoch % 5 == 0 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                vf_n = normalize(v_feat.to(device))
                v_logits = model(vf_n, v_mask.to(device))
                preds = v_logits.argmax(dim=-1).cpu()
                gt_pos = v_lab.argmax(dim=-1)
                has_gt = v_lab.any(dim=1)
                correct = ((preds == gt_pos) & has_gt).float().sum()
                val_r1 = (correct / max(has_gt.sum().item(), 1)).item()

            if val_r1 > best_val_r1:
                best_val_r1 = val_r1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience // 5:
                    break

    model.eval()
    with torch.no_grad():
        vf_n = normalize(v_feat.to(device))
        v_logits = model(vf_n, v_mask.to(device))
        preds = v_logits.argmax(dim=-1).cpu()
        gt_pos = v_lab.argmax(dim=-1)
        has_gt = v_lab.any(dim=1)
        correct = ((preds == gt_pos) & has_gt).float().sum()
        total_with_gt = has_gt.sum().item()
        val_r1_final = (correct / max(total_with_gt, 1)).item()

        top5_correct = 0.0
        for qi in range(v_feat.shape[0]):
            if not has_gt[qi]:
                continue
            topk = v_logits[qi].topk(min(5, v_logits.shape[1])).indices.cpu()
            if gt_pos[qi] in topk:
                top5_correct += 1
        val_r5 = top5_correct / max(total_with_gt, 1)

    return {
        "R@1": val_r1_final,
        "R@5": val_r5,
        "best_val_r1_during_training": best_val_r1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation for union-shortlist reranker")
    parser.add_argument("tri_results_dir", type=str)
    parser.add_argument("legacy_results_dir", type=str)
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--train-cache-split", type=str, default="train_oof")
    parser.add_argument("--val-cache-split", type=str, default="val")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default="v40_ablation")
    args = parser.parse_args()

    tri_dir = Path(args.tri_results_dir)
    cache_dir = tri_dir / "cache"
    diagnostics_dir = tri_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    k = args.shortlist_k
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cache = _load_cache(cache_dir / f"union_shortlist_{args.train_cache_split}_k{k}.npz")
    val_cache = _load_cache(cache_dir / f"union_shortlist_{args.val_cache_split}_k{k}.npz")

    total_features = train_cache["features"].shape[-1]
    all_indices = list(range(total_features))

    train_args = dict(
        train_labels=train_cache["labels"],
        train_shortlists=train_cache["shortlists"],
        val_labels=val_cache["labels"],
        val_shortlists=val_cache["shortlists"],
        input_dim=total_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device=device,
    )

    results: dict[str, Any] = {"groups": {}, "leave_one_out": {}, "single_group": {}}

    # Baseline: all features
    logger.info("Training baseline (all %d features)...", total_features)
    baseline = _train_and_eval(
        train_features=train_cache["features"],
        val_features=val_cache["features"],
        **train_args,
    )
    results["baseline"] = baseline
    logger.info("  Baseline R@1: %.1f%%", baseline["R@1"] * 100)

    # Leave-one-group-out ablation
    for group_name, group_indices in FEATURE_GROUPS.items():
        valid_indices = [i for i in group_indices if i < total_features]
        if not valid_indices:
            continue
        keep = [i for i in all_indices if i not in valid_indices]
        logger.info("Leave-out '%s' (%d features removed, %d kept)...", group_name, len(valid_indices), len(keep))
        ablated = _train_and_eval(
            train_features=_mask_features(train_cache["features"], keep),
            val_features=_mask_features(val_cache["features"], keep),
            **train_args,
        )
        drop = baseline["R@1"] - ablated["R@1"]
        results["leave_one_out"][group_name] = {
            **ablated,
            "n_removed": len(valid_indices),
            "n_kept": len(keep),
            "drop_from_baseline": drop,
        }
        logger.info("  '%s' removed -> R@1: %.1f%% (drop: %.1f pp)", group_name, ablated["R@1"] * 100, drop * 100)

    # Single-group-only ablation
    for group_name, group_indices in FEATURE_GROUPS.items():
        valid_indices = [i for i in group_indices if i < total_features]
        if not valid_indices:
            continue
        logger.info("Single group '%s' only (%d features)...", group_name, len(valid_indices))
        single = _train_and_eval(
            train_features=_mask_features(train_cache["features"], valid_indices),
            val_features=_mask_features(val_cache["features"], valid_indices),
            **train_args,
        )
        results["single_group"][group_name] = {
            **single,
            "n_features": len(valid_indices),
        }
        logger.info("  '%s' only -> R@1: %.1f%%", group_name, single["R@1"] * 100)

    # Rank groups by importance (drop from baseline)
    ranked = sorted(
        results["leave_one_out"].items(),
        key=lambda x: x[1]["drop_from_baseline"],
        reverse=True,
    )
    results["importance_ranking"] = [
        {"group": name, "drop_pp": round(data["drop_from_baseline"] * 100, 2)}
        for name, data in ranked
    ]

    out_path = diagnostics_dir / f"{args.run_tag}_feature_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ablation results to %s", out_path)

    print("\n" + "=" * 60)
    print("FEATURE ABLATION RESULTS")
    print("=" * 60)
    print(f"\nBaseline (all features): R@1 = {baseline['R@1']:.1%}")
    print("\nLeave-one-group-out (sorted by importance):")
    for item in results["importance_ranking"]:
        print(f"  {item['group']:30s}  drop = {item['drop_pp']:+.1f} pp")
    print("\nSingle-group-only:")
    for group_name, data in results["single_group"].items():
        print(f"  {group_name:30s}  R@1 = {data['R@1']:.1%}  ({data['n_features']} features)")


if __name__ == "__main__":
    main()
