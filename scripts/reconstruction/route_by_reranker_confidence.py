#!/usr/bin/env python3
"""Route queries to retrieval/refinement/diffusion based on reranker confidence.

This script replaces heuristic kappa-based routing with reranker-derived
confidence scores. High-confidence queries use retrieval directly (identity
pass), medium-confidence queries use lightweight img2img refinement, and
low-confidence queries use full diffusion.

Usage:
    python scripts/reconstruction/route_by_reranker_confidence.py \
        --reranker-checkpoint cache/v40_oof_tri_gate_reranker_k100_best.pt \
        --cache-dir experimental_results/V35_legacy_teacher_distill/subj01/cache \
        --split shared1000 \
        --shortlist-k 100 \
        --output-dir outputs/v40_confidence_routing

Outputs:
    {output_dir}/routing_assignments.json
    {output_dir}/routing_summary.json
    {output_dir}/per_query_routing.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUTE_CONFIGS = {
    "identity_pass": {
        "description": "Retrieval only, skip diffusion entirely",
        "strength": 0.0,
        "steps": 0,
        "guidance_scale": 0.0,
        "confidence_min": 0.85,
    },
    "low_strength_refine": {
        "description": "Minimal img2img refinement",
        "strength": 0.05,
        "steps": 20,
        "guidance_scale": 3.5,
        "confidence_min": 0.65,
    },
    "guided_refine": {
        "description": "Moderate img2img refinement",
        "strength": 0.14,
        "steps": 30,
        "guidance_scale": 4.5,
        "confidence_min": 0.40,
    },
    "exploratory_refine": {
        "description": "Full diffusion generation",
        "strength": 0.24,
        "steps": 40,
        "guidance_scale": 5.5,
        "confidence_min": 0.0,
    },
}


def _load_reranker(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained reranker checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmri2img.models.union_shortlist_reranker import (
        CandidateReranker,
        PlattCalibrator,
        ShortlistSetTransformerReranker,
        VMFEvidenceReranker,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    family = ckpt.get("model_family", "candidate_mlp")
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt.get("hidden_dim", 64)
    num_layers = ckpt.get("num_layers", 2)
    dropout = ckpt.get("dropout", 0.1)

    if family == "vmf_evidence":
        model = VMFEvidenceReranker(input_dim, hidden_dim, num_layers, dropout)
    elif family == "set_transformer":
        model = ShortlistSetTransformerReranker(input_dim, hidden_dim, num_layers, dropout)
    else:
        model = CandidateReranker(input_dim, hidden_dim, num_layers, dropout)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


def _assign_route(confidence: float) -> str:
    """Assign a routing bucket based on confidence score."""
    for route_name in ["identity_pass", "low_strength_refine", "guided_refine"]:
        if confidence >= ROUTE_CONFIGS[route_name]["confidence_min"]:
            return route_name
    return "exploratory_refine"


def main() -> None:
    parser = argparse.ArgumentParser(description="Route queries by reranker confidence")
    parser.add_argument("--reranker-checkpoint", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="shared1000")
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--calibrate-on-val", action="store_true",
        help="Fit Platt calibration on val cache before routing",
    )
    parser.add_argument("--val-split", type=str, default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    model, ckpt = _load_reranker(Path(args.reranker_checkpoint), device)
    feat_mean = torch.from_numpy(ckpt["feat_mean"]).float().to(device)
    feat_std = torch.from_numpy(ckpt["feat_std"]).float().to(device)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - feat_mean) / feat_std

    # Optional Platt calibration on val
    calibrator = None
    if args.calibrate_on_val:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
        from fmri2img.models.union_shortlist_reranker import PlattCalibrator

        val_path = cache_dir / f"union_shortlist_{args.val_split}_k{args.shortlist_k}.npz"
        if val_path.exists():
            val_data = dict(np.load(val_path, allow_pickle=True))
            v_feat = torch.from_numpy(val_data["features"]).float().to(device)
            v_labels = torch.from_numpy(val_data["labels"]).float().to(device)
            v_mask = torch.from_numpy(val_data["shortlists"] >= 0).to(device)
            with torch.no_grad():
                v_logits = model(normalize(v_feat), v_mask)
            calibrator = PlattCalibrator()
            calibrator.fit(v_logits, v_labels, v_mask)
            logger.info("Platt calibration fitted: T=%.3f, b=%.3f",
                        calibrator.temperature, calibrator.bias)
        else:
            logger.warning("Val cache not found for calibration: %s", val_path)

    # Load eval cache
    eval_path = cache_dir / f"union_shortlist_{args.split}_k{args.shortlist_k}.npz"
    eval_data = dict(np.load(eval_path, allow_pickle=True))
    features = torch.from_numpy(eval_data["features"]).float().to(device)
    labels = torch.from_numpy(eval_data["labels"]).float()
    shortlists = eval_data["shortlists"]
    mask = torch.from_numpy(shortlists >= 0).to(device)
    nsd_ids = eval_data.get("nsd_ids")
    n_queries = features.shape[0]

    with torch.no_grad():
        logits = model(normalize(features), mask)

    if calibrator is not None:
        with torch.no_grad():
            probs = calibrator.calibrate(logits, mask)
            confidences = probs.max(dim=-1).values.cpu().numpy()
    else:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1).values.cpu().numpy()

    predicted_indices = logits.argmax(dim=-1).cpu().numpy()

    # Assign routes
    routes = [_assign_route(float(c)) for c in confidences]
    gt_positions = labels.numpy().argmax(axis=-1)
    has_gt = labels.numpy().any(axis=1)
    correct = (predicted_indices == gt_positions) & has_gt

    # Build per-query routing table
    per_query = []
    for qi in range(n_queries):
        row: dict[str, Any] = {
            "query_index": qi,
            "confidence": float(confidences[qi]),
            "route": routes[qi],
            "predicted_candidate_idx": int(predicted_indices[qi]),
            "correct": bool(correct[qi]),
            "has_gt_in_shortlist": bool(has_gt[qi]),
        }
        if nsd_ids is not None:
            row["nsd_id"] = int(nsd_ids[qi])
        top1_gallery_idx = shortlists[qi, predicted_indices[qi]]
        row["top1_gallery_index"] = int(top1_gallery_idx)
        per_query.append(row)

    # Route statistics
    route_stats: dict[str, dict[str, Any]] = {}
    for route_name in ROUTE_CONFIGS:
        route_mask = [r == route_name for r in routes]
        n_route = sum(route_mask)
        if n_route == 0:
            route_stats[route_name] = {"count": 0, "fraction": 0.0}
            continue
        route_correct = sum(c for c, m in zip(correct, route_mask) if m)
        route_stats[route_name] = {
            "count": n_route,
            "fraction": n_route / n_queries,
            "R@1": route_correct / n_route,
            "mean_confidence": float(np.mean([confidences[i] for i, m in enumerate(route_mask) if m])),
            **ROUTE_CONFIGS[route_name],
        }

    summary = {
        "split": args.split,
        "n_queries": n_queries,
        "overall_R@1": float(correct.sum()) / max(has_gt.sum(), 1),
        "mean_confidence": float(np.mean(confidences)),
        "calibrated": calibrator is not None,
        "route_stats": route_stats,
    }

    if calibrator is not None:
        summary["platt_calibration"] = calibrator.state_dict()

    # Save outputs
    with open(output_dir / "routing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "routing_assignments.json", "w") as f:
        json.dump({"route_configs": ROUTE_CONFIGS, "assignments": per_query}, f, indent=2)

    csv_path = output_dir / "per_query_routing.csv"
    if per_query:
        fieldnames = list(per_query[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query)

    logger.info("Saved routing summary to %s", output_dir / "routing_summary.json")
    logger.info("Saved per-query routing to %s", csv_path)

    print("\n" + "=" * 60)
    print("CONFIDENCE-BASED ROUTING SUMMARY")
    print("=" * 60)
    print(f"\nSplit: {args.split}, Queries: {n_queries}")
    print(f"Overall R@1: {summary['overall_R@1']:.1%}")
    print(f"Mean confidence: {summary['mean_confidence']:.3f}")
    print(f"\nRoute distribution:")
    for route_name, stats in route_stats.items():
        if stats["count"] > 0:
            print(f"  {route_name:25s}  n={stats['count']:4d} ({stats['fraction']:.1%})  "
                  f"R@1={stats['R@1']:.1%}  conf={stats['mean_confidence']:.3f}")
        else:
            print(f"  {route_name:25s}  n=   0")


if __name__ == "__main__":
    main()
