#!/usr/bin/env python3
"""
Evaluate Posterior Predictive Retrieval (PPR) scoring on existing checkpoints.

Loads a trained vMF model, runs inference to get (mu, kappa) predictions,
then compares six scoring regimes:

  1. raw_cosine         — baseline: cos(mu, z)
  2. csls               — CSLS on cosine (current production)
  3. ppr_kappa_only     — kappa * cos(mu, z)
  4. ppr_full           — kappa * cos(mu, z) + log C_d(kappa)
  5. ppr_csls_kappa     — CSLS applied to kappa-weighted scores
  6. ppr_csls_full      — CSLS applied to full PPR log-likelihood scores

This script requires NO retraining — it evaluates existing checkpoints with
the new scoring function to validate the PPR hypothesis cheaply.

Usage:
    python scripts/evaluation/evaluate_ppr.py \
        --experiment-dir experimental_results/V55b_subj01_finetune/subj01 \
        --subject subj01 \
        --output-dir experimental_results/V55b_subj01_finetune/subj01/ppr_evaluation

    # Quick test on legacy expert:
    python scripts/evaluation/evaluate_ppr.py \
        --experiment-dir experimental_results/N1v28a_dual_head/subj01 \
        --subject subj01

    # Also evaluate legacy expert for fusion PPR:
    python scripts/evaluation/evaluate_ppr.py \
        --experiment-dir experimental_results/N1v28a_dual_head/subj01 \
        --legacy-dir experimental_results/N1v28a_dual_head/subj01 \
        --subject subj01
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_ppr")


def _load_config(experiment_dir: Path) -> dict:
    """Load YAML config from experiment directory or find it from checkpoint."""
    config_candidates = [
        experiment_dir / "config.yaml",
        experiment_dir / "config.yml",
    ]
    for p in config_candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)

    ckpt_path = experiment_dir / "checkpoint_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            return ckpt["config"]

    raise FileNotFoundError(
        f"No config found in {experiment_dir}. "
        "Expected config.yaml or config embedded in checkpoint."
    )


def _load_model_and_predict(
    experiment_dir: Path,
    subject: str,
    split: str = "val",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load model and run inference.

    Returns:
        (predictions, ground_truth, kappas, nsd_ids) where:
        - predictions: (N, D) L2-normalized mu predictions
        - ground_truth: (N, D) L2-normalized CLIP embeddings
        - kappas: (N,) concentration parameters
        - nsd_ids: (N,) stimulus IDs for cross-referencing
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from fmri2img.data.preextracted_dataset import PreextractedNSDDataset

    config = _load_config(experiment_dir)
    model_type = config.get("model", {}).get("type", "vmf")

    ckpt_path = experiment_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        ckpt_path = experiment_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {experiment_dir}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    from fmri2img.models.unified_model import create_model

    model_config = config.get("model", {})
    data_config = config.get("data", {})

    feat_path = Path(f"cache/preextracted/subject={subject}/fmri_features.npy")
    if feat_path.exists():
        n_voxels = np.load(feat_path, mmap_mode="r").shape[1]
        model_config.setdefault("encoder", {})["input_dim"] = n_voxels

    model = create_model(model_config)

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if hasattr(model, "load_state_dict"):
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning("Missing keys: %s", incompatible.missing_keys[:5])
        if incompatible.unexpected_keys:
            logger.warning(
                "Unexpected keys: %s", incompatible.unexpected_keys[:5]
            )

    model = model.to(device)
    model.eval()

    dataset = PreextractedNSDDataset(
        subject=subject,
        config=data_config,
        split=split,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    all_preds = []
    all_gts = []
    all_kappas = []
    all_nsd_ids = []

    vmf_is_log = False

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                fmri = batch[0].to(device, dtype=torch.float32)
                gt = batch[1].to(device, dtype=torch.float32) if len(batch) > 1 else None
                nsd_id = batch[-1] if len(batch) > 2 else None
            elif isinstance(batch, dict):
                fmri = batch["fmri"].to(device, dtype=torch.float32)
                gt = batch.get("clip_embedding")
                if gt is not None:
                    gt = gt.to(device, dtype=torch.float32)
                nsd_id = batch.get("nsd_id")
            else:
                fmri = batch.to(device, dtype=torch.float32)
                gt = None
                nsd_id = None

            pred, aux = model(fmri)

            pred_np = pred.cpu().numpy()
            pred_norm = pred_np / (
                np.linalg.norm(pred_np, axis=-1, keepdims=True) + 1e-8
            )
            all_preds.append(pred_norm)

            if gt is not None:
                gt_np = gt.cpu().numpy()
                gt_norm = gt_np / (
                    np.linalg.norm(gt_np, axis=-1, keepdims=True) + 1e-8
                )
                all_gts.append(gt_norm)

            if aux is not None:
                kappa = aux.squeeze(-1)
                if vmf_is_log:
                    kappa = kappa.exp()
                all_kappas.append(kappa.cpu().numpy())

            if nsd_id is not None:
                if isinstance(nsd_id, torch.Tensor):
                    all_nsd_ids.append(nsd_id.cpu().numpy())
                else:
                    all_nsd_ids.append(np.array(nsd_id))

    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_gts, axis=0) if all_gts else None
    kappas = np.concatenate(all_kappas, axis=0) if all_kappas else None
    nsd_ids = np.concatenate(all_nsd_ids, axis=0) if all_nsd_ids else None

    logger.info(
        "Inference complete: %d samples, pred shape %s, kappa range [%.2f, %.2f]",
        len(predictions),
        predictions.shape,
        kappas.min() if kappas is not None else 0,
        kappas.max() if kappas is not None else 0,
    )

    return predictions, ground_truth, kappas, nsd_ids


def _evaluate_split(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    kappas: np.ndarray,
    split_name: str,
    csls_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Run PPR evaluation on a single split."""
    from fmri2img.eval.ppr_scoring import compute_ppr_retrieval_metrics

    logger.info("=== PPR Evaluation: %s (%d samples) ===", split_name, len(predictions))

    results = compute_ppr_retrieval_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        kappas=kappas,
        ks=(1, 5, 10),
        csls_k=csls_k,
    )

    logger.info("--- %s Summary ---", split_name)
    for method, metrics in sorted(results.items()):
        logger.info(
            "  %-20s  R@1=%.3f  R@5=%.3f  R@10=%.3f  MedR=%.1f",
            method,
            metrics["top1_accuracy"],
            metrics["top5_accuracy"],
            metrics["top10_accuracy"],
            metrics["median_rank"],
        )

    return results


def _kappa_bin_analysis(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    kappas: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, list]:
    """Analyze retrieval performance by kappa quantile bins.

    Tests the hypothesis that high-kappa queries are easier to rank.
    """
    from fmri2img.eval.ppr_scoring import _correct_ranks_from_scores, _csls_from_cosine

    cosine_sim = predictions @ ground_truth.T
    csls_scores = _csls_from_cosine(predictions, ground_truth, k=10)

    raw_ranks = _correct_ranks_from_scores(cosine_sim)
    csls_ranks = _correct_ranks_from_scores(csls_scores)

    ppr_scores = kappas[:, None] * cosine_sim
    ppr_ranks = _correct_ranks_from_scores(ppr_scores)

    bin_edges = np.percentile(kappas, np.linspace(0, 100, n_bins + 1))
    bin_results = {
        "bin_edges": bin_edges.tolist(),
        "bins": [],
    }

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (kappas >= lo) & (kappas <= hi)
        else:
            mask = (kappas >= lo) & (kappas < hi)

        n = mask.sum()
        if n == 0:
            continue

        bin_info = {
            "kappa_range": [float(lo), float(hi)],
            "n_samples": int(n),
            "kappa_mean": float(kappas[mask].mean()),
            "raw_r@1": float((raw_ranks[mask] < 1).mean()),
            "csls_r@1": float((csls_ranks[mask] < 1).mean()),
            "ppr_kappa_r@1": float((ppr_ranks[mask] < 1).mean()),
            "raw_mrr": float((1.0 / (raw_ranks[mask] + 1.0)).mean()),
            "csls_mrr": float((1.0 / (csls_ranks[mask] + 1.0)).mean()),
        }
        bin_results["bins"].append(bin_info)
        logger.info(
            "  Kappa bin [%.1f, %.1f]: n=%d  raw_R@1=%.3f  csls_R@1=%.3f  ppr_R@1=%.3f",
            lo, hi, n,
            bin_info["raw_r@1"], bin_info["csls_r@1"], bin_info["ppr_kappa_r@1"],
        )

    return bin_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPR scoring on existing vMF checkpoints"
    )
    parser.add_argument(
        "--experiment-dir", type=str, required=True,
        help="Path to experiment results directory (contains checkpoint_best.pt)",
    )
    parser.add_argument(
        "--legacy-dir", type=str, default=None,
        help="Optional: legacy expert directory for fusion PPR evaluation",
    )
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument("--kappa-bins", type=int, default=5)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "ppr_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Experiment dir: %s", exp_dir)
    logger.info("Output dir: %s", out_dir)

    predictions, ground_truth, kappas, nsd_ids = _load_model_and_predict(
        exp_dir, args.subject, split="val", batch_size=args.batch_size,
    )

    if ground_truth is None:
        logger.error("No ground truth embeddings available. Cannot evaluate.")
        sys.exit(1)

    if kappas is None:
        logger.error(
            "No kappa values from model. PPR requires a vMF model with kappa output."
        )
        sys.exit(1)

    all_results = {}

    val_results = _evaluate_split(
        predictions, ground_truth, kappas,
        split_name="val", csls_k=args.csls_k,
    )
    all_results["val"] = val_results

    logger.info("=== Kappa-Bin Analysis (val) ===")
    kappa_bins = _kappa_bin_analysis(
        predictions, ground_truth, kappas, n_bins=args.kappa_bins,
    )
    all_results["val_kappa_bins"] = kappa_bins

    improvement = {}
    if "csls" in val_results and "ppr_csls_full" in val_results:
        for method in ["ppr_kappa_only", "ppr_full", "ppr_csls_kappa", "ppr_csls_full"]:
            if method in val_results:
                delta = (
                    val_results[method]["top1_accuracy"]
                    - val_results["csls"]["top1_accuracy"]
                )
                improvement[f"{method}_vs_csls"] = {
                    "delta_r@1": float(delta),
                    "ppr_r@1": float(val_results[method]["top1_accuracy"]),
                    "csls_r@1": float(val_results["csls"]["top1_accuracy"]),
                }
    all_results["ppr_vs_csls_improvement"] = improvement

    output_path = out_dir / "ppr_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to: %s", output_path)

    logger.info("")
    logger.info("=== PPR vs CSLS Summary ===")
    for key, vals in improvement.items():
        sign = "+" if vals["delta_r@1"] >= 0 else ""
        logger.info(
            "  %-25s: %s%.3f  (PPR=%.3f, CSLS=%.3f)",
            key, sign, vals["delta_r@1"], vals["ppr_r@1"], vals["csls_r@1"],
        )


if __name__ == "__main__":
    main()
