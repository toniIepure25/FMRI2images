#!/usr/bin/env python3
"""Evaluate PPR + CSLS + raw cosine on all available best checkpoints.

Automatically discovers checkpoint directories, loads models, runs inference
on shared1000, and produces a comparison table. Also runs fusion sweeps
between the best available compact expert and the N1v28a legacy expert.

Usage:
    python scripts/evaluation/evaluate_all_checkpoints_ppr.py \
        --subject subj01 \
        --output-dir experimental_results/ppr_comparison
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ppr_all")


def csls(S, k=10):
    r = np.mean(np.sort(S, axis=1)[:, -k:], axis=1)
    c = np.mean(np.sort(S, axis=0)[-k:, :], axis=0)
    return 2 * S - r[:, None] - c[None, :]


def ranks_from(S):
    d = np.diag(S)
    return (S >= d[:, None]).sum(axis=1) - 1


def compute_retrieval_metrics(preds, gts, label=""):
    sim = preds @ gts.T
    raw_r = ranks_from(sim)
    csls_r = ranks_from(csls(sim))
    return {
        "label": label,
        "n": len(preds),
        "dim": int(preds.shape[1]),
        "raw_r@1": float((raw_r == 0).mean()),
        "raw_r@5": float((raw_r < 5).mean()),
        "csls_r@1": float((csls_r == 0).mean()),
        "csls_r@5": float((csls_r < 5).mean()),
    }


def ppr_score(sim, kappas):
    """PPR scoring: kappa * cosine(mu, z)."""
    return kappas[:, None] * sim


def norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def zscore(M):
    mu = M.mean(axis=1, keepdims=True)
    std = M.std(axis=1, keepdims=True) + 1e-8
    return (M - mu) / std


def discover_checkpoints(results_root, subject):
    """Find all experiment directories with a best checkpoint."""
    results_root = Path(results_root)
    found = []
    if not results_root.exists():
        return found

    for exp_dir in sorted(results_root.iterdir()):
        subj_dir = exp_dir / subject
        ckpt = subj_dir / "checkpoint_best.pt"
        if ckpt.exists():
            found.append({
                "experiment": exp_dir.name,
                "dir": str(subj_dir),
                "checkpoint": str(ckpt),
            })
    return found


def load_saved_predictions(metrics_dir):
    """Load pre-saved predictions from metrics directory."""
    metrics_dir = Path(metrics_dir)
    result = {}

    for split in ["shared1000", "val"]:
        preds_candidates = [
            f"{split}_predictions_compact.npy",
            f"{split}_predictions.npy",
        ]
        gt_candidates = [
            f"{split}_ground_truth_compact.npy",
            f"{split}_ground_truth.npy",
        ]
        kappa_path = metrics_dir / f"{split}_kappas.npy"
        ids_path = metrics_dir / f"{split}_nsd_ids.npy"

        preds_path = None
        for c in preds_candidates:
            p = metrics_dir / c
            if p.exists():
                preds_path = p
                break

        gt_path = None
        for c in gt_candidates:
            p = metrics_dir / c
            if p.exists():
                gt_path = p
                break

        if preds_path and gt_path:
            preds = np.load(preds_path).astype(np.float32)
            gt = np.load(gt_path).astype(np.float32)
            kappas = np.load(kappa_path).astype(np.float32) if kappa_path.exists() else None
            nsd_ids = np.load(ids_path).astype(np.int32) if ids_path.exists() else None

            result[split] = {
                "preds": preds,
                "gt": gt,
                "kappas": kappas,
                "nsd_ids": nsd_ids,
            }
            logger.info(
                "Loaded %s: preds=%s gt=%s kappas=%s",
                split, preds.shape, gt.shape,
                kappas.shape if kappas is not None else "None"
            )

    return result


def evaluate_experiment(exp_info, subject):
    """Evaluate one experiment's predictions with PPR + CSLS + raw."""
    metrics_dir = Path(exp_info["dir"]) / "metrics"
    if not metrics_dir.exists():
        logger.warning("No metrics dir for %s", exp_info["experiment"])
        return None

    splits = load_saved_predictions(metrics_dir)
    if not splits:
        logger.warning("No saved predictions for %s", exp_info["experiment"])
        return None

    results = {"experiment": exp_info["experiment"]}

    for split_name, data in splits.items():
        preds = norm(data["preds"])
        gt = norm(data["gt"])
        kappas = data["kappas"]

        if preds.shape[1] != gt.shape[1]:
            if preds.shape[1] > gt.shape[1]:
                token_dim = gt.shape[1]
                preds_compact = norm(preds[:, :token_dim])
                metrics = compute_retrieval_metrics(preds_compact, gt, f"{split_name}_compact")
            else:
                logger.warning("Pred dim %d < GT dim %d for %s", preds.shape[1], gt.shape[1], exp_info["experiment"])
                continue
        else:
            metrics = compute_retrieval_metrics(preds, gt, split_name)

        results[f"{split_name}_metrics"] = metrics

        if kappas is not None:
            sim = (preds if preds.shape[1] == gt.shape[1] else preds_compact) @ gt.T
            ppr_sim = ppr_score(sim, kappas)
            ppr_csls_sim = csls(ppr_sim)
            ppr_csls_r = ranks_from(ppr_csls_sim)
            results[f"{split_name}_ppr_csls_r@1"] = float((ppr_csls_r == 0).mean())
            results[f"{split_name}_kappa_stats"] = {
                "mean": float(kappas.mean()),
                "std": float(kappas.std()),
                "min": float(kappas.min()),
                "max": float(kappas.max()),
            }

    return results


def run_fusion_sweep(compact_info, legacy_info, split="shared1000"):
    """Run CSLS z-score fusion sweep between compact and legacy experts."""
    compact_dir = Path(compact_info["dir"]) / "metrics"
    legacy_dir = Path(legacy_info["dir"]) / "metrics"

    compact_data = load_saved_predictions(compact_dir).get(split)
    legacy_data = load_saved_predictions(legacy_dir).get(split)

    if compact_data is None or legacy_data is None:
        logger.warning("Cannot run fusion: missing %s predictions", split)
        return None

    c_preds = norm(compact_data["preds"])
    l_preds = norm(legacy_data["preds"])
    gt = norm(compact_data.get("gt", legacy_data["gt"]))
    c_kappas = compact_data.get("kappas")
    c_ids = compact_data.get("nsd_ids")
    l_ids = legacy_data.get("nsd_ids")

    if c_ids is not None and l_ids is not None and not np.array_equal(c_ids, l_ids):
        common = np.intersect1d(c_ids, l_ids)
        c_idx = np.array([np.where(c_ids == cid)[0][0] for cid in common])
        l_idx = np.array([np.where(l_ids == cid)[0][0] for cid in common])
        c_preds = c_preds[c_idx]
        l_preds = l_preds[l_idx]
        gt = gt[c_idx]
        if c_kappas is not None:
            c_kappas = c_kappas[c_idx]

    if c_preds.shape[1] != gt.shape[1]:
        if c_preds.shape[1] > gt.shape[1]:
            c_preds = norm(c_preds[:, :gt.shape[1]])
        else:
            logger.warning("Compact dim %d != GT dim %d", c_preds.shape[1], gt.shape[1])
            return None

    if l_preds.shape[1] != gt.shape[1]:
        if l_preds.shape[1] > gt.shape[1]:
            l_preds = norm(l_preds[:, :gt.shape[1]])
        else:
            logger.warning("Legacy dim %d != GT dim %d", l_preds.shape[1], gt.shape[1])
            return None

    c_sim = c_preds @ gt.T
    l_sim = l_preds @ gt.T
    c_csls = csls(c_sim)
    l_csls = csls(l_sim)
    c_z = zscore(c_csls)
    l_z = zscore(l_csls)

    best_r1, best_alpha = 0, 0
    sweep = []
    for a in np.arange(0.05, 0.96, 0.05):
        fused = a * c_z + (1 - a) * l_z
        r = ranks_from(fused)
        r1 = float((r == 0).mean())
        sweep.append({"alpha": round(float(a), 2), "r@1": r1})
        if r1 > best_r1:
            best_r1, best_alpha = r1, a

    result = {
        "compact": compact_info["experiment"],
        "legacy": legacy_info["experiment"],
        "best_alpha": float(best_alpha),
        "best_r@1": float(best_r1),
        "sweep": sweep,
    }

    if c_kappas is not None:
        c_ppr = c_kappas[:, None] * c_sim
        c_ppr_csls = csls(c_ppr)
        c_ppr_z = zscore(c_ppr_csls)
        best_r1_ppr, best_alpha_ppr = 0, 0
        ppr_sweep = []
        for a in np.arange(0.05, 0.96, 0.05):
            fused = a * c_ppr_z + (1 - a) * l_z
            r = ranks_from(fused)
            r1 = float((r == 0).mean())
            ppr_sweep.append({"alpha": round(float(a), 2), "r@1": r1})
            if r1 > best_r1_ppr:
                best_r1_ppr, best_alpha_ppr = r1, a
        result["ppr_best_alpha"] = float(best_alpha_ppr)
        result["ppr_best_r@1"] = float(best_r1_ppr)
        result["ppr_sweep"] = ppr_sweep

    return result


def main():
    parser = argparse.ArgumentParser(description="PPR evaluation on all checkpoints")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--results-root", type=str, default="experimental_results")
    parser.add_argument("--output-dir", type=str, default="experimental_results/ppr_comparison")
    parser.add_argument("--legacy-experiment", type=str, default="N1v28a_dual_head")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_experiments = [
        "V55b_subj01_finetune",
        "V56a_fusion_distill_fixed",
        "V56c_projection_rdrop",
        "V57a_roi_transformer_dual_head",
        "N1v28a_dual_head",
    ]

    all_results = []
    found_checkpoints = discover_checkpoints(args.results_root, args.subject)
    logger.info("Discovered %d experiments with checkpoints", len(found_checkpoints))

    for exp_info in found_checkpoints:
        if exp_info["experiment"] not in target_experiments:
            continue
        logger.info("Evaluating %s...", exp_info["experiment"])
        result = evaluate_experiment(exp_info, args.subject)
        if result:
            all_results.append(result)
            logger.info(
                "  %s: %s",
                exp_info["experiment"],
                {k: v for k, v in result.items() if "metrics" in k or "ppr" in k}
            )

    legacy_info = None
    for exp_info in found_checkpoints:
        if exp_info["experiment"] == args.legacy_experiment:
            legacy_info = exp_info
            break

    fusion_results = []
    if legacy_info:
        compact_candidates = [
            e for e in found_checkpoints
            if e["experiment"] in target_experiments
            and e["experiment"] != args.legacy_experiment
        ]
        for compact_info in compact_candidates:
            logger.info("Fusion sweep: %s + %s", compact_info["experiment"], legacy_info["experiment"])
            fusion = run_fusion_sweep(compact_info, legacy_info)
            if fusion:
                fusion_results.append(fusion)
                logger.info(
                    "  Best fusion R@1=%.3f (alpha=%.2f), PPR R@1=%.3f",
                    fusion["best_r@1"],
                    fusion["best_alpha"],
                    fusion.get("ppr_best_r@1", 0),
                )

    print("\n" + "=" * 80)
    print("PPR EVALUATION SUMMARY")
    print("=" * 80)
    for r in all_results:
        exp = r["experiment"]
        for key in sorted(r.keys()):
            if key.endswith("_metrics"):
                m = r[key]
                print(f"  {exp:40s} {m.get('label', key):15s}  "
                      f"raw_R@1={m['raw_r@1']:.3f}  csls_R@1={m['csls_r@1']:.3f}")
            if key.endswith("_ppr_csls_r@1"):
                print(f"  {exp:40s} {key:15s}  ppr_csls_R@1={r[key]:.3f}")

    print("\n" + "-" * 80)
    print("FUSION RESULTS")
    print("-" * 80)
    for f in fusion_results:
        line = f"  {f['compact']:30s} + {f['legacy']:20s}  R@1={f['best_r@1']:.3f} (alpha={f['best_alpha']:.2f})"
        if "ppr_best_r@1" in f:
            line += f"  PPR_R@1={f['ppr_best_r@1']:.3f} (alpha={f['ppr_best_alpha']:.2f})"
        print(line)

    print(f"\n  Previous best (V35+N1v28a): 77.2%")

    output = {
        "individual": all_results,
        "fusion": fusion_results,
        "previous_best": 0.772,
    }
    out_path = Path(args.output_dir) / "ppr_all_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
