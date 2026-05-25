#!/usr/bin/env python3
"""Run the fixed triple fusion that matched the pod sweep (~86.3% CSLS R@1 @ k=3).

Combines **z-scored similarity matrices** from three streams (same geometry as
``cross_architecture_fusion.py``):

- **V61a** MC-TTA16 predictions (197K-D token targets)
- **V62a** CLS predictions (768-D)
- **V66a** ROI pretrain CLS (768-D)

Default weights from the pod log tail (grid search in ``cross_architecture_fusion.py``):
``w1=0.7, w2=0.1, w3=0.2`` with **CSLS k=3**.

**Data layout**

- **Local bundle:** run ``sync_fusion_triple_86_from_pod.sh`` (or copy the six
  ``.npy`` files) into ``experimental_results/fusion_triple_86/{v61a_mctta16,v62a,v66a}/``.
- **On the Jupyter pod:** use ``--use-repo-metrics`` so tensors are read from
  ``<repo>/experimental_results/...`` (no multi-GB download over ``kubectl cp``).
  Optionally ``--output-json`` and copy only that small file to your laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

device = (
    "cuda"
    if torch.cuda.is_available() and torch.cuda.device_count() > 0
    else "cpu"
)


def compute_sim_matrix_gpu(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    p = torch.from_numpy(preds).to(device).float()
    g = torch.from_numpy(gts).to(device).float()
    p = p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)
    g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (p @ g.T).cpu().numpy()


def compute_csls(sims: np.ndarray, k: int = 10) -> np.ndarray:
    """Return CSLS-corrected similarity matrix."""
    topk_p = np.partition(-sims, k, axis=1)[:, :k]
    hub_s = -topk_p.mean(axis=1)
    topk_g = np.partition(-(sims.T), k, axis=1)[:, :k]
    hub_t = -topk_g.mean(axis=1)
    return sims - hub_s[:, None] / 2 - hub_t[None, :] / 2


def retrieval_metrics(sims: np.ndarray, csls_k: int = 10) -> dict[str, float]:
    n = sims.shape[0]
    diag = np.array([sims[i, i] for i in range(n)])
    ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n)])
    r1 = float((ranks == 1).mean())
    r5 = float((ranks <= 5).mean())
    mrr = float((1.0 / ranks).mean())

    csls = compute_csls(sims, csls_k)
    diag_c = np.array([csls[i, i] for i in range(n)])
    ranks_c = np.array([(csls[i] > diag_c[i]).sum() + 1 for i in range(n)])
    cr1 = float((ranks_c == 1).mean())
    cr5 = float((ranks_c <= 5).mean())
    cmrr = float((1.0 / ranks_c).mean())

    return {
        "r@1": r1,
        "r@5": r5,
        "mrr": mrr,
        "csls_r@1": cr1,
        "csls_r@5": cr5,
        "csls_mrr": cmrr,
        "median_rank": float(np.median(ranks)),
    }


def zscore_normalize(sims: np.ndarray) -> np.ndarray:
    mu = sims.mean()
    std = sims.std()
    return (sims - mu) / max(std, 1e-8)


def load_pair(pred_path: str, gt_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(pred_path)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    preds = np.load(pred_path)
    gts = np.load(gt_path)
    if preds.shape[1] != gts.shape[1]:
        raise ValueError(
            f"dim mismatch: preds {preds.shape} vs gts {gts.shape} for {pred_path}"
        )
    return preds, gts


def _default_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _paths_from_bundle(bundle: str) -> dict[str, tuple[str, str]]:
    return {
        "V61a_mctta16": (
            os.path.join(bundle, "v61a_mctta16", "shared1000_predictions_mctta16.npy"),
            os.path.join(bundle, "v61a_mctta16", "shared1000_ground_truth.npy"),
        ),
        "V62a": (
            os.path.join(bundle, "v62a", "shared1000_predictions.npy"),
            os.path.join(bundle, "v62a", "shared1000_ground_truth.npy"),
        ),
        "V66a": (
            os.path.join(bundle, "v66a", "shared1000_predictions.npy"),
            os.path.join(bundle, "v66a", "shared1000_ground_truth.npy"),
        ),
    }


def _paths_from_repo_metrics(repo_root: str) -> dict[str, tuple[str, str]]:
    er = os.path.join(repo_root, "experimental_results")
    return {
        "V61a_mctta16": (
            os.path.join(er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_predictions_mctta16.npy"),
            os.path.join(er, "V61a_finetune_difflr", "subj01", "metrics", "shared1000_ground_truth.npy"),
        ),
        "V62a": (
            os.path.join(er, "V62a_cls_retrieval_768d", "subj01", "metrics", "shared1000_predictions.npy"),
            os.path.join(er, "V62a_cls_retrieval_768d", "subj01", "metrics", "shared1000_ground_truth.npy"),
        ),
        "V66a": (
            os.path.join(er, "V66a_roi_pretrain", "subj01", "metrics", "shared1000_predictions.npy"),
            os.path.join(er, "V66a_roi_pretrain", "subj01", "metrics", "shared1000_ground_truth.npy"),
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed triple fusion on shared1000 sims.")
    ap.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Root of fusion_triple_86 layout (default: <repo>/experimental_results/fusion_triple_86)",
    )
    ap.add_argument(
        "--use-repo-metrics",
        action="store_true",
        help="Read npy files from experimental_results/... (for running on the pod with data in PVC)",
    )
    ap.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repo root when using --use-repo-metrics (default: parent of scripts/)",
    )
    ap.add_argument("--w1", type=float, default=0.7, help="Weight for V61a_mctta16 sim matrix")
    ap.add_argument("--w2", type=float, default=0.1, help="Weight for V62a sim matrix")
    ap.add_argument("--w3", type=float, default=0.2, help="Weight for V66a sim matrix")
    ap.add_argument("--csls-k", type=int, default=3, help="CSLS hubness k")
    ap.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write metrics + weights to this path (small file for copying off the pod)",
    )
    ap.add_argument(
        "--save-top1-dir",
        type=str,
        default=None,
        help="Directory to save per-stream and fused CSLS top-1 gallery indices as .npy",
    )
    args = ap.parse_args()

    repo = args.repo_root or _default_repo_root()
    if args.use_repo_metrics:
        paths = _paths_from_repo_metrics(repo)
        bundle = "experimental_results/{V61a,V62a,V66a}/subj01/metrics"
    else:
        bundle = args.bundle or os.path.join(repo, "experimental_results", "fusion_triple_86")
        paths = _paths_from_bundle(bundle)

    w1, w2, w3 = args.w1, args.w2, args.w3
    s = w1 + w2 + w3
    if abs(s - 1.0) > 1e-6:
        print(f"Renormalizing weights (sum was {s:.6f})", file=sys.stderr)
        w1, w2, w3 = w1 / s, w2 / s, w3 / s

    print(f"Device: {device}")
    print(f"Bundle: {bundle}")
    print(f"Weights: V61a_mctta16={w1:.3f}, V62a={w2:.3f}, V66a={w3:.3f}, CSLS k={args.csls_k}")

    sims = {}
    for name, (pp, gp) in paths.items():
        preds, gts = load_pair(pp, gp)
        sims[name] = compute_sim_matrix_gpu(preds, gts)
        m = retrieval_metrics(sims[name], csls_k=args.csls_k)
        print(f"\n  {name} alone: CSLS R@1={m['csls_r@1']:.4f}  R@1={m['r@1']:.4f}")

    z1 = zscore_normalize(sims["V61a_mctta16"])
    z2 = zscore_normalize(sims["V62a"])
    z3 = zscore_normalize(sims["V66a"])
    fused = w1 * z1 + w2 * z2 + w3 * z3
    mf = retrieval_metrics(fused, csls_k=args.csls_k)

    print("\n  --- Triple fusion (z-scored sims) ---")
    print(f"  CSLS R@1={mf['csls_r@1']:.4f}  R@1={mf['r@1']:.4f}  CSLS MRR={mf['csls_mrr']:.4f}")

    if args.save_top1_dir:
        top1_dir = os.path.abspath(args.save_top1_dir)
        os.makedirs(top1_dir, exist_ok=True)
        csls_fused = compute_csls(fused, args.csls_k)
        csls_v61a = compute_csls(sims["V61a_mctta16"], args.csls_k)
        csls_v62a = compute_csls(sims["V62a"], args.csls_k)
        csls_v66a = compute_csls(sims["V66a"], args.csls_k)
        for tag, mat in [
            ("triple_top1_ids", csls_fused),
            ("v61a_top1_ids", csls_v61a),
            ("v62a_top1_ids", csls_v62a),
            ("v66a_top1_ids", csls_v66a),
        ]:
            ids = np.argmax(mat, axis=1).astype(np.int64)
            out_path = os.path.join(top1_dir, f"{tag}.npy")
            np.save(out_path, ids)
            print(f"  Saved {out_path}  shape={ids.shape}")

        gt_ranks_fused = np.array(
            [(csls_fused[i] > csls_fused[i, i]).sum() + 1 for i in range(csls_fused.shape[0])]
        )
        np.save(os.path.join(top1_dir, "triple_gt_ranks.npy"), gt_ranks_fused)
        print(f"  Saved triple_gt_ranks.npy  shape={gt_ranks_fused.shape}")

        for tag, mat in [("v61a_gt_ranks", csls_v61a), ("v62a_gt_ranks", csls_v62a), ("v66a_gt_ranks", csls_v66a)]:
            ranks = np.array([(mat[i] > mat[i, i]).sum() + 1 for i in range(mat.shape[0])])
            np.save(os.path.join(top1_dir, f"{tag}.npy"), ranks)

    if args.output_json:
        out = {
            "triple": "V61a_mctta16+V62a+V66a",
            "weights": {"w1": w1, "w2": w2, "w3": w3},
            "csls_k": args.csls_k,
            "use_repo_metrics": args.use_repo_metrics,
            "repo_root": os.path.abspath(repo),
            "fused": mf,
            "individual": {
                n: retrieval_metrics(sims[n], csls_k=args.csls_k) for n in sims
            },
        }
        out_path = os.path.abspath(args.output_json)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
