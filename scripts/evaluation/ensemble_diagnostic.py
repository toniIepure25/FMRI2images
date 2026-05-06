#!/usr/bin/env python3
"""Quick diagnostic: ensemble existing 197K-D shared1000 predictions on GPU."""

import numpy as np
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

models = {
    "V61a": "experimental_results/V61a_finetune_difflr/subj01/metrics",
    "V60b": "experimental_results/V60b_subj01_finetune_197k/subj01/metrics",
    "V60d": "experimental_results/V60d_subj01_finetune_kappa/subj01/metrics",
}

preds_np = {}
for name, d in models.items():
    p = np.load(f"{d}/shared1000_predictions.npy")
    preds_np[name] = p
    print(f"  Loaded {name}: {p.shape}")

gt_np = np.load("experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_ground_truth.npy")
n_images = gt_np.shape[0]
print(f"  GT: {gt_np.shape}, n_images={n_images}")

gt_t = torch.from_numpy(gt_np).to(device).float()
gt_norms = gt_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
gt_t = gt_t / gt_norms


def compute_metrics_gpu(pred_np, gt_tensor, csls_k=10):
    pred_t = torch.from_numpy(pred_np).to(device).float()
    pred_t = pred_t / pred_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims = pred_t @ gt_tensor.T  # (N, N)

    diag_vals = torch.diagonal(sims)
    ranks = (sims > diag_vals.unsqueeze(1)).sum(dim=1) + 1
    ranks_cpu = ranks.cpu().float()

    raw_r1 = (ranks_cpu == 1).float().mean().item()
    raw_r5 = (ranks_cpu <= 5).float().mean().item()
    raw_r10 = (ranks_cpu <= 10).float().mean().item()
    raw_mrr = (1.0 / ranks_cpu).mean().item()
    pos_sim = diag_vals.mean().item()

    # CSLS
    topk_p, _ = sims.topk(csls_k, dim=1)
    hub_s = topk_p.mean(dim=1)
    topk_g, _ = sims.T.topk(csls_k, dim=1)
    hub_t = topk_g.mean(dim=1)
    csls_sims = 2 * sims - hub_s.unsqueeze(1) - hub_t.unsqueeze(0)

    csls_diag = torch.diagonal(csls_sims)
    csls_ranks = (csls_sims > csls_diag.unsqueeze(1)).sum(dim=1) + 1
    csls_ranks_cpu = csls_ranks.cpu().float()

    csls_r1_val = (csls_ranks_cpu == 1).float().mean().item()
    csls_r5_val = (csls_ranks_cpu <= 5).float().mean().item()
    csls_r10_val = (csls_ranks_cpu <= 10).float().mean().item()

    return {
        "r_at_1": raw_r1, "r_at_5": raw_r5, "r_at_10": raw_r10,
        "mrr": raw_mrr, "pos_sim": pos_sim,
        "csls_r_at_1": csls_r1_val, "csls_r_at_5": csls_r5_val,
        "csls_r_at_10": csls_r10_val,
    }


print("\n" + "=" * 60)
print("INDIVIDUAL MODELS (197K-D)")
print("=" * 60)
for name in ["V61a", "V60b", "V60d"]:
    m = compute_metrics_gpu(preds_np[name], gt_t)
    print(f"  {name:6s}: R@1={m['r_at_1']:.3f}  CSLS_R@1={m['csls_r_at_1']:.3f}  "
          f"R@5={m['r_at_5']:.3f}  CSLS_R@5={m['csls_r_at_5']:.3f}  "
          f"pos_sim={m['pos_sim']:.4f}")

print("\n" + "=" * 60)
print("PAIRWISE ENSEMBLE (average embeddings)")
print("=" * 60)
pairs = [("V61a", "V60b"), ("V61a", "V60d"), ("V60b", "V60d")]
for a, b in pairs:
    ens = (preds_np[a] + preds_np[b]) / 2.0
    m = compute_metrics_gpu(ens, gt_t)
    print(f"  {a}+{b}: CSLS_R@1={m['csls_r_at_1']:.3f}  R@1={m['r_at_1']:.3f}  "
          f"CSLS_R@5={m['csls_r_at_5']:.3f}")

print("\n" + "=" * 60)
print("TRIPLE ENSEMBLE")
print("=" * 60)
ens3 = (preds_np["V61a"] + preds_np["V60b"] + preds_np["V60d"]) / 3.0
m = compute_metrics_gpu(ens3, gt_t)
print(f"  V61a+V60b+V60d (equal): CSLS_R@1={m['csls_r_at_1']:.3f}  "
      f"R@1={m['r_at_1']:.3f}  CSLS_R@5={m['csls_r_at_5']:.3f}")

print("\n" + "=" * 60)
print("WEIGHTED SWEEPS")
print("=" * 60)

print("\n--- V61a + V60b weighted ---")
best_w, best_csls = 0, 0
for w_int in range(30, 95, 5):
    w = w_int / 100.0
    ens = w * preds_np["V61a"] + (1 - w) * preds_np["V60b"]
    m = compute_metrics_gpu(ens, gt_t)
    marker = ""
    if m["csls_r_at_1"] > best_csls:
        best_csls = m["csls_r_at_1"]
        best_w = w
        marker = " ***"
    print(f"  w={w:.2f}: CSLS_R@1={m['csls_r_at_1']:.3f}  R@1={m['r_at_1']:.3f}{marker}")
print(f"  Best: w={best_w:.2f}, CSLS_R@1={best_csls:.3f}")

print("\n--- V61a + V60d weighted ---")
best_w, best_csls = 0, 0
for w_int in range(30, 95, 5):
    w = w_int / 100.0
    ens = w * preds_np["V61a"] + (1 - w) * preds_np["V60d"]
    m = compute_metrics_gpu(ens, gt_t)
    marker = ""
    if m["csls_r_at_1"] > best_csls:
        best_csls = m["csls_r_at_1"]
        best_w = w
        marker = " ***"
    print(f"  w={w:.2f}: CSLS_R@1={m['csls_r_at_1']:.3f}  R@1={m['r_at_1']:.3f}{marker}")
print(f"  Best: w={best_w:.2f}, CSLS_R@1={best_csls:.3f}")

print("\n--- Triple weighted (fine grid) ---")
best_combo, best_csls = None, 0
for w61 in range(40, 85, 5):
    for wb in range(5, 40, 5):
        wd = 100 - w61 - wb
        if wd < 5:
            continue
        w61f = w61 / 100.0
        wbf = wb / 100.0
        wdf = wd / 100.0
        ens = w61f * preds_np["V61a"] + wbf * preds_np["V60b"] + wdf * preds_np["V60d"]
        m = compute_metrics_gpu(ens, gt_t)
        if m["csls_r_at_1"] > best_csls:
            best_csls = m["csls_r_at_1"]
            best_combo = (w61f, wbf, wdf)
if best_combo:
    print(f"  Best triple: V61a={best_combo[0]:.2f}/V60b={best_combo[1]:.2f}"
          f"/V60d={best_combo[2]:.2f} => CSLS_R@1={best_csls:.3f}")
    ens = best_combo[0]*preds_np["V61a"] + best_combo[1]*preds_np["V60b"] + best_combo[2]*preds_np["V60d"]
    m = compute_metrics_gpu(ens, gt_t)
    print(f"    Full metrics: R@1={m['r_at_1']:.3f} R@5={m['r_at_5']:.3f} "
          f"CSLS_R@1={m['csls_r_at_1']:.3f} CSLS_R@5={m['csls_r_at_5']:.3f} "
          f"pos_sim={m['pos_sim']:.4f}")

torch.cuda.empty_cache()
print("\nDone.")
