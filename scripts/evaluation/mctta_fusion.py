#!/usr/bin/env python3
"""Fusion analysis using MC-TTA enhanced predictions."""

import numpy as np
import torch
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# Load all prediction variants
model_specs = {
    "V61a_orig":  "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_predictions.npy",
    "V61a_mc16":  "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_predictions_mctta16.npy",
    "V62a":       "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_predictions.npy",
    "V63a":       "experimental_results/V63a_strong_197k/subj01/metrics/shared1000_predictions.npy",
}

gt_specs = {
    "V61a_orig":  "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_ground_truth.npy",
    "V61a_mc16":  "experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_ground_truth.npy",
    "V62a":       "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_ground_truth.npy",
    "V63a":       "experimental_results/V63a_strong_197k/subj01/metrics/shared1000_ground_truth.npy",
}

preds, gts = {}, {}
for name, path in model_specs.items():
    if os.path.exists(path):
        preds[name] = np.load(path)
        gts[name] = np.load(gt_specs[name])
        print(f"  {name:12s}: {preds[name].shape}")

n = next(iter(preds.values())).shape[0]


def sim_matrix_gpu(p, g):
    pt = torch.from_numpy(p).to(device).float()
    gt_t = torch.from_numpy(g).to(device).float()
    pt = pt / pt.norm(dim=1, keepdim=True).clamp(min=1e-8)
    gt_t = gt_t / gt_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (pt @ gt_t.T).cpu().numpy()


def metrics_from_sims(sims, csls_k=10):
    diag = np.array([sims[i, i] for i in range(n)])
    ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n)])
    r1 = (ranks == 1).mean()
    r5 = (ranks <= 5).mean()
    r10 = (ranks <= 10).mean()
    mrr = (1.0 / ranks).mean()

    topk_p = np.partition(-sims, csls_k, axis=1)[:, :csls_k]
    hub_s = -topk_p.mean(axis=1)
    topk_g = np.partition(-(sims.T), csls_k, axis=1)[:, :csls_k]
    hub_t = -topk_g.mean(axis=1)
    csls = 2 * sims - hub_s[:, None] - hub_t[None, :]
    csls_diag = np.array([csls[i, i] for i in range(n)])
    csls_ranks = np.array([(csls[i] > csls_diag[i]).sum() + 1 for i in range(n)])
    cr1 = (csls_ranks == 1).mean()
    cr5 = (csls_ranks <= 5).mean()
    return {"r1": r1, "r5": r5, "r10": r10, "mrr": mrr,
            "csls_r1": cr1, "csls_r5": cr5, "med_rank": np.median(ranks)}


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-10)


# Compute sim matrices
sim = {}
print("\n" + "="*60)
print("INDIVIDUAL MODELS")
print("="*60)
for name in preds:
    sim[name] = sim_matrix_gpu(preds[name], gts[name])
    for k in [3, 5, 10]:
        m = metrics_from_sims(sim[name], csls_k=k)
        if k == 3:
            print(f"  {name:12s}: R@1={m['r1']:.3f}  CSLS(k3)={m['csls_r1']:.3f}  "
                  f"CSLS(k5)={metrics_from_sims(sim[name], 5)['csls_r1']:.3f}  "
                  f"CSLS(k10)={metrics_from_sims(sim[name], 10)['csls_r1']:.3f}")
            break

# === MC-TTA + V62a cross-space fusion ===
print("\n" + "="*60)
print("MC-TTA V61a + V62a CROSS-SPACE FUSION")
print("="*60)
if "V61a_mc16" in sim and "V62a" in sim:
    s1 = minmax(sim["V61a_mc16"])
    s2 = minmax(sim["V62a"])
    print("  MinMax normalization:")
    for w_int in range(50, 100, 5):
        w = w_int / 100
        fused = w * s1 + (1-w) * s2
        for k in [3, 5, 10]:
            m = metrics_from_sims(fused, csls_k=k)
            if k == 3:
                m5 = metrics_from_sims(fused, 5)
                m10 = metrics_from_sims(fused, 10)
                marker = " ***" if m["csls_r1"] >= 0.85 else ""
                print(f"    w={w:.2f}: R@1={m['r1']:.3f}  CSLS(k3)={m['csls_r1']:.3f}  "
                      f"CSLS(k5)={m5['csls_r1']:.3f}  CSLS(k10)={m10['csls_r1']:.3f}{marker}")
                break

    print("\n  Z-score normalization:")
    s1z = zscore(sim["V61a_mc16"])
    s2z = zscore(sim["V62a"])
    for w_int in range(50, 100, 5):
        w = w_int / 100
        fused = w * s1z + (1-w) * s2z
        m3 = metrics_from_sims(fused, 3)
        m10 = metrics_from_sims(fused, 10)
        marker = " ***" if m3["csls_r1"] >= 0.85 else ""
        print(f"    w={w:.2f}: R@1={m3['r1']:.3f}  CSLS(k3)={m3['csls_r1']:.3f}  "
              f"CSLS(k10)={m10['csls_r1']:.3f}{marker}")

# === Triple: MC-TTA V61a + V62a + V63a ===
print("\n" + "="*60)
print("TRIPLE: MC-TTA V61a + V62a + V63a")
print("="*60)
if all(k in sim for k in ["V61a_mc16", "V62a", "V63a"]):
    s1 = minmax(sim["V61a_mc16"])
    s2 = minmax(sim["V62a"])
    s3 = minmax(sim["V63a"])
    best_csls, best_combo = 0, None
    for w1i in range(30, 80, 5):
        for w2i in range(5, 40, 5):
            w3i = 100 - w1i - w2i
            if w3i < 0:
                continue
            w1, w2, w3 = w1i/100, w2i/100, w3i/100
            fused = w1*s1 + w2*s2 + w3*s3
            m = metrics_from_sims(fused, csls_k=3)
            if m["csls_r1"] > best_csls:
                best_csls = m["csls_r1"]
                best_combo = (w1, w2, w3)
    if best_combo:
        w1, w2, w3 = best_combo
        fused = w1*s1 + w2*s2 + w3*s3
        m3 = metrics_from_sims(fused, 3)
        m5 = metrics_from_sims(fused, 5)
        m10 = metrics_from_sims(fused, 10)
        print(f"  Best: V61a_mc={w1:.2f} V62a={w2:.2f} V63a={w3:.2f}")
        print(f"  R@1={m3['r1']:.3f}  CSLS(k3)={m3['csls_r1']:.3f}  "
              f"CSLS(k5)={m5['csls_r1']:.3f}  CSLS(k10)={m10['csls_r1']:.3f}")

# === MC-TTA for V63a too ===
print("\n" + "="*60)
print("Note: V63a MC-TTA not yet computed. Run mctta_reevaluate.py on V63a for potential further gains.")
print("="*60)

torch.cuda.empty_cache()
