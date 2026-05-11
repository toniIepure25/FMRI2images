#!/usr/bin/env python3
"""Comprehensive fusion analysis: all models, multiple strategies, CSLS-k sweep."""

import numpy as np
import torch
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# --- Load ALL shared1000 predictions ---
model_specs = {
    "V61a":     ("experimental_results/V61a_finetune_difflr/subj01/metrics", "197K"),
    "V63a":     ("experimental_results/V63a_strong_197k/subj01/metrics", "197K"),
    "V60b":     ("experimental_results/V60b_subj01_finetune_197k/subj01/metrics", "197K"),
    "V60d":     ("experimental_results/V60d_subj01_finetune_kappa/subj01/metrics", "197K"),
    "V62a":     ("experimental_results/V62a_cls_retrieval_768d/subj01/metrics", "768"),
    "V62b":     ("experimental_results/V62b_triple_head_768d_197k/subj01/metrics", "768"),
    "V63b":     ("experimental_results/V63b_cls_from_v61a/subj01/metrics", "768"),
}

# Also try MC-TTA predictions
mctta_specs = {
    "V61a_mc":  ("experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_predictions_mctta16.npy", "197K"),
    "V63a_mc":  ("experimental_results/V63a_strong_197k/subj01/metrics/shared1000_predictions_mctta16.npy", "197K"),
}

# V64 models
v64_specs = {
    "V64a":     ("experimental_results/V64a_continued_v61a/subj01/metrics", "197K"),
    "V64b":     ("experimental_results/V64b_frozen_encoder_deep_decoder/subj01/metrics", "197K"),
}
model_specs.update(v64_specs)

preds, gts, dims = {}, {}, {}
for name, (d, dim) in model_specs.items():
    pp = f"{d}/shared1000_predictions.npy"
    gp = f"{d}/shared1000_ground_truth.npy"
    if os.path.exists(pp):
        preds[name] = np.load(pp)
        gts[name] = np.load(gp)
        dims[name] = dim
        print(f"  {name:8s}: {preds[name].shape}  ({dim}-D)")

# Load MC-TTA predictions (already aggregated, use same GT as base model)
for name, (path, dim) in mctta_specs.items():
    base_name = name.replace("_mc", "")
    if os.path.exists(path) and base_name in gts:
        preds[name] = np.load(path)
        gts[name] = gts[base_name]
        dims[name] = dim
        print(f"  {name:8s}: {preds[name].shape}  ({dim}-D) [MC-TTA]")

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


# Compute all similarity matrices
sim = {}
print("\n" + "=" * 70)
print("INDIVIDUAL MODELS")
print("=" * 70)
for name in preds:
    sim[name] = sim_matrix_gpu(preds[name], gts[name])
    m = metrics_from_sims(sim[name])
    print(f"  {name:6s} ({dims[name]:4s}): R@1={m['r1']:.3f}  CSLS_R@1={m['csls_r1']:.3f}  "
          f"R@5={m['r5']:.3f}  MRR={m['mrr']:.3f}")


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


# === PAIRWISE cross-space fusion (197K + 768-D) ===
print("\n" + "=" * 70)
print("PAIRWISE CROSS-SPACE FUSION (best 197K + each 768-D)")
print("=" * 70)
m197k_best = ["V61a", "V63a"]
m768_best = ["V62a", "V62b", "V63b"]

for m197 in m197k_best:
    for m768 in m768_best:
        if m197 not in sim or m768 not in sim:
            continue
        s1 = minmax(sim[m197])
        s2 = minmax(sim[m768])
        best_c, best_w = 0, 0
        for w_int in range(0, 105, 5):
            w = w_int / 100
            fused = w * s1 + (1 - w) * s2
            m = metrics_from_sims(fused)
            if m["csls_r1"] > best_c:
                best_c = m["csls_r1"]
                best_w = w
        fused = best_w * s1 + (1 - best_w) * s2
        m = metrics_from_sims(fused)
        print(f"  {m197}+{m768}: w={best_w:.2f}  CSLS_R@1={best_c:.3f}  "
              f"R@1={m['r1']:.3f}  R@5={m['r5']:.3f}")


# === ALL 197K-D models score fusion ===
print("\n" + "=" * 70)
print("ALL 197K-D MODELS SCORE FUSION")
print("=" * 70)
models_197k = [k for k in sim if dims[k] == "197K"]
if len(models_197k) >= 2:
    sims_197k = {k: minmax(sim[k]) for k in models_197k}
    if len(models_197k) == 4:
        best_c, best_combo = 0, None
        for w0 in range(20, 90, 5):
            for w1 in range(5, 60, 5):
                for w2 in range(0, 40, 5):
                    w3 = 100 - w0 - w1 - w2
                    if w3 < 0:
                        continue
                    wf = [w0/100, w1/100, w2/100, w3/100]
                    fused = sum(wf[i]*sims_197k[models_197k[i]] for i in range(4))
                    m = metrics_from_sims(fused)
                    if m["csls_r1"] > best_c:
                        best_c = m["csls_r1"]
                        best_combo = dict(zip(models_197k, wf))
        if best_combo:
            combo_str = " / ".join(f"{k}={v:.2f}" for k, v in best_combo.items())
            fused = sum(best_combo[k]*sims_197k[k] for k in models_197k)
            m = metrics_from_sims(fused)
            print(f"  Best: {combo_str}")
            print(f"  CSLS_R@1={m['csls_r1']:.3f}  R@1={m['r1']:.3f}  R@5={m['r5']:.3f}")


# === GRAND FUSION: all models cross-space ===
print("\n" + "=" * 70)
print("GRAND FUSION (all models, cross-space)")
print("=" * 70)
all_names = list(sim.keys())
all_sims_norm = {k: minmax(sim[k]) for k in all_names}

# Try best 197K + best 768 + second-best 197K
combos_to_try = [
    ("V61a", "V62a"),
    ("V61a", "V62a", "V63a"),
    ("V61a", "V62a", "V60b"),
    ("V61a", "V62a", "V63a", "V60b"),
]

for combo in combos_to_try:
    valid = [c for c in combo if c in sim]
    if len(valid) < 2:
        continue
    sn = [all_sims_norm[c] for c in valid]
    best_c, best_w = 0, None
    n_models = len(valid)

    if n_models == 2:
        for w0 in range(10, 95, 5):
            wf = [w0/100, (100-w0)/100]
            fused = sum(wf[i]*sn[i] for i in range(2))
            m = metrics_from_sims(fused)
            if m["csls_r1"] > best_c:
                best_c = m["csls_r1"]
                best_w = wf
    elif n_models == 3:
        for w0 in range(20, 85, 5):
            for w1 in range(5, 50, 5):
                w2 = 100 - w0 - w1
                if w2 < 0:
                    continue
                wf = [w0/100, w1/100, w2/100]
                fused = sum(wf[i]*sn[i] for i in range(3))
                m = metrics_from_sims(fused)
                if m["csls_r1"] > best_c:
                    best_c = m["csls_r1"]
                    best_w = wf
    elif n_models == 4:
        for w0 in range(30, 80, 5):
            for w1 in range(5, 40, 5):
                for w2 in range(5, 30, 5):
                    w3 = 100 - w0 - w1 - w2
                    if w3 < 0:
                        continue
                    wf = [w0/100, w1/100, w2/100, w3/100]
                    fused = sum(wf[i]*sn[i] for i in range(4))
                    m = metrics_from_sims(fused)
                    if m["csls_r1"] > best_c:
                        best_c = m["csls_r1"]
                        best_w = wf

    if best_w:
        fused = sum(best_w[i]*sn[i] for i in range(n_models))
        m = metrics_from_sims(fused)
        combo_str = "+".join(f"{valid[i]}({best_w[i]:.2f})" for i in range(n_models))
        print(f"  {combo_str}")
        print(f"    CSLS_R@1={m['csls_r1']:.3f}  R@1={m['r1']:.3f}  R@5={m['r5']:.3f}  MRR={m['mrr']:.3f}")


# === CSLS-k sweep on best fusion ===
print("\n" + "=" * 70)
print("CSLS-k SWEEP on best fusion (V61a 0.75 + V62a 0.25)")
print("=" * 70)
if "V61a" in sim and "V62a" in sim:
    fused = 0.75 * minmax(sim["V61a"]) + 0.25 * minmax(sim["V62a"])
    for k in [3, 5, 7, 10, 15, 20, 30, 50]:
        m = metrics_from_sims(fused, csls_k=k)
        print(f"  k={k:3d}: CSLS_R@1={m['csls_r1']:.3f}  R@1={m['r1']:.3f}")


# === Reciprocal Rank Fusion (RRF) ===
print("\n" + "=" * 70)
print("RECIPROCAL RANK FUSION (RRF)")
print("=" * 70)

def rrf_fuse(sim_list, k_rrf=60):
    n_q = sim_list[0].shape[0]
    fused_scores = np.zeros((n_q, n_q), dtype=np.float64)
    for s in sim_list:
        for i in range(n_q):
            order = np.argsort(-s[i])
            for rank_pos, j in enumerate(order):
                fused_scores[i, j] += 1.0 / (k_rrf + rank_pos + 1)
    return fused_scores

# RRF: V61a + V62a
if "V61a" in sim and "V62a" in sim:
    for k_rrf in [10, 30, 60, 100]:
        rrf_scores = rrf_fuse([sim["V61a"], sim["V62a"]], k_rrf=k_rrf)
        m = metrics_from_sims(rrf_scores)
        print(f"  V61a+V62a (k_rrf={k_rrf:3d}): CSLS_R@1={m['csls_r1']:.3f}  R@1={m['r1']:.3f}")

# RRF: V61a + V62a + V63a
if all(k in sim for k in ["V61a", "V62a", "V63a"]):
    for k_rrf in [10, 30, 60]:
        rrf_scores = rrf_fuse([sim["V61a"], sim["V62a"], sim["V63a"]], k_rrf=k_rrf)
        m = metrics_from_sims(rrf_scores)
        print(f"  V61a+V62a+V63a (k_rrf={k_rrf:3d}): CSLS_R@1={m['csls_r1']:.3f}  R@1={m['r1']:.3f}")

# RRF: all 197K models
if all(k in sim for k in ["V61a", "V63a", "V60b", "V60d"]):
    rrf_scores = rrf_fuse([sim[k] for k in ["V61a", "V63a", "V60b", "V60d"]], k_rrf=60)
    m = metrics_from_sims(rrf_scores)
    print(f"  All 197K (k_rrf=60): CSLS_R@1={m['csls_r1']:.3f}  R@1={m['r1']:.3f}")


torch.cuda.empty_cache()
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
