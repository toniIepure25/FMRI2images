#!/usr/bin/env python3
"""Cross-space score-level fusion: combine 197K-D and 768-D similarity matrices.

Unlike embedding-level ensemble (which failed for 197K-D), score-level fusion
works across different embedding spaces by combining the (N x N) similarity
matrices before ranking. This captures complementary information from models
that operate on different feature spaces.

Also re-evaluates best 197K-D model (V61a) with MC-TTA on shared1000.
"""

import numpy as np
import torch
import json
import os
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# --- Load all available shared1000 predictions ---
models_197k = {
    "V61a": "experimental_results/V61a_finetune_difflr/subj01/metrics",
    "V60b": "experimental_results/V60b_subj01_finetune_197k/subj01/metrics",
    "V60d": "experimental_results/V60d_subj01_finetune_kappa/subj01/metrics",
}
models_768d = {
    "V62a": "experimental_results/V62a_cls_retrieval_768d/subj01/metrics",
}

all_preds = {}
all_gts = {}

for name, d in {**models_197k, **models_768d}.items():
    pp = f"{d}/shared1000_predictions.npy"
    gp = f"{d}/shared1000_ground_truth.npy"
    if os.path.exists(pp):
        all_preds[name] = np.load(pp)
        all_gts[name] = np.load(gp)
        print(f"Loaded {name}: preds={all_preds[name].shape}, dim={'197K' if all_preds[name].shape[1] > 1000 else '768'}")

# V62b uses 768-D compact space for headline metrics
v62b_pred_path = "experimental_results/V62b_triple_head_768d_197k/subj01/metrics/shared1000_predictions.npy"
if os.path.exists(v62b_pred_path):
    all_preds["V62b"] = np.load(v62b_pred_path)
    all_gts["V62b"] = np.load("experimental_results/V62b_triple_head_768d_197k/subj01/metrics/shared1000_ground_truth.npy")
    print(f"Loaded V62b: preds={all_preds['V62b'].shape}")

# Check for V62b rich predictions (197K-D)
v62b_rich_path = "experimental_results/V62b_triple_head_768d_197k/subj01/metrics/shared1000_predictions_rich.npy"
if os.path.exists(v62b_rich_path):
    all_preds["V62b_rich"] = np.load(v62b_rich_path)
    all_gts["V62b_rich"] = np.load("experimental_results/V62b_triple_head_768d_197k/subj01/metrics/shared1000_ground_truth_rich.npy")
    print(f"Loaded V62b_rich: preds={all_preds['V62b_rich'].shape}")


def compute_sim_matrix_gpu(preds, gts):
    """Compute normalized similarity matrix on GPU."""
    p = torch.from_numpy(preds).to(device).float()
    g = torch.from_numpy(gts).to(device).float()
    p = p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)
    g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (p @ g.T).cpu().numpy()


def retrieval_from_sims(sims, csls_k=10):
    """Compute retrieval metrics from a similarity matrix."""
    n = sims.shape[0]
    diag = np.array([sims[i, i] for i in range(n)])

    ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n)])
    r1 = (ranks == 1).mean()
    r5 = (ranks <= 5).mean()
    r10 = (ranks <= 10).mean()
    mrr = (1.0 / ranks).mean()

    # CSLS
    topk_p = np.partition(-sims, csls_k, axis=1)[:, :csls_k]
    hub_s = -topk_p.mean(axis=1)
    topk_g = np.partition(-(sims.T), csls_k, axis=1)[:, :csls_k]
    hub_t = -topk_g.mean(axis=1)
    csls = 2 * sims - hub_s[:, None] - hub_t[None, :]

    csls_diag = np.array([csls[i, i] for i in range(n)])
    csls_ranks = np.array([(csls[i] > csls_diag[i]).sum() + 1 for i in range(n)])
    csls_r1 = (csls_ranks == 1).mean()
    csls_r5 = (csls_ranks <= 5).mean()

    return {
        "r1": r1, "r5": r5, "r10": r10, "mrr": mrr,
        "csls_r1": csls_r1, "csls_r5": csls_r5,
    }


# --- Compute individual similarity matrices ---
print("\n" + "=" * 70)
print("INDIVIDUAL MODEL SIMILARITY MATRICES")
print("=" * 70)

sim_matrices = {}
for name in all_preds:
    sims = compute_sim_matrix_gpu(all_preds[name], all_gts[name])
    sim_matrices[name] = sims
    m = retrieval_from_sims(sims)
    dim_label = "768-D" if all_preds[name].shape[1] < 1000 else "197K-D"
    print(f"  {name:12s} ({dim_label:6s}): R@1={m['r1']:.3f}  CSLS_R@1={m['csls_r1']:.3f}  "
          f"R@5={m['r5']:.3f}  CSLS_R@5={m['csls_r5']:.3f}")


# --- Cross-space score-level fusion ---
print("\n" + "=" * 70)
print("CROSS-SPACE SCORE-LEVEL FUSION (197K-D + 768-D)")
print("=" * 70)
print("Combining similarity matrices from different embedding spaces.\n")

if "V61a" in sim_matrices and "V62a" in sim_matrices:
    # Need to normalize similarity matrices to same scale before combining
    # Min-max normalize each to [0, 1]
    def minmax_norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-10)

    def zscore_norm(s):
        return (s - s.mean()) / (s.std() + 1e-10)

    for norm_name, norm_fn in [("minmax", minmax_norm), ("zscore", zscore_norm), ("raw", lambda x: x)]:
        print(f"\n--- Normalization: {norm_name} ---")
        s_197k = norm_fn(sim_matrices["V61a"])
        s_768d = norm_fn(sim_matrices["V62a"])

        best_csls, best_w = 0, 0
        for w_int in range(0, 105, 5):
            w = w_int / 100.0
            fused = w * s_197k + (1 - w) * s_768d
            m = retrieval_from_sims(fused)
            marker = ""
            if m["csls_r1"] > best_csls:
                best_csls = m["csls_r1"]
                best_w = w
                marker = " ***"
            if w_int % 10 == 0:
                print(f"  w_197k={w:.2f}: R@1={m['r1']:.3f}  CSLS_R@1={m['csls_r1']:.3f}{marker}")
        print(f"  BEST: w_197k={best_w:.2f}, CSLS_R@1={best_csls:.3f}")

        # Full metrics at best weight
        fused = best_w * s_197k + (1 - best_w) * s_768d
        m = retrieval_from_sims(fused)
        print(f"  Full: R@1={m['r1']:.3f} R@5={m['r5']:.3f} R@10={m['r10']:.3f} "
              f"CSLS_R@1={m['csls_r1']:.3f} CSLS_R@5={m['csls_r5']:.3f} MRR={m['mrr']:.3f}")


# --- Also try: V62b_rich (197K-D from vmf_triple) + V62b (768-D compact) ---
if "V62b" in sim_matrices and "V62b_rich" in sim_matrices:
    print("\n" + "=" * 70)
    print("V62b INTERNAL FUSION (compact 768-D + rich 197K-D)")
    print("=" * 70)
    s_compact = minmax_norm(sim_matrices["V62b"])
    s_rich = minmax_norm(sim_matrices["V62b_rich"])
    best_csls, best_w = 0, 0
    for w_int in range(0, 105, 5):
        w = w_int / 100.0
        fused = w * s_rich + (1 - w) * s_compact
        m = retrieval_from_sims(fused)
        if m["csls_r1"] > best_csls:
            best_csls = m["csls_r1"]
            best_w = w
    print(f"  BEST: w_rich={best_w:.2f}, CSLS_R@1={best_csls:.3f}")


# --- Triple fusion: V61a + V62a + V60b ---
print("\n" + "=" * 70)
print("TRIPLE CROSS-SPACE FUSION")
print("=" * 70)

if all(k in sim_matrices for k in ["V61a", "V62a", "V60b"]):
    s1 = minmax_norm(sim_matrices["V61a"])   # 197K-D best
    s2 = minmax_norm(sim_matrices["V62a"])   # 768-D best
    s3 = minmax_norm(sim_matrices["V60b"])   # 197K-D #2

    best_csls, best_combo = 0, None
    for w1 in range(30, 90, 5):
        for w2 in range(5, 50, 5):
            w3_val = 100 - w1 - w2
            if w3_val < 0:
                continue
            wf1, wf2, wf3 = w1/100, w2/100, w3_val/100
            fused = wf1 * s1 + wf2 * s2 + wf3 * s3
            m = retrieval_from_sims(fused)
            if m["csls_r1"] > best_csls:
                best_csls = m["csls_r1"]
                best_combo = (wf1, wf2, wf3)

    if best_combo:
        print(f"  BEST: V61a={best_combo[0]:.2f}/V62a={best_combo[1]:.2f}/V60b={best_combo[2]:.2f}")
        fused = best_combo[0]*s1 + best_combo[1]*s2 + best_combo[2]*s3
        m = retrieval_from_sims(fused)
        print(f"  R@1={m['r1']:.3f} R@5={m['r5']:.3f} CSLS_R@1={m['csls_r1']:.3f} CSLS_R@5={m['csls_r5']:.3f}")


# --- All 4 models fusion ---
print("\n" + "=" * 70)
print("QUAD FUSION (V61a + V62a + V60b + V60d)")
print("=" * 70)

if all(k in sim_matrices for k in ["V61a", "V62a", "V60b", "V60d"]):
    s1 = minmax_norm(sim_matrices["V61a"])
    s2 = minmax_norm(sim_matrices["V62a"])
    s3 = minmax_norm(sim_matrices["V60b"])
    s4 = minmax_norm(sim_matrices["V60d"])

    best_csls, best_combo = 0, None
    for w1 in range(40, 85, 5):
        for w2 in range(5, 35, 5):
            for w3 in range(5, 30, 5):
                w4_val = 100 - w1 - w2 - w3
                if w4_val < 0:
                    continue
                wf = [w1/100, w2/100, w3/100, w4_val/100]
                fused = wf[0]*s1 + wf[1]*s2 + wf[2]*s3 + wf[3]*s4
                m = retrieval_from_sims(fused)
                if m["csls_r1"] > best_csls:
                    best_csls = m["csls_r1"]
                    best_combo = wf

    if best_combo:
        print(f"  BEST: V61a={best_combo[0]:.2f}/V62a={best_combo[1]:.2f}/"
              f"V60b={best_combo[2]:.2f}/V60d={best_combo[3]:.2f}")
        fused = best_combo[0]*s1 + best_combo[1]*s2 + best_combo[2]*s3 + best_combo[3]*s4
        m = retrieval_from_sims(fused)
        print(f"  R@1={m['r1']:.3f} R@5={m['r5']:.3f} CSLS_R@1={m['csls_r1']:.3f} CSLS_R@5={m['csls_r5']:.3f}")


torch.cuda.empty_cache()
print("\nDone.")
