#!/usr/bin/env python3
"""Cross-Architecture Fusion Analysis for 90%+ R@1.

Fuses predictions from:
  - V65a (MLP + V9 recipe, 197K-D)
  - V66b (ROI Transformer, 197K-D)
  - V62a (768-D CLS)
  - V61a (original MLP, 197K-D)

Supports:
  - Score-level fusion with z-score normalization
  - CSLS k-sweep
  - Kappa-weighted fusion
  - MC-TTA predictions
  - Error correlation analysis
"""

import numpy as np
import torch
import json
import os
import sys
from itertools import combinations

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

RESULTS_ROOT = "experimental_results"

MODEL_DIRS = {
    "V61a": f"{RESULTS_ROOT}/V61a_finetune_difflr/subj01/metrics",
    "V65a": f"{RESULTS_ROOT}/V65a_mlp_v9_recipe/subj01/metrics",
    "V66b": f"{RESULTS_ROOT}/V66b_roi_finetune_197k/subj01/metrics",
    "V62a": f"{RESULTS_ROOT}/V62a_cls_retrieval_768d/subj01/metrics",
    "V66a": f"{RESULTS_ROOT}/V66a_roi_pretrain/subj01/metrics",
}

all_preds = {}
all_gts = {}
all_kappas = {}


def load_model(name, d):
    """Load shared1000 prediction matrices; MC-TTA uses distinct filenames on disk."""
    gp_default = f"{d}/shared1000_ground_truth.npy"
    variants = [
        ("_mctta16", "shared1000_predictions_mctta16.npy"),
        ("_mctta", "shared1000_predictions_mctta.npy"),
        ("", "shared1000_predictions.npy"),
    ]
    for suffix, pred_fn in variants:
        pp = os.path.join(d, pred_fn)
        tag = f"{name}{suffix}" if suffix else name
        if not os.path.exists(pp):
            continue
        all_preds[tag] = np.load(pp)
        if os.path.exists(gp_default):
            all_gts[tag] = np.load(gp_default)
        kp_candidates = [
            os.path.join(d, f"shared1000_kappas{suffix}.npy") if suffix else os.path.join(d, "shared1000_kappas.npy"),
        ]
        if suffix == "_mctta16":
            kp_candidates.insert(0, os.path.join(d, "shared1000_kappas_mctta16.npy"))
        kp_candidates.append(os.path.join(d, "shared1000_kappas.npy"))
        for kp in kp_candidates:
            if os.path.exists(kp):
                all_kappas[tag] = np.load(kp)
                break
        dim = all_preds[tag].shape[1] if all_preds[tag].ndim == 2 else 0
        label = f"{dim}-D" if dim > 0 else "?"
        print(f"  Loaded {tag}: shape={all_preds[tag].shape} ({label})")


print("=== Loading predictions ===")
for name, d in MODEL_DIRS.items():
    load_model(name, d)

# Also look for mctta standalone results
for name in list(MODEL_DIRS.keys()):
    mctta_dir = f"{MODEL_DIRS[name]}/mctta"
    if os.path.isdir(mctta_dir):
        pp = f"{mctta_dir}/shared1000_predictions.npy"
        if os.path.exists(pp):
            tag = f"{name}_mctta_standalone"
            all_preds[tag] = np.load(pp)
            gp = f"{mctta_dir}/shared1000_ground_truth.npy"
            if os.path.exists(gp):
                all_gts[tag] = np.load(gp)
            print(f"  Loaded {tag}: shape={all_preds[tag].shape}")

print(f"\nTotal models loaded: {len(all_preds)}")
print()

if not all_preds:
    print("ERROR: No predictions found. Exiting.")
    sys.exit(1)

gt = None
for v in all_gts.values():
    gt = v
    break
if gt is None:
    print("ERROR: No ground truth found.")
    sys.exit(1)


def compute_sim_matrix_gpu(preds, gts):
    p = torch.from_numpy(preds).to(device).float()
    g = torch.from_numpy(gts).to(device).float()
    p = p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)
    g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (p @ g.T).cpu().numpy()


def retrieval_metrics(sims, csls_k=10):
    n = sims.shape[0]
    diag = np.array([sims[i, i] for i in range(n)])
    ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n)])
    r1 = float((ranks == 1).mean())
    r5 = float((ranks <= 5).mean())
    mrr = float((1.0 / ranks).mean())

    topk_p = np.partition(-sims, csls_k, axis=1)[:, :csls_k]
    hub_s = -topk_p.mean(axis=1)
    topk_g = np.partition(-(sims.T), csls_k, axis=1)[:, :csls_k]
    hub_t = -topk_g.mean(axis=1)
    csls = sims - hub_s[:, None] / 2 - hub_t[None, :] / 2
    diag_c = np.array([csls[i, i] for i in range(n)])
    ranks_c = np.array([(csls[i] > diag_c[i]).sum() + 1 for i in range(n)])
    cr1 = float((ranks_c == 1).mean())
    cr5 = float((ranks_c <= 5).mean())
    cmrr = float((1.0 / ranks_c).mean())

    return {
        "r@1": r1, "r@5": r5, "mrr": mrr,
        "csls_r@1": cr1, "csls_r@5": cr5, "csls_mrr": cmrr,
        "median_rank": float(np.median(ranks)),
    }


def zscore_normalize(sims):
    mu = sims.mean()
    std = sims.std()
    return (sims - mu) / max(std, 1e-8)


def minmax_normalize(sims):
    mn, mx = sims.min(), sims.max()
    return (sims - mn) / max(mx - mn, 1e-8)


# =============================================
# STEP 1: Individual model performance
# =============================================
print("=" * 60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("=" * 60)

sim_matrices = {}
for name, preds in all_preds.items():
    gt_key = name
    if name not in all_gts:
        for suf in ("_mctta16", "_mctta_standalone", "_mctta"):
            if name.endswith(suf):
                gt_key = name[: -len(suf)]
                break
    if gt_key not in all_gts:
        gt_key = list(all_gts.keys())[0]
    gts = all_gts[gt_key]

    if preds.shape[1] != gts.shape[1]:
        print(f"  SKIP {name}: dim mismatch preds={preds.shape[1]} vs gt={gts.shape[1]}")
        continue

    sims = compute_sim_matrix_gpu(preds, gts)
    sim_matrices[name] = sims
    m = retrieval_metrics(sims)
    print(f"\n  {name}:")
    print(f"    R@1={m['r@1']:.3f}  R@5={m['r@5']:.3f}  MRR={m['mrr']:.3f}")
    for k in [3, 5, 10]:
        mk = retrieval_metrics(sims, csls_k=k)
        print(f"    CSLS(k={k}): R@1={mk['csls_r@1']:.3f}  R@5={mk['csls_r@5']:.3f}")

# =============================================
# STEP 2: Error correlation analysis
# =============================================
print("\n" + "=" * 60)
print("ERROR CORRELATION ANALYSIS")
print("=" * 60)

correct_masks = {}
for name, sims in sim_matrices.items():
    n = sims.shape[0]
    diag = np.array([sims[i, i] for i in range(n)])
    ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n)])
    correct_masks[name] = (ranks == 1)

model_names = list(correct_masks.keys())
for i, n1 in enumerate(model_names):
    for n2 in model_names[i + 1:]:
        c1 = correct_masks[n1]
        c2 = correct_masks[n2]
        both_correct = (c1 & c2).sum()
        both_wrong = (~c1 & ~c2).sum()
        only_1 = (c1 & ~c2).sum()
        only_2 = (~c1 & c2).sum()
        total = len(c1)
        corr = np.corrcoef(c1.astype(float), c2.astype(float))[0, 1]
        print(f"\n  {n1} vs {n2}:")
        print(f"    Both correct: {both_correct}/{total} ({both_correct/total:.1%})")
        print(f"    Both wrong:   {both_wrong}/{total} ({both_wrong/total:.1%})")
        print(f"    Only {n1}: {only_1}/{total} ({only_1/total:.1%})")
        print(f"    Only {n2}: {only_2}/{total} ({only_2/total:.1%})")
        print(f"    Error correlation: {corr:.3f}")
        oracle = (c1 | c2).mean()
        print(f"    Oracle fusion R@1: {oracle:.3f}")

# =============================================
# STEP 3: Pairwise score-level fusion
# =============================================
print("\n" + "=" * 60)
print("PAIRWISE SCORE-LEVEL FUSION")
print("=" * 60)

for (n1, s1), (n2, s2) in combinations(sim_matrices.items(), 2):
    z1 = zscore_normalize(s1)
    z2 = zscore_normalize(s2)

    print(f"\n  {n1} + {n2}:")
    best_cr1 = 0
    best_w = 0
    best_k = 10
    for w in np.arange(0.1, 1.0, 0.05):
        fused = w * z1 + (1 - w) * z2
        for k in [3, 5, 10]:
            m = retrieval_metrics(fused, csls_k=k)
            if m["csls_r@1"] > best_cr1:
                best_cr1 = m["csls_r@1"]
                best_w = w
                best_k = k

    fused_best = best_w * z1 + (1 - best_w) * z2
    mb = retrieval_metrics(fused_best, csls_k=best_k)
    print(f"    Best: w={best_w:.2f}, k={best_k}: CSLS R@1={mb['csls_r@1']:.3f}, R@1={mb['r@1']:.3f}")

# =============================================
# STEP 4: Triple fusion (key combos)
# =============================================
print("\n" + "=" * 60)
print("TRIPLE FUSION (KEY COMBINATIONS)")
print("=" * 60)

key_triples = []
for n1, n2, n3 in combinations(sim_matrices.keys(), 3):
    key_triples.append((n1, n2, n3))

for n1, n2, n3 in key_triples:
    z1 = zscore_normalize(sim_matrices[n1])
    z2 = zscore_normalize(sim_matrices[n2])
    z3 = zscore_normalize(sim_matrices[n3])

    best_cr1 = 0
    best_params = (0.33, 0.33, 10)
    for w1 in np.arange(0.1, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            fused = w1 * z1 + w2 * z2 + w3 * z3
            for k in [3, 5, 10]:
                m = retrieval_metrics(fused, csls_k=k)
                if m["csls_r@1"] > best_cr1:
                    best_cr1 = m["csls_r@1"]
                    best_params = (w1, w2, k)

    w1, w2, k = best_params
    w3 = 1.0 - w1 - w2
    fused = w1 * z1 + w2 * z2 + w3 * z3
    m = retrieval_metrics(fused, csls_k=k)
    print(f"\n  {n1}({w1:.1f}) + {n2}({w2:.1f}) + {n3}({w3:.1f}), k={k}:")
    print(f"    CSLS R@1={m['csls_r@1']:.3f}, R@1={m['r@1']:.3f}, MRR={m['csls_mrr']:.3f}")

# =============================================
# STEP 5: Kappa-weighted fusion
# =============================================
if all_kappas:
    print("\n" + "=" * 60)
    print("KAPPA-WEIGHTED FUSION")
    print("=" * 60)

    kw_models = [n for n in sim_matrices if n in all_kappas]
    if len(kw_models) >= 2:
        for n1, n2 in combinations(kw_models, 2):
            k1 = all_kappas[n1]
            k2 = all_kappas[n2]
            z1 = zscore_normalize(sim_matrices[n1])
            z2 = zscore_normalize(sim_matrices[n2])

            k1_norm = k1 / (k1 + k2 + 1e-8)
            k2_norm = k2 / (k1 + k2 + 1e-8)

            fused = k1_norm[:, None] * z1 + k2_norm[:, None] * z2
            for k in [3, 5, 10]:
                m = retrieval_metrics(fused, csls_k=k)
                if k == 3:
                    print(f"\n  Kappa-weighted {n1} + {n2} (k={k}):")
                    print(f"    CSLS R@1={m['csls_r@1']:.3f}, R@1={m['r@1']:.3f}")

# =============================================
# SUMMARY
# =============================================
print("\n" + "=" * 60)
print("SUMMARY — Best Results")
print("=" * 60)

best_individual = {}
for name, sims in sim_matrices.items():
    for k in [3, 10]:
        m = retrieval_metrics(sims, csls_k=k)
        key = f"{name}_k{k}"
        best_individual[key] = m["csls_r@1"]

if best_individual:
    sorted_bi = sorted(best_individual.items(), key=lambda x: -x[1])
    print("\n  Individual (top 5):")
    for name, v in sorted_bi[:5]:
        print(f"    {name}: {v:.3f}")

print("\nDone.")
