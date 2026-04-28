#!/usr/bin/env python3
"""Phase 1: 197K-D standalone + fusion evaluation on shared1000."""
import json
import numpy as np


def csls(S, k=10):
    r = np.mean(np.sort(S, axis=1)[:, -k:], axis=1)
    c = np.mean(np.sort(S, axis=0)[-k:, :], axis=0)
    return 2 * S - r[:, None] - c[None, :]


def ranks_from(S):
    d = np.diag(S)
    return (S >= d[:, None]).sum(axis=1) - 1


def compute_metrics(preds, gts, label):
    sim = preds @ gts.T
    raw_r = ranks_from(sim)
    csls_r = ranks_from(csls(sim))
    return {
        "raw_r@1": float((raw_r == 0).mean()),
        "raw_r@5": float((raw_r < 5).mean()),
        "raw_r@10": float((raw_r < 10).mean()),
        "csls_r@1": float((csls_r == 0).mean()),
        "csls_r@5": float((csls_r < 5).mean()),
        "csls_r@10": float((csls_r < 10).mean()),
        "label": label,
        "n": len(preds),
        "dim": int(preds.shape[1]),
    }


def zscore(M):
    mu = M.mean(axis=1, keepdims=True)
    std = M.std(axis=1, keepdims=True) + 1e-8
    return (M - mu) / std


def norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def main():
    n1_dir = "experimental_results/N1v28a_dual_head/subj01/metrics"
    v55b_dir = "experimental_results/V55b_subj01_finetune/subj01/metrics"

    n1_preds = norm(np.load(f"{n1_dir}/shared1000_predictions_compact.npy"))
    n1_gt = norm(np.load(f"{n1_dir}/shared1000_ground_truth.npy"))
    n1_ids = np.load(f"{n1_dir}/shared1000_nsd_ids.npy")

    v55b_preds = norm(np.load(f"{v55b_dir}/shared1000_predictions.npy"))
    v55b_kappas = np.load(f"{v55b_dir}/shared1000_kappas.npy")
    v55b_ids = np.load(f"{v55b_dir}/shared1000_nsd_ids.npy")

    print(f"V55b: {v55b_preds.shape}, N1v28a: {n1_preds.shape}")
    print(f"NSD ID match: {np.array_equal(np.sort(v55b_ids), np.sort(n1_ids))}")

    if not np.array_equal(v55b_ids, n1_ids):
        common = np.intersect1d(v55b_ids, n1_ids)
        c_idx = np.array([np.where(v55b_ids == cid)[0][0] for cid in common])
        l_idx = np.array([np.where(n1_ids == cid)[0][0] for cid in common])
        v55b_preds = v55b_preds[c_idx]
        v55b_kappas = v55b_kappas[c_idx]
        n1_preds = n1_preds[l_idx]
        n1_gt = n1_gt[l_idx]
        print(f"Aligned: {len(common)} common images")

    print()
    m_v55b = compute_metrics(v55b_preds, n1_gt, "V55b_197k")
    print(
        f"V55b 197K-D shared1000:  raw_R@1={m_v55b['raw_r@1']:.3f}  "
        f"csls_R@1={m_v55b['csls_r@1']:.3f}  R@5={m_v55b['csls_r@5']:.3f}"
    )

    m_n1 = compute_metrics(n1_preds, n1_gt, "N1v28a_197k")
    print(
        f"N1v28a 197K-D shared1000: raw_R@1={m_n1['raw_r@1']:.3f}  "
        f"csls_R@1={m_n1['csls_r@1']:.3f}  R@5={m_n1['csls_r@5']:.3f}"
    )

    # --- Fusion sweep ---
    print("\n" + "=" * 60)
    print("FUSION SWEEP (197K-D, CSLS z-score)")
    print("=" * 60)

    v55b_sim = v55b_preds @ n1_gt.T
    n1_sim = n1_preds @ n1_gt.T
    v55b_csls = csls(v55b_sim)
    n1_csls = csls(n1_sim)
    v55b_z = zscore(v55b_csls)
    n1_z = zscore(n1_csls)

    best_r1, best_alpha = 0, 0
    fusion_results = []
    for a in np.arange(0.05, 0.96, 0.05):
        g = 1.0 - a
        fused = a * v55b_z + g * n1_z
        r = ranks_from(fused)
        r1 = float((r == 0).mean())
        r5 = float((r < 5).mean())
        r10 = float((r < 10).mean())
        fusion_results.append({"alpha": round(float(a), 2), "r@1": r1, "r@5": r5, "r@10": r10})
        print(f"  alpha={a:.2f}  R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}")
        if r1 > best_r1:
            best_r1, best_alpha = r1, a

    print(f"\n  BEST FUSION: alpha={best_alpha:.2f}  R@1={best_r1:.3f}")
    print(f"  Previous best (V35+N1v28a): 77.2%")
    print(f"  Delta: {(best_r1 - 0.772) * 100:+.1f}pp")

    # --- PPR fusion ---
    print("\n" + "=" * 60)
    print("PPR FUSION SWEEP (kappa-weighted V55b + N1v28a)")
    print("=" * 60)

    v55b_ppr = v55b_kappas[:, None] * v55b_sim
    v55b_ppr_csls = csls(v55b_ppr)
    v55b_ppr_z = zscore(v55b_ppr_csls)

    best_r1_ppr, best_alpha_ppr = 0, 0
    ppr_results = []
    for a in np.arange(0.05, 0.96, 0.05):
        g = 1.0 - a
        fused = a * v55b_ppr_z + g * n1_z
        r = ranks_from(fused)
        r1 = float((r == 0).mean())
        r5 = float((r < 5).mean())
        ppr_results.append({"alpha": round(float(a), 2), "r@1": r1, "r@5": r5})
        print(f"  alpha={a:.2f}  R@1={r1:.3f}  R@5={r5:.3f}")
        if r1 > best_r1_ppr:
            best_r1_ppr, best_alpha_ppr = r1, a

    print(f"\n  BEST PPR FUSION: alpha={best_alpha_ppr:.2f}  R@1={best_r1_ppr:.3f}")

    # --- Kappa analysis ---
    print("\n  Kappa stats: mean={:.1f} std={:.1f} min={:.1f} max={:.1f}".format(
        v55b_kappas.mean(), v55b_kappas.std(), v55b_kappas.min(), v55b_kappas.max(),
    ))

    results = {
        "V55b_standalone_197k": m_v55b,
        "N1v28a_standalone_197k": m_n1,
        "best_fusion": {"alpha": float(best_alpha), "r@1": float(best_r1)},
        "best_ppr_fusion": {"alpha": float(best_alpha_ppr), "r@1": float(best_r1_ppr)},
        "fusion_sweep": fusion_results,
        "ppr_sweep": ppr_results,
        "previous_best": 0.772,
    }
    with open("experimental_results/phase1_197k_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experimental_results/phase1_197k_results.json")


if __name__ == "__main__":
    main()
