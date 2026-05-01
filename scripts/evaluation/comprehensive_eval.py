#!/usr/bin/env python3
"""Comprehensive shared1000 evaluation across all experiments."""
import json
import numpy as np
import os
from pathlib import Path

base = Path("/home/jovyan/work/FMRI2images/experimental_results")


def cosine_sim(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return a @ b.T


def csls_scores(preds, gallery, k=10):
    sim = cosine_sim(preds, gallery)
    topk_q = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    topk_g = np.sort(sim, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return 2 * sim - topk_q - topk_g


def retrieval_metrics(scores):
    n = scores.shape[0]
    gt = np.arange(n)
    ranks = (scores >= scores[np.arange(n), gt][:, None]).sum(axis=1)
    r1 = float((ranks <= 1).mean())
    r5 = float((ranks <= 5).mean())
    r10 = float((ranks <= 10).mean())
    mrr = float((1.0 / ranks).mean())
    med = float(np.median(ranks))
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr, "MedR": med}


print("=" * 90)
print("SHARED1000 HEAD-TO-HEAD (compact 768-D space, computed fresh)")
print("=" * 90)
print()
print(f"{'Experiment':45s} {'Raw R@1':>8s} {'Raw R@5':>8s} {'CSLS R@1':>9s} {'CSLS R@5':>9s} {'MRR':>6s}")
print("-" * 90)

experiments = [
    "N1v28a_dual_head",
    "V55a_multi_subject_dual_head",
    "V55b_subj01_finetune",
    "V56a_fusion_distill_fixed",
    "V56c_projection_rdrop",
    "V57a_roi_transformer_dual_head",
]

results = {}

for exp in experiments:
    metrics_dir = base / exp / "subj01" / "metrics"

    for candidate in ["shared1000_predictions_compact.npy", "shared1000_predictions.npy"]:
        pred_path = metrics_dir / candidate
        if pred_path.exists():
            break
    else:
        print(f"  {exp}: NO shared1000 predictions")
        continue

    gt_name = candidate.replace("predictions", "ground_truth")
    gt_path = metrics_dir / gt_name

    if not gt_path.exists():
        gt_path = metrics_dir / "shared1000_ground_truth_compact.npy"
    if not gt_path.exists():
        gt_path = metrics_dir / "shared1000_ground_truth.npy"

    preds = np.load(pred_path)
    gts = np.load(gt_path)

    if preds.shape[1] > 768:
        preds = preds[:, :768].copy()
        preds /= np.maximum(np.linalg.norm(preds, axis=-1, keepdims=True), 1e-8)
        gts_use = gts[:, :768].copy() if gts.shape[1] > 768 else gts.copy()
        gts_use /= np.maximum(np.linalg.norm(gts_use, axis=-1, keepdims=True), 1e-8)
    else:
        preds = preds.copy()
        gts_use = gts.copy()

    raw = cosine_sim(preds, gts_use)
    m_raw = retrieval_metrics(raw)
    cs = csls_scores(preds, gts_use)
    m_csls = retrieval_metrics(cs)

    results[exp] = {"raw": m_raw, "csls": m_csls, "dim": preds.shape[1]}

    print(f"  {exp:43s} {m_raw['R@1']*100:7.1f}% {m_raw['R@5']*100:7.1f}% {m_csls['R@1']*100:8.1f}% {m_csls['R@5']*100:8.1f}% {m_csls['MRR']:5.3f}")


# N1v28a full 197K-D
print()
print("=" * 90)
print("N1v28a FULL 197K-D RETRIEVAL (shared1000)")
print("=" * 90)
n1_pred = base / "N1v28a_dual_head/subj01/metrics/shared1000_predictions.npy"
n1_gt = base / "N1v28a_dual_head/subj01/metrics/shared1000_ground_truth.npy"
if n1_pred.exists() and n1_gt.exists():
    p197 = np.load(n1_pred)
    g197 = np.load(n1_gt)
    raw197 = cosine_sim(p197, g197)
    m197_raw = retrieval_metrics(raw197)
    cs197 = csls_scores(p197, g197)
    m197_csls = retrieval_metrics(cs197)
    print(f"  197K-D raw:  R@1={m197_raw['R@1']*100:.1f}%  R@5={m197_raw['R@5']*100:.1f}%  MRR={m197_raw['MRR']:.3f}")
    print(f"  197K-D CSLS: R@1={m197_csls['R@1']*100:.1f}%  R@5={m197_csls['R@5']*100:.1f}%  MRR={m197_csls['MRR']:.3f}")


# V57a rich head
print()
print("=" * 90)
print("V57a RICH HEAD RETRIEVAL (shared1000)")
print("=" * 90)
v57_rich_pred = base / "V57a_roi_transformer_dual_head/subj01/metrics/shared1000_predictions_rich.npy"
v57_rich_gt = base / "V57a_roi_transformer_dual_head/subj01/metrics/shared1000_ground_truth_rich.npy"
if v57_rich_pred.exists() and v57_rich_gt.exists():
    pr = np.load(v57_rich_pred)
    gr = np.load(v57_rich_gt)
    print(f"  Rich shape: preds={pr.shape}, gts={gr.shape}")
    raw_r = cosine_sim(pr, gr)
    m_r_raw = retrieval_metrics(raw_r)
    cs_r = csls_scores(pr, gr)
    m_r_csls = retrieval_metrics(cs_r)
    print(f"  {pr.shape[1]}D raw:  R@1={m_r_raw['R@1']*100:.1f}%  R@5={m_r_raw['R@5']*100:.1f}%  MRR={m_r_raw['MRR']:.3f}")
    print(f"  {pr.shape[1]}D CSLS: R@1={m_r_csls['R@1']*100:.1f}%  R@5={m_r_csls['R@5']*100:.1f}%  MRR={m_r_csls['MRR']:.3f}")


# V57a diagnostics
print()
print("=" * 90)
print("V57a DIAGNOSTICS")
print("=" * 90)
kp = base / "V57a_roi_transformer_dual_head/subj01/metrics/val_kappas.npy"
if kp.exists():
    k = np.load(kp)
    print(f"  Val kappas: mean={k.mean():.1f} std={k.std():.1f} min={k.min():.1f} max={k.max():.1f}")

sp = base / "V57a_roi_transformer_dual_head/subj01/metrics/summary.json"
if sp.exists():
    s = json.load(open(sp))
    print(f"  best_epoch={s.get('best_epoch')} / total={s.get('total_epochs')}")
    print(f"  best_metric (full val CSLS)={s.get('best_checkpoint_metric_value')}")
    print(f"  final_compact_csls_r@1={s.get('final_compact_csls_r@1')}")
    print(f"  wall_time={s.get('wall_time_seconds', 0)/3600:.1f} hours")


# Fusion sweep: V57a compact + N1v28a 768D
print()
print("=" * 90)
print("FUSION: V57a (compact 768D) + N1v28a (768D) on shared1000")
print("=" * 90)
v57_compact = base / "V57a_roi_transformer_dual_head/subj01/metrics/shared1000_predictions_compact.npy"
v57_gt_compact = base / "V57a_roi_transformer_dual_head/subj01/metrics/shared1000_ground_truth_compact.npy"
n1_compact = base / "N1v28a_dual_head/subj01/metrics/shared1000_predictions_compact.npy"
n1_gt_compact = base / "N1v28a_dual_head/subj01/metrics/shared1000_ground_truth_compact.npy"

if all(p.exists() for p in [v57_compact, v57_gt_compact, n1_compact, n1_gt_compact]):
    v57c = np.load(v57_compact)
    v57g = np.load(v57_gt_compact)
    n1c = np.load(n1_compact)
    n1g = np.load(n1_gt_compact)

    # Align by nsd_ids
    v57_ids = np.load(base / "V57a_roi_transformer_dual_head/subj01/metrics/shared1000_nsd_ids.npy")
    n1_ids = np.load(base / "N1v28a_dual_head/subj01/metrics/shared1000_nsd_ids.npy")

    common = sorted(set(v57_ids.tolist()) & set(n1_ids.tolist()))
    print(f"  Common shared1000 nsd_ids: {len(common)}")

    v57_idx = {int(nid): i for i, nid in enumerate(v57_ids)}
    n1_idx = {int(nid): i for i, nid in enumerate(n1_ids)}

    v57_aligned = np.stack([v57c[v57_idx[nid]] for nid in common])
    n1_aligned = np.stack([n1c[n1_idx[nid]] for nid in common])
    gt_aligned = np.stack([v57g[v57_idx[nid]] for nid in common])

    # Normalize if needed
    if n1_aligned.shape[1] > 768:
        n1_aligned = n1_aligned[:, :768].copy()
    v57_aligned /= np.maximum(np.linalg.norm(v57_aligned, axis=-1, keepdims=True), 1e-8)
    n1_aligned /= np.maximum(np.linalg.norm(n1_aligned, axis=-1, keepdims=True), 1e-8)
    gt_aligned /= np.maximum(np.linalg.norm(gt_aligned, axis=-1, keepdims=True), 1e-8)

    # Error analysis: where do they agree/disagree?
    sim_v57 = cosine_sim(v57_aligned, gt_aligned)
    sim_n1 = cosine_sim(n1_aligned, gt_aligned)
    ranks_v57 = (sim_v57 >= sim_v57[np.arange(len(common)), np.arange(len(common))][:, None]).sum(axis=1)
    ranks_n1 = (sim_n1 >= sim_n1[np.arange(len(common)), np.arange(len(common))][:, None]).sum(axis=1)

    v57_correct = ranks_v57 <= 1
    n1_correct = ranks_n1 <= 1
    both_correct = v57_correct & n1_correct
    v57_only = v57_correct & ~n1_correct
    n1_only = ~v57_correct & n1_correct
    neither = ~v57_correct & ~n1_correct

    print(f"  Error analysis (raw cosine, before CSLS):")
    print(f"    Both correct:   {both_correct.sum():4d} ({both_correct.mean()*100:.1f}%)")
    print(f"    V57a only:      {v57_only.sum():4d} ({v57_only.mean()*100:.1f}%)")
    print(f"    N1v28a only:    {n1_only.sum():4d} ({n1_only.mean()*100:.1f}%)")
    print(f"    Neither:        {neither.sum():4d} ({neither.mean()*100:.1f}%)")
    print(f"    Oracle (either): {(v57_correct | n1_correct).sum():4d} ({(v57_correct | n1_correct).mean()*100:.1f}%)")

    # CSLS fusion sweep
    print(f"\n  Fusion sweep (CSLS, alpha = V57a weight):")
    csls_v57 = csls_scores(v57_aligned, gt_aligned)
    csls_n1 = csls_scores(n1_aligned, gt_aligned)

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fused = alpha * csls_v57 + (1 - alpha) * csls_n1
        mf = retrieval_metrics(fused)
        marker = " <-- best" if alpha == 0.0 else ""
        print(f"    alpha={alpha:.1f}: R@1={mf['R@1']*100:.1f}%  R@5={mf['R@5']*100:.1f}%  MRR={mf['MRR']:.3f}{marker}")

print()
print("=" * 90)
print("DONE")
print("=" * 90)
