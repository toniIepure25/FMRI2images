#!/usr/bin/env python3
"""Debug reranking pipeline for V30 triple-head architecture.

Loads saved compact + stage-2 predictions/GTs and runs a comprehensive set of
diagnostics to determine whether the reranking failure is a bug or a
modeling issue.

Usage:
    python scripts/evaluation/debug_reranking.py <results_dir>
    python scripts/evaluation/debug_reranking.py <results_dir> --split shared1000

Checks performed:
    1. Shape & alignment sanity
    2. Norm distributions (stage-2 preds vs stage-2 GTs)
    3. Stage-2-only full-gallery retrieval (R@1, R@5, R@10)
    4. Cosine similarity distributions (positive vs negative pairs)
    5. Two-stage retrieval with standard shortlist
    6. Oracle shortlist reranking (GT always included)
    7. Per-query diagnostics (rank of GT in stage-2 space)
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
_TWO_STAGE_MODULE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D) x (M, D) -> (N, M) cosine similarity."""
    return _l2_normalize(a) @ _l2_normalize(b).T


def _retrieval_at_k(sim: np.ndarray, ks=(1, 5, 10)):
    """Given (N, N) similarity where GT for query i is gallery i, compute R@K."""
    N = sim.shape[0]
    ranks = np.argsort(-sim, axis=1)
    results = {}
    for k in ks:
        hits = sum(1 for i in range(N) if i in ranks[i, :k])
        results[f"R@{k}"] = hits / N
    return results


def _gt_ranks(sim: np.ndarray) -> np.ndarray:
    """For each query i, find the rank of gallery item i (0-indexed)."""
    N = sim.shape[0]
    ranks = np.argsort(-sim, axis=1)
    gt_rank = np.zeros(N, dtype=int)
    for i in range(N):
        gt_rank[i] = int(np.where(ranks[i] == i)[0][0])
    return gt_rank


def _load_two_stage_module():
    """Load the retrieval utility module without importing heavy eval extras."""
    global _TWO_STAGE_MODULE
    if _TWO_STAGE_MODULE is not None:
        return _TWO_STAGE_MODULE

    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fmri2img"
        / "eval"
        / "two_stage_retrieval.py"
    )
    spec = importlib.util.spec_from_file_location("two_stage_retrieval_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load retrieval helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TWO_STAGE_MODULE = module
    return module


def _load_fusion_config(results_dir: Path, metrics_dir: Path) -> dict | None:
    """Load fixed fusion config from the experiment config referenced in summary."""
    summary_path = metrics_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        config_path = summary.get("manifest", {}).get("config_path")
        if not config_path:
            return None
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            cfg_path = results_dir.parent.parent / config_path
        if not cfg_path.exists():
            return None
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        fusion_cfg = config.get("evaluation", {}).get("fusion", {})
        if not fusion_cfg.get("enabled", False):
            return None
        resolved = dict(_load_two_stage_module().DEFAULT_FUSION_CONFIG)
        resolved.update(fusion_cfg)
        return resolved
    except Exception as exc:
        logger.warning("Could not load fusion config from summary: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(results_dir: Path, metrics_dir: Path, split: str = "val"):
    prefix = "shared1000" if split == "shared1000" else "val"

    compact_p_path = metrics_dir / f"{prefix}_predictions_compact.npy"
    compact_g_path = metrics_dir / f"{prefix}_ground_truth_compact.npy"
    rerank_p_path = metrics_dir / f"{prefix}_predictions_rerank.npy"
    rerank_g_path = metrics_dir / f"{prefix}_ground_truth_rerank.npy"
    rich_p_path = metrics_dir / f"{prefix}_predictions_rich.npy"
    rich_g_path = metrics_dir / f"{prefix}_ground_truth_rich.npy"

    if rerank_p_path.exists() and rerank_g_path.exists():
        stage2_p_path = rerank_p_path
        stage2_g_path = rerank_g_path
        stage2_name = "rerank"
    else:
        stage2_p_path = rich_p_path
        stage2_g_path = rich_g_path
        stage2_name = "rich"

    # --- Check file existence ---
    for p in [compact_p_path, compact_g_path, stage2_p_path, stage2_g_path]:
        if not p.exists():
            logger.error("Missing: %s", p)
            return None
        logger.info("Found: %s", p.name)

    cp = np.load(compact_p_path)
    cg = np.load(compact_g_path)
    rp = np.load(stage2_p_path)
    rg = np.load(stage2_g_path)

    report = {}

    # -----------------------------------------------------------------------
    # 1. Shape & alignment sanity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. SHAPE & ALIGNMENT")
    print("=" * 70)
    print(f"  compact_preds:  {cp.shape}")
    print(f"  compact_gts:    {cg.shape}")
    print(f"  {stage2_name}_preds:     {rp.shape}")
    print(f"  {stage2_name}_gts:       {rg.shape}")

    N = cp.shape[0]
    assert cp.shape[0] == cg.shape[0] == rp.shape[0] == rg.shape[0], \
        f"Row count mismatch: {cp.shape[0]} vs {cg.shape[0]} vs {rp.shape[0]} vs {rg.shape[0]}"
    assert cp.shape[1] == cg.shape[1], \
        f"Compact dim mismatch: preds {cp.shape[1]} vs gts {cg.shape[1]}"
    assert rp.shape[1] == rg.shape[1], \
        f"Stage-2 dim mismatch: preds {rp.shape[1]} vs gts {rg.shape[1]}"
    print(f"  N = {N} images, compact_dim = {cp.shape[1]}, {stage2_name}_dim = {rp.shape[1]}")
    print("  [OK] Shapes consistent")

    report["n_images"] = N
    report["compact_dim"] = int(cp.shape[1])
    report[f"{stage2_name}_dim"] = int(rp.shape[1])

    # -----------------------------------------------------------------------
    # 2. Norm distributions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. NORM DISTRIBUTIONS")
    print("=" * 70)

    for name, arr in [("compact_preds", cp), ("compact_gts", cg),
                      (f"{stage2_name}_preds", rp), (f"{stage2_name}_gts", rg)]:
        norms = np.linalg.norm(arr, axis=-1)
        print(f"  {name:20s}  mean={norms.mean():.4f}  std={norms.std():.4f}  "
              f"min={norms.min():.4f}  max={norms.max():.4f}")
        report[f"norm_{name}"] = {
            "mean": float(norms.mean()), "std": float(norms.std()),
            "min": float(norms.min()), "max": float(norms.max()),
        }

    # Check for zero/near-zero norms
    rp_norms = np.linalg.norm(rp, axis=-1)
    rg_norms = np.linalg.norm(rg, axis=-1)
    n_zero_rp = int((rp_norms < 1e-6).sum())
    n_zero_rg = int((rg_norms < 1e-6).sum())
    if n_zero_rp > 0 or n_zero_rg > 0:
        print(f"  [WARN] Near-zero norms: {stage2_name}_preds={n_zero_rp}, {stage2_name}_gts={n_zero_rg}")

    # Norm ratio (are predictions much smaller/larger than GTs?)
    norm_ratio = rp_norms.mean() / max(rg_norms.mean(), 1e-8)
    print(f"  Norm ratio ({stage2_name}_preds / {stage2_name}_gts): {norm_ratio:.4f}")
    report[f"{stage2_name}_norm_ratio"] = float(norm_ratio)

    # -----------------------------------------------------------------------
    # 3. Rich-only full-gallery retrieval
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"3. {stage2_name.upper()}-ONLY FULL-GALLERY RETRIEVAL")
    print("=" * 70)

    rich_sim = _cosine_sim(rp, rg)  # (N, N)
    rich_ret = _retrieval_at_k(rich_sim, ks=(1, 5, 10))
    for k, v in rich_ret.items():
        print(f"  {stage2_name.capitalize()} {k}: {v:.1%}")
    report[f"{stage2_name}_only_retrieval"] = rich_ret

    # Also compute with raw dot product (no normalization)
    rich_dot_sim = rp @ rg.T
    rich_dot_ret = _retrieval_at_k(rich_dot_sim, ks=(1, 5, 10))
    print("  (dot product, no norm):")
    for k, v in rich_dot_ret.items():
        print(f"    {stage2_name.capitalize()}-dot {k}: {v:.1%}")
    report[f"{stage2_name}_only_dot_retrieval"] = rich_dot_ret

    # -----------------------------------------------------------------------
    # 4. Cosine similarity distributions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"4. COSINE SIMILARITY DISTRIBUTIONS ({stage2_name.upper()} SPACE)")
    print("=" * 70)

    diag_sims = np.diag(rich_sim)  # positive pairs
    # Sample off-diagonal for efficiency
    rng = np.random.RandomState(42)
    n_neg = min(50000, N * (N - 1))
    neg_i = rng.randint(0, N, n_neg)
    neg_j = rng.randint(0, N - 1, n_neg)
    neg_j[neg_j >= neg_i] += 1  # avoid diagonal
    neg_sims = rich_sim[neg_i, neg_j]

    print(f"  Positive (diagonal) cosine:  mean={diag_sims.mean():.4f}  "
          f"std={diag_sims.std():.4f}  min={diag_sims.min():.4f}  max={diag_sims.max():.4f}")
    print(f"  Negative (off-diag) cosine:  mean={neg_sims.mean():.4f}  "
          f"std={neg_sims.std():.4f}  min={neg_sims.min():.4f}  max={neg_sims.max():.4f}")

    separability = (diag_sims.mean() - neg_sims.mean()) / max(neg_sims.std(), 1e-8)
    print(f"  Separability (d'):           {separability:.4f}")

    overlap_threshold = neg_sims.mean() + neg_sims.std()
    pos_below_threshold = (diag_sims < overlap_threshold).mean()
    print(f"  Positive pairs below neg_mean+1std: {pos_below_threshold:.1%}")

    report[f"{stage2_name}_sim_positive"] = {
        "mean": float(diag_sims.mean()), "std": float(diag_sims.std()),
    }
    report[f"{stage2_name}_sim_negative"] = {
        "mean": float(neg_sims.mean()), "std": float(neg_sims.std()),
    }
    report[f"{stage2_name}_separability"] = float(separability)

    # -----------------------------------------------------------------------
    # 5. Compact retrieval baseline
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. COMPACT RETRIEVAL BASELINE")
    print("=" * 70)

    compact_sim = _cosine_sim(cp, cg)
    compact_ret = _retrieval_at_k(compact_sim, ks=(1, 5, 10))
    for k, v in compact_ret.items():
        print(f"  Compact {k}: {v:.1%}")
    report["compact_retrieval"] = compact_ret

    # -----------------------------------------------------------------------
    # 6. Two-stage: standard shortlist + rerank
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. TWO-STAGE: SHORTLIST(100) + RERANK")
    print("=" * 70)

    _ts_module = _load_two_stage_module()
    shortlist_retrieval = _ts_module.shortlist_retrieval
    rerank_shortlist = _ts_module.rerank_shortlist
    two_stage_metrics = _ts_module.two_stage_metrics
    ts = two_stage_metrics(cp, cg, rp, rg, shortlist_k=100, ks=(1, 5, 10))
    sl_recall = ts["shortlist_recall"]
    print(f"  Shortlist recall@100:  {sl_recall.get('r@100', 0):.1%}")
    for k, v in ts["compact_raw"].items():
        print(f"  {k}: {v:.1%}")
    for k, v in ts["reranked"].items():
        print(f"  {k}: {v:.1%}")
    print(f"  Rerank gain (pp):     {ts['rerank_gain_over_compact_raw'] * 100:.1f}")
    report["two_stage"] = ts

    # -----------------------------------------------------------------------
    # 6b. Fixed score fusion
    # -----------------------------------------------------------------------
    fusion_cfg = _load_fusion_config(results_dir, metrics_dir)
    if fusion_cfg is not None and stage2_name == "rerank":
        print("\n" + "=" * 70)
        print("6B. FIXED SCORE FUSION")
        print("=" * 70)
        fusion_report = _ts_module.fusion_metrics(
            cp,
            cg,
            rp,
            rg,
            shortlist_k=int(fusion_cfg.get("shortlist_k", 50)),
            ks=(1, 5, 10),
            compact_score=str(fusion_cfg.get("compact_score", "csls")),
            family=str(fusion_cfg.get("family", "normalized_weighted")),
            normalization=str(fusion_cfg.get("normalization", "zscore")),
            alpha=float(fusion_cfg.get("alpha", 0.8)),
            csls_k=int(fusion_cfg.get("csls_k", 10)),
            rerank_mode="cosine",
        )
        fused = fusion_report["fused"]
        print(f"  Compact score:          {fusion_cfg['compact_score']}")
        print(f"  Fusion family:          {fusion_cfg['family']}")
        print(f"  Normalization:          {fusion_cfg['normalization']}")
        print(f"  Shortlist k:            {fusion_cfg['shortlist_k']}")
        print(f"  Alpha:                  {fusion_cfg['alpha']}")
        print(f"  Fused R@1:              {fused.get('fused_r@1', 0):.1%}")
        print(f"  Fused R@5:              {fused.get('fused_r@5', 0):.1%}")
        print(f"  Fused R@10:             {fused.get('fused_r@10', 0):.1%}")
        print(f"  Fused MedR / MRR:       {fused.get('fused_median_rank', 0):.1f} / {fused.get('fused_mrr', 0):.4f}")
        print(f"  Gain over compact CSLS: {fusion_report.get('fused_gain_over_compact_csls', 0) * 100:.1f} pp")
        print(f"  Gain over rerank repl.: {fusion_report.get('fused_gain_over_rerank_replacement', 0) * 100:.1f} pp")
        report["fused_retrieval"] = fusion_report

    tri_fused_metrics_path = metrics_dir / f"{split}_tri_fused_metrics.json"
    if tri_fused_metrics_path.exists():
        print("\n" + "=" * 70)
        print("6C. TRI-EXPERT FUSION")
        print("=" * 70)
        with open(tri_fused_metrics_path, "r") as f:
            tri_report = json.load(f)
        tri_payload = tri_report.get("tri_fused_best", tri_report.get("tri_fused_frozen", tri_report))
        print(f"  Tri-fused R@1:           {tri_payload.get('R@1', 0.0):.1%}")
        print(f"  Tri-fused R@5:           {tri_payload.get('R@5', 0.0):.1%}")
        print(f"  Tri-fused R@10:          {tri_payload.get('R@10', 0.0):.1%}")
        print(f"  Tri-fused MedR / MRR:    {tri_payload.get('median_rank', 0.0):.1f} / {tri_payload.get('MRR', 0.0):.4f}")
        print(f"  Gain over compact CSLS:  {tri_report.get('gain_over_compact_csls', 0.0) * 100:.1f} pp")
        print(f"  Gain over 2-expert:      {tri_report.get('gain_over_two_expert_fusion', 0.0) * 100:.1f} pp")
        report["tri_fused_retrieval"] = tri_report

    tri_gated_metrics_path = metrics_dir / f"{split}_tri_gated_metrics.json"
    if tri_gated_metrics_path.exists():
        print("\n" + "=" * 70)
        print("6D. LEARNED TRI-FUSION GATE")
        print("=" * 70)
        with open(tri_gated_metrics_path, "r") as f:
            tri_gate_report = json.load(f)
        tri_gate_payload = tri_gate_report.get(
            "tri_gated_best",
            tri_gate_report.get("tri_gated_frozen", tri_gate_report),
        )
        print(f"  Gate model:              {tri_gate_payload.get('model_family', 'unknown')}")
        print(f"  Shortlist k:             {tri_gate_payload.get('shortlist_k', 'n/a')}")
        print(f"  Compact shortlist score: {tri_gate_payload.get('compact_shortlist_variant', 'n/a')}")
        print(f"  Tri-gated R@1:           {tri_gate_payload.get('R@1', 0.0):.1%}")
        print(f"  Tri-gated R@5:           {tri_gate_payload.get('R@5', 0.0):.1%}")
        print(f"  Tri-gated R@10:          {tri_gate_payload.get('R@10', 0.0):.1%}")
        print(f"  Tri-gated MedR / MRR:    {tri_gate_payload.get('median_rank', 0.0):.1f} / {tri_gate_payload.get('MRR', 0.0):.4f}")
        print(f"  Gain over compact CSLS:  {tri_gate_report.get('gain_over_compact_csls', 0.0) * 100:.1f} pp")
        print(f"  Gain over fixed tri:     {tri_gate_report.get('gain_over_fixed_tri_fusion', 0.0) * 100:.1f} pp")
        print(f"  Gain over 2-expert:      {tri_gate_report.get('gain_over_two_expert_fusion', 0.0) * 100:.1f} pp")
        report["tri_gated_retrieval"] = tri_gate_report

    # -----------------------------------------------------------------------
    # 7. Oracle shortlist reranking
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. ORACLE SHORTLIST RERANKING")
    print("=" * 70)
    print("  (GT is always in the shortlist — tests reranker in isolation)")

    # Build oracle shortlist: for each query, include GT (=i) plus random others
    oracle_k = 100
    oracle_sl = np.zeros((N, oracle_k), dtype=int)
    rng2 = np.random.RandomState(123)
    for i in range(N):
        candidates = list(range(N))
        candidates.remove(i)
        chosen = rng2.choice(candidates, oracle_k - 1, replace=False)
        # Put GT at a random position in the shortlist
        gt_pos = rng2.randint(0, oracle_k)
        oracle_sl[i] = np.insert(chosen, gt_pos, i)[:oracle_k]

    reranked_oracle = rerank_shortlist(rp, rg, oracle_sl, mode="cosine")

    # Check R@1 on oracle shortlist
    oracle_hits = sum(1 for i in range(N) if reranked_oracle[i, 0] == i)
    oracle_r1 = oracle_hits / N
    # R@5
    oracle_hits5 = sum(1 for i in range(N) if i in reranked_oracle[i, :5])
    oracle_r5 = oracle_hits5 / N
    # R@10
    oracle_hits10 = sum(1 for i in range(N) if i in reranked_oracle[i, :10])
    oracle_r10 = oracle_hits10 / N

    print(f"  Oracle reranked R@1:   {oracle_r1:.1%}")
    print(f"  Oracle reranked R@5:   {oracle_r5:.1%}")
    print(f"  Oracle reranked R@10:  {oracle_r10:.1%}")
    print(f"  (Random baseline R@1 = {1/oracle_k:.1%})")

    report["oracle_rerank"] = {
        "R@1": float(oracle_r1),
        "R@5": float(oracle_r5),
        "R@10": float(oracle_r10),
        "shortlist_size": oracle_k,
        "random_baseline_r1": 1 / oracle_k,
    }

    # -----------------------------------------------------------------------
    # 8. GT rank distribution in rich space
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"8. GT RANK DISTRIBUTION IN {stage2_name.upper()} SPACE")
    print("=" * 70)

    gt_ranks_rich = _gt_ranks(rich_sim)
    print(f"  Mean GT rank:    {gt_ranks_rich.mean():.1f}")
    print(f"  Median GT rank:  {np.median(gt_ranks_rich):.1f}")
    print(f"  Min GT rank:     {gt_ranks_rich.min()}")
    print(f"  Max GT rank:     {gt_ranks_rich.max()}")
    for pct in [25, 50, 75, 90, 95]:
        print(f"  {pct}th percentile: {np.percentile(gt_ranks_rich, pct):.0f}")

    report[f"{stage2_name}_gt_rank"] = {
        "mean": float(gt_ranks_rich.mean()),
        "median": float(np.median(gt_ranks_rich)),
        "p25": float(np.percentile(gt_ranks_rich, 25)),
        "p75": float(np.percentile(gt_ranks_rich, 75)),
        "p95": float(np.percentile(gt_ranks_rich, 95)),
    }

    # Also check compact space GT ranks for comparison
    gt_ranks_compact = _gt_ranks(compact_sim)
    print(f"\n  (Compact space comparison)")
    print(f"  Mean GT rank:    {gt_ranks_compact.mean():.1f}")
    print(f"  Median GT rank:  {np.median(gt_ranks_compact):.1f}")

    report["compact_gt_rank"] = {
        "mean": float(gt_ranks_compact.mean()),
        "median": float(np.median(gt_ranks_compact)),
    }

    # -----------------------------------------------------------------------
    # 9. Embedding collapse check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. EMBEDDING COLLAPSE CHECK")
    print("=" * 70)

    rp_n = _l2_normalize(rp)
    rg_n = _l2_normalize(rg)

    # Inter-prediction similarity (are all predictions the same?)
    if N <= 2000:
        pred_inter = rp_n @ rp_n.T
        np.fill_diagonal(pred_inter, 0)
        pred_inter_mean = pred_inter.sum() / (N * (N - 1))
    else:
        # Sample for large N
        idx = np.random.RandomState(99).choice(N, 2000, replace=False)
        sub = rp_n[idx]
        pred_inter = sub @ sub.T
        np.fill_diagonal(pred_inter, 0)
        pred_inter_mean = pred_inter.sum() / (2000 * 1999)

    print(f"  Mean inter-prediction cosine sim: {pred_inter_mean:.4f}")
    print(f"  (1.0 = total collapse, 0.0 = orthogonal, expect < 0.5 for good discrimination)")

    # Same for GTs
    if N <= 2000:
        gt_inter = rg_n @ rg_n.T
        np.fill_diagonal(gt_inter, 0)
        gt_inter_mean = gt_inter.sum() / (N * (N - 1))
    else:
        idx = np.random.RandomState(99).choice(N, 2000, replace=False)
        sub = rg_n[idx]
        gt_inter = sub @ sub.T
        np.fill_diagonal(gt_inter, 0)
        gt_inter_mean = gt_inter.sum() / (2000 * 1999)

    print(f"  Mean inter-GT cosine sim:         {gt_inter_mean:.4f}")

    report["collapse_check"] = {
        "inter_pred_cosine": float(pred_inter_mean),
        "inter_gt_cosine": float(gt_inter_mean),
    }

    # -----------------------------------------------------------------------
    # 10. Summary & diagnosis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    is_bug = False
    issues = []

    if rich_ret["R@1"] < 0.05:
        issues.append(f"{stage2_name.capitalize()}-only R@1 < 5% → stage-2 head not discriminative in cosine space")
    if separability < 1.0:
        issues.append(f"{stage2_name.capitalize()} separability = {separability:.2f} (< 1.0) → pos/neg overlap heavily")
    if pred_inter_mean > 0.8:
        issues.append(f"Inter-prediction cosine = {pred_inter_mean:.3f} → predictions are COLLAPSING")
    if norm_ratio < 0.01 or norm_ratio > 100:
        issues.append(f"Norm ratio = {norm_ratio:.4f} → extreme norm mismatch (preds vs GTs)")
    if oracle_r1 < 0.05:
        issues.append(f"Oracle rerank R@1 = {oracle_r1:.1%} → reranker fundamentally broken")

    if not issues:
        print(f"  No obvious issues detected. {stage2_name.capitalize()} space seems discriminative.")
        print("  If two-stage still underperforms, check shortlist quality.")
    else:
        for iss in issues:
            print(f"  [!] {iss}")

        if rich_ret["R@1"] < 0.05 and pred_inter_mean > 0.7:
            print(f"\n  CONCLUSION: {stage2_name.capitalize()} outputs collapse in cosine space.")
            print("  Stage-2 predictions are not separating identities after L2-normalization.")
            print("  → Check rerank-target quality, loss routing, and saved prediction alignment.")
        elif rich_ret["R@1"] < 0.05:
            print(f"\n  CONCLUSION: {stage2_name.capitalize()} space is not discriminative for retrieval.")
            print("  Stage-2 predictions carry insufficient identity signal for cosine matching.")
            print("  → Check rerank loss behavior and cache/prediction alignment.")

    report["issues"] = issues
    print("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description="Debug V30 reranking pipeline")
    parser.add_argument("results_dir", type=str,
                        help="Path to experiment results directory")
    parser.add_argument("--split", choices=["val", "shared1000"], default="val",
                        help="Which split to diagnose (default: val)")
    parser.add_argument("--save", action="store_true",
                        help="Save JSON report to diagnostics dir")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    if not metrics_dir.exists():
        # Maybe the user passed the metrics dir directly
        if (results_dir / "val_predictions_compact.npy").exists():
            metrics_dir = results_dir
        else:
            logger.error("Cannot find metrics directory in %s", results_dir)
            return
    if results_dir == metrics_dir and metrics_dir.name == "metrics":
        results_dir = metrics_dir.parent

    report = run_diagnostics(results_dir, metrics_dir, split=args.split)

    if report is not None and args.save:
        diag_dir = results_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_path = diag_dir / f"rerank_debug_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        fused_report = report.get("fused_retrieval")
        if fused_report is not None:
            fused_metrics_path = metrics_dir / f"{args.split}_fused_metrics.json"
            with open(fused_metrics_path, "w") as f:
                json.dump(fused_report, f, indent=2, default=str)
            print(f"Fused metrics saved to {fused_metrics_path}")
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
