"""Geodesic / spherical interpolation in decoded CLIP space.

This module walks between decoded embeddings on the CLIP unit sphere and
measures whether text-probe distributions change smoothly. It is latent-space
trajectory analysis, not a real neural trajectory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr, softmax

from analysis.manifold.figures import (
    PALETTE,
    create_bar_chart,
    create_line_plot,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import cosine_similarity_matrix, safe_normalize, slerp

logger = logging.getLogger(__name__)

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": (
        "Latent-space interpolation requires at least two decoded embeddings "
        "and real CLIP text probe embeddings."
    ),
}


def slerp_path(
    z_a: np.ndarray,
    z_b: np.ndarray,
    n_steps: int = 21,
) -> np.ndarray:
    """Spherical linear interpolation path between two unit vectors."""
    ts = np.linspace(0.0, 1.0, int(n_steps))
    return safe_normalize(np.stack([slerp(z_a, z_b, float(t)) for t in ts]))


def text_probe_distribution(
    embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    temperature: float = 0.07,
) -> np.ndarray:
    """Softmax distribution over text probes."""
    scores = cosine_similarity_matrix(safe_normalize(embeddings), safe_normalize(text_embeddings))
    return softmax(scores / temperature, axis=1)


def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Jensen-Shannon divergence between adjacent distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m), axis=1) + 0.5 * np.sum(rel_entr(q, m), axis=1)


def compute_semantic_velocity(probs: np.ndarray) -> np.ndarray:
    """Semantic velocity as JS divergence between consecutive probe distributions."""
    if probs.shape[0] < 2:
        return np.array([], dtype=float)
    return js_distance(probs[:-1], probs[1:])


def compute_path_smoothness(velocity: np.ndarray) -> float:
    """Smoothness score in (0, 1], high when semantic velocity is small/stable."""
    if velocity.size == 0:
        return float("nan")
    return float(1.0 / (1.0 + float(np.mean(velocity)) + float(np.std(velocity))))


def select_interpolation_pairs(
    z_pred: np.ndarray,
    correct: Optional[np.ndarray] = None,
    top_concepts: Optional[np.ndarray] = None,
    max_pairs: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select mixed interpolation pairs from correctness/concept/distance groups."""
    rng = np.random.RandomState(seed)
    z = safe_normalize(z_pred)
    n = z.shape[0]
    max_pairs = min(int(max_pairs), max(n * (n - 1) // 2, 1))
    pairs: Dict[Tuple[int, int], str] = {}

    def add_pair(i: int, j: int, label: str) -> None:
        if i == j or len(pairs) >= max_pairs:
            return
        key = tuple(sorted((int(i), int(j))))
        pairs.setdefault(key, label)

    if top_concepts is not None and len(top_concepts) == n:
        concepts = np.asarray(top_concepts)
        for concept in np.unique(concepts):
            idx = np.where(concepts == concept)[0]
            if len(idx) >= 2:
                i, j = rng.choice(idx, 2, replace=False)
                add_pair(i, j, "same_concept")
                if len(pairs) >= max_pairs // 4:
                    break
        unique = np.unique(concepts)
        if len(unique) >= 2:
            for _ in range(max_pairs // 4 + 1):
                c1, c2 = rng.choice(unique, 2, replace=False)
                i = rng.choice(np.where(concepts == c1)[0])
                j = rng.choice(np.where(concepts == c2)[0])
                add_pair(i, j, "different_concept")

    if correct is not None and len(correct) == n:
        corr = np.asarray(correct, dtype=bool)
        if corr.sum() >= 2:
            i, j = rng.choice(np.where(corr)[0], 2, replace=False)
            add_pair(i, j, "both_correct")
        if corr.any() and (~corr).any():
            i = rng.choice(np.where(corr)[0])
            j = rng.choice(np.where(~corr)[0])
            add_pair(i, j, "one_correct_one_incorrect")

    sims = z @ z.T
    dists = 1.0 - sims
    tri = np.triu_indices(n, k=1)
    far_order = np.argsort(-dists[tri])
    for ord_idx in far_order[: max_pairs]:
        add_pair(tri[0][ord_idx], tri[1][ord_idx], "large_distance")
        if len(pairs) >= max_pairs:
            break

    while len(pairs) < max_pairs:
        i, j = rng.choice(n, 2, replace=False)
        add_pair(i, j, "random")

    return [{"i": i, "j": j, "pair_type": label} for (i, j), label in pairs.items()]


def run_interpolation(
    z_pred: np.ndarray,
    gallery: np.ndarray,
    correct: Optional[np.ndarray] = None,
    text_embeddings: Optional[np.ndarray] = None,
    text_concepts: Optional[List[str]] = None,
    n_steps: int = 21,
    n_pairs: int = 20,
    seed: int = 42,
    temperature: float = 0.07,
    pair_strategy: str = "mixed",
) -> Dict[str, Any]:
    """Run latent-space interpolation analysis."""
    if z_pred.shape[0] < 2:
        return {**_UNAVAILABLE, "reason": "At least two decoded embeddings are required."}
    if text_embeddings is None:
        return {**_UNAVAILABLE, "reason": "CLIP text probe embeddings are unavailable."}

    z_pred = safe_normalize(z_pred)
    gallery = safe_normalize(gallery)
    text_embeddings = safe_normalize(text_embeddings)
    endpoint_probs = text_probe_distribution(z_pred, text_embeddings, temperature)
    endpoint_top = np.argmax(endpoint_probs, axis=1)
    pairs = select_interpolation_pairs(
        z_pred,
        correct=correct,
        top_concepts=endpoint_top,
        max_pairs=n_pairs,
        seed=seed,
    )

    cases: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    smoothness: List[float] = []
    velocity_means: List[float] = []
    abrupt_rates: List[float] = []
    transition_counts: List[int] = []
    same_smooth: List[float] = []
    diff_smooth: List[float] = []

    for pi, pair in enumerate(pairs):
        i, j = pair["i"], pair["j"]
        path = slerp_path(z_pred[i], z_pred[j], n_steps=n_steps)
        probs = text_probe_distribution(path, text_embeddings, temperature)
        velocity = compute_semantic_velocity(probs)
        sm = compute_path_smoothness(velocity)
        top_idx = np.argmax(probs, axis=1)
        top_names = [text_concepts[int(k)] if text_concepts else str(int(k)) for k in top_idx]
        concept_transitions = int(np.sum(top_idx[1:] != top_idx[:-1]))
        threshold = float(np.mean(velocity) + 2.0 * np.std(velocity)) if velocity.size else float("inf")
        abrupt = int(np.sum(velocity > threshold)) if velocity.size else 0
        abrupt_rate = float(abrupt / max(len(velocity), 1))
        scores = cosine_similarity_matrix(path, gallery)
        top_gallery = np.argsort(-scores, axis=1)[:, :5]

        smoothness.append(sm)
        velocity_means.append(float(np.mean(velocity)) if velocity.size else float("nan"))
        abrupt_rates.append(abrupt_rate)
        transition_counts.append(concept_transitions)
        if endpoint_top[i] == endpoint_top[j]:
            same_smooth.append(sm)
        else:
            diff_smooth.append(sm)

        case = {
            "pair_index": pi,
            "pair": [int(i), int(j)],
            "pair_type": pair["pair_type"],
            "endpoint_concepts": [top_names[0], top_names[-1]],
            "smoothness": sm,
            "semantic_velocity_mean": velocity_means[-1],
            "semantic_velocity_max": float(np.max(velocity)) if velocity.size else 0.0,
            "abrupt_transition_count": abrupt,
            "top_concept_transition_count": concept_transitions,
            "endpoint_consistency": bool(top_idx[0] == endpoint_top[i] and top_idx[-1] == endpoint_top[j]),
            "top_concepts": top_names,
            "velocity": velocity.tolist(),
        }
        cases.append(case)

        ts = np.linspace(0.0, 1.0, n_steps)
        for step, t in enumerate(ts):
            path_rows.append(
                {
                    "pair_index": pi,
                    "i": int(i),
                    "j": int(j),
                    "pair_type": pair["pair_type"],
                    "t": float(t),
                    "top_concept": top_names[step],
                    "top_concept_idx": int(top_idx[step]),
                    "top_concept_probability": float(probs[step, int(top_idx[step])]),
                    "semantic_velocity_from_previous": float(velocity[step - 1]) if step > 0 and velocity.size else None,
                    "top5_gallery_indices": top_gallery[step].tolist(),
                }
            )

    smooth_arr = np.asarray(smoothness, dtype=float)
    vel_arr = np.asarray(velocity_means, dtype=float)
    abrupt_arr = np.asarray(abrupt_rates, dtype=float)

    return {
        "status": "computed",
        "n_pairs": int(len(pairs)),
        "n_steps": int(n_steps),
        "pair_strategy": pair_strategy,
        "mean_smoothness": float(np.nanmean(smooth_arr)),
        "median_smoothness": float(np.nanmedian(smooth_arr)),
        "mean_semantic_velocity": float(np.nanmean(vel_arr)),
        "abrupt_transition_rate": float(np.nanmean(abrupt_arr)),
        "top_concept_transition_count": int(np.sum(transition_counts)),
        "path_monotonicity": None,
        "same_concept_smoothness_mean": float(np.mean(same_smooth)) if same_smooth else None,
        "different_concept_smoothness_mean": float(np.mean(diff_smooth)) if diff_smooth else None,
        "cases": cases,
        "_path_rows": path_rows,
    }


def _plot_concept_probabilities(rows: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    groups = list(df.groupby("pair_index"))[:4]
    series = {}
    for pair_idx, g in groups:
        series[f"pair {pair_idx}"] = (g["t"].to_numpy(), g["top_concept_probability"].to_numpy())
    fig = create_line_plot(
        series,
        title="Interpolation Top-Concept Probability",
        xlabel="t",
        ylabel="top concept probability",
    )
    return save_figure(fig, output_path, **save_kw)


def _plot_semantic_velocity(cases: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    cases_with_vel = [c for c in cases if c.get("velocity")][:5]
    if not cases_with_vel:
        return []
    series = {}
    for c in cases_with_vel:
        vel = np.asarray(c["velocity"], dtype=float)
        t = np.linspace(0, 1, len(vel))
        series[f"pair {c['pair_index']}"] = (t, vel)
    fig = create_line_plot(
        series,
        title="Semantic Velocity Along Latent Interpolation",
        xlabel="t",
        ylabel="JS divergence",
    )
    return save_figure(fig, output_path, **save_kw)


def _plot_top_concept_paths(rows: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    concepts = {c: i for i, c in enumerate(sorted(df["top_concept"].unique()))}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pair_idx, g in list(df.groupby("pair_index"))[:6]:
        y = [concepts[c] for c in g["top_concept"]]
        ax.step(g["t"], y, where="mid", label=f"pair {pair_idx}")
    ax.set_yticks(list(concepts.values()))
    ax.set_yticklabels(list(concepts.keys()), fontsize=7)
    ax.set_xlabel("t")
    ax.set_title("Interpolation Top-Concept Paths")
    ax.legend(fontsize=7, ncol=2)
    return save_figure(fig, output_path, **save_kw)


def _plot_example_grid(cases: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    rows = []
    for c in cases[:8]:
        rows.append(
            [
                c["pair_index"],
                c["pair_type"],
                " -> ".join(c["endpoint_concepts"]),
                f"{c['smoothness']:.3f}",
                c["top_concept_transition_count"],
            ]
        )
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["pair", "type", "endpoint concepts", "smoothness", "transitions"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("Interpolation Example Grid")
    return save_figure(fig, output_path, **save_kw)


def save_interpolation_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    """Persist interpolation metrics, cases, paths, and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = results.pop("_path_rows", [])
    cases = results.get("cases", [])
    metrics = {k: v for k, v in results.items() if k != "cases"}
    with open(output_dir / "interpolation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(output_dir / "interpolation_cases.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "interpolation_paths.csv", index=False)

    figs: List[Path] = []
    if results.get("status") == "computed":
        figs_dir = output_dir / "figures"
        figs += _plot_concept_probabilities(
            rows,
            figs_dir / "figure_interpolation_concept_probabilities",
            formats=save_formats,
        )
        figs += _plot_semantic_velocity(
            cases,
            figs_dir / "figure_interpolation_semantic_velocity",
            formats=save_formats,
        )
        figs += _plot_top_concept_paths(
            rows,
            figs_dir / "figure_interpolation_top_concept_paths",
            formats=save_formats,
        )
        figs += _plot_example_grid(
            cases,
            figs_dir / "figure_interpolation_example_grid",
            formats=save_formats,
        )

    logger.info("Interpolation results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}

