"""Render clip_shared_space_alignment.png — Figure 2.2.

Conceptual illustration of CLIP's shared image/text embedding space, with a
decoded fMRI embedding landing inside the correct semantic neighbourhood.
The figure is illustrative only -- it is not a projection of real data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

OUT_PATH = Path(__file__).with_name("clip_shared_space_alignment.png")
RNG = np.random.default_rng(7)


CLUSTERS = [
    # (label, centre, semi-axes, colour, light-fill-colour)
    ("Animals",            (-2.7,  2.0), (1.10, 0.85), "#1f77b4", "#dde7f4"),
    ("Urban scenes",       ( 2.7,  2.0), (1.30, 0.85), "#d62728", "#f4dada"),
    ("Transport",          ( 0.0,  0.0), (1.35, 0.90), "#2ca02c", "#d8eddb"),
    ("Food / kitchen",     ( 2.7, -2.0), (1.30, 0.85), "#ff7f0e", "#fae2cd"),
    ("Outdoor landscapes", (-2.7, -2.0), (1.40, 0.85), "#9467bd", "#e6dff1"),
]


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=180)
    ax.set_xlim(-5.0, 5.5)
    ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal")

    # Axes box and tick-free axes
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Semantic dimension 1", fontsize=10, color="#333")
    ax.set_ylabel("Semantic dimension 2", fontsize=10, color="#333")

    # Draw the clusters
    for label, centre, (sx, sy), border_c, fill_c in CLUSTERS:
        e = Ellipse(centre, width=2 * sx, height=2 * sy,
                    angle=0, facecolor=fill_c, edgecolor=border_c,
                    linewidth=1.4, linestyle="--", alpha=0.85)
        ax.add_patch(e)

        # 3 image embeddings (circles) + 2 text embeddings (squares)
        pts_img = np.array(centre) + 0.55 * RNG.standard_normal((3, 2)) * np.array([sx, sy])
        pts_txt = np.array(centre) + 0.55 * RNG.standard_normal((2, 2)) * np.array([sx, sy])
        ax.scatter(pts_img[:, 0], pts_img[:, 1], s=44, c=border_c,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(pts_txt[:, 0], pts_txt[:, 1], s=44, facecolors="white",
                   edgecolors=border_c, linewidths=1.5, marker="s", zorder=3)

        # Label above the ellipse with a soft tag background
        lx, ly = centre[0], centre[1] + sy + 0.18
        ax.text(lx, ly, label, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=border_c,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=border_c, linewidth=1.0))

    # --- Decoded fMRI embedding: star inside the "Transport" cluster ---
    decoded_xy = (-0.3, 0.20)   # inside Transport
    target_xy = ( 0.55, 0.05)   # the matched image embedding within Transport
    ax.scatter([decoded_xy[0]], [decoded_xy[1]],
               s=300, marker="*", c="#6f00b3",
               edgecolors="white", linewidths=1.2, zorder=5)

    # arrow showing the alignment between decoded fMRI star and matched image embedding
    arr = FancyArrowPatch(
        decoded_xy, target_xy,
        arrowstyle="-|>", mutation_scale=12,
        color="#6f00b3", lw=1.6, alpha=0.85, zorder=4,
    )
    ax.add_patch(arr)
    ax.text(decoded_xy[0] - 0.65, decoded_xy[1] + 0.55,
            "decoded brain\nembedding",
            ha="center", va="center",
            fontsize=9.5, color="#6f00b3", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="#6f00b3", linewidth=1.0))
    ax.text(target_xy[0] + 0.95, target_xy[1] - 0.30,
            "matched image /\ntext pair",
            ha="left", va="center",
            fontsize=8.5, color="#333333", style="italic")

    # --- Legend (top-right, outside cluster zones) ---
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="None",
                   markersize=8, markerfacecolor="#555", markeredgecolor="white",
                   label="Image embedding"),
        plt.Line2D([], [], marker="s", linestyle="None",
                   markersize=8, markerfacecolor="white", markeredgecolor="#555",
                   label="Text embedding"),
        plt.Line2D([], [], marker="*", linestyle="None",
                   markersize=14, markerfacecolor="#6f00b3", markeredgecolor="white",
                   label="Decoded fMRI embedding"),
    ]
    # Legend placed outside the plot area to the right so it never overlaps cluster labels.
    ax.legend(handles=legend_handles, loc="center left",
              frameon=True, fontsize=9.5,
              bbox_to_anchor=(1.02, 0.85))

    # --- Caption box at lower-left ---
    ax.text(-4.85, -3.35,
            "Nearby points encode shared semantics; the decoded brain\n"
            "embedding should land inside the correct image neighbourhood.",
            ha="left", va="bottom", fontsize=9, color="#333333", style="italic",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fafafa", edgecolor="#cccccc", linewidth=0.8))

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
