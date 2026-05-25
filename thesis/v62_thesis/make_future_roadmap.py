"""Render future_roadmap.png for the thesis.

Print-friendly horizontal roadmap with four phases (CURRENT, NEAR-TERM,
MEDIUM-TERM, LONG-TERM) plus an EXPLORATORY call-out underneath MEDIUM-TERM.
The CURRENT box reflects the final exported triple-fusion retrieval system
(V61a_mctta16 + V62a + V66a, 86.3% SHARED1000 CSLS R@1), with the earlier
V35+N1v28a fixed-fusion milestone retained only in text elsewhere.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PATH = Path(__file__).with_name("future_roadmap.png")

PHASES = [
    {
        "x": 0.05, "y": 0.55, "w": 0.20, "h": 0.30,
        "header": "CURRENT", "header_color": "#3a8d99",
        "fill": "#e6f1f3", "edge": "#a9c9cf",
        "title": "Frozen production\nsystem",
        "body": "Triple score fusion\nV61a$_{\\mathrm{mctta16}}$ + V62a + V66a\n86.3 % SHARED1000\nCSLS R@1",
    },
    {
        "x": 0.28, "y": 0.55, "w": 0.20, "h": 0.30,
        "header": "NEAR-TERM", "header_color": "#5f8d4e",
        "fill": "#ecf3e7", "edge": "#bcd2ad",
        "title": "Cross-subject\npretraining",
        "body": "Stronger alignment\nfor shared\nrepresentations",
    },
    {
        "x": 0.51, "y": 0.55, "w": 0.20, "h": 0.30,
        "header": "MEDIUM-TERM", "header_color": "#b78b3c",
        "fill": "#f7efd9", "edge": "#dec88b",
        "title": "Fusion-aware\ntraining",
        "body": "Learn complementarity\nrather than only\nexporting scores",
    },
    {
        "x": 0.74, "y": 0.55, "w": 0.20, "h": 0.30,
        "header": "LONG-TERM", "header_color": "#8a6fb0",
        "fill": "#efeaf5", "edge": "#c4b6db",
        "title": "Richer\nreconstruction",
        "body": "Stronger quantitative\nimage metrics",
    },
]

EXPLORATORY = {
    "x": 0.40, "y": 0.10, "w": 0.30, "h": 0.32,
    "header": "Perception vs. imagination",
    "header_color": "#b25b66",
    "fill": "#f6e3e6", "edge": "#d8a5ad",
    "body": "Extension to imagination /\nperception comparison and\nmore ambitious decoding tasks",
}

LEGEND = [
    ("Current system", "#e6f1f3", "#a9c9cf"),
    ("Near-term", "#ecf3e7", "#bcd2ad"),
    ("Medium-term", "#f7efd9", "#dec88b"),
    ("Long-term", "#efeaf5", "#c4b6db"),
    ("Exploratory", "#f6e3e6", "#d8a5ad"),
]


def draw_box(ax, x, y, w, h, fill, edge):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.2, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.0))

    # Phase headers (above each box) and box bodies.
    for p in PHASES:
        cx = p["x"] + p["w"] / 2
        ax.text(cx, p["y"] + p["h"] + 0.04, p["header"],
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=p["header_color"])
        draw_box(ax, p["x"], p["y"], p["w"], p["h"], p["fill"], p["edge"])
        ax.text(cx, p["y"] + p["h"] - 0.05, p["title"],
                ha="center", va="top",
                fontsize=11, fontweight="bold",
                color=p["header_color"])
        ax.text(cx, p["y"] + p["h"] / 2 - 0.04, p["body"],
                ha="center", va="center",
                fontsize=9, color="#222222")

    # Arrows linking the four phases horizontally.
    arrow_y = 0.55 + 0.30 / 2
    for i in range(len(PHASES) - 1):
        x_start = PHASES[i]["x"] + PHASES[i]["w"]
        x_end = PHASES[i + 1]["x"]
        ax.annotate(
            "", xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
            arrowprops=dict(arrowstyle="-", color="#888888", linewidth=1.0),
        )

    # Exploratory branch (under MEDIUM-TERM).
    e = EXPLORATORY
    draw_box(ax, e["x"], e["y"], e["w"], e["h"], e["fill"], e["edge"])
    cx = e["x"] + e["w"] / 2
    ax.text(cx, e["y"] + e["h"] - 0.05, e["header"],
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color=e["header_color"])
    ax.text(cx, e["y"] + e["h"] / 2 - 0.04, e["body"],
            ha="center", va="center",
            fontsize=9, color="#222222")
    # Connector from MEDIUM-TERM box down to exploratory box.
    mid_phase = PHASES[2]
    ax.annotate(
        "",
        xy=(cx, e["y"] + e["h"]),
        xytext=(mid_phase["x"] + mid_phase["w"] / 2, mid_phase["y"]),
        arrowprops=dict(arrowstyle="-", color="#b25b66", linewidth=0.8,
                        linestyle="-"),
    )

    # Legend (bottom-left).
    legend_x = 0.015
    legend_y_top = 0.30
    legend_h = 0.025
    legend_w = 0.022
    for i, (label, fill, edge) in enumerate(LEGEND):
        y = legend_y_top - i * (legend_h + 0.012)
        draw_box(ax, legend_x, y, legend_w, legend_h, fill, edge)
        ax.text(legend_x + legend_w + 0.008, y + legend_h / 2, label,
                ha="left", va="center", fontsize=9, color="#222222")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
