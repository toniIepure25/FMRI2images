"""Render model_architecture.png — Figure 3.2.

Trainable decoder architecture shared by the V60--V66 lineage: shared encoder
on the preprocessed voxel vector, three task-specific heads (compact / rerank /
rich). The diagram is presented as the *trainable* architecture; the
inference-time triple score fusion sits on top of this and is shown
separately in Figure 5.3.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).with_name("model_architecture.png")

FIG_W, FIG_H = 11.5, 5.8

C_INPUT = "#dfe7f3"
C_ENC = "#f5e8d3"
C_HEAD_C = "#e0e7f4"
C_HEAD_R = "#ecdde7"
C_HEAD_G = "#dde9da"
C_OUT = "#d8e8e6"
C_BORDER = "#7e8694"


def box(ax, x, y, w, h, text, *, face, title=None,
        border=C_BORDER, lw=1.0, fontsize=10, title_size=11):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=border, facecolor=face,
    )
    ax.add_patch(rect)
    cx = x + w / 2
    if title is not None:
        ax.text(cx, y + h - 0.22, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold")
        ax.text(cx, y + h / 2 - 0.22, text, ha="center", va="center",
                fontsize=fontsize)
    else:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize)


def arrow(ax, p1, p2, *, color="#3a3a3a", lw=1.3):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        color=color, lw=lw, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_axis_off()

    # Title at top
    ax.text(FIG_W / 2, FIG_H - 0.25,
            "Trainable multi-head decoder shared by the V60–V66 lineage",
            ha="center", va="top", fontsize=12.5, fontweight="bold")
    ax.text(FIG_W / 2, FIG_H - 0.65,
            "(Inference-time score fusion across V61a, V62a and V66a is shown in Figure 5.3.)",
            ha="center", va="top", fontsize=10, color="#555555", style="italic")

    # --- Input ---
    x_in, w_in = 0.40, 2.20
    y_mid = (FIG_H - 1.50) / 2 - 0.20
    h_in = 1.10
    box(ax, x_in, y_mid, w_in, h_in,
        "$\\sim$15.7k voxels\nper-session z-scored",
        face=C_INPUT, title="Preprocessed fMRI")

    # --- Encoder ---
    x_enc, w_enc = x_in + w_in + 0.50, 2.30
    box(ax, x_enc, y_mid, w_enc, h_in,
        "residual MLP backbone\nshared across all three heads",
        face=C_ENC, title="Shared encoder")

    arrow(ax, (x_in + w_in, y_mid + h_in / 2),
              (x_enc, y_mid + h_in / 2))

    # --- Three heads (stacked vertically) ---
    x_head = x_enc + w_enc + 0.50
    w_head = 4.40                     # wider so body text fits inside the box
    h_head = 1.10                     # slightly taller for two-line body
    gap = 0.20

    y_head_top = y_mid + h_in / 2 + h_head + gap / 2
    y_head_mid = y_mid + h_in / 2 - h_head / 2
    y_head_bot = y_mid + h_in / 2 - h_head - h_head / 2 - gap / 2

    box(ax, x_head, y_head_top, w_head, h_head,
        "768-D compact target\nshortlist formation",
        face=C_HEAD_C, title="Compact retrieval head", title_size=10.5,
        fontsize=9.5)
    box(ax, x_head, y_head_mid, w_head, h_head,
        "2048-D PCA target\nshortlist-local correction",
        face=C_HEAD_R, title="Dedicated rerank head", title_size=10.5,
        fontsize=9.5)
    box(ax, x_head, y_head_bot, w_head, h_head,
        "high-dimensional target\nregression for qualitative decoding",
        face=C_HEAD_G, title="Rich regression head", title_size=10.5,
        fontsize=9.5)

    # encoder -> heads (fan-out)
    enc_right = (x_enc + w_enc, y_mid + h_in / 2)
    for y_h in (y_head_top, y_head_mid, y_head_bot):
        arrow(ax, enc_right, (x_head, y_h + h_head / 2), color="#4a4a4a")

    # --- caption strip at the bottom ---
    ax.text(FIG_W / 2, 0.18,
            "The compact head feeds retrieval; the rerank head re-scores the shortlist; "
            "the rich head supports regression and the optional qualitative reconstruction add-on.",
            ha="center", va="bottom", fontsize=9, color="#333333")

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
