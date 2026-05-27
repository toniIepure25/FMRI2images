"""Render model_architecture.png -- Figure 3.2.

Trainable decoder architecture shared by the V60--V66 lineage: shared encoder
on the preprocessed voxel vector, three task-specific heads (compact / rerank /
rich). The diagram is presented as the *trainable* architecture; the
inference-time triple score fusion sits on top of this and is shown
separately in Figure 5.3.

Routing: a single arrow leaves the encoder into a small "fan junction"
which then sends one horizontal arrow into each of the three heads.
This makes it visually unambiguous that there is exactly one encoder
feeding three parallel heads, rather than three diagonal arrows emerging
from the encoder edge.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).with_name("model_architecture.png")

FIG_W, FIG_H = 11.5, 6.8

C_INPUT = "#dfe7f3"
C_ENC = "#f5e8d3"
C_HEAD_C = "#e0e7f4"
C_HEAD_R = "#ecdde7"
C_HEAD_G = "#dde9da"
C_BORDER = "#7e8694"
C_BUS = "#4a4a4a"


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
        ax.text(cx, y + h - 0.18, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold")
        ax.text(cx, y + h - 0.55, text, ha="center", va="top",
                fontsize=fontsize, linespacing=1.35)
    else:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, linespacing=1.35)


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

    # ===== Title strip at the top =====
    ax.text(FIG_W / 2, FIG_H - 0.25,
            "Trainable multi-head decoder shared by the V60–V66 lineage",
            ha="center", va="top", fontsize=12.5, fontweight="bold")
    ax.text(FIG_W / 2, FIG_H - 0.70,
            "(Inference-time score fusion across V61a, V62a and V66a is shown in Figure 5.3.)",
            ha="center", va="top", fontsize=10, color="#555555", style="italic")

    # ===== Vertical centre, accounting for title strip and footer =====
    center_y = FIG_H / 2 - 0.10

    # ===== Input + Encoder (left column) =====
    h_block = 1.15
    y_block = center_y - h_block / 2

    x_in, w_in = 0.40, 2.20
    box(ax, x_in, y_block, w_in, h_block,
        "$\\sim$15.7k voxels\nper-session z-scored",
        face=C_INPUT, title="Preprocessed fMRI")

    x_enc, w_enc = x_in + w_in + 0.50, 2.30
    box(ax, x_enc, y_block, w_enc, h_block,
        "residual MLP backbone\nshared across all three heads",
        face=C_ENC, title="Shared encoder")

    arrow(ax, (x_in + w_in, center_y),
              (x_enc, center_y))

    enc_right_x = x_enc + w_enc
    enc_mid_y = center_y

    # ===== Three head boxes (right column, stacked vertically) =====
    x_head = enc_right_x + 0.95            # leave room for the fan junction
    w_head = 4.20
    h_head = 1.05
    head_gap = 0.22

    # Symmetric vertical placement: top above center, middle at center, bottom below.
    y_head_top = center_y + h_head / 2 + head_gap         # bottom-left y of top box
    y_head_mid = center_y - h_head / 2
    y_head_bot = center_y - h_head / 2 - h_head - head_gap

    box(ax, x_head, y_head_top, w_head, h_head,
        "768-D compact target\nshortlist formation",
        face=C_HEAD_C, title="Compact retrieval head", title_size=10.5,
        fontsize=9.5)
    box(ax, x_head, y_head_mid, w_head, h_head,
        "2048-D PCA target\nshortlist-local correction",
        face=C_HEAD_R, title="Dedicated rerank head", title_size=10.5,
        fontsize=9.5)
    box(ax, x_head, y_head_bot, w_head, h_head,
        "high-dimensional target\nregression + qualitative decoding",
        face=C_HEAD_G, title="Rich regression head", title_size=10.5,
        fontsize=9.5)

    # ===== Fan junction: encoder -> single trunk -> bracket -> three heads =====
    junction_x = enc_right_x + 0.45

    # 1) horizontal trunk out of the encoder
    ax.plot([enc_right_x, junction_x], [enc_mid_y, enc_mid_y],
            color=C_BUS, lw=1.6, solid_capstyle="round")
    # 2) vertical bracket spanning the head centers
    bracket_top_y = y_head_top + h_head / 2
    bracket_bot_y = y_head_bot + h_head / 2
    ax.plot([junction_x, junction_x], [bracket_bot_y, bracket_top_y],
            color=C_BUS, lw=1.6, solid_capstyle="round")
    # small junction dot at the trunk/bracket meeting point
    ax.plot([junction_x], [enc_mid_y], marker="o", markersize=5,
            markerfacecolor=C_BUS, markeredgecolor=C_BUS)
    # 3) one horizontal arrow from the bracket into each head's left side
    for y_h in (y_head_top, y_head_mid, y_head_bot):
        arrow(ax, (junction_x, y_h + h_head / 2),
                  (x_head, y_h + h_head / 2),
              color=C_BUS, lw=1.3)

    # ===== Footer caption strip =====
    ax.text(FIG_W / 2, 0.22,
            "The compact head feeds retrieval; the rerank head re-scores the shortlist; "
            "the rich head supports regression and the qualitative reconstruction add-on.",
            ha="center", va="bottom", fontsize=9, color="#333333")

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
