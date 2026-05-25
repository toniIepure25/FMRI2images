"""Render thesis_pipeline.png — Figure 1.1.

Pipeline overview that places the cross-architecture triple-fusion exported
system as the visual headline of the diagram (bottom row, dark border), and
shows the V35+N1v28a fixed fusion as a smaller milestone strip below. The
reconstruction add-on is included as a small "optional" side-branch.

Run from inside thesis/v62_thesis/ with `python make_thesis_pipeline.py`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).with_name("thesis_pipeline.png")

FIG_W, FIG_H = 12.0, 7.8

# Colours
C_DATA = "#dfe7f3"
C_PRE = "#e3eddc"
C_ENC = "#f5e8d3"
C_EXPERT = "#e3eaf6"
C_FUSION_HEAD = "#fce5e8"
C_OUTPUT = "#d8e8e6"
C_MILESTONE = "#ece2f0"
C_DIFFUSION = "#f5f0e0"

C_BORDER_DEFAULT = "#9aa3b2"
C_BORDER_HEAD = "#1c1c1c"
C_BORDER_MILESTONE = "#8077a0"


def box(ax, x, y, w, h, text, *, face, border=C_BORDER_DEFAULT, lw=1.0,
        fontsize=10, weight="normal", title=None, title_size=11,
        title_color="black"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=border, facecolor=face,
    )
    ax.add_patch(rect)
    cx = x + w / 2
    if title is not None:
        ax.text(cx, y + h - 0.22, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=title_color)
        ax.text(cx, y + h / 2 - 0.18, text, ha="center", va="center",
                fontsize=fontsize)
    else:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight)


def arrow(ax, p1, p2, *, ls="-", color="#3a3a3a", lw=1.3):
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>", mutation_scale=14,
        linestyle=ls, color=color, lw=lw,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_axis_off()

    # ----- TOP ROW: data -> preprocessing -> encoder -> multi-head -----
    y_top = 5.80
    h_top = 1.15
    box_w = 2.40
    gap = 0.45
    x0 = 0.40

    xs = [x0 + i * (box_w + gap) for i in range(4)]

    box(ax, xs[0], y_top, box_w, h_top,
        "NSD 7T fMRI + stimuli\nsubjects 01 / 02 / 05 / 07",
        face=C_DATA, title="Data")
    box(ax, xs[1], y_top, box_w, h_top,
        "voxel masking, caching\nper-session z-score\nstimulus-level splits",
        face=C_PRE, title="Preprocessing")
    box(ax, xs[2], y_top, box_w, h_top,
        "shared MLP backbone\nused by the V60--V66 lineage",
        face=C_ENC, title="fMRI encoder")
    box(ax, xs[3], y_top, box_w, h_top,
        "compact 768-D  ·  rerank 2048-D PCA\nrich regression head",
        face=C_ENC, title="Multi-head decoder")

    for i in range(3):
        arrow(ax,
              (xs[i] + box_w, y_top + h_top / 2),
              (xs[i + 1], y_top + h_top / 2))

    # ----- MIDDLE ROW: three frozen experts -----
    y_mid = 3.85
    h_mid = 1.20
    exp_w = 3.40
    gap_e = 0.30
    total_exp_w = 3 * exp_w + 2 * gap_e
    x_exp0 = (FIG_W - total_exp_w) / 2

    exp_x = [x_exp0 + i * (exp_w + gap_e) for i in range(3)]

    box(ax, exp_x[0], y_mid, exp_w, h_mid,
        "197K ViT-L/14 token target\nMC-TTA-16 at inference\n(strongest single expert)",
        face=C_EXPERT, title="Expert 1 -- V61a")
    box(ax, exp_x[1], y_mid, exp_w, h_mid,
        "768-D CLS target\ncomplementary geometry",
        face=C_EXPERT, title="Expert 2 -- V62a")
    box(ax, exp_x[2], y_mid, exp_w, h_mid,
        "768-D ROI-pretrained CLS\nanatomically-structured prior",
        face=C_EXPERT, title="Expert 3 -- V66a")

    # arrows from decoder (top row) into each expert via a clean bracket-shaped bus
    decoder_x_center = xs[3] + box_w / 2
    bus_y_top = y_mid + h_mid + 0.40   # well below decoder, well above experts
    bus_x_left = exp_x[0] + exp_w / 2
    bus_x_right = exp_x[-1] + exp_w / 2
    # vertical segment from decoder centre down to the bus
    ax.plot([decoder_x_center, decoder_x_center], [y_top, bus_y_top],
            color="#4a4a4a", lw=1.6, solid_capstyle="round")
    # horizontal bus spanning the three expert lanes (extend to also reach the decoder x)
    bus_left = min(bus_x_left, decoder_x_center)
    bus_right = max(bus_x_right, decoder_x_center)
    ax.plot([bus_left, bus_right], [bus_y_top, bus_y_top],
            color="#4a4a4a", lw=1.6, solid_capstyle="round")
    # short arrow drops into each expert top (with arrow heads)
    for ex in exp_x:
        arrow(ax,
              (ex + exp_w / 2, bus_y_top),
              (ex + exp_w / 2, y_mid + h_mid),
              color="#4a4a4a")

    # ----- HEADLINE ROW: frozen score fusion -> SHARED1000 output -----
    y_head = 1.55
    h_head = 1.30

    fuse_w = 5.20
    out_w = 4.20
    gap_f = 0.40
    total_head_w = fuse_w + gap_f + out_w
    x_fuse0 = (FIG_W - total_head_w) / 2

    x_fuse = x_fuse0
    x_out = x_fuse + fuse_w + gap_f

    box(ax, x_fuse, y_head, fuse_w, h_head,
        "per-expert CSLS  (k = 3)   then  z-score\n"
        r"$s_{\mathrm{final}}(i,j) = 0.7\,\widetilde{s}_1 + 0.1\,\widetilde{s}_2 + 0.2\,\widetilde{s}_3$"
        "\nweights chosen on validation and frozen",
        face=C_FUSION_HEAD, border="#9d8a90", lw=1.3,
        title="Frozen score-level fusion", title_size=11.5)

    box(ax, x_out, y_head, out_w, h_head,
        "CSLS R@1 = 86.3 %\nR@5 = 98.0 %  ·  MRR = 0.914",
        face=C_OUTPUT, border=C_BORDER_HEAD, lw=2.2,
        title="SHARED1000 headline endpoint",
        title_size=11.5, fontsize=10.5, weight="bold")

    # arrows from each expert down into the fusion box via a second clean bracket-shaped bus
    fusion_x_center = x_fuse + fuse_w / 2
    bus_y_bot = y_mid - 0.40           # well below experts, well above fusion box
    # short drop from each expert bottom to the bus (no arrow head; arrowhead is the
    # single descending arrow that follows)
    for ex in exp_x:
        ax.plot([ex + exp_w / 2, ex + exp_w / 2],
                [y_mid, bus_y_bot],
                color="#7a5a60", lw=1.6, solid_capstyle="round")
    # horizontal bus collecting the three expert lanes
    bus_left = min(exp_x[0] + exp_w / 2, fusion_x_center)
    bus_right = max(exp_x[-1] + exp_w / 2, fusion_x_center)
    ax.plot([bus_left, bus_right], [bus_y_bot, bus_y_bot],
            color="#7a5a60", lw=1.6, solid_capstyle="round")
    # single arrow from the bus centre down into the fusion box
    arrow(ax, (fusion_x_center, bus_y_bot),
              (fusion_x_center, y_head + h_head),
          color="#7a5a60", lw=1.6)
    # arrow from fusion box to headline output
    arrow(ax, (x_fuse + fuse_w, y_head + h_head / 2),
              (x_out, y_head + h_head / 2),
          color=C_BORDER_HEAD, lw=1.8)

    # ----- BOTTOM ROW: milestone strip + reconstruction add-on side by side -----
    y_mil = 0.10
    h_mil = 0.85
    total_bot = total_head_w
    mil_w = total_bot * 0.62
    diff_w = total_bot * 0.32
    gap_bot = total_bot - mil_w - diff_w
    x_mil = x_fuse0
    x_diff = x_mil + mil_w + gap_bot

    box(ax, x_mil, y_mil, mil_w, h_mil,
        "V35 compact-CSLS + N1v28a legacy-CSLS\n"
        "SHARED1000 R@1 = 77.2 %  -- retained as methodological milestone",
        face=C_MILESTONE, border=C_BORDER_MILESTONE, lw=1.0,
        title=None, fontsize=9)

    box(ax, x_diff, y_mil, diff_w, h_mil,
        "SDXL 1.0 + IP-Adapter + CLIP injection\n"
        r"$n=141$ examples  ·  2-way AlexNet(5) acc. $86.2\%$",
        face=C_DIFFUSION, border="#a89878", lw=1.0,
        title=None, fontsize=8.5)
    arrow(ax, (x_out + out_w / 2, y_head),
              (x_diff + diff_w / 2, y_mil + h_mil),
          ls=":", color="#7a6a4a")

    # ----- section labels (small, left margin, with breathing space) -----
    label_kw = dict(fontsize=9, fontweight="bold", color="#3a3a3a", ha="left")
    ax.text(0.30, y_top + h_top + 0.20, "MAIN  PIPELINE", **label_kw)
    ax.text(0.30, y_mid + h_mid + 0.20, "THREE  FROZEN  EXPERTS", **label_kw)
    ax.text(x_fuse0, y_head + h_head + 0.20,
            "FINAL  EXPORTED  FUSION   (headline)", **label_kw)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
