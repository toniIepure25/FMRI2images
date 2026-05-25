"""Render thesis_pipeline.png -- Figure 1.1.

Conceptually-correct pipeline overview:

    NSD + 7T fMRI  ->  Preprocessing
                       (single shared preprocessing stage)
                              |
                              |  (broadcast)
              +---------------+---------------+
              |               |               |
            V61a            V62a            V66a       (three INDEPENDENTLY
        encoder + head   encoder + head   encoder + head   trained experts,
              |               |               |          no shared weights)
              CSLS           CSLS            CSLS
              z-score        z-score         z-score
              |               |               |
              +---------------+---------------+
                              |
                Frozen score-level weighted sum
                  w1=0.7  w2=0.1  w3=0.2
                              |
                              v
                SHARED1000 headline endpoint
                CSLS R@1 = 86.3 %

The headline output box has a heavy border. The earlier V35 + N1v28a
fixed-fusion milestone is shown as a smaller strip at the bottom; the
SDXL + IP-Adapter reconstruction add-on is a separate strip to its right.
The three experts are explicitly drawn as parallel, independent pipelines
to avoid the previous misleading "shared multi-head decoder feeding three
experts" depiction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).with_name("thesis_pipeline.png")

FIG_W, FIG_H = 12.5, 8.4

C_DATA = "#dfe7f3"
C_PRE = "#e3eddc"
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
        ax.text(cx, y + h / 2 - 0.20, text, ha="center", va="center",
                fontsize=fontsize)
    else:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight)


def arrow(ax, p1, p2, *, ls="-", color="#3a3a3a", lw=1.4):
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

    # ----- TOP ROW: data + preprocessing (centered, single shared stage) -----
    y_top = 6.80
    h_top = 1.20

    data_w = 3.20
    pre_w = 3.20
    top_gap = 0.40
    total_top = data_w + top_gap + pre_w
    x_data0 = (FIG_W - total_top) / 2
    x_data = x_data0
    x_pre = x_data + data_w + top_gap

    box(ax, x_data, y_top, data_w, h_top,
        "NSD 7T fMRI + stimuli\nsubject 01 (subj01)",
        face=C_DATA, title="Data")
    box(ax, x_pre, y_top, pre_w, h_top,
        "voxel masking, caching\nper-session z-score\nstimulus-level splits",
        face=C_PRE, title="Preprocessing")
    arrow(ax, (x_data + data_w, y_top + h_top / 2),
              (x_pre, y_top + h_top / 2))

    # ----- MIDDLE ROW: three INDEPENDENT expert pipelines (parallel) -----
    y_mid = 4.55
    h_mid = 1.55       # taller so we can list "encoder + head" stack
    exp_w = 3.60
    gap_e = 0.30
    total_exp_w = 3 * exp_w + 2 * gap_e
    x_exp0 = (FIG_W - total_exp_w) / 2

    exp_x = [x_exp0 + i * (exp_w + gap_e) for i in range(3)]

    box(ax, exp_x[0], y_mid, exp_w, h_mid,
        "independently trained encoder + head\n"
        "197K ViT-L/14 token target\n"
        "MC-TTA-16 at inference\n"
        "(strongest single expert)",
        face=C_EXPERT, title="Expert 1 -- V61a")
    box(ax, exp_x[1], y_mid, exp_w, h_mid,
        "independently trained encoder + head\n"
        "768-D ViT-L/14 CLS target\n"
        "complementary geometry",
        face=C_EXPERT, title="Expert 2 -- V62a")
    box(ax, exp_x[2], y_mid, exp_w, h_mid,
        "independently trained encoder + head\n"
        "768-D ROI-pretrained CLS target\n"
        "anatomically-structured prior",
        face=C_EXPERT, title="Expert 3 -- V66a")

    # ----- bus from Preprocessing down to the three experts (broadcast) -----
    pre_center_x = x_pre + pre_w / 2
    bus_y_top = y_mid + h_mid + 0.40
    # vertical drop from preprocessing centre to bus
    ax.plot([pre_center_x, pre_center_x], [y_top, bus_y_top],
            color="#4a4a4a", lw=1.6, solid_capstyle="round")
    # horizontal bus reaching all three expert lanes
    bus_left = min(exp_x[0] + exp_w / 2, pre_center_x)
    bus_right = max(exp_x[-1] + exp_w / 2, pre_center_x)
    ax.plot([bus_left, bus_right], [bus_y_top, bus_y_top],
            color="#4a4a4a", lw=1.6, solid_capstyle="round")
    # three short drops with arrow heads into each expert top
    for ex in exp_x:
        arrow(ax, (ex + exp_w / 2, bus_y_top),
                  (ex + exp_w / 2, y_mid + h_mid),
              color="#4a4a4a")
    # tiny label on the bus to emphasise broadcast
    ax.text(pre_center_x + 0.10, bus_y_top + 0.10,
            "shared preprocessed input  ·  no shared weights downstream",
            fontsize=8.5, color="#4a4a4a", ha="left", va="bottom", style="italic")

    # ----- HEADLINE ROW: frozen score fusion -> SHARED1000 output -----
    y_head = 1.95
    h_head = 1.40

    fuse_w = 5.60
    out_w = 4.40
    gap_f = 0.40
    total_head_w = fuse_w + gap_f + out_w
    x_fuse0 = (FIG_W - total_head_w) / 2
    x_fuse = x_fuse0
    x_out = x_fuse + fuse_w + gap_f

    box(ax, x_fuse, y_head, fuse_w, h_head,
        "per-expert CSLS  ($k=3$)   then  $z$-score\n"
        r"$s_{\mathrm{final}}(i,j) = 0.7\,\widetilde{s}_1 + 0.1\,\widetilde{s}_2 + 0.2\,\widetilde{s}_3$"
        "\nweights chosen on validation and frozen\n"
        "inference-time only -- no unified latent embedding",
        face=C_FUSION_HEAD, border="#9d8a90", lw=1.3,
        title="Frozen score-level fusion", title_size=11.5, fontsize=9.5)

    box(ax, x_out, y_head, out_w, h_head,
        "CSLS R@1 = 86.3 %\nR@5 = 98.0 %  ·  MRR = 0.914\n"
        "primary SHARED1000\nbenchmark endpoint",
        face=C_OUTPUT, border=C_BORDER_HEAD, lw=2.2,
        title="SHARED1000 retrieval",
        title_size=11.5, fontsize=10, weight="bold")

    # bracket-shaped bus from experts down into fusion box
    fusion_x_center = x_fuse + fuse_w / 2
    bus_y_bot = y_mid - 0.45
    for ex in exp_x:
        ax.plot([ex + exp_w / 2, ex + exp_w / 2],
                [y_mid, bus_y_bot],
                color="#7a5a60", lw=1.6, solid_capstyle="round")
    bus_left = min(exp_x[0] + exp_w / 2, fusion_x_center)
    bus_right = max(exp_x[-1] + exp_w / 2, fusion_x_center)
    ax.plot([bus_left, bus_right], [bus_y_bot, bus_y_bot],
            color="#7a5a60", lw=1.6, solid_capstyle="round")
    arrow(ax, (fusion_x_center, bus_y_bot),
              (fusion_x_center, y_head + h_head),
          color="#7a5a60", lw=1.6)
    arrow(ax, (x_fuse + fuse_w, y_head + h_head / 2),
              (x_out, y_head + h_head / 2),
          color=C_BORDER_HEAD, lw=1.8)

    # ----- BOTTOM ROW: milestone strip + reconstruction add-on side by side -----
    y_mil = 0.20
    h_mil = 0.95
    total_bot = total_head_w
    mil_w = total_bot * 0.60
    diff_w = total_bot * 0.34
    gap_bot = total_bot - mil_w - diff_w
    x_mil = x_fuse0
    x_diff = x_mil + mil_w + gap_bot

    box(ax, x_mil, y_mil, mil_w, h_mil,
        "V35 compact-CSLS + N1v28a legacy-CSLS\n"
        "SHARED1000 R@1 = 77.2 %  -- methodological milestone",
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

    # ----- section labels (small, breathing space) -----
    label_kw = dict(fontsize=9, fontweight="bold", color="#3a3a3a", ha="left")
    ax.text(0.30, y_top + h_top + 0.20,
            "SHARED  PREPROCESSING", **label_kw)
    ax.text(0.30, y_mid + h_mid + 0.65,
            "THREE  INDEPENDENTLY  TRAINED  RETRIEVAL  EXPERTS", **label_kw)
    ax.text(x_fuse0, y_head + h_head + 0.25,
            "FINAL  EXPORTED  FUSION   (inference-time, score-level only)", **label_kw)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
