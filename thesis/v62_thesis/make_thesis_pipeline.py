"""Render thesis_pipeline.png -- Figure 1.1.

Three independently trained retrieval experts (V61a, V62a, V66a) consume
the same preprocessed fMRI input but share no weights downstream. Their
per-expert CSLS score matrices are z-score normalised and combined by a
fixed weighted sum (0.7 / 0.1 / 0.2), reaching 86.3% CSLS R@1 on SHARED1000.
The V35 + N1v28a fixed-fusion milestone (77.2%) and the SDXL + IP-Adapter
reconstruction add-on are shown as auxiliary strips at the bottom.

Layout rules:
- No section labels overlap any arrow or bus.
- Body text inside each box is top-anchored so it cannot spill below the box.
- Buses are routed with clear vertical space above and below each row.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = Path(__file__).with_name("thesis_pipeline.png")

FIG_W, FIG_H = 13.0, 9.4

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
C_BUS_TOP = "#4a4a4a"
C_BUS_BOT = "#7a5a60"


def box(ax, x, y, w, h, text, *, face, border=C_BORDER_DEFAULT, lw=1.0,
        fontsize=10, weight="normal", title=None, title_size=11,
        title_color="black"):
    """Round rectangle with optional bold title at the top.

    Body text is top-anchored, so longer content extends DOWN inside the box
    rather than spilling past the box edges.
    """
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=border, facecolor=face,
    )
    ax.add_patch(rect)
    cx = x + w / 2
    if title is not None:
        title_y = y + h - 0.18
        ax.text(cx, title_y, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=title_color)
        body_y = y + h - 0.55
        ax.text(cx, body_y, text, ha="center", va="top",
                fontsize=fontsize, fontweight=weight, linespacing=1.35)
    else:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, linespacing=1.35)


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

    # ===== TOP ROW: data + preprocessing (single shared stage) =====
    y_top = 8.00
    h_top = 1.20
    data_w = 3.40
    pre_w = 3.40
    top_gap = 0.50
    total_top = data_w + top_gap + pre_w
    x_data = (FIG_W - total_top) / 2
    x_pre = x_data + data_w + top_gap

    box(ax, x_data, y_top, data_w, h_top,
        "NSD 7T fMRI + stimuli\nsubject 01 (subj01)",
        face=C_DATA, title="Data")
    box(ax, x_pre, y_top, pre_w, h_top,
        "voxel masking, caching\nper-session z-score\nstimulus-level splits",
        face=C_PRE, title="Preprocessing")
    arrow(ax, (x_data + data_w, y_top + h_top / 2),
              (x_pre, y_top + h_top / 2))

    # ===== MIDDLE ROW: three INDEPENDENT expert pipelines =====
    y_mid = 5.10
    h_mid = 1.85
    exp_w = 3.70
    gap_e = 0.40
    total_exp_w = 3 * exp_w + 2 * gap_e
    x_exp0 = (FIG_W - total_exp_w) / 2
    exp_x = [x_exp0 + i * (exp_w + gap_e) for i in range(3)]

    box(ax, exp_x[0], y_mid, exp_w, h_mid,
        "independently trained\nencoder + head\n"
        "197K ViT-L/14 token target\n"
        "MC-TTA-16 at inference",
        face=C_EXPERT, title="Expert 1 - V61a")
    box(ax, exp_x[1], y_mid, exp_w, h_mid,
        "independently trained\nencoder + head\n"
        "768-D ViT-L/14 CLS target\n"
        "complementary geometry",
        face=C_EXPERT, title="Expert 2 - V62a")
    box(ax, exp_x[2], y_mid, exp_w, h_mid,
        "independently trained\nencoder + head\n"
        "768-D ROI-pretrained CLS\n"
        "anatomically-structured prior",
        face=C_EXPERT, title="Expert 3 - V66a")

    # ===== BROADCAST BUS: preprocessing -> three experts =====
    pre_center_x = x_pre + pre_w / 2
    bus_y_top = y_mid + h_mid + 0.55       # safely above experts
    # vertical drop from preprocessing
    ax.plot([pre_center_x, pre_center_x], [y_top, bus_y_top],
            color=C_BUS_TOP, lw=1.8, solid_capstyle="round")
    # horizontal bus
    bus_left = min(exp_x[0] + exp_w / 2, pre_center_x)
    bus_right = max(exp_x[-1] + exp_w / 2, pre_center_x)
    ax.plot([bus_left, bus_right], [bus_y_top, bus_y_top],
            color=C_BUS_TOP, lw=1.8, solid_capstyle="round")
    # arrowed drop into each expert top
    for ex in exp_x:
        arrow(ax, (ex + exp_w / 2, bus_y_top),
                  (ex + exp_w / 2, y_mid + h_mid),
              color=C_BUS_TOP, lw=1.5)
    # bus annotation: placed on the LEFT half of the bus, above it, so it
    # never crosses the vertical drop (which sits on the right half).
    ann_x_center = (bus_left + pre_center_x) / 2
    ax.text(ann_x_center, bus_y_top + 0.18,
            "broadcast: no shared weights downstream",
            fontsize=9, color="#4a4a4a",
            ha="center", va="bottom", style="italic")

    # ===== HEADLINE ROW: frozen score-level fusion + SHARED1000 output =====
    y_head = 2.20
    h_head = 1.55
    fuse_w = 6.00
    out_w = 4.40
    gap_f = 0.50
    total_head_w = fuse_w + gap_f + out_w
    x_fuse = (FIG_W - total_head_w) / 2
    x_out = x_fuse + fuse_w + gap_f

    box(ax, x_fuse, y_head, fuse_w, h_head,
        "per-expert CSLS  ($k=3$),  then  $z$-score\n"
        r"$s_{\mathrm{final}}(i,j) = 0.7\,\widetilde{s}_1 + 0.1\,\widetilde{s}_2 + 0.2\,\widetilde{s}_3$"
        "\nweights chosen on validation, then frozen\n"
        "inference-time only - no unified latent embedding",
        face=C_FUSION_HEAD, border="#9d8a90", lw=1.3,
        title="Frozen score-level fusion", title_size=12, fontsize=10)

    box(ax, x_out, y_head, out_w, h_head,
        "CSLS R@1 = 86.3 %\nR@5 = 98.0 %  ·  MRR = 0.914\n"
        "primary SHARED1000\nbenchmark endpoint",
        face=C_OUTPUT, border=C_BORDER_HEAD, lw=2.2,
        title="SHARED1000 retrieval", title_size=12,
        fontsize=10.5, weight="bold")

    # bracket-shaped bus from experts down into fusion box
    fusion_x_center = x_fuse + fuse_w / 2
    bus_y_bot = y_mid - 0.55
    for ex in exp_x:
        ax.plot([ex + exp_w / 2, ex + exp_w / 2],
                [y_mid, bus_y_bot],
                color=C_BUS_BOT, lw=1.6, solid_capstyle="round")
    bot_left = min(exp_x[0] + exp_w / 2, fusion_x_center)
    bot_right = max(exp_x[-1] + exp_w / 2, fusion_x_center)
    ax.plot([bot_left, bot_right], [bus_y_bot, bus_y_bot],
            color=C_BUS_BOT, lw=1.6, solid_capstyle="round")
    arrow(ax, (fusion_x_center, bus_y_bot),
              (fusion_x_center, y_head + h_head),
          color=C_BUS_BOT, lw=1.6)
    arrow(ax, (x_fuse + fuse_w, y_head + h_head / 2),
              (x_out, y_head + h_head / 2),
          color=C_BORDER_HEAD, lw=1.8)

    # ===== BOTTOM ROW: milestone strip + reconstruction add-on =====
    y_mil = 0.35
    h_mil = 1.15
    total_bot = total_head_w
    mil_w = total_bot * 0.58
    diff_w = total_bot * 0.36
    gap_bot = total_bot - mil_w - diff_w
    x_mil = x_fuse
    x_diff = x_mil + mil_w + gap_bot

    box(ax, x_mil, y_mil, mil_w, h_mil,
        "V35 compact-CSLS + N1v28a legacy-CSLS\n"
        "SHARED1000 R@1 = 77.2 %",
        face=C_MILESTONE, border=C_BORDER_MILESTONE, lw=1.0,
        title="Methodological milestone (predecessor)",
        title_size=10.5, fontsize=9.5)

    box(ax, x_diff, y_mil, diff_w, h_mil,
        "SDXL 1.0 + IP-Adapter + CLIP injection\n"
        r"$n=141$  ·  2-way AlexNet(5) = $86.2\%$",
        face=C_DIFFUSION, border="#a89878", lw=1.0,
        title="Reconstruction add-on",
        title_size=10.5, fontsize=9)
    # Stronger dotted connector from the SHARED1000 output box down to
    # the SDXL strip, making the "headline retrieval feeds the add-on"
    # relationship readable.
    arrow(ax, (x_out + out_w / 2, y_head),
              (x_diff + diff_w / 2, y_mil + h_mil),
          ls=":", color="#7a6a4a", lw=1.7)
    # label on the dotted connector clarifying what flows from the
    # headline retrieval into the reconstruction add-on
    ax.text((x_out + out_w / 2 + x_diff + diff_w / 2) / 2 + 0.10,
            (y_head + y_mil + h_mil) / 2,
            "top-1 image\nas IP-Adapter anchor",
            fontsize=8, color="#7a6a4a", ha="left", va="center",
            style="italic", linespacing=1.25)

    # ===== Stage numbers down the LEFT margin (1..4) =====
    stage_x = 0.20
    stage_kw = dict(ha="left", va="center", fontsize=11, fontweight="bold",
                    color="#555555")
    ax.text(stage_x, y_top + h_top / 2, "1", **stage_kw)
    ax.text(stage_x, y_mid + h_mid / 2, "2", **stage_kw)
    ax.text(stage_x, y_head + h_head / 2, "3", **stage_kw)
    ax.text(stage_x, y_mil + h_mil / 2, "4", **stage_kw)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
