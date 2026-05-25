"""Render triple_fusion_architecture.png for the thesis.

Visual diagram of the final exported triple-fusion retrieval system:
three retrieval experts (V61a token-space with MC-TTA-16, V62a 768-D CLS,
V66a 768-D ROI) each producing a CSLS-corrected score at neighbourhood
size k=3, z-score normalized, then combined with frozen weights
0.7/0.1/0.2 into the SHARED1000 retrieval ranking that reaches 86.3%
CSLS R@1 (R@5 = 98.0%, MRR = 0.914).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PATH = Path(__file__).with_name("triple_fusion_architecture.png")

EXPERT_FILL = "#eaf0f7"
EXPERT_EDGE = "#9cb2cf"
EXPERT_TITLE = "#3f4f70"

CSLS_FILL = "#f3ecdc"
CSLS_EDGE = "#cdb583"
CSLS_TITLE = "#7a6128"

ZSCORE_FILL = "#ecf3e7"
ZSCORE_EDGE = "#a8c89c"
ZSCORE_TITLE = "#4a6b3a"

FUSION_FILL = "#f6e3e6"
FUSION_EDGE = "#c79aa3"
FUSION_TITLE = "#7a3d4a"

OUTPUT_FILL = "#e6f1f3"
OUTPUT_EDGE = "#7fa6ae"
OUTPUT_TITLE = "#2c5b65"

ARROW = "#666666"

EXPERTS = [
    {
        "cx": 0.18,
        "title": "V61a token-space",
        "body": "197K-D ViT-L/14 target\n+ MC-TTA-16 inference\n(strongest single)",
        "weight": "$w_1 = 0.7$",
    },
    {
        "cx": 0.50,
        "title": "V62a 768-D CLS",
        "body": "768-D ViT-L/14 CLS target\n(complementary)",
        "weight": "$w_2 = 0.1$",
    },
    {
        "cx": 0.82,
        "title": "V66a 768-D ROI",
        "body": "768-D ROI-pretrained\n(complementary)",
        "weight": "$w_3 = 0.2$",
    },
]


def draw_box(ax, cx, cy, w, h, fill, edge, lw=1.2):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)


def vertical_arrow(ax, cx, y_top, y_bot, color=ARROW):
    ax.annotate(
        "", xy=(cx, y_bot), xytext=(cx, y_top),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.1),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 8.4))

    # Layout rows (top-down y-coordinates).
    y_expert = 0.88
    y_csls = 0.71
    y_z = 0.55
    y_fusion = 0.36
    y_output = 0.14

    expert_w, expert_h = 0.22, 0.13
    csls_w, csls_h = 0.18, 0.09
    z_w, z_h = 0.18, 0.09
    fusion_w, fusion_h = 0.46, 0.13
    output_w, output_h = 0.50, 0.15

    # 1. Three expert boxes at the top.
    for exp in EXPERTS:
        cx = exp["cx"]
        draw_box(ax, cx, y_expert, expert_w, expert_h, EXPERT_FILL, EXPERT_EDGE)
        ax.text(cx, y_expert + 0.030, exp["title"],
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=EXPERT_TITLE)
        ax.text(cx, y_expert - 0.022, exp["body"],
                ha="center", va="center",
                fontsize=9.5, color="#222222")

    # 2. CSLS k=3 row.
    for exp in EXPERTS:
        cx = exp["cx"]
        draw_box(ax, cx, y_csls, csls_w, csls_h, CSLS_FILL, CSLS_EDGE)
        ax.text(cx, y_csls + 0.015, "CSLS",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=CSLS_TITLE)
        ax.text(cx, y_csls - 0.018, "$k = 3$",
                ha="center", va="center",
                fontsize=10, color="#222222")
        vertical_arrow(ax, cx, y_expert - expert_h / 2, y_csls + csls_h / 2)

    # 3. z-score normalization row.
    for exp in EXPERTS:
        cx = exp["cx"]
        draw_box(ax, cx, y_z, z_w, z_h, ZSCORE_FILL, ZSCORE_EDGE)
        ax.text(cx, y_z + 0.015, "z-score",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=ZSCORE_TITLE)
        ax.text(cx, y_z - 0.018, "per-expert",
                ha="center", va="center",
                fontsize=9.5, color="#222222")
        vertical_arrow(ax, cx, y_csls - csls_h / 2, y_z + z_h / 2)

    # 4. Fusion box (centered horizontally).
    fusion_cx = 0.50
    draw_box(ax, fusion_cx, y_fusion, fusion_w, fusion_h, FUSION_FILL, FUSION_EDGE)
    ax.text(fusion_cx, y_fusion + 0.040, "Frozen weighted sum",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=FUSION_TITLE)
    ax.text(fusion_cx, y_fusion + 0.005,
            r"$s_{\mathrm{final}}(i,j) = 0.7\,\tilde{s}_{1,\mathrm{CSLS}} + 0.1\,\tilde{s}_{2,\mathrm{CSLS}} + 0.2\,\tilde{s}_{3,\mathrm{CSLS}}$",
            ha="center", va="center",
            fontsize=11, color="#222222")
    ax.text(fusion_cx, y_fusion - 0.038,
            "weights selected on validation and then frozen",
            ha="center", va="center",
            fontsize=9, style="italic", color="#555555")

    # Arrows from each z-score box into the fusion box, with weight labels.
    for exp in EXPERTS:
        cx = exp["cx"]
        ax.annotate(
            "",
            xy=(cx, y_fusion + fusion_h / 2),
            xytext=(cx, y_z - z_h / 2),
            arrowprops=dict(arrowstyle="->", color=ARROW, linewidth=1.1),
        )
        # Weight label midway down the arrow.
        midy = (y_fusion + fusion_h / 2 + y_z - z_h / 2) / 2
        ax.text(cx + 0.020, midy, exp["weight"],
                ha="left", va="center",
                fontsize=10, color=FUSION_TITLE)

    # 5. Final output box.
    draw_box(ax, 0.50, y_output, output_w, output_h, OUTPUT_FILL, OUTPUT_EDGE)
    ax.text(0.50, y_output + 0.043, "SHARED1000 retrieval ranking",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=OUTPUT_TITLE)
    ax.text(0.50, y_output - 0.005,
            r"CSLS R@1 $= 86.3\%$" "\n"
            r"R@5 $= 98.0\%$ $\cdot$ MRR $= 0.914$ $\cdot$ median rank $1$",
            ha="center", va="center",
            fontsize=10.5, color="#222222")

    vertical_arrow(ax, 0.50,
                   y_fusion - fusion_h / 2,
                   y_output + output_h / 2)

    # Title.
    ax.text(0.50, 1.00,
            "Final exported triple-fusion retrieval architecture",
            ha="center", va="top",
            fontsize=14, fontweight="bold", color="#222222")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_aspect("auto")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
