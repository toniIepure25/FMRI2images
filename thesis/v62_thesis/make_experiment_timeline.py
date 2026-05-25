"""Render experiment_timeline.png for the thesis.

Print-friendly horizontal milestone timeline. Matches the style of the previous
matplotlib-rendered timeline; the only substantive change is that the rightmost
milestone is now the cross-architecture triple fusion (final exported system),
with V35 plus N1v28a relabelled as the preceding methodological milestone.
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT_PATH = Path(__file__).with_name("experiment_timeline.png")

# (x, label_above (True) or below (False), title, body)
MILESTONES = [
    (0.05, True, "v4", "Bug fixes\npipeline becomes valid"),
    (0.18, False, "v17", "Clean restored baseline\n56.3% CSLS (VAL)"),
    (0.31, True, "v23a", "Wider encoder\n59.0% CSLS (VAL)"),
    (0.44, False, "v26a", "Token targets\n69.6% CSLS"),
    (0.57, True, "v28a", "Dual-head MindEye-style\n70.3% CSLS"),
    (0.70, False, "V30/V32", "Compact+rerank wave\nfusion validated"),
    (0.82, True, "V35 + N1v28a", "Fixed-fusion milestone\n77.2% SHARED1000 CSLS R@1"),
    (0.95, False, "Triple fusion",
        "V61a$_{\\mathrm{mctta16}}$ + V62a + V66a\nFinal exported system\n86.3% SHARED1000 CSLS R@1"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.0))

    ax.hlines(0.5, 0.0, 1.0, colors="#333333", linewidth=2.0)

    for x, above, title, body in MILESTONES:
        ax.plot([x], [0.5], marker="o", markersize=11,
                markerfacecolor="#1f77b4", markeredgecolor="#1f77b4")
        if above:
            ax.vlines(x, 0.5, 0.72, colors="#888888", linewidth=0.8)
            text_y = 0.74
            va = "bottom"
        else:
            ax.vlines(x, 0.28, 0.5, colors="#888888", linewidth=0.8)
            text_y = 0.26
            va = "top"
        is_final = "Final exported" in body
        weight = "bold" if is_final else "normal"
        ax.text(x, text_y, f"{title}\n{body}", ha="center", va=va,
                fontsize=10, fontweight=weight)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    ax.set_title("Main experimental milestones of the project",
                 fontsize=14, fontweight="bold", pad=18)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
