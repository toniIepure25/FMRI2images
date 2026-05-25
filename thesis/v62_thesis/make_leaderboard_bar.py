"""Render leaderboard_bar.png for the thesis.

Print-friendly horizontal bar chart of SHARED1000 CSLS Recall@1 across the
methodological milestones used in Chapter 5. The triple fusion is rendered
as the final/highest bar; V35 + N1v28a is preserved as a milestone bar.

Numbers come from the paper-grade SHARED1000 endpoint reported in
docs/POD_MODEL_RESULTS_AND_LIMITATIONS.md and
experimental_results/fusion_triple_86/triple_fusion_shared1000_86.json.
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT_PATH = Path(__file__).with_name("leaderboard_bar.png")

# (label, csls_r1_percent, is_final_or_milestone)
# Ordered top-to-bottom so the final exported system sits at the top of the chart.
ROWS = [
    ("Triple fusion V61a$_{\\mathrm{mctta16}}$ + V62a + V66a", 86.3, "final"),
    ("V61a + MC-TTA-16", 83.1, "single"),
    ("V61a (single 197K)", 79.1, "single"),
    ("V64a (continued V61a)", 78.6, "single"),
    ("V35 + N1v28a fixed fusion", 77.2, "milestone"),
    ("V30e compact + rerank fusion", 51.5, "wave"),
    ("V30e compact head", 49.2, "wave"),
    ("V62a (768-D CLS)", 47.5, "complement"),
    ("V66a (768-D ROI pretrain)", 39.1, "complement"),
]

COLOR = {
    "final": "#222222",
    "milestone": "#666666",
    "single": "#999999",
    "wave": "#bdbdbd",
    "complement": "#bdbdbd",
}


def main() -> None:
    labels = [row[0] for row in ROWS][::-1]
    values = [row[1] for row in ROWS][::-1]
    kinds = [row[2] for row in ROWS][::-1]
    colors = [COLOR[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.6)

    for bar, value, kind in zip(bars, values, kinds):
        weight = "bold" if kind == "final" else "normal"
        ax.text(
            value + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight=weight,
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("SHARED1000 CSLS Recall@1 (%)")
    ax.set_title(
        "Progression of SHARED1000 CSLS Recall@1 toward the final triple-fusion system",
        fontsize=10,
        pad=10,
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{int(v)}%"))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
