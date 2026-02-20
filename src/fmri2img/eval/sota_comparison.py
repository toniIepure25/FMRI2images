"""
SOTA Comparison Table for Brain Decoding Literature
====================================================

Centralised repository of published numbers from competing methods
evaluated on the Natural Scenes Dataset (NSD) Subject 01.

These numbers are used to place our method in context without
requiring re-running external codebases.

Sources:
    - MindEye       (Scotti et al., NeurIPS 2023)
    - Brain Diffuser (Ozcelik & VanRullen, 2023)
    - MindEye2      (Scotti et al., 2024)
    - Takagi & Nishimoto (CVPR 2023)
    - UniBrain      (Mai & Zhang, 2023)
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ---- Published results on NSD subj01 (982 test images) ----
# All values taken from the respective papers / supplementary material.
# Keys follow the naming convention used by MindEye2 Table 1.

SOTA_TABLE: Dict[str, Dict[str, Optional[float]]] = {
    "MindEye": {
        "reference": "Scotti et al., NeurIPS 2023",
        "clip_model": "ViT-L/14",
        "test_size": 982,
        "retrieval_r1": 0.052,
        "retrieval_r5": 0.148,
        "twoafc": 0.838,
        "pixcorr": 0.254,
        "ssim": 0.328,
        "alexnet_early": 0.781,
        "alexnet_late": 0.848,
        "clip_i": 0.710,
        "fid": None,
    },
    "Brain_Diffuser": {
        "reference": "Ozcelik & VanRullen, 2023",
        "clip_model": "ViT-L/14",
        "test_size": 982,
        "retrieval_r1": None,
        "retrieval_r5": None,
        "twoafc": 0.840,
        "pixcorr": 0.254,
        "ssim": 0.356,
        "alexnet_early": 0.768,
        "alexnet_late": 0.842,
        "clip_i": 0.700,
        "fid": None,
    },
    "MindEye2": {
        "reference": "Scotti et al., 2024",
        "clip_model": "ViT-L/14",
        "test_size": 982,
        "retrieval_r1": 0.039,
        "retrieval_r5": 0.115,
        "twoafc": 0.929,
        "pixcorr": 0.322,
        "ssim": 0.362,
        "alexnet_early": 0.822,
        "alexnet_late": 0.886,
        "clip_i": 0.737,
        "fid": None,
    },
    "Takagi_Nishimoto": {
        "reference": "Takagi & Nishimoto, CVPR 2023",
        "clip_model": "ViT-L/14",
        "test_size": 982,
        "retrieval_r1": None,
        "retrieval_r5": None,
        "twoafc": 0.780,
        "pixcorr": 0.210,
        "ssim": 0.310,
        "alexnet_early": 0.730,
        "alexnet_late": 0.810,
        "clip_i": 0.650,
        "fid": None,
    },
}


def get_comparison_table(
    our_results: Dict[str, float],
    method_name: str = "Ours (vMF-NCE)",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Return comparison table augmented with our results.

    Args:
        our_results: dict with same keys as SOTA_TABLE entries
        method_name: display name for our method

    Returns:
        combined table
    """
    table = dict(SOTA_TABLE)
    table[method_name] = our_results
    return table


def print_comparison_table(table: Dict[str, Dict[str, Optional[float]]]):
    """Pretty-print a Markdown comparison table."""
    metrics = ["twoafc", "retrieval_r1", "pixcorr", "ssim",
               "alexnet_early", "alexnet_late", "clip_i"]
    header = "| Method | " + " | ".join(m.replace("_", " ").title() for m in metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)

    rows = [header, sep]
    for name, vals in table.items():
        cells = [name]
        for m in metrics:
            v = vals.get(m)
            cells.append(f"{v:.3f}" if v is not None else "-")
        rows.append("| " + " | ".join(cells) + " |")

    print("\n".join(rows))


def to_latex_table(table: Dict[str, Dict[str, Optional[float]]]) -> str:
    """Generate LaTeX tabular for the paper."""
    metrics = ["twoafc", "retrieval_r1", "pixcorr", "ssim",
               "alexnet_early", "alexnet_late", "clip_i"]
    ncols = len(metrics) + 1
    lines = [
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(m.replace("_", r"\_") for m in metrics) + r" \\",
        r"\midrule",
    ]
    for name, vals in table.items():
        cells = [name.replace("_", r"\_")]
        for m in metrics:
            v = vals.get(m)
            cells.append(f"{v:.3f}" if v is not None else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
