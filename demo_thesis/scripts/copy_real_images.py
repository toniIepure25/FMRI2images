#!/usr/bin/env python3
"""Copy real evaluation images from results/final_outputs into public/assets/cases.

Reads data/demo_cases.json, resolves sources under results/final_outputs/, copies
standardized filenames, updates JSON paths, and mirrors JSON to public/data/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# demo_thesis/ (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "final_outputs"
BEST = RESULTS / "best_system"
QUAL_SHARED = BEST / "qualitatives" / "shared1000"
RECON_SHARED = BEST / "reconstructions" / "shared1000"
ADDON_ROOT = RESULTS / "true_diffusion_addon" / "reconstructions"
ADDON_BUCKETS = ("perfect", "good", "near_good", "hard")
DEMO_CASES_SRC = ROOT / "data" / "demo_cases.json"
DEMO_CASES_PUBLIC = ROOT / "public" / "data" / "demo_cases.json"


def find_qualitative_panel(case_id: str) -> Path | None:
    for sub in ("best_cases", "medium_cases", "hard_cases"):
        p = QUAL_SHARED / sub / f"{case_id}.png"
        if p.is_file():
            return p
    return None


def find_addon_dir(case_id: str) -> Path | None:
    stem = f"{case_id}_ground_truth.png"
    for bucket in ADDON_BUCKETS:
        p = ADDON_ROOT / bucket / stem
        if p.is_file():
            return ADDON_ROOT / bucket
    return None


def copy_if_src(dst: Path, src: Path | None, label: str, log: list[str]) -> bool:
    if src is None or not src.is_file():
        log.append(f"  {label}: (no source)")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log.append(f"  {label}: <- {src.relative_to(ROOT)}")
    return True


def main() -> int:
    if not DEMO_CASES_SRC.is_file():
        print(f"Missing {DEMO_CASES_SRC}", file=sys.stderr)
        return 1

    with open(DEMO_CASES_SRC, encoding="utf-8") as f:
        cases: list[dict] = json.load(f)

    all_logs: list[str] = []

    for case in cases:
        case_id = case["id"]
        dest_dir = ROOT / "public" / "assets" / "cases" / case_id
        log: list[str] = [f"{case_id}:"]

        addon_dir = find_addon_dir(case_id)
        qual = find_qualitative_panel(case_id)
        comp_recon = RECON_SHARED / f"{case_id}_comparison.png"
        top1 = RECON_SHARED / f"{case_id}_top1_reconstruction.png"

        if addon_dir:
            log.append(f"  (addon bucket: {addon_dir.name})")

        # target.png — ground truth from diffusion addon when available
        gt = addon_dir / f"{case_id}_ground_truth.png" if addon_dir else None
        if not copy_if_src(dest_dir / "target.png", gt, "target.png", log):
            existing = dest_dir / "target.png"
            if existing.is_file():
                log.append(f"  target.png: (kept existing, no addon GT)")

        # reconstruction.png — diffusion_final preferred, else best_system top-1 recon
        recon_src = None
        if addon_dir:
            df = addon_dir / f"{case_id}_diffusion_final.png"
            if df.is_file():
                recon_src = df
        if recon_src is None and top1.is_file():
            recon_src = top1
        copy_if_src(dest_dir / "reconstruction.png", recon_src, "reconstruction.png", log)

        # retrieved_1.png — only from addon (best_system has no separate top-1 tile)
        ret1 = addon_dir / f"{case_id}_retrieval_top1.png" if addon_dir else None
        if not copy_if_src(dest_dir / "retrieved_1.png", ret1, "retrieved_1.png", log):
            if (dest_dir / "retrieved_1.png").is_file():
                log.append(f"  retrieved_1.png: (kept existing)")

        # topk_strip, diffusion_prior, diffusion_final — addon only
        strip = addon_dir / f"{case_id}_topk_strip.png" if addon_dir else None
        prior = addon_dir / f"{case_id}_diffusion_prior.png" if addon_dir else None
        final = addon_dir / f"{case_id}_diffusion_final.png" if addon_dir else None

        copy_if_src(dest_dir / "topk_strip.png", strip, "topk_strip.png", log)
        copy_if_src(dest_dir / "diffusion_prior.png", prior, "diffusion_prior.png", log)
        if not copy_if_src(dest_dir / "diffusion_final.png", final, "diffusion_final.png", log):
            # Expose final under standard name using same pixels as reconstruction
            if recon_src is not None and recon_src.is_file():
                copy_if_src(
                    dest_dir / "diffusion_final.png",
                    recon_src,
                    "diffusion_final.png (alias of recon source)",
                    log,
                )

        # comparison_panel — qualitatives mosaic first, else shared1000 comparison
        panel_src = qual if qual and qual.is_file() else comp_recon if comp_recon.is_file() else None
        copy_if_src(dest_dir / "comparison_panel.png", panel_src, "comparison_panel.png", log)

        # --- JSON paths ---
        base = f"/assets/cases/{case_id}"
        case["targetImage"] = f"{base}/target.png"
        case["reconstructionImage"] = f"{base}/reconstruction.png"
        if case.get("retrievedImages"):
            for item in case["retrievedImages"]:
                if item.get("rank") == 1:
                    item["image"] = f"{base}/retrieved_1.png"
                    break
        case["comparisonPanel"] = f"{base}/comparison_panel.png"
        case["topkStrip"] = f"{base}/topk_strip.png"
        case["diffusionPrior"] = f"{base}/diffusion_prior.png"
        case["diffusionFinal"] = f"{base}/diffusion_final.png"

        all_logs.extend(log)
        all_logs.append("")

    DEMO_CASES_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_CASES_SRC, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
        f.write("\n")
    shutil.copy2(DEMO_CASES_SRC, DEMO_CASES_PUBLIC)

    print("Copy + JSON update complete.\n")
    print("\n".join(all_logs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
