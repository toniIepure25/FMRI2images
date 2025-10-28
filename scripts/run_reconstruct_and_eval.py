#!/usr/bin/env python3
"""
Orchestrator: Generate reconstructions + evaluate in correct CLIP space.

This script combines decode_diffusion.py and eval_reconstruction.py into a single
workflow that ensures evaluation is performed in the same CLIP space as generation:
- No adapter → 512-D evaluation (ViT-B/32)
- With adapter → 768/1024-D evaluation (target CLIP)

Outputs:
- Reconstructed images (PNG)
- Evaluation metrics (CSV, JSON)
- Visualization grid (PNG)
- Thesis-ready Markdown summary

Usage:
    # No adapter (512-D)
    python scripts/run_reconstruct_and_eval.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --output-dir outputs/recon/subj01/auto \\
        --report-dir outputs/reports/subj01 \\
        --limit 64

    # With adapter (1024-D)
    python scripts/run_reconstruct_and_eval.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --use-adapter \\
        --adapter checkpoints/clip_adapter/subj01/adapter.pt \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --output-dir outputs/recon/subj01/auto_adapter \\
        --report-dir outputs/reports/subj01 \\
        --limit 64
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def print_banner(text: str) -> None:
    """Print a clear section banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def check_sd_cache(model_id: str = "runwayml/stable-diffusion-v1-5") -> bool:
    """
    Check if SD model is cached by running check_hf_cache.py.
    
    Returns:
        True if model is cached, False otherwise.
    """
    script_path = Path(__file__).parent / "check_hf_cache.py"
    if not script_path.exists():
        print(f"Warning: {script_path} not found, skipping cache check.")
        return True  # Assume cached if check script missing
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse output to check if model is cached
        # check_hf_cache.py should print "✓" if cached, "✗" if not
        if "✓" in result.stdout and model_id in result.stdout:
            return True
        elif "✗" in result.stdout and model_id in result.stdout:
            return False
        
        # If unclear, assume not cached to be safe
        return False
    except Exception as e:
        print(f"Warning: Failed to check cache: {e}")
        return True  # Assume cached on error to continue


def load_adapter_metadata(adapter_path: Path) -> Dict[str, Any]:
    """
    Load adapter metadata to get target dimension.
    
    Returns:
        Dictionary with 'target_dim' and other metadata.
    """
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    ckpt = torch.load(adapter_path, map_location="cpu")
    
    if "metadata" not in ckpt:
        raise ValueError(f"Adapter missing metadata: {adapter_path}")
    
    metadata = ckpt["metadata"]
    if "target_dim" not in metadata:
        raise ValueError(f"Adapter metadata missing target_dim: {adapter_path}")
    
    return metadata


def run_decode(
    encoder: str,
    ckpt_path: Path,
    output_dir: Path,
    limit: int,
    steps: int,
    device: str,
    use_adapter: bool,
    adapter_path: Optional[Path],
    model_id: Optional[str],
    clip_target_dim: Optional[int],
    subject: str,
    index_root: Optional[Path],
    index_file: Optional[Path],
) -> int:
    """
    Run decode_diffusion.py to generate reconstructions.
    
    Returns:
        Exit code (0 = success).
    """
    print_banner("Step 1/3: Generate Reconstructions")
    
    script_path = Path(__file__).parent / "decode_diffusion.py"
    if not script_path.exists():
        print(f"ERROR: decode_diffusion.py not found at {script_path}")
        return 1
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--encoder", encoder,
        "--ckpt", str(ckpt_path),
        "--output-dir", str(output_dir),
        "--limit", str(limit),
        "--steps", str(steps),
        "--device", device,
        "--subject", subject,
    ]
    
    # Add index specification
    if index_file:
        cmd.extend(["--index-file", str(index_file)])
    elif index_root:
        cmd.extend(["--index-root", str(index_root)])
    
    # Add adapter if specified
    if use_adapter:
        if not adapter_path:
            print("ERROR: --use-adapter requires --adapter")
            return 1
        if not model_id:
            print("ERROR: --use-adapter requires --model-id")
            return 1
        
        cmd.extend([
            "--clip-adapter", str(adapter_path),
            "--model-id", model_id,
        ])
        
        if clip_target_dim:
            cmd.extend(["--clip-target-dim", str(clip_target_dim)])
    
    print("Command:", " ".join(cmd))
    print()
    
    # Run decode
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nERROR: decode_diffusion.py failed with exit code {result.returncode}")
        return result.returncode
    
    print(f"\n✓ Decoding complete: {output_dir}")
    return 0


def run_eval(
    subject: str,
    recon_dir: Path,
    clip_cache: Path,
    report_dir: Path,
    use_adapter: bool,
    model_id: Optional[str],
    index_root: Optional[Path],
    index_file: Optional[Path],
    limit: int,
) -> int:
    """
    Run eval_reconstruction.py to evaluate generated images.
    
    Returns:
        Exit code (0 = success).
    """
    print_banner("Step 2/3: Evaluate Reconstructions")
    
    script_path = Path(__file__).parent / "eval_reconstruction.py"
    if not script_path.exists():
        print(f"ERROR: eval_reconstruction.py not found at {script_path}")
        return 1
    
    # Create report directory
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    csv_path = report_dir / "recon_eval.csv"
    json_path = report_dir / "recon_eval.json"
    fig_path = report_dir / "recon_grid.png"
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--subject", subject,
        "--recon-dir", str(recon_dir),
        "--clip-cache", str(clip_cache),
        "--out-csv", str(csv_path),
        "--out-json", str(json_path),
        "--out-fig", str(fig_path),
        "--limit", str(limit),
    ]
    
    # Add index specification
    if index_file:
        cmd.extend(["--index-file", str(index_file)])
    elif index_root:
        cmd.extend(["--index-root", str(index_root)])
    
    # Add adapter settings if specified
    if use_adapter:
        if not model_id:
            print("ERROR: --use-adapter requires --model-id for evaluation")
            return 1
        
        cmd.extend([
            "--use-adapter",
            "--model-id", model_id,
        ])
    
    print("Command:", " ".join(cmd))
    print()
    
    # Run eval
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nERROR: eval_reconstruction.py failed with exit code {result.returncode}")
        return result.returncode
    
    print(f"\n✓ Evaluation complete: {report_dir}")
    return 0


def create_markdown_summary(
    report_dir: Path,
    subject: str,
    encoder: str,
    use_adapter: bool,
    model_id: Optional[str],
    clip_dim: int,
    ckpt_path: Path,
    adapter_path: Optional[Path],
    recon_dir: Path,
    limit: int,
) -> int:
    """
    Create thesis-ready Markdown summary from evaluation results.
    
    Returns:
        Exit code (0 = success).
    """
    print_banner("Step 3/3: Generate Markdown Summary")
    
    json_path = report_dir / "recon_eval.json"
    csv_path = report_dir / "recon_eval.csv"
    fig_path = report_dir / "recon_grid.png"
    
    if not json_path.exists():
        print(f"ERROR: Evaluation JSON not found: {json_path}")
        return 1
    
    # Load evaluation results
    with open(json_path) as f:
        results = json.load(f)
    
    # Extract metrics
    clipscore = results.get("clipscore", {})
    retrieval = results.get("retrieval", {})
    ranking = results.get("ranking", {})
    
    # Generate Markdown
    md_path = report_dir / "recon_eval_summary.md"
    
    with open(md_path, "w") as f:
        # Header
        f.write("# Reconstruction Evaluation Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"- **Subject:** `{subject}`\n")
        f.write(f"- **Encoder:** `{encoder}`\n")
        f.write(f"- **Checkpoint:** `{ckpt_path}`\n")
        f.write(f"- **Adapter:** {'Yes' if use_adapter else 'No'}")
        if use_adapter:
            f.write(f" (`{adapter_path}`)")
        f.write("\n")
        f.write(f"- **Model ID:** `{model_id or 'N/A (default)'}`\n")
        f.write(f"- **CLIP Space:** **{clip_dim}-D** ")
        f.write(f"({'ViT-B/32' if clip_dim == 512 else 'target CLIP'})\n")
        f.write(f"- **Test Samples:** {results.get('n_samples', 'N/A')} / {results.get('n_test_total', 'N/A')}\n")
        f.write(f"- **Output Directory:** `{recon_dir}`\n\n")
        
        # Important note
        f.write("---\n\n")
        f.write("**Note:** Evaluated in **")
        if clip_dim == 512:
            f.write("512-D CLIP space (ViT-B/32)")
        elif clip_dim == 768:
            f.write("768-D CLIP space (target for SD-1.5)")
        elif clip_dim == 1024:
            f.write("1024-D CLIP space (target for SD-2.1)")
        else:
            f.write(f"{clip_dim}-D CLIP space")
        f.write("** — matched to generation space.\n\n")
        
        # Metrics table
        f.write("## Results\n\n")
        f.write("| Metric | Value | Interpretation |\n")
        f.write("|--------|-------|----------------|\n")
        
        # CLIPScore
        cs_mean = clipscore.get("mean", 0.0)
        cs_std = clipscore.get("std", 0.0)
        if cs_mean >= 0.7:
            cs_qual = "Excellent"
        elif cs_mean >= 0.5:
            cs_qual = "Good"
        elif cs_mean >= 0.3:
            cs_qual = "Moderate"
        else:
            cs_qual = "Poor"
        f.write(f"| **CLIPScore** | {cs_mean:.3f} ± {cs_std:.3f} | {cs_qual} |\n")
        
        # Retrieval@K
        r1 = retrieval.get("R@1", 0.0)
        r5 = retrieval.get("R@5", 0.0)
        r10 = retrieval.get("R@10", 0.0)
        f.write(f"| **R@1** | {r1:.3f} | {r1*100:.1f}% top-1 correct |\n")
        f.write(f"| **R@5** | {r5:.3f} | {r5*100:.1f}% in top-5 |\n")
        f.write(f"| **R@10** | {r10:.3f} | {r10*100:.1f}% in top-10 |\n")
        
        # Ranking
        mean_rank = ranking.get("mean_rank", 0.0)
        median_rank = ranking.get("median_rank", 0.0)
        mrr = ranking.get("mrr", 0.0)
        f.write(f"| **Mean Rank** | {mean_rank:.2f} | Avg position in gallery |\n")
        f.write(f"| **Median Rank** | {median_rank:.1f} | Median position |\n")
        f.write(f"| **MRR** | {mrr:.3f} | Mean reciprocal rank |\n")
        
        # Quality interpretation
        f.write("\n### Quality Assessment\n\n")
        if cs_mean >= 0.6 and r1 >= 0.3:
            quality = "**Very Good** — strong semantic alignment with ground truth"
        elif cs_mean >= 0.5 and r1 >= 0.2:
            quality = "**Good** — reasonable semantic preservation"
        elif cs_mean >= 0.4:
            quality = "**Moderate** — some semantic similarity preserved"
        else:
            quality = "**Poor** — low semantic alignment"
        f.write(f"{quality}\n\n")
        
        # Output files
        f.write("## Output Files\n\n")
        f.write(f"- **CSV (per-sample):** `{csv_path}`\n")
        f.write(f"- **JSON (aggregate):** `{json_path}`\n")
        f.write(f"- **Visualization Grid:** `{fig_path}`\n")
        f.write(f"- **Generated Images:** `{recon_dir}/`\n\n")
        
        # Citation
        f.write("## Methodology\n\n")
        f.write("**Metrics:**\n")
        f.write("- **CLIPScore:** Cosine similarity between generated and GT image embeddings (Hessel et al. 2021)\n")
        f.write("- **Retrieval@K:** Proportion of samples where GT appears in top-K retrievals\n")
        f.write("- **Mean Rank:** Average position of GT in ranked retrieval list\n")
        f.write("- **MRR:** Mean reciprocal rank (1/rank)\n\n")
        
        f.write("**CLIP Space Consistency:**\n")
        if use_adapter:
            f.write(f"Generated with CLIP adapter → evaluated in {clip_dim}-D target space (consistent).\n")
        else:
            f.write(f"Generated without adapter → evaluated in 512-D space (consistent).\n")
        f.write("\n")
        
        # Comparison context
        f.write("## Baseline Comparison\n\n")
        f.write("| Method | CLIPScore | R@1 | Notes |\n")
        f.write("|--------|-----------|-----|-------|\n")
        f.write("| **NN Retrieval** | 0.90-0.95 | 0.70-0.80 | Strong baseline (existing images) |\n")
        f.write("| **Diffusion (literature)** | 0.60-0.80 | 0.30-0.60 | Novel generation |\n")
        f.write(f"| **This Run** | {cs_mean:.2f} | {r1:.2f} | Current results |\n\n")
        
        f.write("*Note: Lower scores for diffusion-based methods don't necessarily indicate worse quality—they reflect the trade-off between semantic similarity and perceptual novelty.*\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write(f"*Generated by `run_reconstruct_and_eval.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"✓ Markdown summary created: {md_path}")
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}\n")
    print(f"CLIPScore: {cs_mean:.3f} ± {cs_std:.3f}")
    print(f"R@1:       {r1:.3f}  ({r1*100:.1f}%)")
    print(f"R@5:       {r5:.3f}  ({r5*100:.1f}%)")
    print(f"R@10:      {r10:.3f}  ({r10*100:.1f}%)")
    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"MRR:       {mrr:.3f}")
    print(f"\n{'='*80}\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--subject", type=str, default="subj01",
                        help="NSD subject ID (default: subj01)")
    parser.add_argument("--encoder", type=str, required=True,
                        choices=["ridge", "mlp"],
                        help="Encoder type: ridge or mlp")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to encoder checkpoint")
    parser.add_argument("--clip-cache", type=Path, required=True,
                        help="Path to CLIP embeddings cache (parquet)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for generated images")
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="Directory for evaluation reports")
    
    # Index
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument("--index-root", type=Path,
                             help="Root directory containing subject-specific indices")
    index_group.add_argument("--index-file", type=Path,
                             help="Direct path to index parquet file")
    
    # Optional
    parser.add_argument("--limit", type=int, default=64,
                        help="Number of test samples to process (default: 64)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for inference (default: auto)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion steps (default: 50)")
    
    # Adapter
    parser.add_argument("--use-adapter", action="store_true",
                        help="Use CLIP adapter for dimension alignment")
    parser.add_argument("--adapter", type=Path,
                        help="Path to CLIP adapter checkpoint (required if --use-adapter)")
    parser.add_argument("--model-id", type=str,
                        help="Diffusion model ID (e.g., stabilityai/stable-diffusion-2-1)")
    
    args = parser.parse_args()
    
    # Validation
    if args.use_adapter and not args.adapter:
        print("ERROR: --use-adapter requires --adapter")
        return 1
    
    if args.use_adapter and not args.model_id:
        print("ERROR: --use-adapter requires --model-id")
        return 1
    
    if not args.ckpt.exists():
        print(f"ERROR: Encoder checkpoint not found: {args.ckpt}")
        return 1
    
    if not args.clip_cache.exists():
        print(f"ERROR: CLIP cache not found: {args.clip_cache}")
        return 1
    
    # Load adapter metadata if using adapter
    clip_target_dim = None
    if args.use_adapter:
        try:
            metadata = load_adapter_metadata(args.adapter)
            clip_target_dim = metadata["target_dim"]
            print(f"✓ Adapter metadata loaded: target_dim={clip_target_dim}")
        except Exception as e:
            print(f"ERROR: Failed to load adapter metadata: {e}")
            return 1
    
    # Determine CLIP dimension for evaluation
    clip_dim = clip_target_dim if args.use_adapter else 512
    
    # Print configuration
    print_banner("Reconstruct & Evaluate Workflow")
    print(f"Subject:       {args.subject}")
    print(f"Encoder:       {args.encoder}")
    print(f"Checkpoint:    {args.ckpt}")
    print(f"Adapter:       {'Yes' if args.use_adapter else 'No'}")
    if args.use_adapter:
        print(f"  Path:        {args.adapter}")
        print(f"  Model ID:    {args.model_id}")
        print(f"  Target Dim:  {clip_target_dim}")
    print(f"CLIP Space:    {clip_dim}-D")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Report Dir:    {args.report_dir}")
    print(f"Limit:         {args.limit}")
    print(f"Steps:         {args.steps}")
    print()
    
    # Check SD cache
    model_id_for_cache = args.model_id or "runwayml/stable-diffusion-v1-5"
    print(f"Checking SD cache for: {model_id_for_cache}")
    
    if not check_sd_cache(model_id_for_cache):
        print()
        print("=" * 80)
        print("  WARNING: Stable Diffusion model not cached!")
        print("=" * 80)
        print()
        print(f"Model '{model_id_for_cache}' does not appear to be cached locally.")
        print()
        print("To download the model, run:")
        print(f"  make download-sd MODEL={model_id_for_cache}")
        print()
        print("Or manually:")
        print(f"  python scripts/download_sd_model.py --model-id {model_id_for_cache}")
        print()
        print("=" * 80)
        print()
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 1
    else:
        print(f"✓ SD model cached: {model_id_for_cache}\n")
    
    # Step 1: Decode
    exit_code = run_decode(
        encoder=args.encoder,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        limit=args.limit,
        steps=args.steps,
        device=args.device,
        use_adapter=args.use_adapter,
        adapter_path=args.adapter,
        model_id=args.model_id,
        clip_target_dim=clip_target_dim,
        subject=args.subject,
        index_root=args.index_root,
        index_file=args.index_file,
    )
    
    if exit_code != 0:
        return exit_code
    
    # Step 2: Evaluate
    exit_code = run_eval(
        subject=args.subject,
        recon_dir=args.output_dir,
        clip_cache=args.clip_cache,
        report_dir=args.report_dir,
        use_adapter=args.use_adapter,
        model_id=args.model_id,
        index_root=args.index_root,
        index_file=args.index_file,
        limit=args.limit,
    )
    
    if exit_code != 0:
        return exit_code
    
    # Step 3: Create Markdown summary
    exit_code = create_markdown_summary(
        report_dir=args.report_dir,
        subject=args.subject,
        encoder=args.encoder,
        use_adapter=args.use_adapter,
        model_id=args.model_id,
        clip_dim=clip_dim,
        ckpt_path=args.ckpt,
        adapter_path=args.adapter,
        recon_dir=args.output_dir,
        limit=args.limit,
    )
    
    if exit_code != 0:
        return exit_code
    
    print(f"\n{'='*80}")
    print("  ALL STEPS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📁 Images:  {args.output_dir}")
    print(f"📊 Reports: {args.report_dir}")
    print(f"📝 Summary: {args.report_dir}/recon_eval_summary.md")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
