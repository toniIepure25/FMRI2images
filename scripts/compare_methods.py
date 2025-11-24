#!/usr/bin/env python3
"""
Compare reconstruction results across different methods.
"""
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_evaluation_results(eval_dir: Path) -> dict:
    """Load evaluation results from a directory."""
    summary_file = eval_dir / "summary.json"
    metrics_file = eval_dir / "metrics_per_sample.csv"
    
    if not summary_file.exists():
        logger.warning(f"No summary found in {eval_dir}")
        return None
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
        summary['n_samples_with_gt'] = len(metrics_df)
    
    summary['method'] = eval_dir.parent.name
    return summary


def compare_methods(eval_dirs: list[Path], output_dir: Path):
    """Compare multiple reconstruction methods."""
    logger.info("=" * 80)
    logger.info("COMPARING RECONSTRUCTION METHODS")
    logger.info("=" * 80)
    
    # Load all results
    results = []
    for eval_dir in eval_dirs:
        result = load_evaluation_results(eval_dir)
        if result:
            results.append(result)
    
    if not results:
        logger.error("No valid evaluation results found!")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Sort by mean SSIM (higher is better)
    df = df.sort_values('mean_ssim', ascending=False)
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 80)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "SSIM↑", "PSNR↑", "LPIPS↓", "N"
    ))
    print("-" * 70)
    
    for _, row in df.iterrows():
        ssim = row.get('mean_ssim', 0)
        psnr = row.get('mean_psnr', 0)
        lpips = row.get('mean_lpips', 0)
        n = row.get('n_samples_with_gt', row.get('n_samples', 0))
        
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
            row['method'][:24], ssim, psnr, lpips, n
        ))
    
    # Save comparison table
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "method_comparison.csv", index=False)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # SSIM comparison
    ax = axes[0]
    ax.barh(df['method'], df['mean_ssim'])
    ax.set_xlabel('SSIM (higher is better)')
    ax.set_title('Structural Similarity')
    ax.set_xlim(0, 1)
    
    # PSNR comparison
    ax = axes[1]
    ax.barh(df['method'], df['mean_psnr'])
    ax.set_xlabel('PSNR (dB) (higher is better)')
    ax.set_title('Peak Signal-to-Noise Ratio')
    
    # LPIPS comparison
    ax = axes[2]
    ax.barh(df['method'], df['mean_lpips'])
    ax.set_xlabel('LPIPS (lower is better)')
    ax.set_title('Perceptual Similarity')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n✓ Comparison saved to {output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare reconstruction methods")
    parser.add_argument("--eval-dirs", nargs="+", required=True, help="Evaluation directories to compare")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    eval_dirs = [Path(d) for d in args.eval_dirs]
    compare_methods(eval_dirs, Path(args.output_dir))


if __name__ == "__main__":
    main()
