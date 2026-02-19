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

Usage Examples:

    # Basic: No adapter (512-D), matched gallery
    python scripts/run_reconstruct_and_eval.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --output-dir outputs/recon/subj01/mlp_512d \\
        --report-dir outputs/reports/subj01 \\
        --limit 64

    # With adapter (1024-D), test gallery
    python scripts/run_reconstruct_and_eval.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --use-adapter \\
        --adapter checkpoints/clip_adapter/subj01/adapter.pt \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --output-dir outputs/recon/subj01/mlp_1024d \\
        --report-dir outputs/reports/subj01 \\
        --gallery test \\
        --image-source hdf5 \\
        --limit 32
        
    # All galleries mode
    python scripts/run_reconstruct_and_eval.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --output-dir outputs/recon/subj01/mlp_all \\
        --report-dir outputs/reports/subj01 \\
        --all-galleries \\
        --limit 32

Makefile Integration:
    make reconstruct-and-eval SUBJ=subj01 ENCODER=mlp GALLERY=test
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.nsd_index_builder import NSDIndexBuilder
from fmri2img.data.torch_dataset import NSDIterableDataset
from fmri2img.generation.decode_diffusion import (
    setup_diffusion_pipeline,
    generate_image_from_clip_embedding,
)
from fmri2img.inference.pipeline import SamplingConfig, run_probabilistic_trials, TrialResult
from fmri2img.models.encoders import load_probabilistic_encoder
from fmri2img.utils.clip_utils import load_clip_model, encode_images
from fmri2img.eval.stimulus import parse_stimulus_id_from_filename, resolve_from_index

# Import manifest utilities for reproducibility tracking
from fmri2img.utils.manifest import gather_env_info, write_manifest, hash_file


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
    Load adapter metadata to get target dimension using robust loader.
    
    Returns:
        Dictionary with 'target_dim' and other metadata.
    """
    from fmri2img.models.clip_adapter import load_adapter
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    try:
        _, metadata = load_adapter(str(adapter_path), map_location="cpu")
        
        # Get target_dim (prefer target_dim, fallback to out_dim)
        target_dim = metadata.get("target_dim", metadata.get("out_dim"))
        if target_dim is None:
            raise ValueError(f"Adapter metadata missing target_dim: {adapter_path}")
        
        # Ensure it's in metadata
        metadata["target_dim"] = target_dim
        
        return metadata
    except Exception as e:
        raise ValueError(f"Failed to load adapter metadata from {adapter_path}: {e}")


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
    preproc_enabled: bool,
    preproc_path: Optional[str],
) -> Tuple[int, Optional[int]]:
    """
    Run decode_diffusion.py to generate reconstructions.
    
    Returns:
        Tuple of (exit_code, detected_target_dim)
    """
    print_banner("Step 1/3: Generate Reconstructions")
    
    script_path = Path(__file__).parent / "decode_diffusion.py"
    if not script_path.exists():
        # Fallback to top-level scripts/ for backwards compatibility
        alt_path = Path(__file__).resolve().parents[3] / "scripts" / "decode_diffusion.py"
        if alt_path.exists():
            script_path = alt_path
        else:
            print(f"ERROR: decode_diffusion.py not found at {script_path} or {alt_path}")
            return 1, None
    
    # Auto-detect target_dim from adapter if using adapter
    detected_target_dim = None
    if use_adapter and adapter_path:
        try:
            metadata = load_adapter_metadata(adapter_path)
            detected_target_dim = metadata["target_dim"]
            print(f"   Auto-detected adapter target_dim: {detected_target_dim}D")
            
            # Override clip_target_dim if not explicitly set
            if clip_target_dim is None:
                clip_target_dim = detected_target_dim
            elif clip_target_dim != detected_target_dim:
                print(f"   ⚠️  WARNING: --clip-target-dim={clip_target_dim} but adapter uses {detected_target_dim}D")
                print(f"   Using adapter's dimension: {detected_target_dim}D")
                clip_target_dim = detected_target_dim
        except Exception as e:
            print(f"ERROR: Failed to load adapter metadata: {e}")
            return 1, None
    
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
            return 1, None
        if not model_id:
            print("ERROR: --use-adapter requires --model-id")
            return 1, None
        
        cmd.extend([
            "--clip-adapter", str(adapter_path),
            "--model-id", model_id,
        ])
        
        if clip_target_dim:
            cmd.extend(["--clip-target-dim", str(clip_target_dim)])
    
    # Add preprocessing if enabled
    if preproc_enabled:
        cmd.append("--use-preproc")
        if preproc_path:
            cmd.extend(["--preproc-dir", preproc_path])
    
    print("Command:", " ".join(cmd))
    print()
    
    # Run decode
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nERROR: decode_diffusion.py failed with exit code {result.returncode}")
        return result.returncode, None
    
    print(f"\n✓ Decoding complete: {output_dir}")
    return 0, detected_target_dim


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
    gallery: str,
    image_source: str,
    nsd_hdf5: Optional[Path],
) -> int:
    """
    Run eval_reconstruction.py to evaluate generated images for a specific gallery.
    
    Args:
        gallery: Retrieval gallery type (matched, test, all)
        image_source: Source for visualization images (auto, s3, png, hdf5)
        nsd_hdf5: Optional path to NSD HDF5 file
        
    Returns:
        Exit code (0 = success).
    """
    print_banner(f"Evaluate Reconstructions (gallery={gallery})")
    
    script_path = Path(__file__).parent / "eval_reconstruction.py"
    if not script_path.exists():
        print(f"ERROR: eval_reconstruction.py not found at {script_path}")
        return 1
    
    # Create report directory
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Output paths with gallery suffix
    csv_path = report_dir / f"recon_eval_{gallery}.csv"
    json_path = report_dir / f"recon_eval_{gallery}.json"
    fig_path = report_dir / f"recon_grid_{gallery}.png"
    
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
        "--gallery", gallery,
        "--image-source", image_source,
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
    
    # Add NSD HDF5 if provided
    if nsd_hdf5:
        cmd.extend(["--nsd-hdf5", str(nsd_hdf5)])
    
    print("Command:", " ".join(cmd))
    print()
    
    # Run eval
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nERROR: eval_reconstruction.py failed with exit code {result.returncode}")
        return result.returncode
    
    print(f"\n✓ Evaluation complete ({gallery}): {report_dir}")
    return 0


def build_sample_entries(
    recon_dir: Path,
    subject: str,
    limit: int,
    index_root: Optional[Path],
    index_file: Optional[Path],
    steps: int,
    guidance: float,
    sampling_policy: str,
    seed_base: Optional[int],
) -> list[dict]:
    """Create per-sample manifest entries mapping recon filenames to stimulus IDs."""
    img_root = recon_dir if recon_dir.name != "images" else recon_dir
    images = sorted(img_root.glob("*.png"))
    if (recon_dir / "images").exists():
        images = sorted((recon_dir / "images").glob("*.png"))
    if limit:
        images = images[:limit]

    index_df = None
    if index_file:
        index_df = read_subject_index(str(index_file), subject, allow_fallback_index=True)
    elif index_root:
        index_df = read_subject_index(str(index_root), subject, allow_fallback_index=True)

    entries = []
    for i, path in enumerate(images):
        parsed_id = parse_stimulus_id_from_filename(path)
        nsd_id = None
        nsdId = None
        if parsed_id is not None and index_df is not None:
            stim = resolve_from_index(parsed_id, index_df)
            if stim:
                nsd_id = stim.nsd_id
                nsdId = stim.nsdId
        entries.append({
            "recon_filename": path.name,
            "recon_path": str(path),
            "parsed_id_from_filename": parsed_id,
            "nsd_id": nsd_id,
            "nsdId": nsdId,
            "seed": seed_base + i if seed_base is not None else None,
            "steps": steps,
            "guidance_scale": guidance,
            "sampling_policy": sampling_policy,
            "trial_index": i,
        })
    return entries


def append_samples_to_manifest(manifest_path: Path, samples: list[dict]) -> None:
    if not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        return
    data["samples"] = samples
    manifest_path.write_text(json.dumps(data, indent=2))


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
    gallery: str = "matched",
) -> int:
    """
    Create thesis-ready Markdown summary from evaluation results.
    
    Args:
        gallery: Gallery type used for primary summary
        
    Returns:
        Exit code (0 = success).
    """
    print_banner("Step 3/3: Generate Markdown Summary")
    
    json_path = report_dir / f"recon_eval_{gallery}.json"
    csv_path = report_dir / f"recon_eval_{gallery}.csv"
    fig_path = report_dir / f"recon_grid_{gallery}.png"
    
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


def write_trial_results(
    trial_results: List[TrialResult],
    output_dir: Path,
    csv_name: str = "trial_results.csv",
    parquet_name: Optional[str] = "trial_results.parquet",
) -> Tuple[Path, Optional[Path]]:
    """Persist TrialResult rows to CSV (and optional parquet)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    parquet_path = output_dir / parquet_name if parquet_name else None

    rows = [asdict(tr) for tr in trial_results]
    fieldnames = list(rows[0].keys()) if rows else []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if parquet_path is not None:
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    return csv_path, parquet_path


def run_calibration_assets(trial_csv: Path, run_dir: Path) -> None:
    """Run calibration post-processing to produce paper assets."""
    try:
        from fmri2img.eval.calibration import generate_calibration_assets

        generate_calibration_assets(trial_csv, run_dir)
    except Exception as e:
        print(f"⚠️  Calibration assets skipped: {e}")


def probabilistic_reconstruct(
    args: argparse.Namespace,
    galleries: list[str],
    preproc_enabled: bool,
    preproc_path: Optional[Path],
    clip_dim: int,
) -> Tuple[Path, Path]:
    """End-to-end probabilistic reconstruction with adaptive K and logging."""
    if args.selection_rule == "oracle" and not args.allow_oracle:
        print("ERROR: --selection-rule=oracle requires --allow-oracle")
        sys.exit(1)

    index_path = args.index_file or args.index_root
    if index_path is None:
        raise ValueError("index_root or index_file is required for probabilistic mode")

    # Load preprocessing (optional)
    preprocessor = None
    if preproc_enabled:
        preproc_dir = Path(preproc_path) if preproc_path else Path("outputs/preproc") / args.subject
        preprocessor = NSDPreprocessor(args.subject, out_dir=preproc_dir.parent)
        preprocessor.out_dir = preproc_dir  # ensure correct dir
        if not preprocessor.meta_path.exists():
            raise FileNotFoundError(f"Preprocessing artifacts not found at {preproc_dir}")
        preprocessor.load_artifacts()

    # Load CLIP cache (target embeddings)
    clip_cache = CLIPCache(str(args.clip_cache)).load()

    # Stream NSD samples
    dataset = NSDIterableDataset(
        index_path_or_root=str(index_path),
        subject=args.subject,
        shuffle=False,
        limit=args.limit,
        preprocessor=preprocessor,
        clip_cache=clip_cache,
    )

    X_list: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    stimulus_ids: List[int] = []

    for sample in dataset:
        if "clip" not in sample:
            continue
        x = sample["fmri"]
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
        else:
            x_t = torch.tensor(x).float()
        if x_t.dim() > 1:
            x_t = x_t.reshape(-1)
        X_list.append(x_t)
        targets.append(torch.from_numpy(sample["clip"]).float())
        stimulus_ids.append(int(sample["nsdId"]))

    if not X_list:
        raise RuntimeError("No samples loaded for probabilistic reconstruction")

    X = torch.stack(X_list).to(args.device)
    target_tensor = torch.stack(targets).to(args.device)

    # Load probabilistic encoder
    model, prob_meta = load_probabilistic_encoder(str(args.ckpt), map_location=args.device)
    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        outputs, _ = model(X, sample=False, return_kl=False)
    final_out = outputs["final"]
    mu = final_out.mu
    logvar = final_out.logvar
    uncertainties = final_out.uncertainty

    # Resolve adaptive quantile parameters with validation
    adaptive_quantiles = tuple(args.adaptive_quantile) if args.adaptive_quantile else ()
    adaptive_k_bins = tuple(args.adaptive_k_bins) if args.adaptive_k_bins else ()

    if args.sampling_policy == "adaptive_quantile":
        if not adaptive_quantiles:
            print("=" * 80)
            print("ERROR: --sampling-policy adaptive_quantile requires --adaptive-quantile values")
            print("=" * 80)
            print()
            print("Example: --adaptive-quantile 0.0 0.1 0.3 0.6 0.85 1.0")
            return 1
        if len(adaptive_quantiles) < 2:
            print("=" * 80)
            print("ERROR: --adaptive-quantile must have at least two quantile boundaries")
            print("=" * 80)
            print()
            print(f"Provided: {adaptive_quantiles}")
            print("Need: >=2 values (len(K_bins) must be len(quantiles)-1)")
            return 1
        if adaptive_k_bins and len(adaptive_k_bins) != len(adaptive_quantiles) - 1:
            print("=" * 80)
            print("ERROR: --adaptive-k-bins length must be len(adaptive-quantile)-1")
            print("=" * 80)
            print()
            print(f"Quantiles: {adaptive_quantiles}")
            print(f"K bins:   {adaptive_k_bins}")
            print()
            print("Example: --adaptive-quantile 0.0 0.1 0.3 0.6 0.85 1.0 --adaptive-k-bins 16 8 4 2 1")
            return 1
        if not adaptive_k_bins:
            needed = len(adaptive_quantiles) - 1
            if len(args.k_set) >= needed:
                adaptive_k_bins = tuple(args.k_set[:needed])
            else:
                print("=" * 80)
                print("ERROR: Not enough K values for adaptive quantile bins")
                print("=" * 80)
                print()
                print(f"Quantiles: {adaptive_quantiles} (needs {needed} K bins)")
                print(f"Provided k_set: {args.k_set}")
                print()
                print("Solution: pass --adaptive-k-bins explicitly with length len(quantiles)-1")
                return 1
    else:
        adaptive_quantiles = ()
        adaptive_k_bins = ()

    # Sampling configuration
    sampling_cfg = SamplingConfig(
        sampling_policy=args.sampling_policy,
        k_base=args.k_base,
        k_set=tuple(args.k_set),
        adaptive_quantile=(adaptive_quantiles, adaptive_k_bins, args.adaptive_budget_correction),
        adaptive_mapping=(tuple(args.adaptive_mapping), tuple(args.k_set)) if args.adaptive_mapping else ((), ()),
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
        variance_floor=args.variance_floor,
        clip_space=prob_meta.get("clip_space", "normalized"),
        enforce_budget=not getattr(args, "no_enforce_budget", False),
    )

    # Diffusion + CLIP image encoder
    model_id = args.model_id or "stabilityai/stable-diffusion-2-1"
    pipe = setup_diffusion_pipeline(
        model_id=model_id,
        device=args.device,
        dtype_str="float16" if args.device == "cuda" else "float32",
        scheduler_name="dpm",
        fail_if_missing=False,
    )

    clip_model, preprocess, _ = load_clip_model(device=args.device)

    def generate_fn(z: torch.Tensor, seed: int):
        z_np = z.detach().cpu().numpy()
        return generate_image_from_clip_embedding(
            pipe,
            z_np,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=seed,
        )

    def embed_image_fn(img) -> torch.Tensor:
        emb = encode_images(
            clip_model,
            preprocess,
            [img],
            device=args.device,
            normalize=sampling_cfg.clip_space == "normalized",
        )
        emb_t = torch.from_numpy(emb[0]).to(mu.device)

        # Align embedding dimension with model output (mu/logvar)
        mu_dim = mu.shape[-1]
        if emb_t.shape[-1] != mu_dim:
            if emb_t.shape[-1] > mu_dim:
                emb_t = emb_t[..., :mu_dim]
            else:
                pad = torch.zeros(*emb_t.shape[:-1], mu_dim - emb_t.shape[-1], device=emb_t.device, dtype=emb_t.dtype)
                emb_t = torch.cat([emb_t, pad], dim=-1)
            # Re-normalize if working in normalized clip space
            if sampling_cfg.clip_space == "normalized":
                emb_t = torch.nn.functional.normalize(emb_t, dim=-1)
        return emb_t

    trial_results, chosen_images, budget_stats = run_probabilistic_trials(
        mus=mu,
        logvars=logvar,
        targets=target_tensor,
        uncertainties=uncertainties,
        sampling_cfg=sampling_cfg,
        selection_rule=args.selection_rule,
        allow_oracle=args.allow_oracle,
        generate_fn=generate_fn,
        embed_image_fn=embed_image_fn,
        seed_base=args.seed_base,
        guidance_scale=args.guidance_scale,
        diffusion_steps=args.steps,
        stimulus_ids=stimulus_ids,
        subject=args.subject,
        split="test",
    )

    # Persist images
    recon_dir = args.output_dir / "images"
    recon_dir.mkdir(parents=True, exist_ok=True)
    for img, stim_id in zip(chosen_images, stimulus_ids):
        if hasattr(img, "save"):
            img.save(recon_dir / f"{stim_id}.png")

    trial_csv, trial_parquet = write_trial_results(trial_results, args.output_dir, csv_name=args.trial_csv)
    run_calibration_assets(trial_csv, args.output_dir)

    # Persist budget stats
    budget_path = args.output_dir / "budget_stats.json"
    with open(budget_path, "w") as f:
        json.dump(budget_stats, f, indent=2)
    print(f"✓ Budget stats written: {budget_path}")

    return recon_dir, trial_csv


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--subject", type=str, default="subj01",
                        help="NSD subject ID (default: subj01)")
    parser.add_argument("--encoder", type=str, required=True,
                        choices=["ridge", "mlp", "two_stage", "prob"],
                        help="Encoder type (ridge/mlp/two_stage/prob)")
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
    parser.add_argument("--allow-fallback-index", action="store_true", help="Permit collapsed/synthetic indices (not for paper runs)")
    parser.add_argument("--skip-index-validation", action="store_true", help="Skip header/contiguity validation of index (not recommended)")
    
    # Optional
    parser.add_argument("--limit", type=int, default=64,
                        help="Number of test samples to process (default: 64)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for inference (default: auto)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion steps (default: 50)")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Classifier-free guidance scale for diffusion")

    # Probabilistic inference
    parser.add_argument("--probabilistic", action="store_true",
                        help="Enable probabilistic sampling + adaptive K")
    parser.add_argument("--selection-rule", type=str, default="cosine",
                        choices=["cosine", "likelihood", "oracle"],
                        help="Selection rule for candidate images")
    parser.add_argument("--allow-oracle", action="store_true",
                        help="Permit oracle selection (uses ground truth)")
    parser.add_argument("--sampling-policy", type=str, default="fixed_k",
                        choices=["fixed_k", "adaptive_quantile", "adaptive_mapping"],
                        help="K allocation policy")
    parser.add_argument("--k-base", type=int, default=1,
                        help="Base K for fixed_k or average budget")
    parser.add_argument("--k-set", type=int, nargs="+", default=[1, 2, 4],
                        help="Candidate Ks for adaptive policies")
    parser.add_argument("--no-enforce-budget", action="store_true",
                        help="Disable budget enforcement for adaptive K policies (default: enforce)")
    parser.add_argument("--adaptive-quantile", type=float, nargs="+",
                        metavar=("Q0", "Q1"),
                        help="Quantile thresholds for adaptive_quantile policy (>=2 values; K bins must have length len(quantiles)-1)")
    parser.add_argument("--adaptive-k-bins", type=int, nargs="+",
                        metavar=("K0", "K1"),
                        help="K values per quantile bucket for adaptive_quantile policy (length must be len(quantiles)-1)")
    parser.add_argument("--adaptive-budget-correction", type=str, default="deterministic",
                        choices=["deterministic", "none"],
                        help="Budget correction strategy for adaptive quantile allocator")
    parser.add_argument("--adaptive-mapping", type=float, nargs=3,
                        metavar=("U_LOW", "U_MED", "U_HIGH"),
                        help="Uncertainty thresholds for adaptive_mapping policy")
    parser.add_argument("--logvar-min", type=float, default=-8.0,
                        help="Clamp minimum for log-variance")
    parser.add_argument("--logvar-max", type=float, default=2.0,
                        help="Clamp maximum for log-variance")
    parser.add_argument("--variance-floor", type=float, default=1e-6,
                        help="Variance floor for sampling stability")
    parser.add_argument("--trial-csv", type=str, default="trial_results.csv",
                        help="Filename for per-trial CSV (under output-dir)")
    parser.add_argument("--seed-base", type=int, default=123,
                        help="Base seed for probabilistic sampling")
    
    # Retrieval gallery configuration
    parser.add_argument("--gallery", type=str, default="matched",
                        choices=["matched", "test", "all"],
                        help="Retrieval gallery type (default: matched)")
    parser.add_argument("--all-galleries", action="store_true",
                        help="Run evaluation for all gallery types (matched, test, all)")
    
    # Image source configuration
    parser.add_argument("--image-source", type=str, default="hdf5",
                        choices=["hdf5", "files"],
                        help="Source for visualization images: 'hdf5' (NSD HDF5 file) or 'files' (PNG/S3) (default: hdf5)")
    parser.add_argument("--nsd-hdf5", type=Path,
                        help="Path to NSD HDF5 file (optional, for HDF5 image source)")
    
    # Adapter
    parser.add_argument("--use-adapter", action="store_true",
                        help="Use CLIP adapter for dimension alignment")
    parser.add_argument("--adapter", type=Path,
                        help="Path to CLIP adapter checkpoint (required if --use-adapter)")
    parser.add_argument("--model-id", type=str,
                        help="Diffusion model ID (e.g., stabilityai/stable-diffusion-2-1)")
    
    # Preprocessing
    parser.add_argument("--use-preproc", action="store_true",
                        help="Force enable preprocessing (overrides auto-detection)")
    parser.add_argument("--no-preproc", action="store_true",
                        help="Force disable preprocessing (overrides auto-detection)")
    parser.add_argument("--skip-sd-cache-check", action="store_true",
                        help="Skip Stable Diffusion cache check and do not prompt")
    parser.add_argument("--preproc-dir", type=Path,
                        help="Path to preprocessing directory (auto-discovered if not provided)")
    
    args = parser.parse_args()
    
    # ============================================================================
    # VALIDATION: Strict checks with actionable error messages
    # ============================================================================
    
    # Check adapter requirements
    if args.use_adapter:
        if not args.adapter:
            print("=" * 80)
            print("ERROR: --use-adapter requires --adapter PATH")
            print("=" * 80)
            print()
            print("You must provide the path to a CLIP adapter checkpoint.")
            print()
            print("Example:")
            print("  --use-adapter \\")
            print("  --adapter checkpoints/clip_adapter/subj01/adapter.pt \\")
            print("  --model-id stabilityai/stable-diffusion-2-1")
            print()
            return 1
        
        if not args.model_id:
            print("=" * 80)
            print("ERROR: --use-adapter requires --model-id MODEL_ID")
            print("=" * 80)
            print()
            print("You must specify the diffusion model ID when using an adapter.")
            print()
            print("Common model IDs:")
            print("  stabilityai/stable-diffusion-2-1")
            print("  runwayml/stable-diffusion-v1-5")
            print()
            print("Example:")
            print("  --use-adapter \\")
            print("  --adapter checkpoints/clip_adapter/subj01/adapter.pt \\")
            print("  --model-id stabilityai/stable-diffusion-2-1")
            print()
            return 1
        
        if not args.adapter.exists():
            print("=" * 80)
            print(f"ERROR: Adapter checkpoint not found")
            print("=" * 80)
            print()
            print(f"Path: {args.adapter}")
            print()
            print("Solution:")
            print("  1. Check the path is correct")
            print("  2. Train an adapter first: make train-adapter SUBJ=subj01")
            print()
            return 1
    
    # Check required files exist
    if not args.ckpt.exists():
        print("=" * 80)
        print(f"ERROR: Encoder checkpoint not found")
        print("=" * 80)
        print()
        print(f"Path: {args.ckpt}")
        print()
        print("Solution:")
        print(f"  Train the {args.encoder} encoder first:")
        print(f"    make train-{args.encoder} SUBJ={args.subject}")
        print()
        return 1
    
    if not args.clip_cache.exists():
        print("=" * 80)
        print(f"ERROR: CLIP cache not found")
        print("=" * 80)
        print()
        print(f"Path: {args.clip_cache}")
        print()
        print("Solution:")
        print("  Build the CLIP cache first:")
        print("    make build-clip-cache")
        print()
        return 1

    index_path = args.index_file or args.index_root
    if index_path and not args.skip_index_validation:
        try:
            df = read_subject_index(str(index_path), args.subject, allow_fallback_index=args.allow_fallback_index)
            NSDIndexBuilder().validate_index(df)
        except Exception as e:
            print("=" * 80)
            print("ERROR: Index validation failed")
            print("=" * 80)
            print()
            print(f"Path: {index_path}")
            print(f"Reason: {e}")
            print()
            print("Rebuild the canonical index or pass --allow-fallback-index/--skip-index-validation (not for paper runs).")
            return 1
    
    # Load encoder metadata to determine preprocessing requirements
    encoder_ckpt = torch.load(args.ckpt, map_location="cpu")
    encoder_meta = encoder_ckpt.get("meta", {})
    
    # Determine if preprocessing should be enabled
    preproc_meta = encoder_meta.get("preproc", {})
    preproc_trained_with = preproc_meta.get("used_preproc", False)
    expected_dim = encoder_meta.get("input_dim")
    
    # Resolve preprocessing flag
    if args.use_preproc and args.no_preproc:
        print("ERROR: Cannot specify both --use-preproc and --no-preproc")
        return 1
    
    if args.use_preproc:
        preproc_enabled = True
    elif args.no_preproc:
        preproc_enabled = False
    else:
        # Auto-detect from metadata
        preproc_enabled = preproc_trained_with
        # Heuristic: if model expects a small feature dim (<=1024) and no metadata flag,
        # assume preprocessing was used.
        if not preproc_enabled and expected_dim and expected_dim <= 1024:
            preproc_enabled = True
    
    # Determine preprocessing path
    preproc_path = None
    if preproc_enabled:
        # Priority: CLI arg > metadata > auto-discover
        if args.preproc_dir:
            preproc_path = args.preproc_dir
        elif preproc_meta.get("path"):
            preproc_path = Path(preproc_meta["path"])
        else:
            # Auto-discover preprocessing directory
            preproc_base = Path("outputs/preproc") / args.subject
            if preproc_base.exists():
                candidates = []
                # Consider base meta.json
                base_meta = preproc_base / "meta.json"
                if base_meta.exists():
                    try:
                        import json
                        with open(base_meta) as f:
                            base_meta_json = json.load(f)
                        pca_k = base_meta_json.get("pca_components")
                        if expected_dim and pca_k == expected_dim:
                            candidates.append((preproc_base, preproc_base.stat().st_mtime))
                    except Exception:
                        pass

                for subdir in preproc_base.iterdir():
                    if not subdir.is_dir():
                        continue
                    meta_json = subdir / "meta.json"
                    if not meta_json.exists():
                        continue
                    
                    try:
                        import json
                        with open(meta_json) as f:
                            preproc_meta_json = json.load(f)
                        
                        # Check if dimensions match
                        pca_k = preproc_meta_json.get("pca_components")
                        if expected_dim and pca_k == expected_dim:
                            candidates.append((subdir, subdir.stat().st_mtime))
                        elif not expected_dim:
                            # No expected dim, add all candidates
                            candidates.append((subdir, subdir.stat().st_mtime))
                    except Exception:
                        continue
                
                if candidates:
                    # Pick most recent
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    preproc_path = candidates[0][0]
                    print(f"✓ Auto-discovered preprocessing directory: {preproc_path}")
                else:
                    print(f"ERROR: Preprocessing enabled but no valid directory found")
                    print(f"       Searched in: {preproc_base}")
                    print(f"       Expected dim: {expected_dim}")
                    print(f"       Solution: Provide --preproc-dir explicitly")
                    return 1
            else:
                print(f"ERROR: Preprocessing enabled but directory not found: {preproc_base}")
                print(f"       Solution: Provide --preproc-dir explicitly")
                return 1
    
    # Print preprocessing banner
    if preproc_enabled:
        preproc_k = preproc_meta.get("k", encoder_meta.get("input_dim", "unknown"))
        preproc_thr = preproc_meta.get("reliability_thr", "unknown")
        print(f"✓ Preprocessing: ENABLED (k={preproc_k}, thr={preproc_thr})")
        if preproc_path:
            print(f"  Path: {preproc_path}")
    else:
        print("✓ Preprocessing: DISABLED")
    
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
    
    detected_target_dim = clip_target_dim

    # Determine galleries to evaluate
    if args.all_galleries:
        galleries = ["matched", "test", "all"]
        print(f"✓ Running all galleries: {', '.join(galleries)}")
    else:
        galleries = [args.gallery]
        print(f"✓ Running single gallery: {args.gallery}")
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # REPRODUCIBILITY: Gather environment info and write manifest
    # ============================================================================
    print_banner("Gathering Environment Info for Reproducibility")
    
    # Gather environment information
    env_info = gather_env_info()
    print(f"✓ Environment: Python {env_info['python_version']}, "
          f"PyTorch {env_info['torch_version']}, "
          f"CUDA {env_info['cuda_version']}")
    print(f"✓ Git: {env_info['git_commit'][:8]} on {env_info['git_branch']}")
    if env_info['git_dirty']:
        print("  ⚠️  Warning: Git working directory has uncommitted changes")
    
    # Create manifest with input file hashes
    manifest_data = {
        "script": "run_reconstruct_and_eval.py",
        "timestamp": datetime.now().isoformat(),
        "environment": env_info,
        "config": {
            "subject": args.subject,
            "encoder": args.encoder,
            "encoder_ckpt": str(args.ckpt),
            "use_adapter": args.use_adapter,
            "adapter": str(args.adapter) if args.adapter else None,
            "model_id": args.model_id,
            "clip_target_dim": clip_target_dim,
            "limit": args.limit,
            "steps": args.steps,
            "galleries": galleries,
            "image_source": args.image_source,
            "preprocessing_enabled": preproc_enabled,
            "preprocessing_path": str(preproc_path) if preproc_path else None,
            "probabilistic": bool(args.probabilistic),
            "sampling_policy": args.sampling_policy,
            "k_base": args.k_base,
            "k_set": args.k_set,
            "selection_rule": args.selection_rule,
            "allow_oracle": args.allow_oracle,
            "enforce_budget": not getattr(args, "no_enforce_budget", False),
        },
        "input_hashes": {}
    }
    
    # Hash input files for reproducibility
    try:
        manifest_data["input_hashes"]["encoder_ckpt"] = hash_file(str(args.ckpt))
        print(f"✓ Hashed encoder checkpoint: {manifest_data['input_hashes']['encoder_ckpt'][:16]}...")
        
        if args.adapter and args.adapter.exists():
            manifest_data["input_hashes"]["adapter"] = hash_file(str(args.adapter))
            print(f"✓ Hashed adapter: {manifest_data['input_hashes']['adapter'][:16]}...")
        
        if args.clip_cache.exists():
            manifest_data["input_hashes"]["clip_cache"] = hash_file(str(args.clip_cache))
            print(f"✓ Hashed CLIP cache: {manifest_data['input_hashes']['clip_cache'][:16]}...")
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to hash some input files: {e}")
    
    # Write manifest to output directory (config goes in second param)
    manifest_path = args.output_dir / "manifest.json"
    write_manifest(
        str(manifest_path),
        manifest_data,
        cli_args=sys.argv,
        env_info=env_info,
        input_hashes=manifest_data.get("input_hashes", {}),
    )
    print(f"✓ Manifest written: {manifest_path}")
    print()
    
    # Print configuration banner
    print_banner("Reconstruct & Evaluate Workflow")
    print("CONFIGURATION")
    print("-" * 80)
    print(f"  Subject:         {args.subject}")
    print(f"  Encoder:         {args.encoder}")
    print(f"  Checkpoint:      {args.ckpt.name}")
    print()
    print(f"  Adapter:         {'✓ ENABLED' if args.use_adapter else '✗ Disabled'}")
    if args.use_adapter:
        print(f"    Path:          {args.adapter.name}")
        print(f"    Model ID:      {args.model_id}")
        print(f"    Target Dim:    {clip_target_dim}D")
    print()
    print(f"  CLIP Space:      {clip_dim}D ({('ViT-B/32' if clip_dim == 512 else f'{clip_target_dim}D target')})")
    print(f"  Galleries:       {', '.join(galleries)}")
    print(f"  Image Source:    {args.image_source}")
    if args.nsd_hdf5:
        print(f"  NSD HDF5:        {args.nsd_hdf5}")
    print()
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  Report Dir:      {args.report_dir}")
    print(f"  Limit:           {args.limit} samples")
    print(f"  Diffusion Steps: {args.steps}")
    if args.index_root:
        print(f"  Index Root:      {args.index_root}")
    elif args.index_file:
        print(f"  Index File:      {args.index_file}")
    print("-" * 80)
    print()
    
    # Check SD cache with robust method
    model_id_for_cache = args.model_id or "runwayml/stable-diffusion-v1-5"
    print(f"Checking SD cache for: {model_id_for_cache}")
    print()
    
    def _is_sd_cached(mid: str) -> bool:
        """Robust check: try constructing the pipeline offline."""
        try:
            from diffusers import StableDiffusionPipeline
            StableDiffusionPipeline.from_pretrained(mid, local_files_only=True)
            return True
        except Exception:
            return False
    
    cached = True
    if not args.skip_sd_cache_check:
        cached = _is_sd_cached(model_id_for_cache)
    
    if not cached and not args.skip_sd_cache_check:
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
        if args.skip_sd_cache_check:
            print("⊘ SD cache check skipped (--skip-sd-cache-check)")
        else:
            print(f"✓ Stable Diffusion appears cached (local_files_only=True succeeded)")
        print()
    
    # Probabilistic branch (adaptive sampling + trial logging)
    # Resolve device 'auto' for downstream torch ops in both branches
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.probabilistic:
        recon_dir, trial_csv = probabilistic_reconstruct(
            args=args,
            galleries=galleries,
            preproc_enabled=preproc_enabled,
            preproc_path=preproc_path,
            clip_dim=clip_dim,
        )
    else:
        # Step 1: Decode
        exit_code, detected_target_dim = run_decode(
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
            preproc_enabled=preproc_enabled,
            preproc_path=str(preproc_path) if preproc_path else None,
        )
        
        if exit_code != 0:
            return exit_code
        
        # Determine recon_dir (where images are actually stored)
        recon_dir = args.output_dir / "images"
        if not recon_dir.exists():
            # Fallback: decode_diffusion.py might have written directly to output_dir
            recon_dir = args.output_dir

    # Update manifest with per-sample mapping (deterministic based on images on disk)
    samples = build_sample_entries(
        recon_dir=recon_dir,
        subject=args.subject,
        limit=args.limit,
        index_root=args.index_root,
        index_file=args.index_file,
        steps=args.steps,
        guidance=args.guidance_scale,
        sampling_policy=args.sampling_policy if args.probabilistic else "deterministic",
        seed_base=args.seed_base if args.probabilistic else None,
    )
    append_samples_to_manifest(args.output_dir / "manifest.json", samples)
    
    # Step 2: Evaluate (loop over galleries)
    for gallery in galleries:
        exit_code = run_eval(
            subject=args.subject,
            recon_dir=recon_dir,
            clip_cache=args.clip_cache,
            report_dir=args.report_dir,
            use_adapter=args.use_adapter,
            model_id=args.model_id,
            index_root=args.index_root,
            index_file=args.index_file,
            limit=args.limit,
            gallery=gallery,
            image_source=args.image_source,
            nsd_hdf5=args.nsd_hdf5,
        )
        
        if exit_code != 0:
            return exit_code
    
    # Step 3: Create Markdown summary (use first gallery for backward compatibility)
    primary_gallery = galleries[0]
    exit_code = create_markdown_summary(
        report_dir=args.report_dir,
        subject=args.subject,
        encoder=args.encoder,
        use_adapter=args.use_adapter,
        model_id=args.model_id,
        clip_dim=clip_dim,
        ckpt_path=args.ckpt,
        adapter_path=args.adapter,
        recon_dir=recon_dir,
        limit=args.limit,
        gallery=primary_gallery,
    )
    
    if exit_code != 0:
        return exit_code
    
    print(f"\n{'='*80}")
    print("  ✓ ALL STEPS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📁 Generated Images:     {recon_dir}")
    print(f"📊 Evaluation Reports:   {args.report_dir}")
    print(f"📝 Markdown Summary:     {args.report_dir}/recon_eval_summary.md")
    print()
    print("Evaluation outputs per gallery:")
    for gallery in galleries:
        print(f"  • {gallery:8s} → CSV, JSON, PNG")
    print()
    
    # BOLD NOTE about CLIP space
    if args.use_adapter and detected_target_dim:
        print("🔍 " + "=" * 76)
        print(f"   NOTE: Evaluation performed in {detected_target_dim}D CLIP space")
        print(f"         (matching generation with adapter)")
        print("=" * 80)
    else:
        print("🔍 " + "=" * 76)
        print(f"   NOTE: Evaluation performed in 512D CLIP space (ViT-B/32)")
        print(f"         (no adapter used)")
        print("=" * 80)
    print()
    
    print("Next steps:")
    print(f"  • View summary:    cat {args.report_dir}/recon_eval_summary.md")
    print(f"  • View grid:       open {args.report_dir}/recon_grid_{galleries[0]}.png")
    print(f"  • Compare evals:   python scripts/compare_evals.py --report-dir {args.report_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
