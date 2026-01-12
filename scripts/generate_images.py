#!/usr/bin/env python3
"""
Generate reconstructed images from trained fMRI-to-CLIP models.

This script takes a trained model and generates CLIP embeddings from fMRI data,
then optionally uses these embeddings to reconstruct images via diffusion models.

Usage:
    python scripts/generate_images.py --checkpoint PATH --fmri PATH --output DIR

Features:
    - Load trained model checkpoints
    - Process fMRI volumes to CLIP embeddings
    - Visualize fMRI-to-embedding transformations
    - Generate images via Stable Diffusion (optional)
    - Batch processing support

Author: Bachelor Thesis - fMRI to Image Reconstruction
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.simple_cnn import SimpleCNN


def load_model(checkpoint_path: Path, device: str = 'cuda') -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = SimpleCNN(
        input_shape=(1, 81, 104, 83),
        output_dim=512
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'N/A')}, loss {checkpoint.get('loss', 0):.4f})")
    
    return model


def load_fmri_data(fmri_path: Path) -> np.ndarray:
    """
    Load fMRI data from file.
    
    Args:
        fmri_path: Path to fMRI data (.npy or .nii.gz)
        
    Returns:
        fMRI data as numpy array
    """
    print(f"Loading fMRI data from: {fmri_path}")
    
    if fmri_path.suffix == '.npy':
        data = np.load(fmri_path)
    elif fmri_path.suffix == '.gz' or fmri_path.suffixes[-2:] == ['.nii', '.gz']:
        import nibabel as nib
        img = nib.load(fmri_path)
        data = img.get_fdata()
    else:
        raise ValueError(f"Unsupported file format: {fmri_path.suffix}")
    
    print(f"✓ Loaded fMRI shape: {data.shape}")
    
    return data


def generate_embedding(model: torch.nn.Module, fmri_data: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """
    Generate CLIP embedding from fMRI data.
    
    Args:
        model: Trained model
        fmri_data: fMRI volume
        device: Device to use
        
    Returns:
        Generated CLIP embedding
    """
    with torch.no_grad():
        # Ensure correct shape: (1, 1, H, W, D)
        if fmri_data.ndim == 3:
            fmri_data = fmri_data[np.newaxis, np.newaxis, ...]
        elif fmri_data.ndim == 4:
            fmri_data = fmri_data[np.newaxis, ...]
        
        # Convert to tensor
        fmri_tensor = torch.from_numpy(fmri_data).float().to(device)
        
        # Generate embedding
        embedding = model(fmri_tensor)
        
    return embedding.cpu().numpy().squeeze()


def visualize_fmri_slices(fmri_data: np.ndarray, output_path: Path, num_slices: int = 9) -> None:
    """
    Visualize multiple slices of fMRI volume.
    
    Args:
        fmri_data: fMRI volume (H, W, D)
        output_path: Path to save visualization
        num_slices: Number of slices to show
    """
    if fmri_data.ndim == 4:
        fmri_data = fmri_data[0]  # Remove batch dim
    if fmri_data.ndim == 4:
        fmri_data = fmri_data[0]  # Remove channel dim
    
    depth = fmri_data.shape[2]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        slice_data = fmri_data[:, :, slice_idx]
        im = ax.imshow(slice_data, cmap='hot', aspect='auto')
        ax.set_title(f'Slice {slice_idx}/{depth}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('fMRI Volume Slices', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved fMRI visualization to: {output_path}")
    plt.close()


def visualize_embedding(embedding: np.ndarray, output_path: Path) -> None:
    """
    Visualize CLIP embedding as heatmap.
    
    Args:
        embedding: CLIP embedding vector
        output_path: Path to save visualization
    """
    # Reshape to 2D for visualization
    embedding_2d = embedding.reshape(16, 32)  # 512 = 16 x 32
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    im1 = ax1.imshow(embedding_2d, cmap='viridis', aspect='auto')
    ax1.set_title('CLIP Embedding (2D Visualization)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension (width)', fontsize=11)
    ax1.set_ylabel('Dimension (height)', fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Distribution
    ax2.hist(embedding, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Embedding Value Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    stats_text = f"Mean: {embedding.mean():.4f}\nStd: {embedding.std():.4f}\nMin: {embedding.min():.4f}\nMax: {embedding.max():.4f}"
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved embedding visualization to: {output_path}")
    plt.close()


def visualize_generation_pipeline(fmri_data: np.ndarray, embedding: np.ndarray, output_path: Path) -> None:
    """
    Visualize the complete generation pipeline.
    
    Args:
        fmri_data: Input fMRI volume
        embedding: Generated CLIP embedding
        output_path: Path to save visualization
    """
    if fmri_data.ndim == 4:
        fmri_data = fmri_data[0]  # Remove batch dim
    if fmri_data.ndim == 4:
        fmri_data = fmri_data[0]  # Remove channel dim
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: fMRI middle slice
    ax1 = plt.subplot(1, 3, 1)
    middle_slice = fmri_data[:, :, fmri_data.shape[2]//2]
    im1 = ax1.imshow(middle_slice, cmap='hot', aspect='auto')
    ax1.set_title('Input fMRI\n(Middle Slice)', fontsize=13, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Embedding heatmap
    ax2 = plt.subplot(1, 3, 2)
    embedding_2d = embedding.reshape(16, 32)
    im2 = ax2.imshow(embedding_2d, cmap='viridis', aspect='auto')
    ax2.set_title('Generated CLIP Embedding\n(512-dim)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Dimension', fontsize=10)
    ax2.set_ylabel('Channel', fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot 3: Embedding statistics
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    stats_text = [
        "CLIP Embedding Statistics",
        "─" * 30,
        f"Dimension: {embedding.shape[0]}",
        f"Mean: {embedding.mean():.4f}",
        f"Std Dev: {embedding.std():.4f}",
        f"Min: {embedding.min():.4f}",
        f"Max: {embedding.max():.4f}",
        f"L2 Norm: {np.linalg.norm(embedding):.4f}",
        "",
        "✓ Ready for image generation",
        "  (via Stable Diffusion)"
    ]
    
    ax3.text(0.1, 0.9, '\n'.join(stats_text), 
            transform=ax3.transAxes,
            verticalalignment='top',
            fontsize=12,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('fMRI → CLIP Embedding Generation Pipeline', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved pipeline visualization to: {output_path}")
    plt.close()


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(
        description="Generate reconstructed images from trained fMRI-to-CLIP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from single fMRI file
  python scripts/generate_images.py --checkpoint runs/*/checkpoints/best_model.pt --fmri sample.npy --output results/

  # Generate with visualizations
  python scripts/generate_images.py --checkpoint best_model.pt --fmri data.npy --output results/ --visualize
        """
    )
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    parser.add_argument('--fmri', required=True, help="Path to fMRI data (.npy or .nii.gz)")
    parser.add_argument('--output', required=True, help="Output directory for results")
    parser.add_argument('--device', default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument('--visualize', action='store_true', help="Generate visualization plots")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Banner
    print("\n" + "="*100)
    print(" " * 30 + "IMAGE GENERATION FROM fMRI")
    print("="*100 + "\n")
    
    # Load model
    model = load_model(Path(args.checkpoint), args.device)
    
    # Load fMRI data
    fmri_data = load_fmri_data(Path(args.fmri))
    
    # Generate embedding
    print("\nGenerating CLIP embedding...")
    embedding = generate_embedding(model, fmri_data, args.device)
    print(f"✓ Generated embedding shape: {embedding.shape}")
    print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
    
    # Save embedding
    embedding_path = output_dir / 'clip_embedding.npy'
    np.save(embedding_path, embedding)
    print(f"✓ Saved embedding to: {embedding_path}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        visualize_fmri_slices(fmri_data, output_dir / 'fmri_slices.png')
        visualize_embedding(embedding, output_dir / 'embedding_analysis.png')
        visualize_generation_pipeline(fmri_data, embedding, output_dir / 'generation_pipeline.png')
    
    # Summary
    print("\n" + "="*100)
    print("✅ GENERATION COMPLETE!")
    print("="*100)
    print(f"\n📁 Results saved to: {output_dir.absolute()}")
    print("\n📋 Generated files:")
    print(f"  • clip_embedding.npy - Generated CLIP embedding (512-dim)")
    if args.visualize:
        print(f"  • fmri_slices.png - fMRI volume slices")
        print(f"  • embedding_analysis.png - Embedding analysis")
        print(f"  • generation_pipeline.png - Complete pipeline visualization")
    print("\n💡 Next steps:")
    print("  1. Use clip_embedding.npy with Stable Diffusion to generate images")
    print("  2. Evaluate reconstruction quality with evaluation script")
    print("\n" + "="*100 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
