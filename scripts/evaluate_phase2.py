#!/usr/bin/env python3
"""
Comprehensive evaluation of Phase 2 multi-task model.

Evaluates:
1. CLIP embedding prediction quality (cosine similarity)
2. Layer-wise performance (layer_4, layer_8, layer_12, final)
3. Text-CLIP prediction quality
4. Comparison with baseline models
5. Per-sample analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.models.encoders import MultiLayerTwoStageEncoder, load_multilayer_two_stage_encoder
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.models.train_utils import train_val_test_split
from fmri2img.models.ridge import evaluate_predictions
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_multilayer_clip_cache(cache_path: str):
    """Load multi-layer CLIP cache."""
    logger.info(f"Loading multi-layer CLIP cache from {cache_path}...")
    df = pd.read_parquet(cache_path)
    
    cache_dict = {}
    for _, row in df.iterrows():
        nsd_id = int(row['nsdId'])
        cache_dict[nsd_id] = {
            'layer_4': np.array(row['layer_4'], dtype=np.float32),
            'layer_8': np.array(row['layer_8'], dtype=np.float32),
            'layer_12': np.array(row['layer_12'], dtype=np.float32),
            'final': np.array(row['final'], dtype=np.float32)
        }
    
    logger.info(f"  Loaded {len(cache_dict)} multi-layer embeddings")
    return cache_dict


def load_text_clip_cache(cache_path: str):
    """Load text-CLIP cache."""
    logger.info(f"Loading text-CLIP cache from {cache_path}...")
    df = pd.read_parquet(cache_path)
    
    cache_dict = {}
    for _, row in df.iterrows():
        # Handle both column name formats
        nsd_col = 'nsd_id' if 'nsd_id' in df.columns else 'nsdId'
        nsd_id = int(row[nsd_col])
        text_emb = np.array(row['text_clip_embedding'], dtype=np.float32)
        cache_dict[nsd_id] = text_emb
    
    logger.info(f"  Loaded {len(cache_dict)} text-CLIP embeddings")
    return cache_dict


def extract_features_and_targets(
    df, nifti_loader, preprocessor, multilayer_cache, text_clip_cache, desc="data"
):
    """Extract fMRI features and all targets (multi-layer + text-CLIP)."""
    from collections import defaultdict
    
    X_list = []
    Y_dict_lists = {'layer_4': [], 'layer_8': [], 'layer_12': [], 'final': []}
    if text_clip_cache is not None:
        Y_dict_lists['text'] = []
    nsd_ids_list = []
    
    # Group by file
    samples_by_file = defaultdict(list)
    for idx, row in df.iterrows():
        nsd_id = int(row["nsdId"])
        
        # Skip if no multi-layer embedding
        if nsd_id not in multilayer_cache:
            continue
        
        # Skip if text-CLIP required but not available
        if text_clip_cache is not None and nsd_id not in text_clip_cache:
            continue
        
        samples_by_file[row["beta_path"]].append({
            'beta_index': int(row["beta_index"]),
            'nsdId': nsd_id
        })
    
    logger.info(f"Extracting {desc}: {len(df)} samples from {len(samples_by_file)} files")
    
    # Process files
    for beta_path, samples in tqdm(samples_by_file.items(), desc=f"Loading {desc}"):
        try:
            img = nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            
            for sample in samples:
                beta_index = sample['beta_index']
                nsd_id = sample['nsdId']
                
                # Extract volume
                vol = data_4d[..., beta_index].astype(np.float32)
                
                # Preprocess
                if preprocessor and preprocessor.is_fitted_:
                    vol_z = preprocessor.transform_T0(vol)
                    features = preprocessor.transform(vol_z)
                else:
                    features = vol.flatten()
                
                # Get targets
                y_dict = multilayer_cache[nsd_id].copy()
                if text_clip_cache is not None:
                    y_dict['text'] = text_clip_cache[nsd_id]
                
                X_list.append(features)
                for layer_name in Y_dict_lists:
                    Y_dict_lists[layer_name].append(y_dict[layer_name])
                nsd_ids_list.append(nsd_id)
                
        except Exception as e:
            logger.warning(f"Failed to load {beta_path}: {e}")
            continue
    
    X = np.vstack(X_list)
    Y_dict = {k: np.vstack(v) for k, v in Y_dict_lists.items()}
    nsd_ids = np.array(nsd_ids_list)
    
    logger.info(f"  Extracted {len(X)} valid samples")
    for layer, emb in Y_dict.items():
        logger.info(f"    {layer}: {emb.shape}")
    
    return X, Y_dict, nsd_ids


def evaluate_layer(Y_true, Y_pred, layer_name):
    """Evaluate predictions for a single layer."""
    metrics = evaluate_predictions(Y_true, Y_pred, normalize=True)
    
    return {
        'layer': layer_name,
        'cosine': metrics['cosine'],
        'cosine_std': metrics['cosine_std'],
        'mse': metrics['mse'],
        'r2': metrics.get('r2', 0.0),
        'dim': Y_true.shape[1]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--multilayer-cache", default="cache/clip_embeddings/nsd_clipcache_multilayer.parquet")
    parser.add_argument("--text-clip-cache", default="cache/clip_embeddings/text_clip.parquet")
    parser.add_argument("--preproc-dir", default="outputs/preproc")
    parser.add_argument("--pca-k", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.subject / "phase2_multitask"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model, meta = load_multilayer_two_stage_encoder(args.model_path, map_location=device)
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded: {model.__class__.__name__}")
    
    # Load preprocessing
    logger.info("Setting up preprocessing...")
    preprocessor = NSDPreprocessor(args.subject, out_dir=args.preproc_dir)
    preprocessor.load_artifacts()
    pca_k = preprocessor.pca_info_.get('k_eff', preprocessor.pca_.n_components_ if preprocessor.pca_ else None)
    logger.info(f"Preprocessing ready: PCA k={pca_k if pca_k else 'disabled'}")
    
    # Load data
    logger.info(f"Loading index for {args.subject}...")
    df = read_subject_index(args.index_root, args.subject)
    train_df, val_df, test_df = train_val_test_split(df, random_seed=42)
    
    # Load caches
    multilayer_cache = load_multilayer_clip_cache(args.multilayer_cache)
    text_clip_cache = load_text_clip_cache(args.text_clip_cache)
    
    # Load NIfTI loader
    fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(fs)
    
    # Extract test data
    logger.info("=" * 80)
    logger.info("EXTRACTING TEST DATA")
    logger.info("=" * 80)
    X_test, Y_test_dict, nsd_ids_test = extract_features_and_targets(
        test_df, nifti_loader, preprocessor, multilayer_cache, text_clip_cache, desc="test"
    )
    
    # Create dataset
    test_dataset_dict = {
        'X': torch.from_numpy(X_test).float(),
        'Y': {k: torch.from_numpy(v).float() for k, v in Y_test_dict.items()},
        'nsd_ids': nsd_ids_test
    }
    
    # Run inference
    logger.info("=" * 80)
    logger.info("RUNNING INFERENCE")
    logger.info("=" * 80)
    
    all_preds = {k: [] for k in Y_test_dict.keys()}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), args.batch_size), desc="Inference"):
            X_batch = test_dataset_dict['X'][i:i+args.batch_size].to(device)
            Y_pred_dict = model(X_batch)
            
            for layer_name in all_preds.keys():
                if layer_name in Y_pred_dict:
                    all_preds[layer_name].append(Y_pred_dict[layer_name].cpu().numpy())
    
    # Stack predictions (only for layers that have predictions)
    Y_pred_dict = {k: np.vstack(v) for k, v in all_preds.items() if len(v) > 0}
    
    # Evaluate each layer
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    results = []
    for layer_name in ['layer_4', 'layer_8', 'layer_12', 'final', 'text']:
        if layer_name in Y_test_dict and layer_name in Y_pred_dict:
            result = evaluate_layer(
                Y_test_dict[layer_name],
                Y_pred_dict[layer_name],
                layer_name
            )
            results.append(result)
            
            logger.info(f"\n{layer_name.upper()}:")
            logger.info(f"  Cosine Similarity: {result['cosine']:.4f}")
            logger.info(f"  MSE: {result['mse']:.6f}")
            logger.info(f"  R²: {result['r2']:.4f}")
            logger.info(f"  Dimensions: {result['dim']}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "layer_performance.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n✅ Results saved to {results_path}")
    
    # Create visualization
    logger.info("\nCreating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cosine similarity by layer
    ax = axes[0]
    sns.barplot(data=results_df, x='layer', y='cosine', ax=ax, palette='viridis')
    ax.set_title('Cosine Similarity by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, row in results_df.iterrows():
        ax.text(i, row['cosine'] + 0.02, f"{row['cosine']:.3f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MSE by layer
    ax = axes[1]
    sns.barplot(data=results_df, x='layer', y='mse', ax=ax, palette='rocket')
    ax.set_title('MSE by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "layer_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Visualization saved to {plot_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Best layer (cosine): {results_df.loc[results_df['cosine'].idxmax(), 'layer']} "
                f"({results_df['cosine'].max():.4f})")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Results: {output_dir}")
    
    # Save summary
    summary = {
        'model_path': str(args.model_path),
        'subject': args.subject,
        'n_test_samples': len(X_test),
        'pca_k': args.pca_k,
        'results': results_df.to_dict('records')
    }
    
    import json
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Summary saved to {summary_path}")
    
    logger.info("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
