#!/usr/bin/env python3
"""
Phase 2 Verification Script: Multi-Task Semantics (Image + Text CLIP)

Tests all Phase 2 components:
1. Text-CLIP cache format validation
2. MultiLayerTwoStageEncoder text head creation
3. Forward pass with text prediction
4. MultiLayerLoss text-CLIP weighting
5. Dataset loading with text targets
6. End-to-end training integration

Run before training with Phase 2 enabled to ensure all components are configured correctly.

Usage:
    python scripts/verify_phase2_multitask.py [--text-clip-cache cache/clip_embeddings/text_clip.parquet]
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fmri2img.models.encoders import MultiLayerTwoStageEncoder
from src.fmri2img.training.losses import MultiLayerLoss


def test_text_clip_cache(cache_path: Path) -> dict:
    """Test 1: Validate text-CLIP cache format."""
    logger.info("=" * 70)
    logger.info("TEST 1: Text-CLIP Cache Format Validation")
    logger.info("=" * 70)
    
    if not cache_path.exists():
        logger.warning(f"❌ Cache file not found: {cache_path}")
        logger.info(f"   To generate cache, run:")
        logger.info(f"   python scripts/build_text_clip_cache.py --image-dir cache/stimuli --output {cache_path}")
        return {'status': 'SKIP', 'reason': 'Cache not found'}
    
    try:
        df = pd.read_parquet(cache_path)
        logger.info(f"✅ Loaded cache with {len(df)} entries")
        
        # Check required columns
        required_cols = ['nsdId', 'text_clip_embedding']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"❌ Missing columns: {missing}")
            return {'status': 'FAIL', 'reason': f'Missing columns: {missing}'}
        
        # Check embedding format
        sample_emb = np.array(df.iloc[0]['text_clip_embedding'])
        if sample_emb.shape != (512,):
            logger.error(f"❌ Expected shape (512,), got {sample_emb.shape}")
            return {'status': 'FAIL', 'reason': f'Wrong shape: {sample_emb.shape}'}
        
        logger.info(f"✅ Embedding shape: {sample_emb.shape} (correct)")
        logger.info(f"✅ Embedding dtype: {sample_emb.dtype}")
        logger.info(f"✅ NSD IDs range: {df['nsdId'].min()} - {df['nsdId'].max()}")
        
        # Build cache dict for later use
        cache_dict = {}
        for _, row in df.iterrows():
            cache_dict[int(row['nsdId'])] = np.array(row['text_clip_embedding'], dtype=np.float32)
        
        return {'status': 'PASS', 'cache_dict': cache_dict, 'count': len(cache_dict)}
        
    except Exception as e:
        logger.error(f"❌ Failed to load cache: {e}")
        return {'status': 'FAIL', 'reason': str(e)}


def test_encoder_text_head(predict_text_clip: bool = True) -> dict:
    """Test 2: MultiLayerTwoStageEncoder text head creation."""
    logger.info("=" * 70)
    logger.info("TEST 2: MultiLayerTwoStageEncoder Text Head")
    logger.info("=" * 70)
    
    try:
        # Test both shared and independent backbone modes
        for shared_head_backbone in [True, False]:
            mode = "shared" if shared_head_backbone else "independent"
            logger.info(f"\nTesting {mode} backbone mode...")
            
            model = MultiLayerTwoStageEncoder(
                input_dim=512,
                latent_dim=1024,
                n_blocks=4,
                dropout=0.2,
                head_type='mlp',
                head_hidden_dim=1536,
                shared_head_backbone=shared_head_backbone,
                predict_text_clip=predict_text_clip
            )
            
            # Check heads
            expected_heads = ['layer_4', 'layer_8', 'layer_12', 'final']
            if predict_text_clip:
                expected_heads.append('text')
            
            actual_heads = list(model.heads.keys())
            if set(actual_heads) != set(expected_heads):
                logger.error(f"❌ {mode}: Expected heads {expected_heads}, got {actual_heads}")
                return {'status': 'FAIL', 'reason': f'{mode} mode has wrong heads'}
            
            logger.info(f"✅ {mode}: Correct heads present: {actual_heads}")
            
            # Check text head dimensions
            if predict_text_clip:
                text_head = model.heads['text']
                # Get output dim by checking the last layer
                if hasattr(text_head, 'out_features'):
                    out_dim = text_head.out_features
                elif hasattr(text_head, 'projection'):
                    out_dim = text_head.projection.out_features
                else:
                    logger.warning(f"⚠️  {mode}: Cannot determine text head output dim")
                    out_dim = None
                
                if out_dim != 512:
                    logger.error(f"❌ {mode}: Text head output dim {out_dim}, expected 512")
                    return {'status': 'FAIL', 'reason': f'{mode} text head wrong dimension'}
                
                logger.info(f"✅ {mode}: Text head output dim = 512 (correct)")
        
        return {'status': 'PASS'}
        
    except Exception as e:
        logger.error(f"❌ Failed to create encoder: {e}")
        return {'status': 'FAIL', 'reason': str(e)}


def test_forward_pass() -> dict:
    """Test 3: Forward pass with text prediction."""
    logger.info("=" * 70)
    logger.info("TEST 3: Forward Pass with Text Prediction")
    logger.info("=" * 70)
    
    try:
        model = MultiLayerTwoStageEncoder(
            input_dim=512,
            latent_dim=1024,
            n_blocks=4,
            dropout=0.0,
            predict_text_clip=True
        )
        model.eval()
        
        # Create dummy input
        batch_size = 8
        x = torch.randn(batch_size, 512)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x)
        
        # Check output keys
        expected_keys = {'layer_4', 'layer_8', 'layer_12', 'final', 'text'}
        if set(outputs.keys()) != expected_keys:
            logger.error(f"❌ Expected keys {expected_keys}, got {outputs.keys()}")
            return {'status': 'FAIL', 'reason': 'Wrong output keys'}
        
        logger.info(f"✅ Output keys: {list(outputs.keys())}")
        
        # Check shapes
        for key, tensor in outputs.items():
            expected_shape = (batch_size, 768) if key.startswith('layer_') else (batch_size, 512)
            if tensor.shape != expected_shape:
                logger.error(f"❌ {key}: Expected shape {expected_shape}, got {tensor.shape}")
                return {'status': 'FAIL', 'reason': f'{key} wrong shape'}
            logger.info(f"✅ {key}: shape {tensor.shape} (correct)")
        
        # Check values are not NaN/Inf
        for key, tensor in outputs.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.error(f"❌ {key}: Contains NaN or Inf")
                return {'status': 'FAIL', 'reason': f'{key} has invalid values'}
        
        logger.info(f"✅ All outputs have valid values (no NaN/Inf)")
        
        return {'status': 'PASS'}
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return {'status': 'FAIL', 'reason': str(e)}


def test_loss_weighting() -> dict:
    """Test 4: MultiLayerLoss text-CLIP weighting."""
    logger.info("=" * 70)
    logger.info("TEST 4: MultiLayerLoss Text-CLIP Weighting")
    logger.info("=" * 70)
    
    try:
        # Test different text weights
        for text_weight in [0.0, 0.3, 0.5, 1.0]:
            logger.info(f"\nTesting text_weight={text_weight}...")
            
            criterion = MultiLayerLoss(
                layer_weights={'layer_4': 0.15, 'layer_8': 0.2, 'layer_12': 0.25, 'final': 0.4},
                text_clip_weight=text_weight
            )
            
            # Create dummy predictions and targets
            batch_size = 4
            pred_dict = {
                'layer_4': torch.randn(batch_size, 768),
                'layer_8': torch.randn(batch_size, 768),
                'layer_12': torch.randn(batch_size, 768),
                'final': torch.randn(batch_size, 512),
                'text': torch.randn(batch_size, 512)
            }
            
            target_dict = {
                'layer_4': torch.randn(batch_size, 768),
                'layer_8': torch.randn(batch_size, 768),
                'layer_12': torch.randn(batch_size, 768),
                'final': torch.randn(batch_size, 512),
                'text': torch.randn(batch_size, 512)
            }
            
            # Compute loss
            loss, components = criterion(pred_dict, target_dict, return_components=True)
            
            # Check components
            expected_keys = {'layer_4', 'layer_8', 'layer_12', 'final', 'text', 'image_total'}
            if not expected_keys.issubset(components.keys()):
                logger.error(f"❌ Missing loss components. Expected {expected_keys}, got {components.keys()}")
                return {'status': 'FAIL', 'reason': 'Missing loss components'}
            
            logger.info(f"✅ Loss components: {list(components.keys())}")
            logger.info(f"   Total loss: {loss.item():.6f}")
            logger.info(f"   Image total: {components['image_total']:.6f}")
            logger.info(f"   Text loss: {components['text']:.6f}")
            
            # Verify weighting formula: total = (1-w)*image + w*text
            expected_total = (1.0 - text_weight) * components['image_total'] + text_weight * components['text']
            if not torch.isclose(loss, torch.tensor(expected_total), atol=1e-5):
                logger.error(f"❌ Loss weighting incorrect. Expected {expected_total:.6f}, got {loss.item():.6f}")
                return {'status': 'FAIL', 'reason': 'Incorrect loss weighting'}
            
            logger.info(f"✅ Weighting formula verified: (1-{text_weight})*{components['image_total']:.6f} + {text_weight}*{components['text']:.6f} = {loss.item():.6f}")
        
        return {'status': 'PASS'}
        
    except Exception as e:
        logger.error(f"❌ Loss test failed: {e}")
        return {'status': 'FAIL', 'reason': str(e)}


def test_dataset_integration(cache_dict: dict = None) -> dict:
    """Test 5: Dataset loading with text targets."""
    logger.info("=" * 70)
    logger.info("TEST 5: Dataset Loading with Text Targets")
    logger.info("=" * 70)
    
    if cache_dict is None:
        logger.warning("❌ No cache dict provided, skipping test")
        return {'status': 'SKIP', 'reason': 'No cache dict'}
    
    try:
        # Simulate multi-layer cache
        nsd_ids = list(cache_dict.keys())[:10]  # Use first 10 samples
        
        multilayer_cache = {}
        for nsd_id in nsd_ids:
            multilayer_cache[nsd_id] = {
                'layer_4': np.random.randn(768).astype(np.float32),
                'layer_8': np.random.randn(768).astype(np.float32),
                'layer_12': np.random.randn(768).astype(np.float32),
                'final': np.random.randn(512).astype(np.float32)
            }
        
        # Add text targets
        for nsd_id in nsd_ids:
            if nsd_id in cache_dict:
                multilayer_cache[nsd_id]['text'] = cache_dict[nsd_id]
        
        # Check combined dict
        sample_dict = multilayer_cache[nsd_ids[0]]
        if 'text' not in sample_dict:
            logger.error(f"❌ Text key not added to multi-layer dict")
            return {'status': 'FAIL', 'reason': 'Text key missing'}
        
        logger.info(f"✅ Combined dict keys: {list(sample_dict.keys())}")
        logger.info(f"✅ Text shape: {sample_dict['text'].shape}")
        
        # Create simple dataset
        class MultiTaskDataset(torch.utils.data.Dataset):
            def __init__(self, cache_dict):
                self.nsd_ids = list(cache_dict.keys())
                self.cache = cache_dict
            
            def __len__(self):
                return len(self.nsd_ids)
            
            def __getitem__(self, idx):
                nsd_id = self.nsd_ids[idx]
                targets = self.cache[nsd_id]
                # Simulate fMRI input
                x = torch.randn(512)
                y_dict = {k: torch.from_numpy(v).float() for k, v in targets.items()}
                return x, y_dict
        
        dataset = MultiTaskDataset(multilayer_cache)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Test loading one batch
        x_batch, y_batch = next(iter(loader))
        
        logger.info(f"✅ Batch loaded successfully")
        logger.info(f"   X shape: {x_batch.shape}")
        logger.info(f"   Y keys: {list(y_batch.keys())}")
        for key, tensor in y_batch.items():
            logger.info(f"   {key}: {tensor.shape}")
        
        if 'text' not in y_batch:
            logger.error(f"❌ Text not in batch targets")
            return {'status': 'FAIL', 'reason': 'Text missing from batch'}
        
        return {'status': 'PASS'}
        
    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")
        return {'status': 'FAIL', 'reason': str(e)}


def test_end_to_end() -> dict:
    """Test 6: End-to-end training step."""
    logger.info("=" * 70)
    logger.info("TEST 6: End-to-End Training Step")
    logger.info("=" * 70)
    
    try:
        # Create model
        model = MultiLayerTwoStageEncoder(
            input_dim=512,
            latent_dim=1024,
            n_blocks=4,
            dropout=0.0,
            predict_text_clip=True
        )
        
        # Create loss
        criterion = MultiLayerLoss(
            layer_weights={'layer_4': 0.15, 'layer_8': 0.2, 'layer_12': 0.25, 'final': 0.4},
            text_clip_weight=0.3
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create dummy batch
        batch_size = 8
        x = torch.randn(batch_size, 512)
        y_dict = {
            'layer_4': torch.randn(batch_size, 768),
            'layer_8': torch.randn(batch_size, 768),
            'layer_12': torch.randn(batch_size, 768),
            'final': torch.randn(batch_size, 512),
            'text': torch.randn(batch_size, 512)
        }
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward
        pred_dict = model(x)
        
        # Loss
        loss, components = criterion(pred_dict, y_dict, return_components=True)
        
        # Backward
        loss.backward()
        
        # Check gradients
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        if not has_grads:
            logger.error(f"❌ No gradients computed")
            return {'status': 'FAIL', 'reason': 'No gradients'}
        
        logger.info(f"✅ Gradients computed successfully")
        
        # Update
        optimizer.step()
        
        logger.info(f"✅ Training step completed")
        logger.info(f"   Loss: {loss.item():.6f}")
        logger.info(f"   Components: {components}")
        
        return {'status': 'PASS'}
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        return {'status': 'FAIL', 'reason': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 2 Multi-Task Semantics")
    parser.add_argument("--text-clip-cache", type=str,
                       default="cache/clip_embeddings/text_clip.parquet",
                       help="Path to text-CLIP cache")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PHASE 2 VERIFICATION: MULTI-TASK SEMANTICS (IMAGE + TEXT CLIP)")
    logger.info("=" * 70)
    
    # Run tests
    results = {}
    cache_dict = None
    
    # Test 1: Cache validation
    result1 = test_text_clip_cache(Path(args.text_clip_cache))
    results['cache_validation'] = result1
    if result1['status'] == 'PASS':
        cache_dict = result1.get('cache_dict')
    
    # Test 2: Encoder text head
    results['encoder_text_head'] = test_encoder_text_head(predict_text_clip=True)
    
    # Test 3: Forward pass
    results['forward_pass'] = test_forward_pass()
    
    # Test 4: Loss weighting
    results['loss_weighting'] = test_loss_weighting()
    
    # Test 5: Dataset integration
    results['dataset_integration'] = test_dataset_integration(cache_dict)
    
    # Test 6: End-to-end
    results['end_to_end'] = test_end_to_end()
    
    # Summary
    logger.info("=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    failed = sum(1 for r in results.values() if r['status'] == 'FAIL')
    skipped = sum(1 for r in results.values() if r['status'] == 'SKIP')
    
    for test_name, result in results.items():
        status_symbol = {
            'PASS': '✅',
            'FAIL': '❌',
            'SKIP': '⏭️'
        }.get(result['status'], '❓')
        
        logger.info(f"{status_symbol} {test_name}: {result['status']}")
        if result['status'] == 'FAIL':
            logger.info(f"   Reason: {result.get('reason', 'Unknown')}")
        elif result['status'] == 'SKIP':
            logger.info(f"   Reason: {result.get('reason', 'Unknown')}")
    
    logger.info("")
    logger.info(f"Total: {len(results)} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    
    if failed > 0:
        logger.error("❌ VERIFICATION FAILED")
        sys.exit(1)
    elif skipped == len(results):
        logger.warning("⚠️  ALL TESTS SKIPPED")
        sys.exit(2)
    else:
        logger.info("✅ VERIFICATION PASSED")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Generate text-CLIP cache if not exists:")
        logger.info(f"   python scripts/build_text_clip_cache.py --image-dir cache/stimuli --output {args.text_clip_cache}")
        logger.info("")
        logger.info("2. Train with Phase 2 enabled:")
        logger.info("   python scripts/train_two_stage.py \\")
        logger.info("       --config configs/sota_two_stage.yaml \\")
        logger.info("       --multi-layer --predict-text-clip \\")
        logger.info("       --text-clip-weight 0.3")
        sys.exit(0)


if __name__ == "__main__":
    main()
