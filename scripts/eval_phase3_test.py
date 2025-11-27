#!/usr/bin/env python3
"""
Quick evaluation script for Phase 3 probabilistic model on test set.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.encoders import load_probabilistic_encoder
from fmri2img.models.ridge import evaluate_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load model
    checkpoint_path = Path("checkpoints/clip_adapter/subj01/phase3_probabilistic")
    logger.info(f"Loading model from {checkpoint_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_probabilistic_encoder(str(checkpoint_path), map_location=device)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    
    # Load test data from training
    import pickle
    test_data_file = Path("outputs/preproc/.temp_phase3_test_data.pkl")
    
    if not test_data_file.exists():
        logger.error(f"Test data file not found: {test_data_file}")
        logger.info("The test data should have been saved during training.")
        return
    
    with open(test_data_file, 'rb') as f:
        data = pickle.load(f)
    
    X_test = torch.from_numpy(data['X']).float()
    Y_test_final = torch.from_numpy(data['Y']['final']).float()
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Evaluate in batches
    batch_size = 128
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size].to(device)
            Y_batch = Y_test_final[i:i+batch_size]
            
            # Get predictions (use mean, not sampling)
            Y_pred_dict, _ = model(X_batch, sample=False, return_kl=False)
            
            all_preds.append(Y_pred_dict['final'].cpu().numpy())
            all_targets.append(Y_batch.numpy())
    
    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_targets)
    
    # Compute metrics
    metrics = evaluate_predictions(Y_true, Y_pred, normalize=True)
    
    print("=" * 80)
    print("PHASE 3 PROBABILISTIC MODEL - FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Test Cosine:  {metrics['cosine']:.4f}")
    print(f"Test Samples: {len(Y_true)}")
    print("=" * 80)
    
    # Save results
    results = {
        'test_cosine': metrics['cosine'],
        'test_samples': len(Y_true),
        'model_path': str(checkpoint_path),
        'best_epoch': meta.get('epoch', 'unknown')
    }
    
    import json
    results_file = Path("outputs/eval/phase3_probabilistic_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
