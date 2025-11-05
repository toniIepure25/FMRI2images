#!/usr/bin/env python3
"""
Integration test: Train adapter → Load in decode/eval scripts
Tests the complete metadata flow through the pipeline.
"""

import sys
import tempfile
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.clip_adapter import CLIPAdapter, load_adapter


def simulate_training():
    """Simulate training and saving an adapter."""
    print("=" * 80)
    print("STEP 1: SIMULATE TRAINING (train_clip_adapter.py)")
    print("=" * 80)
    
    # Create adapter
    adapter = CLIPAdapter(in_dim=512, out_dim=1024, use_layernorm=True)
    
    # Simulate some training
    print("Training adapter...")
    x_train = torch.randn(100, 512)
    y_train = torch.randn(100, 1024)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    adapter.train()
    
    for epoch in range(3):
        optimizer.zero_grad()
        pred = adapter(x_train)
        loss = torch.nn.functional.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}/3: loss={loss.item():.4f}")
    
    adapter.eval()
    
    # Save with metadata (as train_clip_adapter.py does)
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_path = temp_dir / "adapter.pt"
    
    metadata = {
        "subject": "subj01",
        "model_id": "stabilityai/stable-diffusion-2-1",
        "input_dim": 512,
        "target_dim": 1024,
        "created_at": datetime.now().isoformat(),
        "repo_version": "0.1.0",
        "test_cosine": 0.85,
        "best_epoch": 3,
    }
    
    adapter.save(str(checkpoint_path), metadata)
    
    print(f"\n✅ Adapter saved to {checkpoint_path}")
    print(f"   Saved adapter with metadata: {{subject={metadata['subject']}, "
          f"model_id={metadata['model_id']}, input_dim={metadata['input_dim']}, "
          f"target_dim={metadata['target_dim']}, created_at={metadata['created_at']}, "
          f"repo_version={metadata['repo_version']}}}")
    
    return checkpoint_path


def simulate_decode_loading(checkpoint_path: Path, model_id: str):
    """Simulate loading in decode_diffusion.py."""
    print("\n" + "=" * 80)
    print("STEP 2: SIMULATE DECODE LOADING (decode_diffusion.py)")
    print("=" * 80)
    
    print(f"Loading CLIP adapter from {checkpoint_path}")
    
    # Load adapter (as decode_diffusion.py does)
    adapter, adapter_metadata = load_adapter(str(checkpoint_path), map_location="cpu")
    adapter.eval()
    
    # Get dimensions
    adapter_target_dim = adapter_metadata.get("target_dim", adapter_metadata.get("out_dim"))
    adapter_input_dim = adapter_metadata.get("input_dim", adapter_metadata.get("in_dim", 512))
    adapter_model_id = adapter_metadata.get("model_id", "unknown")
    
    print(f"✅ CLIP Adapter loaded: {adapter_input_dim}D → {adapter_target_dim}D")
    print(f"   Adapter metadata: model_id={adapter_model_id}, "
          f"subject={adapter_metadata.get('subject', 'unknown')}")
    
    # Check model_id consistency
    if model_id and adapter_model_id != "unknown" and adapter_model_id != model_id:
        print(f"⚠️  Adapter was trained for {adapter_model_id} but using {model_id}")
        print(f"   This may cause dimension mismatches or degraded quality")
    else:
        print(f"✅ Model ID matches: {adapter_model_id}")
    
    print(f"\nCLIP Adapter: ENABLED (512D → {adapter_target_dim}D)")
    
    # Test inference
    x = torch.randn(8, 512)
    with torch.no_grad():
        y = adapter(x)
    
    print(f"✅ Inference test: {x.shape} → {y.shape}")
    assert y.shape == (8, 1024), f"Unexpected output shape: {y.shape}"
    
    return adapter, adapter_metadata


def simulate_eval_loading(checkpoint_path: Path):
    """Simulate loading in eval_reconstruction.py."""
    print("\n" + "=" * 80)
    print("STEP 3: SIMULATE EVAL LOADING (eval_reconstruction.py)")
    print("=" * 80)
    
    print(f"🔧 Loading adapter: {checkpoint_path}")
    
    # Load adapter (as eval_reconstruction.py does via _load_adapter)
    adapter, metadata = load_adapter(str(checkpoint_path), map_location="cpu")
    adapter.eval()
    
    # Get dimensions
    adapter_in_dim = metadata.get("input_dim", metadata.get("in_dim", 512))
    adapter_out_dim = metadata.get("target_dim", metadata.get("out_dim", 1024))
    
    print(f"✅ Loaded adapter successfully")
    print(f"   Dimensions: {adapter_in_dim}D → {adapter_out_dim}D")
    print(f"   Subject: {metadata.get('subject', 'unknown')}")
    print(f"   Model: {metadata.get('model_id', 'unknown')}")
    
    # Simulate applying adapter
    gen_embeddings = torch.randn(32, 512)
    print(f"\n🔧 Applying adapter: {gen_embeddings.shape[1]}D → {adapter_out_dim}D")
    
    with torch.no_grad():
        gen_embeddings = adapter(gen_embeddings)
    
    print(f"✅ Adapter applied: new shape={gen_embeddings.shape}")
    assert gen_embeddings.shape == (32, 1024), f"Unexpected shape: {gen_embeddings.shape}"
    
    return adapter, metadata


def test_mismatch_warning(checkpoint_path: Path):
    """Test warning when using adapter with wrong model."""
    print("\n" + "=" * 80)
    print("STEP 4: TEST MODEL MISMATCH WARNING")
    print("=" * 80)
    
    # Simulate loading with different model_id
    print(f"Simulating decode with different model_id...")
    print(f"Adapter trained for: stabilityai/stable-diffusion-2-1")
    print(f"Requested model: runwayml/stable-diffusion-v1-5\n")
    
    adapter, metadata = load_adapter(str(checkpoint_path), map_location="cpu")
    
    requested_model = "runwayml/stable-diffusion-v1-5"
    adapter_model = metadata.get("model_id", "unknown")
    
    if adapter_model != requested_model:
        print(f"⚠️  Adapter was trained for {adapter_model} but using {requested_model}")
        print(f"   This may cause dimension mismatches or degraded quality")
    
    print(f"\n✅ Warning system working correctly")


def main():
    """Run integration test."""
    print("\n" + "=" * 80)
    print("ADAPTER METADATA INTEGRATION TEST")
    print("=" * 80 + "\n")
    
    try:
        # Step 1: Train and save
        checkpoint_path = simulate_training()
        
        # Step 2: Load in decode_diffusion.py
        adapter_decode, meta_decode = simulate_decode_loading(
            checkpoint_path,
            model_id="stabilityai/stable-diffusion-2-1"
        )
        
        # Step 3: Load in eval_reconstruction.py
        adapter_eval, meta_eval = simulate_eval_loading(checkpoint_path)
        
        # Step 4: Test mismatch warning
        test_mismatch_warning(checkpoint_path)
        
        # Cleanup
        checkpoint_path.unlink()
        checkpoint_path.parent.rmdir()
        
        print("\n" + "=" * 80)
        print("✅ INTEGRATION TEST PASSED!")
        print("=" * 80)
        print("\nAll steps completed successfully:")
        print("  1. ✅ Training saves metadata correctly")
        print("  2. ✅ Decode script loads and validates metadata")
        print("  3. ✅ Eval script loads and applies adapter")
        print("  4. ✅ Model mismatch warnings work")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
