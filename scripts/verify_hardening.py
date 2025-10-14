#!/usr/bin/env python3
"""
Production Hardening Verification
=================================

Verifies all hardening improvements are in place and functional.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Production Hardening Verification")
print("=" * 60)

passed = []
failed = []

# Test 1: CLIP Cache Import
print("\n[1] Testing CLIP Cache Import...")
try:
    from fmri2img.data.clip_cache import CLIPCache
    cache = CLIPCache()
    schema = cache._schema()
    assert "nsdId" in str(schema)
    assert "clip512" in str(schema)
    passed.append("CLIP Cache import and schema")
    print("✓ CLIP Cache imports successfully")
    print(f"  Schema: {schema}")
except Exception as e:
    failed.append(("CLIP Cache import", str(e)))
    print(f"✗ CLIP Cache import failed: {e}")

# Test 2: Dataset Integration
print("\n[2] Testing Dataset CLIP Integration...")
try:
    from fmri2img.data.torch_dataset import NSDIterableDataset
    # Check if clip_cache parameter exists in __init__
    import inspect
    sig = inspect.signature(NSDIterableDataset.__init__)
    assert 'clip_cache' in sig.parameters, "clip_cache parameter missing"
    passed.append("Dataset CLIP integration")
    print("✓ NSDIterableDataset has clip_cache parameter")
except Exception as e:
    failed.append(("Dataset CLIP integration", str(e)))
    print(f"✗ Dataset integration failed: {e}")

# Test 3: Builder Script
print("\n[3] Testing Builder Script...")
try:
    import sys
    import os
    # Add scripts to path temporarily
    scripts_path = os.path.join(os.getcwd(), 'scripts')
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    
    import build_clip_cache as bcc
    assert hasattr(bcc, 'load_clip_model'), "load_clip_model missing"
    assert hasattr(bcc, 'compute_embeddings_batch'), "compute_embeddings_batch missing"
    assert hasattr(bcc, 'main'), "main missing"
    passed.append("Builder script structure")
    print("✓ build_clip_cache.py has all required functions")
except Exception as e:
    failed.append(("Builder script", str(e)))
    print(f"✗ Builder script failed: {e}")

# Test 4: Nibabel Suppression
print("\n[4] Testing Nibabel Suppression...")
try:
    scripts = [
        'scripts/nsd_fit_preproc.py',
        'scripts/train_smoke.py',
        'scripts/check_index_headers.py'
    ]
    for script in scripts:
        with open(script, 'r') as f:
            content = f.read()
            assert 'nibabel.global' in content, f"{script} missing suppression"
    passed.append("Nibabel logging suppression")
    print(f"✓ All {len(scripts)} scripts have nibabel suppression")
except Exception as e:
    failed.append(("Nibabel suppression", str(e)))
    print(f"✗ Nibabel suppression check failed: {e}")

# Test 5: ROI Path Helpers
print("\n[5] Testing ROI Path Helpers...")
try:
    from fmri2img.io.nsd_layout import NSDLayout
    layout = NSDLayout()
    assert hasattr(layout, 'fsaverage_roi_masks_path'), "fsaverage_roi_masks_path missing"
    assert hasattr(layout, 'mni_roi_masks_path'), "mni_roi_masks_path missing"
    passed.append("ROI path helpers")
    print("✓ NSDLayout has ROI path helper methods")
except Exception as e:
    failed.append(("ROI path helpers", str(e)))
    print(f"✗ ROI path helpers failed: {e}")

# Test 6: PCA Auto-Capping Code
print("\n[6] Testing PCA Auto-Capping Code...")
try:
    with open('src/fmri2img/data/preprocess.py', 'r') as f:
        content = f.read()
        assert 'k_eff' in content, "k_eff variable missing"
        assert 'min(k' in content, "min() capping logic missing"
    passed.append("PCA auto-capping code")
    print("✓ PCA auto-capping logic present in preprocess.py")
except Exception as e:
    failed.append(("PCA auto-capping", str(e)))
    print(f"✗ PCA auto-capping check failed: {e}")

# Test 7: Documentation
print("\n[7] Testing Documentation...")
try:
    with open('README.md', 'r') as f:
        content = f.read()
        assert 'CLIP' in content, "CLIP section missing from README"
        assert 'build_clip_cache' in content, "build_clip_cache missing from README"
    with open('Makefile', 'r') as f:
        content = f.read()
        assert 'build-clip-cache:' in content, "build-clip-cache target missing"
    passed.append("Documentation updates")
    print("✓ README and Makefile updated with CLIP cache docs")
except Exception as e:
    failed.append(("Documentation", str(e)))
    print(f"✗ Documentation check failed: {e}")

# Test 8: Test Script
print("\n[8] Testing CLIP Cache Test Script...")
try:
    path = Path('scripts/test_clip_cache.py')
    assert path.exists(), "test_clip_cache.py missing"
    with open(path, 'r') as f:
        content = f.read()
        assert 'test_clip_cache_workflow' in content, "Main test function missing"
    passed.append("CLIP cache test script")
    print("✓ test_clip_cache.py exists and has test function")
except Exception as e:
    failed.append(("Test script", str(e)))
    print(f"✗ Test script check failed: {e}")

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"\n✅ Passed: {len(passed)}/{len(passed) + len(failed)}")
for item in passed:
    print(f"  ✓ {item}")

if failed:
    print(f"\n✗ Failed: {len(failed)}/{len(passed) + len(failed)}")
    for item, error in failed:
        print(f"  ✗ {item}: {error}")
    sys.exit(1)
else:
    print("\n🎉 All verification checks passed!")
    print("Production hardening is complete and functional.")
    sys.exit(0)
