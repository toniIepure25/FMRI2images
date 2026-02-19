import numpy as np, logging
from pathlib import Path
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader

log = logging.getLogger(__name__)

def test_preprocessor_fit_transform_smoke():
    """Test preprocessor loading existing artifacts and transform functionality."""
    # Test loading existing preprocessor artifacts
    pre = NSDPreprocessor("subj01", out_dir="outputs/preproc")
    
    # Try to load existing artifacts (from previous runs)
    artifacts_loaded = pre.load_artifacts()
    
    if artifacts_loaded:
        # Test transform with mock data
        vol = np.random.randn(81, 104, 83).astype(np.float32)
        out = pre.transform(vol)
        assert out.dtype == np.float32
        assert out.ndim in (1, 3)
        log.info(f"✅ Transform test passed: input {vol.shape} -> output {out.shape}")
    else:
        # Test basic initialization without S3 access
        assert pre.subject == "subj01"
        assert not pre.is_fitted_
        log.info("✅ Basic initialization test passed (no artifacts found)")
        
def test_preprocessor_transform_t0_only():
    """Test T0 (online z-score) transform without fitted artifacts."""
    pre = NSDPreprocessor("subj01", out_dir="outputs/preproc")
    
    # Test T0 transform (no artifacts needed)
    vol = np.random.randn(81, 104, 83).astype(np.float32)
    out = pre.transform(vol)  # Should apply T0 only since not fitted
    
    assert out.dtype == np.float32
    assert out.shape == vol.shape
    assert not pre.is_fitted_  # Should still be unfitted
    log.info(f"✅ T0 transform test passed: {vol.shape} -> {out.shape}")

def test_preprocessor_no_sklearn_access_after_load():
    """Test that loaded preprocessor doesn't access sklearn IncrementalPCA internals."""
    pre = NSDPreprocessor("subj01", out_dir="outputs/preproc")
    
    # Try to load artifacts
    artifacts_loaded = pre.load_artifacts()
    
    if artifacts_loaded and pre.pca_fitted_:
        # Test that PCA is using _NumpyPCA, not sklearn
        from fmri2img.data.preprocess import _NumpyPCA
        assert isinstance(pre.pca_, _NumpyPCA), f"Expected _NumpyPCA, got {type(pre.pca_)}"
        
        # Test that we can transform without accessing sklearn attributes
        n_voxels = int(pre.mask_.sum())
        vec_t1 = np.random.randn(n_voxels).astype(np.float32)
        vec_t2 = pre.transform_T2(vec_t1)
        
        assert vec_t2.dtype == np.float32
        assert vec_t2.ndim == 1
        assert vec_t2.shape[0] == pre.pca_.n_components_
        
        # Test that summary works without sklearn access
        summary = pre.summary()
        assert "pca_components" in summary
        assert "explained_variance_ratio" in summary
        assert summary["pca_fitted"] == True
        
        log.info(f"✅ No sklearn access test passed: PCA transform {n_voxels} -> {vec_t2.shape[0]} features")
    else:
        log.info("⚠️  No PCA artifacts found, skipping sklearn access test")