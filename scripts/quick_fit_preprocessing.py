#!/usr/bin/env python3
"""
Quick script to fit preprocessing for a subject using config defaults.
"""

import sys
from pathlib import Path
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fmri2img.data.preprocess import NSDPreprocessor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def fit_preprocessing_for_subject(subject="subj01"):
    """Fit preprocessing pipeline for a subject."""
    
    index_path = f"data/indices/nsd_index/subject={subject}/index.parquet"
    output_dir = f"outputs/preproc/{subject}"
    
    log.info(f"Fitting preprocessing for {subject}")
    log.info(f"  Index: {index_path}")
    log.info(f"  Output: {output_dir}")
    
    # Check if index exists
    if not Path(index_path).exists():
        log.error(f"❌ Index not found: {index_path}")
        log.error(f"   Build it first: python scripts/build_full_index.py")
        return False
    
    # Load index and get training split
    import pandas as pd
    log.info("Loading index...")
    index_df = pd.read_parquet(index_path)
    
    # Check for split column
    if 'split' not in index_df.columns:
        log.warning("Index doesn't have 'split' column, using first 80% as training")
        n = len(index_df)
        train_end = int(0.8 * n)
        index_df['split'] = 'test'
        index_df.loc[:train_end-1, 'split'] = 'train'
    
    train_df = index_df[index_df['split'] == 'train'].copy()
    log.info(f"Loaded {len(train_df)} training samples from {len(index_df)} total")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Fit preprocessing
    log.info("Fitting preprocessing pipeline (this may take several minutes)...")
    preprocessor = NSDPreprocessor(subject=subject, out_dir="outputs/preproc")
    
    # Create loader factory
    from src.fmri2img.io.s3 import NIfTILoader, get_s3_filesystem
    def loader_factory():
        s3_fs = get_s3_filesystem()
        loader = NIfTILoader(s3_fs)
        def get_volume(loader, row):
            img = loader.load(row['beta_path'])
            data = img.get_fdata()
            return data[..., row['beta_index']]
        return loader, get_volume
    
    try:
        preprocessor.fit(
            train_df=train_df,
            loader_factory=loader_factory,
            reliability_mode="soft_weight",
            reliability_threshold=0.1,
            min_variance=1e-6
        )
        
        # Save
        output_path = Path(output_dir) / "preprocessor.pkl"
        preprocessor.save(str(output_path))
        
        log.info(f"✅ Preprocessing fitted and saved to: {output_path}")
        
        # Print summary
        summary = preprocessor.summary()
        log.info(f"\nPreprocessing Summary:")
        log.info(f"  Voxels kept: {summary.get('n_voxels_kept', 'N/A'):,}")
        log.info(f"  PCA fitted: {summary.get('pca_fitted', False)}")
        if summary.get('pca_fitted'):
            log.info(f"  PCA components: {summary.get('pca_components', 'N/A')}")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Error fitting preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    subject = sys.argv[1] if len(sys.argv) > 1 else "subj01"
    success = fit_preprocessing_for_subject(subject)
    sys.exit(0 if success else 1)
