# Utility Scripts

System utilities, validation, and health checks.

**17 utility scripts**

## Health Checks

- **`preflight.py`** ⭐ - Quick system preflight checks
  - Verify Python environment
  - Check dependencies
  - Validate data paths
  - Fast sanity check before experiments
  
- **`doctor.py`** ⭐ - Comprehensive health diagnostics
  - Deep system validation
  - GPU checks
  - Dataset integrity
  - Configuration validation
  
- `smoke.py` - Quick smoke test
  - Import all modules
  - Check basic functionality
  - Quick health check

- `train_smoke.py` - Training smoke test

## Validation

- **`verify_dataset.py`** ⭐ - Dataset verification
  - Validate NSD data structure
  - Check file integrity
  - Verify CLIP embeddings
  
- `validate_checkpoints.py` - Validate checkpoint files
  - Check checkpoint format
  - Verify model state
  - Ensure compatibility
  
- `validate_index.py` - Validate index files
  - Check index file format
  - Verify data completeness
  
- `check_index_headers.py` - Validate index file headers
  - Quick header validation
  - Schema checks

## Other Utilities

- **`fetch_models.py`** ⭐ - Fetch HuggingFace models
  - Download pretrained models
  - Cache management
  
- `archive_experiment.py` - Archive experiment results
  - Package experiment outputs
  - Compress and organize
  
- `write_run_manifest.py` - Write experiment manifests
  - Generate metadata
  - Track configurations

- `quick_status.py` - Quick project status check

- `download_sd_model.py` - Download Stable Diffusion models

## Usage

```bash
# Quick preflight check
python scripts/utils/preflight.py

# Comprehensive health check
python scripts/utils/doctor.py

# Verify dataset
python scripts/utils/verify_dataset.py --subject subj01

# Fetch models
python scripts/utils/fetch_models.py

# Archive experiment
python scripts/utils/archive_experiment.py --run-dir outputs/exp0/
```
