# System Configurations

**Infrastructure and Dataset Settings**

---

## ⚙️ System Configuration Files

### `data.yaml` - Dataset Configuration

**Purpose**: NSD dataset paths, preprocessing, and data loading settings

```yaml
dataset:
  name: nsd  # Natural Scenes Dataset
  root: data/nsddata/
  cache_dir: cache/
  
subjects:
  - subj01
  - subj02
  - subj03
  - subj04
  - subj05
  - subj06
  - subj07
  - subj08

preprocessing:
  pca:
    n_components: 1000
    explained_variance: 0.95
  normalization: zscore
  
data_loading:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
```

**Key Settings**:

- **Paths**: Dataset root, cache directories
- **Subjects**: Which subjects to include
- **Preprocessing**: PCA, normalization methods
- **Loading**: Batch size, workers, memory options

**Common Overrides**:
```bash
# Change cache location
--override "dataset.cache_dir=/mnt/fast_ssd/cache/"

# Use subset of subjects
--override "subjects=[subj01,subj02]"

# Adjust preprocessing
--override "preprocessing.pca.n_components=500"
```

---

### `clip.yaml` - CLIP Model Configuration

**Purpose**: CLIP model selection and feature extraction settings

```yaml
model:
  name: openai/clip-vit-large-patch14
  pretrained: true
  
embedding:
  dimension: 768
  normalize: true
  projection: linear
  
cache:
  enabled: true
  path: cache/clip_embeddings/
  batch_size: 256
```

**Supported Models**:
- `openai/clip-vit-base-patch32` (512-D)
- `openai/clip-vit-large-patch14` (768-D) ⭐ Recommended
- `openai/clip-vit-large-patch14-336` (768-D, higher res)

**Key Settings**:

- **Model**: Which CLIP variant to use
- **Embedding**: Dimension, normalization
- **Cache**: Caching strategy for embeddings

**Common Overrides**:
```bash
# Use different CLIP model
--override "model.name=openai/clip-vit-base-patch32"

# Disable caching
--override "cache.enabled=false"

# Change cache location
--override "cache.path=/mnt/ssd/clip_cache/"
```

---

### `logging.yaml` - Logging Configuration

**Purpose**: Logging levels, formats, and output destinations

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
console:
  enabled: true
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  colorize: true
  
file:
  enabled: true
  path: logs/
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  rotation: daily
  retention: 30  # days
  
experiment_tracking:
  wandb:
    enabled: false
    project: fmri2img
    entity: null
  tensorboard:
    enabled: true
    log_dir: logs/tensorboard/
```

**Log Levels**:
- `DEBUG`: Detailed diagnostic information
- `INFO`: General operational messages ⭐ Default
- `WARNING`: Warning messages (non-critical issues)
- `ERROR`: Error messages
- `CRITICAL`: Critical failures

**Key Settings**:

- **Console**: Terminal output settings
- **File**: Log file configuration
- **Tracking**: W&B, TensorBoard integration

**Common Overrides**:
```bash
# Enable debug logging
--override "logging.level=DEBUG"

# Enable W&B
--override "logging.experiment_tracking.wandb.enabled=true"

# Change log directory
--override "logging.file.path=logs/my_experiment/"
```

---

## 🔧 Common Use Cases

### **Local Development**
```yaml
# Use local data paths
dataset:
  root: data/nsddata/
  cache_dir: cache/

# Verbose logging
logging:
  level: DEBUG
  console.enabled: true
```

### **Remote Server / HPC**
```yaml
# Use mounted storage
dataset:
  root: /mnt/shared/nsddata/
  cache_dir: /mnt/fast_ssd/cache/

# File logging only (no console clutter)
logging:
  level: INFO
  console.enabled: false
  file.enabled: true
```

### **Cloud / S3 Storage**
```yaml
# S3-backed storage
dataset:
  root: s3://my-bucket/nsddata/
  cache_dir: /tmp/cache/
  s3_options:
    region: us-west-2
    parallel_downloads: 4
```

---

## 📝 Configuration Tips

### **Performance Optimization**

```yaml
# Faster data loading
data_loading:
  num_workers: 8          # More workers (if CPU available)
  prefetch_factor: 4      # More prefetching
  pin_memory: true        # Faster GPU transfer
  persistent_workers: true # Keep workers alive
```

### **Memory Optimization**

```yaml
# Reduce memory usage
preprocessing:
  pca.n_components: 500   # Fewer components
data_loading:
  batch_size: 16          # Smaller batches
  num_workers: 2          # Fewer workers
```

### **Debugging**

```yaml
# Maximum verbosity
logging:
  level: DEBUG
  console.enabled: true
  file.enabled: true

# Single worker (easier debugging)
data_loading:
  num_workers: 0
```

---

## 🐛 Troubleshooting

### **Data Loading Errors**

**Issue**: "NSD data not found"
```bash
# Check paths
--override "dataset.root=/correct/path/to/nsddata/"

# Verify data exists
ls /correct/path/to/nsddata/
```

**Issue**: "CLIP cache missing"
```bash
# Build cache first
python scripts/build_clip_cache.py

# Or disable caching
--override "cache.enabled=false"
```

### **Performance Issues**

**Issue**: "Data loading slow"
```bash
# Increase workers
--override "data_loading.num_workers=8"

# Enable prefetching
--override "data_loading.prefetch_factor=4"
```

**Issue**: "Out of memory"
```bash
# Reduce batch size
--override "data_loading.batch_size=16"

# Reduce PCA components
--override "preprocessing.pca.n_components=500"
```

### **Logging Issues**

**Issue**: "Too much log output"
```bash
# Reduce logging level
--override "logging.level=WARNING"

# Disable console logging
--override "logging.console.enabled=false"
```

**Issue**: "Logs not saving"
```bash
# Check log directory exists
mkdir -p logs/

# Verify permissions
chmod 755 logs/
```

---

## 📚 Environment-Specific Configs

### **Local Machine**
```bash
# Use local paths
python script.py \
    --config configs/system/data.yaml \
    --override "dataset.root=data/nsddata/"
```

### **HPC Cluster**
```bash
# Use mounted storage
python script.py \
    --config configs/system/data.yaml \
    --override "dataset.root=/scratch/user/nsddata/" \
    --override "dataset.cache_dir=/tmp/cache/"
```

### **Docker Container**
```bash
# Use mounted volumes
docker run -v /host/data:/data \
    fmri2img \
    --config configs/system/data.yaml \
    --override "dataset.root=/data/nsddata/"
```

---

## 🔐 Security Considerations

### **Credentials**

Never commit credentials to configs:

```yaml
# ❌ BAD: Hardcoded credentials
s3:
  access_key: AKIAIOSFODNN7EXAMPLE
  secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# ✅ GOOD: Use environment variables
s3:
  access_key: ${AWS_ACCESS_KEY_ID}
  secret_key: ${AWS_SECRET_ACCESS_KEY}
```

### **Paths**

Use relative paths or environment variables:

```yaml
# ✅ GOOD: Relative paths
dataset:
  root: data/nsddata/
  cache_dir: cache/

# ✅ GOOD: Environment variables
dataset:
  root: ${NSD_DATA_ROOT}
  cache_dir: ${CACHE_DIR}
```

---

## 📚 Related Documentation

- **[Main Config README](../README.md)** - Overview
- **[NSD Dataset Guide](../../docs/technical/NSD_Dataset_Guide.md)** - Dataset details
- **[Data Validation](../../docs/technical/DATA_VALIDATION_REAL_VS_FALLBACK.md)** - Validation
- **[Usage Examples](../../USAGE_EXAMPLES.md)** - Complete commands

---

**Last Updated**: December 7, 2025  
**Status**: Production-Ready  
**Note**: Customize paths for your environment
