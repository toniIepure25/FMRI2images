"""
Deployment Scripts for fMRI-to-Image Reconstruction
====================================================

This directory contains production-ready deployment scripts for running
training jobs, evaluation pipelines, and reconstruction workflows.

Scripts Overview
----------------

1. run_training.sh
   Purpose: Launch full training pipeline for all models
   Usage: ./scripts/deployment/run_training.sh [subject] [config]
   Features:
   - Multi-model training orchestration
   - Automatic checkpoint management
   - Comprehensive logging
   - Error handling and recovery

2. run_with_adapter.sh
   Purpose: Run reconstruction with CLIP adapter
   Usage: ./scripts/deployment/run_with_adapter.sh [checkpoint]
   Features:
   - Adapter-based image reconstruction
   - Stable Diffusion integration
   - Batch processing support
   - Quality metrics computation

3. run_phase3_probabilistic.sh
   Purpose: Run probabilistic multilayer training
   Usage: ./scripts/deployment/run_phase3_probabilistic.sh
   Features:
   - InfoNCE loss training
   - Multilayer architecture
   - Probabilistic embeddings

Deployment Workflow
-------------------

Step 1: Environment Setup
```bash
# Activate environment
conda activate fmri2img

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

Step 2: Data Preparation
```bash
# Build indices
python scripts/build_full_index.py --subjects all

# Build CLIP cache
python scripts/build_clip_cache.py --cache-dir cache/clip_embeddings
```

Step 3: Model Training
```bash
# Train all models for subject 01
./scripts/deployment/run_training.sh subj01

# Or train specific model
python scripts/train_mlp.py \
    --config configs/mlp_standard.yaml \
    --subject subj01
```

Step 4: Evaluation
```bash
# Compare all models
python scripts/compare_evals.py \
    --models ridge mlp two_stage \
    --subject subj01 \
    --output outputs/eval/comparison.json
```

Step 5: Reconstruction
```bash
# Generate images with best model
./scripts/deployment/run_with_adapter.sh \
    checkpoints/two_stage/best.pt
```

Production Checklist
--------------------

Prerequisites:
- [ ] Python 3.10+ installed
- [ ] CUDA 11.8+ available (for GPU)
- [ ] 32GB+ RAM (for full dataset)
- [ ] 100GB+ disk space (for data + checkpoints)

Data:
- [ ] NSD dataset downloaded (82GB)
- [ ] Indices built (data/indices/*.pkl)
- [ ] CLIP cache built (cache/clip_embeddings/)
- [ ] Validation data prepared

Environment:
- [ ] Dependencies installed (requirements.txt)
- [ ] Environment activated
- [ ] GPU available and tested
- [ ] Logging directory created (logs/)

Configuration:
- [ ] Config files validated (configs/*.yaml)
- [ ] Hyperparameters tuned
- [ ] Paths correctly set
- [ ] Seed fixed for reproducibility

Monitoring:
- [ ] Logging enabled
- [ ] Checkpoint directory configured
- [ ] Evaluation metrics defined
- [ ] Alert mechanisms (optional)

Script Configuration
--------------------

### run_training.sh

Environment Variables:
- SUBJECT: Subject to train on (default: subj01)
- CONFIG_DIR: Configuration directory (default: configs/)
- CHECKPOINT_DIR: Checkpoint save location (default: checkpoints/)
- LOG_DIR: Log directory (default: logs/)

Example:
```bash
export SUBJECT=subj01
export CONFIG_DIR=configs/
export CHECKPOINT_DIR=checkpoints/
./scripts/deployment/run_training.sh
```

### run_with_adapter.sh

Environment Variables:
- CHECKPOINT: Path to adapter checkpoint
- OUTPUT_DIR: Output directory for images
- NUM_SAMPLES: Number of samples to reconstruct
- SD_MODEL: Stable Diffusion model version

Example:
```bash
export CHECKPOINT=checkpoints/clip_adapter/best.pt
export OUTPUT_DIR=outputs/reconstructions/
export NUM_SAMPLES=100
./scripts/deployment/run_with_adapter.sh
```

Performance Optimization
------------------------

GPU Optimization:
```bash
# Set optimal batch size for your GPU
export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=64  # Adjust based on VRAM

# Enable TF32 for A100 GPUs
export TORCH_ALLOW_TF32=1
```

Memory Optimization:
```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable gradient checkpointing
export ENABLE_CHECKPOINTING=1
```

CPU Optimization:
```bash
# Set number of workers for data loading
export NUM_WORKERS=4
export OMP_NUM_THREADS=8
```

Error Handling
--------------

Common Errors:

1. "CUDA out of memory"
   Solution: Reduce batch size
   ```bash
   python scripts/train_mlp.py --batch-size 32
   ```

2. "NSD index not found"
   Solution: Build indices first
   ```bash
   python scripts/build_full_index.py
   ```

3. "CLIP cache missing"
   Solution: Build cache
   ```bash
   python scripts/build_clip_cache.py
   ```

4. "Checkpoint not found"
   Solution: Check path or train model first
   ```bash
   ls checkpoints/mlp/
   ```

Logging
-------

Log Structure:
```
logs/
  ├── training/
  │   └── mlp_subj01_20251207_120530.log
  ├── evaluation/
  │   └── eval_subj01_20251207_130045.log
  └── reconstruction/
      └── recon_20251207_140010.log
```

Log Levels:
- DEBUG: Detailed information for debugging
- INFO: General informational messages
- WARNING: Warning messages (non-critical)
- ERROR: Error messages
- CRITICAL: Critical errors (program termination)

Viewing Logs:
```bash
# Real-time monitoring
tail -f logs/training/mlp_subj01_*.log

# Search for errors
grep "ERROR" logs/training/*.log

# View last 100 lines
tail -n 100 logs/training/mlp_subj01_*.log
```

Monitoring & Alerts
-------------------

TensorBoard (optional):
```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard/

# View at http://localhost:6006
```

Weights & Biases (optional):
```bash
# Initialize W&B
wandb login

# Training will automatically log to W&B
python scripts/train_mlp.py --use-wandb
```

Email Alerts (optional):
```bash
# Configure email alerts
export ALERT_EMAIL=user@example.com

# Training script will send alerts on completion/failure
./scripts/deployment/run_training.sh
```

Backup & Recovery
-----------------

Checkpoint Backup:
```bash
# Backup checkpoints to S3 (example)
aws s3 sync checkpoints/ s3://my-bucket/fmri2img/checkpoints/

# Or use rsync
rsync -avz checkpoints/ backup_server:/backups/checkpoints/
```

Recovery:
```bash
# Resume from checkpoint
python scripts/train_mlp.py \
    --resume checkpoints/mlp/last.pt

# Or specify epoch
python scripts/train_mlp.py \
    --resume checkpoints/mlp/epoch_50.pt
```

Multi-Node Deployment (Advanced)
---------------------------------

Distributed Training:
```bash
# On node 0 (master)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train_mlp.py

# On node 1
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train_mlp.py
```

Performance Benchmarks
----------------------

Model       | Training Time | GPU Memory | Throughput
------------|---------------|------------|------------
Ridge       | 5 min         | N/A        | N/A
MLP         | 2 hours       | 8GB        | 1000 samples/s
Two-Stage   | 4 hours       | 12GB       | 600 samples/s
Adapter     | 1 hour        | 6GB        | 800 samples/s

*Benchmarks on NVIDIA A100 40GB, batch size 64*

Security Considerations
-----------------------

1. Environment Variables:
   - Never commit .env files with secrets
   - Use .env.example as template
   - Keep API keys in secure storage

2. Data Privacy:
   - NSD data should be stored securely
   - Follow data usage agreements
   - Anonymize outputs if needed

3. Model Checkpoints:
   - Don't commit large checkpoints to git
   - Use Git LFS or external storage
   - Version checkpoints appropriately

Related Documentation
---------------------
- Quick Start: ../../START_HERE.md
- Usage Examples: ../../USAGE_EXAMPLES.md
- Training Guides: ../../docs/guides/
- Configuration: ../../configs/README.md

Support
-------
For deployment issues:
1. Check logs in logs/ directory
2. Review error handling section above
3. Consult USAGE_EXAMPLES.md
4. Open GitHub issue with details

Maintainers
-----------
Bachelor Thesis Project
Last Updated: December 7, 2025
Status: Production-Ready
"""
